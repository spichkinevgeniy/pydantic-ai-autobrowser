import logging
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client | None:
    if not settings.GOOGLE_API_KEY:
        return None
    return genai.Client(api_key=settings.GOOGLE_API_KEY)


class ImageAnalyzer:
    ss_analysis_history: list[str] = []

    def __init__(self, image1_path: Path, image2_path: Path, next_step: str) -> None:
        self.image1_path = Path(image1_path)
        self.image2_path = Path(image2_path)
        self.next_step = next_step
        self.client: genai.Client | None = None

    @classmethod
    def get_formatted_history(cls) -> str:
        if not cls.ss_analysis_history:
            return "No previous SS analysis responses."

        formatted_history = "Previous SS Analysis Responses:\n"
        for idx, response in enumerate(cls.ss_analysis_history, 1):
            formatted_history += f"{idx}. {response}\n\n"
        return formatted_history.strip()

    @classmethod
    def clear_history(cls) -> None:
        cls.ss_analysis_history.clear()

    def _validate_images(self) -> None:
        for path in (self.image1_path, self.image2_path):
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            try:
                with Image.open(path) as image:
                    image.load()
            except Exception as exc:
                raise ValueError(f"Invalid image file {path}: {exc}") from exc

    def _build_prompt(self) -> str:
        history_str = self.get_formatted_history()
        return f"""
You are an excellent screenshot analysis agent. Analyze these two webpage screenshots in detail, considering that this was the action that was intended to be performed next: {self.next_step}.

Previous SS Analysis Responses:
{history_str}

<rules>
1. You have been provided 2 screenshots, one is the state of the webpage before the action was performed and the other is the state of the webpage after the action was performed.
2. If the action was successfully performed, you should be able to see the expected changes in the webpage.
3. We do not need a generic description of what changed. We need an inference on whether the intended action was successfully performed or not.
4. If the action was successfully performed, explicitly say so and describe the visible changes that confirm it.
5. If the action was not successfully performed, explicitly say so and describe the visible evidence that indicates failure or incomplete execution.
6. Confirm whether the current action caused any previous actions to fail or not satisfy the request.
7. If the Browser Agent says the action succeeded, visually confirm whether it actually did.

<special_case>
1. If the action is searching through an API and the webpage does not visibly change, state that the screenshot was unchanged.
2. In that search case, an unchanged screenshot does not automatically mean failure.
</special_case>

<output_rules>
1. Explicitly mention whether the action was successfully performed.
2. Check whether what we actually wanted was achieved.
3. Specify whether any new elements appeared or disappeared.
4. Mention elements related to the intended action.
5. If the current action caused previous actions to fail, explicitly mention which elements were affected and how.
6. Explicitly confirm whether the Browser Agent's success claim matches the visual evidence.
</output_rules>
</rules>
""".strip()

    async def analyze_images(self) -> str:
        """Analyze before/after screenshots with Gemini multimodal input."""
        if not settings.ENABLE_SS_ANALYSIS:
            logger.info("Screenshot analysis disabled in settings")
            return ""

        self.client = get_gemini_client()
        if self.client is None:
            logger.warning("Skipping screenshot analysis because GOOGLE_API_KEY is not configured")
            return ""

        try:
            self._validate_images()
        except Exception:
            logger.exception("Screenshot validation failed")
            return ""

        try:
            response = await self.client.aio.models.generate_content(
                model=settings.SCREENSHOT_ANALYSIS_MODEL,
                contents=[
                    types.Part.from_text(text=self._build_prompt()),
                    types.Part.from_bytes(data=self.image1_path.read_bytes(), mime_type="image/png"),
                    types.Part.from_bytes(data=self.image2_path.read_bytes(), mime_type="image/png"),
                ],
            )
        except Exception:
            logger.exception("Gemini screenshot analysis failed")
            return ""

        analysis = (response.text or "").strip()
        if analysis:
            self.ss_analysis_history.append(analysis)
            self.ss_analysis_history[:] = self.ss_analysis_history[-5:]
        return analysis
