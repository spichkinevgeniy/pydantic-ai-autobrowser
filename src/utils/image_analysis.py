import logging
from asyncio import wait_for
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image
from src.config import settings

logger = logging.getLogger(__name__)

SCREENSHOT_ANALYSIS_TIMEOUT_SECONDS = 25.0


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
3. We do not need generic description of what you see in the screenshots that has changed, we need the information and inference on whether the action was successfully performed or not.
4. If the action was successfully performed, then you need to convey that information and along with that information, you also need to provide information on what changes you see in the screenshots that might have resulted from the action.
5. If the action was not successfully performed, then you need to convey that information and along with that information, you also need to provide information on what changes you see in the screenshots that might have resulted from the action that indicate the tool call was not executed.
6. You also need to confirm whether the current action caused any previous actions to fail or not satisfy the request. For example : Entering text in the first field and pressing enter can take us to the next field but any failure in the first field that has occured should be reported.
7. The Browser Agent will execute an action such an entering fields or clicking buttons and then say that "the action was successfull" but you have to visually confirm whether the text was entered and completed or it has just been entered and we need further action to complete the text entry.

<special_case>
1. One special case is that when the action is searching, we are using SERP API so it will be that the webpage does not change at all. In that case, you need to provide information that the screenshot was unchanged.
2. So if the action is searching then you need to provide information that the SS was unchanged. The screenshot being unchanges in the case of search is a special case and does not conclude failure of the search action.
</special_case>

<output rules>
1. You need to explicitly mention whether the action was successfully performed or not.
2. You need to check in with what we actually wanted and if that was achieved according to the changes in the screenshots.
3. You also need to specify if any new elements have appeared or any elements have disappeared.
4. You need to also explictly mention about any elements related to the action or the element of interest in the screenshots.
5. If the current action caused any previous actions to fail, then you need to explicitly mention that and tell the Critique exactly what fields were affected and in precisely what manner.
6. You also need to explicitly confirm and reassure the Critique that if Browser Agent is saying this action was executed successfully, then was it actually executed successfully or not.

</output_rules>
</rules>
"""

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
            response = await wait_for(
                self.client.aio.models.generate_content(
                    model=settings.SCREENSHOT_ANALYSIS_MODEL,
                    contents=[
                        types.Part.from_text(text=self._build_prompt()),
                        types.Part.from_bytes(
                            data=self.image1_path.read_bytes(),
                            mime_type="image/png",
                        ),
                        types.Part.from_bytes(
                            data=self.image2_path.read_bytes(),
                            mime_type="image/png",
                        ),
                    ],
                ),
                timeout=SCREENSHOT_ANALYSIS_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Gemini screenshot analysis failed")
            return ""

        analysis = (response.text or "").strip()
        if analysis:
            self.ss_analysis_history.append(analysis)
            self.ss_analysis_history[:] = self.ss_analysis_history[-5:]
        return analysis
