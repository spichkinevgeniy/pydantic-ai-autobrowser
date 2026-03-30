"""Application configuration loaded from environment variables and `.env`."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    GOOGLE_API_KEY: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    MODEL_NAME: str = Field(default="gemini-2.5-flash", alias="MODEL_NAME")
    TRANSIENT_RETRY_ATTEMPTS: int = Field(default=3, alias="TRANSIENT_RETRY_ATTEMPTS")
    TRANSIENT_RETRY_BASE_DELAY_SECONDS: float = Field(
        default=5.0,
        alias="TRANSIENT_RETRY_BASE_DELAY_SECONDS",
    )
    SCREENSHOT_ANALYSIS_MODEL: str = Field(
        default="gemini-2.5-flash",
        alias="SCREENSHOT_ANALYSIS_MODEL",
    )
    LOGFIRE_TOKEN: str | None = Field(default=None, alias="LOGFIRE_TOKEN")
    LOGFIRE_SERVICE_NAME: str = Field(default="auto-browser-demo2", alias="LOGFIRE_SERVICE_NAME")

    SCREENSHOT_DIR: Path = Field(
        default=ROOT_DIR / "logs" / "screenshots",
        alias="SCREENSHOT_DIR",
    )
    ENABLE_SCREENSHOTS: bool = Field(default=True, alias="ENABLE_SCREENSHOTS")
    ENABLE_SS_ANALYSIS: bool = Field(default=True, alias="ENABLE_SS_ANALYSIS")
    SCREENSHOT_FULL_PAGE: bool = Field(default=False, alias="SCREENSHOT_FULL_PAGE")
    PLAYWRIGHT_USER_DATA_DIR: Path = Field(
        default=ROOT_DIR / "logs" / "playwright-user-data",
        alias="PLAYWRIGHT_USER_DATA_DIR",
    )

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
