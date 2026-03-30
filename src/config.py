"""Конфигурация приложения и чтение настроек из окружения и `.env`."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    GOOGLE_API_KEY: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    MODEL_NAME: str = Field(default="gemini-2.5-flash", alias="MODEL_NAME")
    MODEL_TIMEOUT_SECONDS: float = Field(default=30.0, alias="MODEL_TIMEOUT_SECONDS")
    LOGFIRE_TOKEN: str | None = Field(default=None, alias="LOGFIRE_TOKEN")
    LOGFIRE_SERVICE_NAME: str = Field(default="auto-browser-demo2", alias="LOGFIRE_SERVICE_NAME")

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
