"""Конфигурация приложения и чтение настроек из окружения и `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash", alias="MODEL_NAME")

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )




settings = Settings()
