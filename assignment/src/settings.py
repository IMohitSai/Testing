from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Environment variables (set locally in .env, and on Vercel in Project Settings):

    OPENROUTER_API_KEY=...
    OPENROUTER_MODEL=openai/gpt-oss-20b:free
    OPENROUTER_FALLBACK_MODEL=openai/gpt-oss-120b:free
    OPENROUTER_SITE_URL=https://your-vercel-domain.vercel.app   (optional)
    OPENROUTER_APP_NAME=PlacementSprint                          (optional)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Keep optional to avoid cold-start crashes if env isn't set yet.
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    openrouter_model: str = Field(
        default="openai/gpt-oss-20b:free",
        alias="OPENROUTER_MODEL",
    )
    openrouter_fallback_model: str = Field(
        default="openai/gpt-oss-120b:free",
        alias="OPENROUTER_FALLBACK_MODEL",
    )

    # Optional attribution headers for OpenRouter
    site_url: str | None = Field(default=None, alias="OPENROUTER_SITE_URL")
    app_name: str | None = Field(default="PlacementSprint", alias="OPENROUTER_APP_NAME")
