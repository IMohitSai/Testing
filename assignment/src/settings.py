from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Keep it optional to prevent config errors from crashing cold starts.
    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")

    openrouter_model: str = Field("openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL")
    openrouter_fallback_model: str = Field(
        "openai/gpt-oss-120b:free", alias="OPENROUTER_FALLBACK_MODEL"
    )

    # Optional attribution headers for OpenRouter
    site_url: str | None = Field(None, alias="OPENROUTER_SITE_URL")
    app_name: str | None = Field("PlacementSprint", alias="OPENROUTER_APP_NAME")
