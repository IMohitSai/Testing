from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Environment variables (set locally in .env, and on Vercel in Project Settings):

    OPENROUTER_API_KEY=...
    OPENROUTER_MODEL=meta-llama/llama-3.2-3b-instruct:free
    OPENROUTER_FALLBACK_MODEL=deepseek/deepseek-chat-v3.1:free
    OPENROUTER_SITE_URL=https://your-vercel-domain.vercel.app   (optional, for OpenRouter attribution)
    OPENROUTER_APP_NAME=PlacementSprint                          (optional, for OpenRouter attribution)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
    "openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL"
)
    openrouter_fallback_model: str = Field(
    "openai/gpt-oss-120b:free", alias="OPENROUTER_FALLBACK_MODEL"
)


    # Optional attribution headers for OpenRouter leaderboards
    site_url: str | None = Field(None, alias="OPENROUTER_SITE_URL")
    app_name: str | None = Field("PlacementSprint", alias="OPENROUTER_APP_NAME")
