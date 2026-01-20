from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Keep API key optional so the app can boot on Vercel even if env vars are mis-set.
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    openrouter_model: str = Field(default="openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL")
    openrouter_fallback_model: str = Field(default="openai/gpt-oss-120b:free", alias="OPENROUTER_FALLBACK_MODEL")

    # Optional attribution (OpenRouter works fine without these)
    app_name: str | None = Field(default="PlacementSprint", alias="OPENROUTER_APP_NAME")
    site_url: str | None = Field(default=None, alias="OPENROUTER_SITE_URL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
