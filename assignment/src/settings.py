from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    openrouter_model: str = Field("openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL")
    openrouter_fallback_model: str = Field("openai/gpt-oss-120b:free", alias="OPENROUTER_FALLBACK_MODEL")

    app_name: str = Field("PlacementSprint", alias="OPENROUTER_APP_NAME")
    site_url: str | None = Field(None, alias="OPENROUTER_SITE_URL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
