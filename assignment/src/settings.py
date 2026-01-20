from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    # Pick models you actually have access to in OpenRouter (free models may be blocked by privacy settings)
    openrouter_model: str = Field(default="openai/gpt-oss-20b:free", alias="OPENROUTER_MODEL")
    openrouter_fallback_model: str = Field(default="openai/gpt-oss-120b:free", alias="OPENROUTER_FALLBACK_MODEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
