"""Runtime settings for the LLM-backed agent."""
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / '.env', env_file_encoding='utf-8',
                                      extra='ignore', populate_by_name=True)

    # Any OpenAI-compatible endpoint. The older KIMI_* names are still read.
    llm_api_key: SecretStr | None = Field(default=None, validation_alias=AliasChoices('LLM_API_KEY', 'KIMI_API_KEY'))
    llm_base_url: str | None = Field(default=None, validation_alias=AliasChoices('LLM_BASE_URL', 'KIMI_BASE_URL'))
    llm_model: str = Field(default='kimi-k2.6', validation_alias=AliasChoices('LLM_MODEL', 'KIMI_MODEL'))
    # None leaves the provider default; some reasoning models reject a fixed temperature.
    llm_temperature: float | None = Field(default=None, ge=0, le=2,
                                          validation_alias=AliasChoices('LLM_TEMPERATURE'))
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300,
                                       validation_alias=AliasChoices('LLM_TIMEOUT_SECONDS'))
    # Graph steps per chat turn; each model call and each tool round is one step.
    max_steps: int = Field(default=20, ge=3, le=50, validation_alias=AliasChoices('PROMETHEUS_MAX_STEPS'))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
