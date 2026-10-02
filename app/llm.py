"""The one place that builds the chat model, so providers can change without touching the graph."""
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings


class MissingAPIKeyError(RuntimeError):
    """No API key was configured for the chat model."""


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise MissingAPIKeyError('LLM_API_KEY is required')
    return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                      temperature=settings.llm_temperature, timeout=settings.llm_timeout_seconds, max_retries=1)
