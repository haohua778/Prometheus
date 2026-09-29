"""The one place that builds the chat model, so providers can change without touching the graph."""
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings


class ModelUnavailable(RuntimeError):
    """The live model cannot be used; `code` is safe to show to API callers."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def build_chat_model(settings: Settings) -> BaseChatModel:
    if not settings.allow_live:
        raise ModelUnavailable('live_calls_disabled')
    if settings.llm_api_key is None or not settings.llm_api_key.get_secret_value().strip():
        raise ModelUnavailable('missing_api_key')
    return ChatOpenAI(model=settings.llm_model, api_key=settings.llm_api_key, base_url=settings.llm_base_url,
                      temperature=settings.llm_temperature, timeout=settings.llm_timeout_seconds, max_retries=1)
