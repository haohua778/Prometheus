"""Offline test doubles. No test calls a live model."""
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.config import Settings


class ScriptedChatModel(BaseChatModel):
    """Returns scripted replies in order and records every prompt it was given."""

    replies: list[AIMessage]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> 'ScriptedChatModel':
        self.bound_tools = [item.name for item in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.prompts.append(list(messages))
        reply = self.replies[min(len(self.prompts), len(self.replies)) - 1]
        # A fresh copy per call: reusing one message id would make the graph replace, not append.
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy(deep=True, update={'id': None}))])


def tool_call(name: str, args: dict[str, Any], call_id: str = 'call_1') -> AIMessage:
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id}])


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def offline_settings() -> Settings:
    return Settings(_env_file=None, llm_api_key=None)
