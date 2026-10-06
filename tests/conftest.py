"""Offline test doubles. No test calls a live model."""
import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from app.config import Settings


class ScriptedChatModel(BaseChatModel):
    """Record prompts and return scripted replies or errors, repeating the last when exhausted."""

    replies: list[Exception | AIMessage]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'scripted'

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> 'ScriptedChatModel':
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.prompts.append(list(messages))
        reply_index = min(len(self.prompts), len(self.replies)) - 1
        scripted_reply = self.replies[reply_index]
        if isinstance(scripted_reply, Exception):
            raise scripted_reply
        # A fresh copy per call: reusing one message id would make the graph replace, not append.
        response_message = scripted_reply.model_copy(deep=True, update={'id': None})
        return ChatResult(generations=[ChatGeneration(message=response_message)])


def make_review_message(verdict: str = 'PASS', feedback: str = '') -> AIMessage:
    return AIMessage(json.dumps({'verdict': verdict, 'feedback': feedback}, ensure_ascii=False))


def make_tool_call_message(tool_name: str, tool_arguments: dict[str, Any], tool_call_id: str = 'call_1') -> AIMessage:
    return AIMessage(content='', tool_calls=[{'name': tool_name, 'args': tool_arguments, 'id': tool_call_id}])


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
