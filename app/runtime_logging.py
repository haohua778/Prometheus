"""Structured operational events, without prompts, tool arguments or provider error bodies."""
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

request_id_context: ContextVar[str | None] = ContextVar('request_id', default=None)
thread_id_context: ContextVar[str | None] = ContextVar('thread_id', default=None)
logger = logging.getLogger('prometheus')


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'event': record.getMessage(),
            'request_id': request_id_context.get(),
            'thread_id': thread_id_context.get(),
            **getattr(record, 'event_fields', {}),
        }, ensure_ascii=False)


def configure_logging(level: str) -> None:
    # Configure only our namespace; leave Uvicorn's handlers under the host's control.
    if not any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={'event_fields': fields})


class RuntimeLogCallback(AsyncCallbackHandler):
    """Observe actual model/tool execution through LangChain's existing callback contract."""

    def __init__(self) -> None:
        self.started: dict[UUID, float] = {}

    def _start(self, kind: str, run_id: UUID, parent_run_id: UUID | None, **fields: Any) -> None:
        self.started[run_id] = perf_counter()
        log_event(f'{kind}.started', span_id=str(run_id),
                  parent_span_id=str(parent_run_id) if parent_run_id else None, **fields)

    def _end(self, kind: str, run_id: UUID, *, error: BaseException | None = None,
             failed: bool = False, **fields: Any) -> None:
        started = self.started.pop(run_id, None)
        failed = failed or error is not None
        log_event(f'{kind}.failed' if failed else f'{kind}.completed',
                  level=logging.ERROR if failed else logging.INFO, span_id=str(run_id),
                  duration_ms=round((perf_counter() - started) * 1000, 2) if started is not None else None,
                  **({'error_type': type(error).__name__} if error else {}), **fields)

    async def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *,
                                  run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._start('llm', run_id, parent_run_id)

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        # Copy only numeric token counters; response metadata may contain provider payloads.
        tokens: dict[str, int] = {}
        if response.generations and response.generations[0]:
            message = getattr(response.generations[0][0], 'message', None)
            usage = getattr(message, 'usage_metadata', None) or {}
            tokens = {key: usage[key] for key in ('input_tokens', 'output_tokens', 'total_tokens')
                      if type(usage.get(key)) is int}
        self._end('llm', run_id, **tokens)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end('llm', run_id, error=error)

    async def on_tool_start(self, serialized: dict[str, Any], input_str: str, *,
                            run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        self._start('tool', run_id, parent_run_id, tool_name=serialized.get('name'))

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        # ToolNode may turn a tool failure into a ToolMessage instead of raising it.
        if getattr(output, 'status', None) == 'error':
            self._end('tool', run_id, failed=True, outcome='error_result')
        else:
            self._end('tool', run_id)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end('tool', run_id, error=error)
