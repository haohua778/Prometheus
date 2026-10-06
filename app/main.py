"""HTTP entry point: a health check and one chat endpoint over the agent graph."""
import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APITimeoutError

from app.agent.graph import build_graph
from app.agent.review import ReviewFormatError
from app.agent.state import create_turn_state
from app.agent.tools import TOOLS
from app.config import Settings, get_settings
from app.llm import MissingAPIKeyError, build_chat_model
from app.schemas import ChatRequest, ChatResponse
from app.runtime_logging import (RuntimeLogCallback, configure_logging, log_event,
                                 request_id_context, thread_id_context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log_event('service.started', model=settings.llm_model, tool_count=len(TOOLS), max_steps=settings.max_steps)
    try:
        yield
    finally:
        log_event('service.stopped')


app = FastAPI(title='Prometheus', lifespan=lifespan)


@app.middleware('http')
async def request_logging(request: Request, call_next):
    request_id = uuid4().hex
    token = request_id_context.set(request_id)
    started = perf_counter()
    status_code = 500
    log_event('http.started', method=request.method)
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers['X-Request-ID'] = request_id
        return response
    except asyncio.CancelledError:
        status_code = 499
        log_event('http.cancelled', level=logging.WARNING)
        raise
    except Exception as exc:
        # Stack locations are useful; exception strings and locals can contain document text or secrets.
        log_event('http.failed', level=logging.ERROR, error_type=type(exc).__name__,
                  stack=[{'file': frame.filename, 'line': frame.lineno, 'function': frame.name}
                         for frame in traceback.extract_tb(exc.__traceback__)])
        raise
    finally:
        route = request.scope.get('route')
        log_event('http.completed', level=logging.ERROR if status_code >= 500 else
                  logging.WARNING if status_code >= 400 else logging.INFO,
                  method=request.method, route=getattr(route, 'path', '<unmatched>'),
                  status_code=status_code, duration_ms=round((perf_counter() - started) * 1000, 2))
        request_id_context.reset(token)


def get_agent_graph(settings: Annotated[Settings, Depends(get_settings)]) -> CompiledStateGraph:
    try:
        return build_graph(build_chat_model(settings), TOOLS)
    except MissingAPIKeyError as exc:
        log_event('chat.failed', level=logging.ERROR, error_code='missing_api_key', error_type=type(exc).__name__)
        raise HTTPException(status_code=503, detail={'code': 'missing_api_key'}) from exc


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest,
               agent_graph: Annotated[CompiledStateGraph, Depends(get_agent_graph)],
               settings: Annotated[Settings, Depends(get_settings)]) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    thread_token = thread_id_context.set(thread_id)
    run_config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': settings.max_steps,
                  'callbacks': [RuntimeLogCallback()]}
    turn_input = create_turn_state(request.message)
    started = perf_counter()
    log_event('chat.started', model=settings.llm_model, max_steps=settings.max_steps)
    try:
        final_state = await agent_graph.ainvoke(turn_input, config=run_config)
    except GraphRecursionError as exc:
        log_event('chat.failed', level=logging.ERROR, error_code='step_limit_reached', error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail={'code': 'step_limit_reached'}) from exc
    except ReviewFormatError as exc:
        log_event('chat.failed', level=logging.ERROR, error_code='review_output_invalid', error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail={'code': 'review_output_invalid'}) from exc
    except APITimeoutError as exc:
        log_event('chat.failed', level=logging.ERROR, error_code='upstream_timeout', error_type=type(exc).__name__)
        raise HTTPException(status_code=504, detail={'code': 'upstream_timeout'}) from exc
    except APIError as exc:
        log_event('chat.failed', level=logging.ERROR, error_code='upstream_error', error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail={'code': 'upstream_error'}) from exc
    else:
        log_event('chat.completed', duration_ms=round((perf_counter() - started) * 1000, 2),
                  review_count=final_state.get('iteration', 0))
        return ChatResponse(thread_id=thread_id, reply=final_state['answer'])
    finally:
        thread_id_context.reset(thread_token)
