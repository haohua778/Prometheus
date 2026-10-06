"""HTTP entry point: a health check and one chat endpoint over the agent graph."""
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
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

app = FastAPI(title='Prometheus')


def get_agent_graph(settings: Annotated[Settings, Depends(get_settings)]) -> CompiledStateGraph:
    try:
        return build_graph(build_chat_model(settings), TOOLS)
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=503, detail={'code': 'missing_api_key'}) from exc


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest,
               agent_graph: Annotated[CompiledStateGraph, Depends(get_agent_graph)],
               settings: Annotated[Settings, Depends(get_settings)]) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    run_config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': settings.max_steps}
    turn_input = create_turn_state(request.message)
    try:
        final_state = await agent_graph.ainvoke(turn_input, config=run_config)
    except GraphRecursionError as exc:
        raise HTTPException(status_code=500, detail={'code': 'step_limit_reached'}) from exc
    except ReviewFormatError as exc:
        raise HTTPException(status_code=502, detail={'code': 'review_output_invalid'}) from exc
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail={'code': 'upstream_timeout'}) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail={'code': 'upstream_error'}) from exc
    return ChatResponse(thread_id=thread_id, reply=final_state['answer'])
