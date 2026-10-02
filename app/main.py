"""HTTP entry point: a health check and one chat endpoint over the agent graph."""
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from openai import APIError, APITimeoutError
from pydantic import BaseModel, Field

from app.agent.graph import build_graph
from app.agent.tools import TOOLS
from app.config import Settings, get_settings
from app.llm import ModelUnavailable, build_chat_model

app = FastAPI(title='Charon')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    thread_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


def get_agent(settings: Annotated[Settings, Depends(get_settings)]) -> CompiledStateGraph:
    try:
        return build_graph(build_chat_model(settings), TOOLS)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail={'code': exc.code}) from exc


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest,
               agent: Annotated[CompiledStateGraph, Depends(get_agent)],
               settings: Annotated[Settings, Depends(get_settings)]) -> ChatResponse:
    thread_id = request.thread_id or uuid4().hex
    config = {'configurable': {'thread_id': thread_id}, 'recursion_limit': settings.max_steps}
    # Each turn starts a fresh review cycle; with a checkpointer these would otherwise carry over.
    turn = {'messages': [HumanMessage(request.message)], 'iteration': 0, 'review': None, 'decision': None}
    try:
        state = await agent.ainvoke(turn, config=config)
    except GraphRecursionError as exc:
        raise HTTPException(status_code=500, detail={'code': 'step_limit_reached'}) from exc
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail={'code': 'upstream_timeout'}) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail={'code': 'upstream_error'}) from exc
    return ChatResponse(thread_id=thread_id, reply=state['answer'])
