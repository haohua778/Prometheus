"""Prometheus's agent loop: the model answers or calls tools, and tool results go back to the model.

A plain reply is held as a draft while the model reviews it. Only the accepted draft, or the last
draft once the review limit is reached, is committed to the thread history.
"""
from collections.abc import Sequence
import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from app.agent.prompts import REVISION_PROMPT, SYSTEM_PROMPT
from app.agent.review import review_draft
from app.agent.state import AgentState
from app.runtime_logging import log_event

# Reviews per turn; after the last one the current draft is committed whatever the verdict.
MAX_REVIEWS_PER_TURN = 3


def _require_draft(state: AgentState, node_name: str) -> AIMessage:
    draft = state.get('draft')
    if draft is None:
        raise RuntimeError(f'{node_name} expected a draft reply, but the state has none')
    return draft


def _needs_another_revision(decision: Literal['continue', 'stop'] | None, review_count: int) -> bool:
    return decision == 'continue' and review_count < MAX_REVIEWS_PER_TURN


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool] = (), *,
                checkpointer: BaseCheckpointSaver | None = None,
                store: BaseStore | None = None) -> CompiledStateGraph:
    """Compile the agent and its review step.

    An optional checkpointer saves each thread's state, including messages and review fields.
    The optional store is available to graph nodes for cross-thread data; current nodes do not use it.
    """
    available_tools = list(tools)
    agent_model = model.bind_tools(available_tools) if available_tools else model

    async def generate_response(state: AgentState) -> dict[str, Any]:
        prompt_messages = [SystemMessage(SYSTEM_PROMPT), *state['messages']]
        if state.get('decision') == 'continue':
            # The rejected draft and the reviewer's notes go into this prompt only, not into the thread history.
            draft = _require_draft(state, 'agent')
            prompt_messages.append(HumanMessage(REVISION_PROMPT.format(draft=draft.text, review=state['review'])))
        model_response = await agent_model.ainvoke(prompt_messages)
        if model_response.tool_calls:
            # Tool calls join the history now; the draft and review notes stay until the revision is written.
            return {'messages': [model_response]}
        return {'draft': model_response}

    async def review_response(state: AgentState) -> dict[str, Any]:
        draft = _require_draft(state, 'reflect')
        # Raises ReviewFormatError on a malformed verdict; an unreadable review never passes the draft.
        review = await review_draft(model, state['messages'], draft)
        review_decision = 'continue' if review.verdict == 'REVISE' else 'stop'
        review_count = state.get('iteration', 0) + 1
        limit_reached = review.verdict == 'REVISE' and review_count >= MAX_REVIEWS_PER_TURN
        log_event('review.limit_reached' if limit_reached else 'review.completed',
                  level=logging.WARNING if limit_reached else logging.INFO,
                  verdict=review.verdict, review_count=review_count)
        review_outcome = {'review': review.feedback, 'decision': review_decision, 'iteration': review_count}
        if _needs_another_revision(review_decision, review_count):
            return review_outcome
        return {**review_outcome, 'messages': [draft], 'answer': str(draft.text), 'draft': None}

    def route_after_review(state: AgentState) -> str:
        if _needs_another_revision(state['decision'], state['iteration']):
            return 'agent'
        return END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node('agent', generate_response)
    graph_builder.add_node('reflect', review_response)
    graph_builder.add_edge(START, 'agent')
    if available_tools:
        graph_builder.add_node('tools', ToolNode(available_tools))
        # A plain reply only updates draft, so messages[-1] is then the user's message or a finished
        # tool result and tools_condition routes to reflect; only new tool calls are written to messages
        # and route to tools.
        graph_builder.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: 'reflect'})
        graph_builder.add_edge('tools', 'agent')
    else:
        graph_builder.add_edge('agent', 'reflect')
    graph_builder.add_conditional_edges('reflect', route_after_review, ['agent', END])
    return graph_builder.compile(checkpointer=checkpointer, store=store)
