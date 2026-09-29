"""Charon's agent loop: the model answers or calls tools, and tool results go back to the model."""
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from app.agent.prompts import SYSTEM_PROMPT


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool] = (), *,
                checkpointer: BaseCheckpointSaver | None = None,
                store: BaseStore | None = None) -> CompiledStateGraph:
    """Compile the agent.

    A checkpointer keeps each thread's messages between calls (short-term memory);
    a store holds facts shared across threads (long-term memory). Both are optional.
    """
    tools = list(tools)
    bound = model.bind_tools(tools) if tools else model

    async def agent(state: MessagesState) -> dict[str, Any]:
        response = await bound.ainvoke([SystemMessage(SYSTEM_PROMPT), *state['messages']])
        return {'messages': [response]}

    builder = StateGraph(MessagesState)
    builder.add_node('agent', agent)
    builder.add_edge(START, 'agent')
    if tools:
        builder.add_node('tools', ToolNode(tools))
        builder.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: END})
        builder.add_edge('tools', 'agent')
    else:
        builder.add_edge('agent', END)
    return builder.compile(checkpointer=checkpointer, store=store)
