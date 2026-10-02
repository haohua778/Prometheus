"""Charon's agent loop: the model answers or calls tools, and tool results go back to the model.

Before a final answer is returned, the model reviews it and may send it back for revision.
"""
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, get_buffer_string
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore

from app.agent.prompts import REVIEW_PROMPT, REVISION_PROMPT, SYSTEM_PROMPT

# Reviews per turn; after the last one the current answer is returned whatever the verdict.
MAX_ITERATIONS = 3


class AgentState(MessagesState):
    answer: str                                   # the reply under review
    review: str | None                            # reviewer's notes
    decision: Literal['continue', 'stop'] | None  # 'continue' = revise again, 'stop' = accept
    iteration: int                                # reviews done this turn


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool] = (), *,
                checkpointer: BaseCheckpointSaver | None = None,
                store: BaseStore | None = None) -> CompiledStateGraph:
    """Compile the agent and its review step.

    A checkpointer keeps each thread's messages between calls (short-term memory);
    a store holds facts shared across threads (long-term memory). Both are optional.
    """
    tools = list(tools)
    bound = model.bind_tools(tools) if tools else model

    async def agent(state: AgentState) -> dict[str, Any]:
        prompt = [SystemMessage(SYSTEM_PROMPT), *state['messages']]
        if state.get('decision') == 'continue':
            # Reviewer feedback goes into this prompt only, not into the thread history.
            prompt.append(HumanMessage(REVISION_PROMPT.format(review=state['review'])))
        response = await bound.ainvoke(prompt)
        return {'messages': [response]}

    async def reflect(state: AgentState) -> dict[str, Any]:
        transcript = get_buffer_string(state['messages'], format='xml')
        response = await model.ainvoke([SystemMessage(REVIEW_PROMPT), HumanMessage(transcript)])
        text = str(response.text).strip()
        # Fail open: anything but an explicit REVISE accepts the answer.
        decision = 'continue' if text.upper().startswith('REVISE') else 'stop'
        return {'answer': str(state['messages'][-1].text), 'review': text, 'decision': decision,
                'iteration': state.get('iteration', 0) + 1}

    def after_review(state: AgentState) -> str:
        return 'agent' if state['decision'] == 'continue' and state['iteration'] < MAX_ITERATIONS else END

    builder = StateGraph(AgentState)
    builder.add_node('agent', agent)
    builder.add_node('reflect', reflect)
    builder.add_edge(START, 'agent')
    if tools:
        builder.add_node('tools', ToolNode(tools))
        builder.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: 'reflect'})
        builder.add_edge('tools', 'agent')
    else:
        builder.add_edge('agent', 'reflect')
    builder.add_conditional_edges('reflect', after_review, ['agent', END])
    return builder.compile(checkpointer=checkpointer, store=store)
