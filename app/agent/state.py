"""Agent state fields and initialization for a new chat turn.

TypedDict describes the structure; it does not validate values or supply defaults.
A checkpointer saves these fields, including the per-turn review state.
"""
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Merge new messages into the thread history; matching message IDs are replaced.
    messages: Annotated[list[AnyMessage], add_messages]
    # Reply awaiting review; kept out of messages until reflect commits it. None once committed.
    draft: NotRequired[AIMessage | None]
    # Latest reply committed to the analyst; with a checkpointer it is the previous turn's reply
    # until the current turn commits. Absent before the first commit.
    answer: NotRequired[str]
    # Validated feedback from the latest review; empty on PASS, reset to None for a new turn.
    review: NotRequired[str | None]
    # continue requests another revision; stop accepts the reply.
    decision: NotRequired[Literal['continue', 'stop'] | None]
    # Number of valid reviews completed this turn; format retries do not count.
    iteration: NotRequired[int]


def create_turn_state(message: str) -> AgentState:
    """Build the current question's input with fresh draft and review fields.

    When configured, a checkpointer restores state; add_messages merges the new message.
    Use this function for each new turn, including direct calls to a graph with a checkpointer.
    The graph does not automatically reset fields omitted from its input.
    """
    return {
        'messages': [HumanMessage(message)],
        'draft': None,
        'review': None,
        'decision': None,
        'iteration': 0,
    }
