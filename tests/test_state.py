import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import MAX_REVIEWS_PER_TURN, build_graph
from app.agent.state import create_turn_state
from tests.conftest import ScriptedChatModel, make_review_message


def test_new_turn_starts_without_a_draft_or_review():
    state = create_turn_state('new question')

    assert [message.content for message in state['messages']] == ['new question']
    assert state['draft'] is None
    assert state['review'] is None
    assert state['decision'] is None
    assert state['iteration'] == 0


@pytest.mark.anyio
async def test_new_turn_keeps_history_and_gets_its_own_review_cycle():
    first_turn_drafts = [f'old draft {number}' for number in range(1, MAX_REVIEWS_PER_TURN + 1)]
    first_turn_replies = [
        message for draft in first_turn_drafts
        for message in (AIMessage(draft), make_review_message('REVISE', 'Fix the old answer.'))
    ]
    model = ScriptedChatModel(
        replies=first_turn_replies
        + [AIMessage('new draft'), make_review_message('REVISE', 'Add units.'), AIMessage('new final'), make_review_message()]
    )
    graph = build_graph(model, checkpointer=InMemorySaver())
    run_config = {'configurable': {'thread_id': 'same-thread'}}

    await graph.ainvoke(create_turn_state('first question'), config=run_config)
    final_state = await graph.ainvoke(create_turn_state('second question'), config=run_config)

    second_turn_prompt = model.prompts[2 * MAX_REVIEWS_PER_TURN]
    assert [message.content for message in second_turn_prompt[1:]] == [
        'first question', first_turn_drafts[-1], 'second question'
    ]
    assert [message.content for message in final_state['messages']] == [
        'first question', first_turn_drafts[-1], 'second question', 'new final'
    ]
    assert final_state['answer'] == 'new final'
    assert final_state['draft'] is None
    assert final_state['iteration'] == 2
    assert len(model.prompts) == 2 * MAX_REVIEWS_PER_TURN + 4
