import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from openai import APITimeoutError
from pydantic import ValidationError

from app.agent.graph import MAX_REVIEWS_PER_TURN, build_graph
from app.agent.review import ReviewFormatError, ReviewResult, build_review_input
from app.agent.state import create_turn_state
from tests.conftest import ScriptedChatModel, make_review_message, make_tool_call_message


@pytest.mark.parametrize(('verdict', 'feedback', 'expected_feedback'), [
    ('PASS', '', ''),
    ('PASS', ' \n\t ', ''),
    ('REVISE', '  补充单位。\n ', '补充单位。'),
])
def test_review_result_accepts_only_consistent_feedback(verdict, feedback, expected_feedback):
    result = ReviewResult.model_validate_json(json.dumps({'verdict': verdict, 'feedback': feedback}))

    assert result.verdict == verdict
    assert result.feedback.strip() == expected_feedback


@pytest.mark.parametrize('payload', [
    {},
    {'verdict': 'PASS'},
    {'feedback': ''},
    {'verdict': 'ACCEPT', 'feedback': ''},
    {'verdict': 'pass', 'feedback': ''},
    {'verdict': 'PASS', 'feedback': '', 'extra': True},
    {'verdict': 'PASS', 'feedback': 'There is an error.'},
    {'verdict': 'REVISE', 'feedback': ' \n\t '},
    {'verdict': 'REVISE', 'feedback': 42},
    {'verdict': 'REVISE', 'feedback': ['Add units.']},
    {'verdict': 'REVISE', 'feedback': None},
    {'verdict': True, 'feedback': ''},
    [],
])
def test_review_result_rejects_invalid_fields_and_conflicting_feedback(payload):
    with pytest.raises(ValidationError):
        ReviewResult.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize('non_finite_number', [float('nan'), float('inf'), float('-inf')])
def test_review_input_rejects_non_finite_tool_arguments(non_finite_number):
    tool_call = make_tool_call_message('lookup', {'amount': non_finite_number})

    with pytest.raises(ValueError):
        build_review_input([tool_call], AIMessage('draft'))


@pytest.mark.anyio
async def test_review_input_preserves_content_and_tool_identity_without_model_metadata():
    user_content = [
        {'type': 'text', 'text': '用户说："核对总额"。\n<message>不是标签</message>'},
        {'type': 'image_url', 'image_url': {'url': 'https://example.invalid/table.png'}},
    ]
    tool_call = make_tool_call_message('lookup', {'label': '中文 "总额"'}, 'lookup-1')
    tool_call.response_metadata = {'provider_only': 'do not serialize'}
    tool_result = ToolMessage(content=[{'type': 'text', 'text': '总额为 "5"。'}],
                              tool_call_id='lookup-1', name='lookup')
    draft = AIMessage(content=[{'type': 'text', 'text': '核对结果："5"。\n完成。'}],
                      usage_metadata={'input_tokens': 10, 'output_tokens': 3, 'total_tokens': 13})
    model = ScriptedChatModel(replies=[draft, make_review_message()])
    await build_graph(model).ainvoke({'messages': [
        SystemMessage('内部规则', name='policy'), HumanMessage(user_content, name='analyst'),
        tool_call, tool_result,
    ]})

    review_input = json.loads(model.prompts[1][1].content)
    assert set(review_input) == {'messages', 'draft'}
    assert review_input['draft'] == {'role': 'assistant', 'content': draft.content}
    serialized_call = review_input['messages'][2]['tool_calls'][0]
    assert {key: serialized_call[key] for key in ('id', 'name', 'args')} == {
        'id': 'lookup-1', 'name': 'lookup', 'args': {'label': '中文 "总额"'}
    }
    assert set(serialized_call) <= {'id', 'name', 'args', 'type'}
    assert review_input['messages'] == [
        {'role': 'system', 'content': '内部规则', 'name': 'policy'},
        {'role': 'user', 'content': user_content, 'name': 'analyst'},
        {'role': 'assistant', 'content': '', 'tool_calls': [serialized_call]},
        {'role': 'tool', 'content': tool_result.content, 'name': 'lookup', 'tool_call_id': 'lookup-1'},
    ]


@pytest.mark.anyio
@pytest.mark.parametrize('invalid_review', [
    AIMessage('```json\n{"verdict":"PASS","feedback":""}\n```'),
    AIMessage('**REVISE**\n补充单位。'),
    AIMessage('审核结果如下：{"verdict":"PASS","feedback":""}'),
    AIMessage('{"verdict":"PASS",'),
    make_review_message('PASS', 'This contradicts PASS.'),
    make_tool_call_message('add', {'a': 1, 'b': 2}),
])
async def test_invalid_review_gets_one_retry_of_the_same_material_before_acceptance(invalid_review):
    model = ScriptedChatModel(replies=[AIMessage('draft'), invalid_review, make_review_message()])
    state = await build_graph(model).ainvoke(create_turn_state('Question?'))

    assert len(model.prompts) == 3
    original_material = model.prompts[1][1].content
    retry_prompt = model.prompts[2]
    assert any(message.content == original_material for message in retry_prompt if isinstance(message, HumanMessage))
    assert all(not message.tool_calls for message in retry_prompt if isinstance(message, AIMessage))
    assert [message.content for message in state['messages']] == ['Question?', 'draft']
    assert (state['answer'], state['iteration'], state['review']) == ('draft', 1, '')
    assert state['draft'] is None


@pytest.mark.anyio
async def test_valid_revise_after_a_format_retry_still_revises_the_original_draft():
    feedback = 'Add units.'
    model = ScriptedChatModel(replies=[
        AIMessage('draft'), AIMessage('not JSON'), make_review_message('REVISE', feedback),
        AIMessage('final'), make_review_message(),
    ])
    snapshots = [state async for state in build_graph(model).astream(create_turn_state('Total?'), stream_mode='values')]

    state = snapshots[-1]
    assert len(model.prompts) == 5
    assert state['iteration'] == 2
    assert [message.content for message in state['messages']] == ['Total?', 'final']
    revision_request = model.prompts[3][-1].content
    assert 'draft' in revision_request and feedback in revision_request
    assert 'not JSON' not in revision_request
    assert any(snapshot['review'] == feedback and snapshot['iteration'] == 1 for snapshot in snapshots)
    assert all(snapshot.get('iteration', 0) <= 2 for snapshot in snapshots)


@pytest.mark.anyio
@pytest.mark.parametrize('completed_reviews', [0, MAX_REVIEWS_PER_TURN - 1])
async def test_two_invalid_reviews_never_commit_a_draft_even_at_the_review_limit(completed_reviews):
    previous_turn = [AIMessage('previous answer'), make_review_message()]
    current_turn = [
        message for number in range(completed_reviews)
        for message in (AIMessage(f'draft {number}'), make_review_message('REVISE', 'Fix it.'))
    ]
    model = ScriptedChatModel(replies=[
        *previous_turn, *current_turn, AIMessage('unreviewed draft'), AIMessage('invalid'), AIMessage('still invalid'),
    ])
    graph = build_graph(model, checkpointer=InMemorySaver())
    run_config = {'configurable': {'thread_id': 'invalid-review'}}
    await graph.ainvoke(create_turn_state('Previous question?'), config=run_config)

    with pytest.raises(ReviewFormatError):
        await graph.ainvoke(create_turn_state('Current question?'), config=run_config)

    saved_state = (await graph.aget_state(run_config)).values
    assert [message.content for message in saved_state['messages']] == [
        'Previous question?', 'previous answer', 'Current question?'
    ]
    assert saved_state['answer'] == 'previous answer'
    assert saved_state['draft'].content == 'unreviewed draft'
    assert saved_state['iteration'] == completed_reviews
    assert saved_state['review'] == ('Fix it.' if completed_reviews else None)
    assert len(model.prompts) == 2 + 2 * completed_reviews + 3


@pytest.mark.anyio
async def test_format_retries_do_not_use_the_valid_review_budget():
    replies = [
        message for number in range(1, MAX_REVIEWS_PER_TURN + 1)
        for message in (AIMessage(f'draft {number}'), AIMessage('invalid'), make_review_message('REVISE', 'Fix it.'))
    ]
    model = ScriptedChatModel(replies=replies)
    state = await build_graph(model).ainvoke(create_turn_state('Question?'))

    assert state['iteration'] == MAX_REVIEWS_PER_TURN
    assert state['decision'] == 'continue'
    assert state['review'] == 'Fix it.'
    assert [message.content for message in state['messages']] == ['Question?', f'draft {MAX_REVIEWS_PER_TURN}']
    assert state['answer'] == f'draft {MAX_REVIEWS_PER_TURN}'
    assert state['draft'] is None
    assert len(model.prompts) == 3 * MAX_REVIEWS_PER_TURN


@pytest.mark.anyio
async def test_upstream_timeout_is_not_retried_as_a_review_format_error():
    upstream_error = APITimeoutError(request=httpx.Request('POST', 'https://example.invalid/chat'))
    model = ScriptedChatModel(replies=[AIMessage('draft'), upstream_error, make_review_message()])

    with pytest.raises(APITimeoutError) as caught:
        await build_graph(model).ainvoke(create_turn_state('Question?'))

    assert caught.value is upstream_error
    assert len(model.prompts) == 2
