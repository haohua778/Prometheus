import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.agent.graph import MAX_REVIEWS_PER_TURN, build_graph
from app.agent.prompts import REVIEW_PROMPT, SYSTEM_PROMPT
from tests.conftest import ScriptedChatModel, add, make_review_message, make_tool_call_message

pytestmark = pytest.mark.anyio


async def test_answers_directly_without_tools():
    reply = AIMessage('Hello, analyst.', response_metadata={'finish_reason': 'stop'},
                      usage_metadata={'input_tokens': 10, 'output_tokens': 3, 'total_tokens': 13})
    model = ScriptedChatModel(replies=[reply, make_review_message()])
    snapshots = [state async for state in build_graph(model).astream(
        {'messages': [HumanMessage('Hi')]}, stream_mode='values'
    )]
    state = snapshots[-1]

    draft_states = [snapshot for snapshot in snapshots if snapshot.get('draft') is not None]
    assert len(draft_states) == 1
    assert [message.content for message in draft_states[0]['messages']] == ['Hi']
    final_reply = state['messages'][-1]
    assert final_reply.content == 'Hello, analyst.'
    assert [message.content for message in state['messages']] == ['Hi', 'Hello, analyst.']
    assert final_reply.id is not None
    assert final_reply.id == draft_states[0]['draft'].id
    assert final_reply.usage_metadata == reply.usage_metadata
    assert final_reply.response_metadata == reply.response_metadata
    assert state['draft'] is None
    prompt_messages = model.prompts[0]
    assert isinstance(prompt_messages[0], SystemMessage) and prompt_messages[0].content == SYSTEM_PROMPT
    assert prompt_messages[1].content == 'Hi'
    assert model.bound_tool_names == []
    assert state['answer'] == 'Hello, analyst.'
    assert state['review'] == ''
    assert len(model.prompts) == 2


async def test_tool_result_goes_back_to_the_model():
    model = ScriptedChatModel(replies=[make_tool_call_message('add', {'a': 2, 'b': 3}), AIMessage('2 + 3 = 5.'),
                                       make_review_message()])
    state = await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('What is 2 + 3?')]})

    message_types = [type(message) for message in state['messages']]
    assert message_types == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert state['messages'][2].content == '5'
    assert state['messages'][-1].content == '2 + 3 = 5.'
    assert model.bound_tool_names == ['add']
    assert isinstance(model.prompts[1][-1], ToolMessage)
    assert len(model.prompts) == 3


async def test_step_limit_stops_a_tool_loop():
    model = ScriptedChatModel(replies=[make_tool_call_message('add', {'a': 1, 'b': 1})])
    with pytest.raises(GraphRecursionError):
        await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('Loop')]},
                                                config={'recursion_limit': 5})


def test_graph_connects_tools_and_review():
    graph_diagram = build_graph(ScriptedChatModel(replies=[AIMessage('x')]), [add]).get_graph()
    graph_edges = {(edge.source, edge.target) for edge in graph_diagram.edges}
    assert {'agent', 'tools'} <= set(graph_diagram.nodes)
    assert {('__start__', 'agent'), ('agent', 'tools'), ('tools', 'agent'), ('agent', 'reflect'),
            ('reflect', 'agent'), ('reflect', '__end__')} <= graph_edges


async def test_checkpointer_keeps_a_thread_history():
    model = ScriptedChatModel(replies=[AIMessage('Noted.'), make_review_message(), AIMessage('You said blue.'),
                                       make_review_message()])
    graph = build_graph(model, checkpointer=InMemorySaver())
    run_config = {'configurable': {'thread_id': 't1'}}
    await graph.ainvoke({'messages': [HumanMessage('My color is blue.')]}, config=run_config)
    await graph.ainvoke({'messages': [HumanMessage('What did I say?')]}, config=run_config)

    second_turn_messages = [message.content for message in model.prompts[2][1:]]
    assert second_turn_messages == ['My color is blue.', 'Noted.', 'What did I say?']


async def test_threads_are_separate_with_a_checkpointer():
    model = ScriptedChatModel(replies=[AIMessage('A'), make_review_message(), AIMessage('B'), make_review_message()])
    graph = build_graph(model, checkpointer=InMemorySaver())
    await graph.ainvoke({'messages': [HumanMessage('first')]}, config={'configurable': {'thread_id': 'a'}})
    await graph.ainvoke({'messages': [HumanMessage('second')]}, config={'configurable': {'thread_id': 'b'}})

    assert [message.content for message in model.prompts[2][1:]] == ['second']


async def test_review_passes_the_final_answer():
    model = ScriptedChatModel(
        replies=[make_tool_call_message('add', {'a': 2, 'b': 3}), AIMessage('2 + 3 = 5.'), make_review_message()]
    )
    state = await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('What is 2 + 3?')]})

    assert (state['answer'], state['decision'], state['iteration']) == ('2 + 3 = 5.', 'stop', 1)
    review_system_message, review_input_message = model.prompts[-1]
    assert isinstance(review_system_message, SystemMessage) and review_system_message.content == REVIEW_PROMPT
    review_input = json.loads(review_input_message.content)
    assert review_input['draft'] == {'role': 'assistant', 'content': '2 + 3 = 5.'}
    assert [message['role'] for message in review_input['messages']] == ['user', 'assistant', 'tool']
    tool_call = review_input['messages'][1]['tool_calls'][0]
    tool_result = review_input['messages'][2]
    assert tool_call['name'] == 'add' and tool_call['args'] == {'a': 2, 'b': 3}
    assert tool_call['id'] == tool_result['tool_call_id']
    assert tool_result['content'] == '5'


@pytest.mark.parametrize('revision_count', [1, 2])
async def test_review_revises_only_the_current_draft_and_keeps_only_the_final_answer(revision_count):
    drafts = [f'draft {number}' for number in range(1, revision_count + 1)]
    reviews = [f'Fix issue {number}.' for number in range(1, revision_count + 1)]
    replies = [
        message for draft, review in zip(drafts, reviews)
        for message in (AIMessage(draft), make_review_message('REVISE', review))
    ]
    model = ScriptedChatModel(replies=[*replies, AIMessage('final'), make_review_message()])
    state = await build_graph(model).ainvoke({'messages': [HumanMessage('Total?')]})

    assert (state['answer'], state['iteration']) == ('final', revision_count + 1)
    assert [(message.type, message.content) for message in state['messages']] == [
        ('human', 'Total?'), ('ai', 'final')
    ]
    assert state['draft'] is None
    for index, (draft, review) in enumerate(zip(drafts, reviews)):
        review_input = json.loads(model.prompts[2 * index + 1][-1].content)
        assert review_input == {
            'messages': [{'role': 'user', 'content': 'Total?'}],
            'draft': {'role': 'assistant', 'content': draft},
        }
        revision_prompt = model.prompts[2 * index + 2]
        revision_request = revision_prompt[-1]
        assert isinstance(revision_request, HumanMessage)
        assert draft in revision_request.content and review in revision_request.content
        assert [message.content for message in revision_prompt[1:-1]] == ['Total?']
        assert all(old_draft not in revision_request.content for old_draft in drafts[:index])
        assert all(old_review not in revision_request.content for old_review in reviews[:index])


async def test_revision_can_call_a_tool_without_losing_the_draft_or_review():
    draft = 'The total is unverified.'
    review = 'Use add to verify the total.'
    model = ScriptedChatModel(replies=[
        AIMessage(draft), make_review_message('REVISE', review), make_tool_call_message('add', {'a': 2, 'b': 3}),
        AIMessage('The verified total is 5.'), make_review_message(),
    ])
    state = await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('What is 2 + 3?')]})

    assert [type(message) for message in state['messages']] == [
        HumanMessage, AIMessage, ToolMessage, AIMessage
    ]
    tool_call, tool_result, final_reply = state['messages'][1:]
    assert tool_call.tool_calls[0]['id'] == tool_result.tool_call_id
    assert tool_result.content == '5'
    assert final_reply.content == state['answer'] == 'The verified total is 5.'
    assert state['draft'] is None
    assert state['iteration'] == 2
    for revision_prompt in model.prompts[2:4]:
        revision_request = revision_prompt[-1]
        assert isinstance(revision_request, HumanMessage)
        assert draft in revision_request.content and review in revision_request.content
    assert [type(message) for message in model.prompts[3][1:-1]] == [
        HumanMessage, AIMessage, ToolMessage
    ]
    assert model.prompts[3][-1].content == model.prompts[2][-1].content
    review_input = json.loads(model.prompts[-1][-1].content)
    assert review_input['messages'][-1]['role'] == 'tool'
    assert review_input['messages'][-1]['content'] == '5'
    assert review_input['draft'] == {'role': 'assistant', 'content': 'The verified total is 5.'}
    assert draft not in json.dumps(review_input)


async def test_review_stops_at_review_limit():
    drafts = [f'draft {number}' for number in range(1, MAX_REVIEWS_PER_TURN + 1)]
    replies = [message for draft in drafts for message in (AIMessage(draft), make_review_message('REVISE', 'Wrong.'))]
    model = ScriptedChatModel(replies=replies)
    state = await build_graph(model).ainvoke({'messages': [HumanMessage('Total?')]})

    assert (state['iteration'], state['decision']) == (MAX_REVIEWS_PER_TURN, 'continue')
    assert [message.content for message in state['messages']] == ['Total?', drafts[-1]]
    assert state['answer'] == drafts[-1]
    assert state['review'] == 'Wrong.'
    assert state['draft'] is None
    assert len(model.prompts) == 2 * MAX_REVIEWS_PER_TURN
