import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.agent.graph import MAX_ITERATIONS, build_graph
from app.agent.prompts import REVIEW_PROMPT, SYSTEM_PROMPT
from tests.conftest import ScriptedChatModel, add, tool_call

pytestmark = pytest.mark.anyio


async def test_answers_directly_without_tools():
    model = ScriptedChatModel(replies=[AIMessage('Hello, analyst.'), AIMessage('PASS')])
    state = await build_graph(model).ainvoke({'messages': [HumanMessage('Hi')]})

    assert state['messages'][-1].content == 'Hello, analyst.'
    prompt = model.prompts[0]
    assert isinstance(prompt[0], SystemMessage) and prompt[0].content == SYSTEM_PROMPT
    assert prompt[1].content == 'Hi'
    assert model.bound_tools == []
    assert state['answer'] == 'Hello, analyst.'
    assert len(model.prompts) == 2


async def test_tool_result_goes_back_to_the_model():
    model = ScriptedChatModel(replies=[tool_call('add', {'a': 2, 'b': 3}), AIMessage('2 + 3 = 5.'),
                                       AIMessage('PASS')])
    state = await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('What is 2 + 3?')]})

    kinds = [type(message) for message in state['messages']]
    assert kinds == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert state['messages'][2].content == '5'
    assert state['messages'][-1].content == '2 + 3 = 5.'
    assert model.bound_tools == ['add']
    assert isinstance(model.prompts[1][-1], ToolMessage)
    assert len(model.prompts) == 3


async def test_step_limit_stops_a_tool_loop():
    model = ScriptedChatModel(replies=[tool_call('add', {'a': 1, 'b': 1})])
    with pytest.raises(GraphRecursionError):
        await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('Loop')]},
                                                config={'recursion_limit': 5})


def test_graph_shape():
    graph = build_graph(ScriptedChatModel(replies=[AIMessage('x')]), [add]).get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert {'agent', 'tools'} <= set(graph.nodes)
    assert {('__start__', 'agent'), ('agent', 'tools'), ('tools', 'agent'), ('agent', 'reflect'),
            ('reflect', 'agent'), ('reflect', '__end__')} <= edges


async def test_checkpointer_keeps_a_thread_history():
    model = ScriptedChatModel(replies=[AIMessage('Noted.'), AIMessage('PASS'), AIMessage('You said blue.'),
                                       AIMessage('PASS')])
    graph = build_graph(model, checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 't1'}}
    await graph.ainvoke({'messages': [HumanMessage('My color is blue.')]}, config=config)
    await graph.ainvoke({'messages': [HumanMessage('What did I say?')]}, config=config)

    second_prompt = [message.content for message in model.prompts[2][1:]]
    assert second_prompt == ['My color is blue.', 'Noted.', 'What did I say?']


async def test_threads_are_separate_with_a_checkpointer():
    model = ScriptedChatModel(replies=[AIMessage('A'), AIMessage('PASS'), AIMessage('B'), AIMessage('PASS')])
    graph = build_graph(model, checkpointer=InMemorySaver())
    await graph.ainvoke({'messages': [HumanMessage('first')]}, config={'configurable': {'thread_id': 'a'}})
    await graph.ainvoke({'messages': [HumanMessage('second')]}, config={'configurable': {'thread_id': 'b'}})

    assert [message.content for message in model.prompts[2][1:]] == ['second']


async def test_review_passes_the_final_answer():
    model = ScriptedChatModel(replies=[tool_call('add', {'a': 2, 'b': 3}), AIMessage('2 + 3 = 5.'), AIMessage('PASS')])
    state = await build_graph(model, [add]).ainvoke({'messages': [HumanMessage('What is 2 + 3?')]})

    assert (state['answer'], state['decision'], state['iteration']) == ('2 + 3 = 5.', 'stop', 1)
    system, transcript = model.prompts[-1]
    assert isinstance(system, SystemMessage) and system.content == REVIEW_PROMPT
    assert 'name="add"' in transcript.content and '<message type="tool">5</message>' in transcript.content


async def test_review_sends_the_answer_back_for_revision():
    model = ScriptedChatModel(replies=[AIMessage('draft'), AIMessage('REVISE\nAdd units.'), AIMessage('final'),
                                       AIMessage('PASS')])
    state = await build_graph(model).ainvoke({'messages': [HumanMessage('Total?')]})

    assert (state['answer'], state['iteration']) == ('final', 2)
    feedback = model.prompts[2][-1]
    assert isinstance(feedback, HumanMessage) and 'Add units.' in feedback.content
    assert all('Add units.' not in message.content for message in state['messages'])


async def test_review_stops_after_max_iterations():
    model = ScriptedChatModel(replies=[AIMessage('draft'), AIMessage('REVISE\nWrong.')] * MAX_ITERATIONS)
    state = await build_graph(model).ainvoke({'messages': [HumanMessage('Total?')]})

    assert (state['iteration'], state['decision']) == (MAX_ITERATIONS, 'continue')
    assert len(model.prompts) == 2 * MAX_ITERATIONS
