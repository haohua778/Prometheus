import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import MAX_REVIEWS_PER_TURN, build_graph
from app.config import Settings, get_settings
from app.llm import build_chat_model
from app.main import app, get_agent_graph
from tests.conftest import ScriptedChatModel, make_review_message, add, make_tool_call_message


@pytest.fixture
def client(offline_settings):
    app.dependency_overrides[get_settings] = lambda: offline_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def override_agent_model(model: ScriptedChatModel, tools=()) -> None:
    app.dependency_overrides[get_agent_graph] = lambda: build_graph(model, tools)


def test_health(client):
    assert client.get('/health').json() == {'status': 'ok'}


def test_chat_returns_the_final_reply_and_a_thread_id(client):
    model = ScriptedChatModel(replies=[make_tool_call_message('add', {'a': 20, 'b': 22}), AIMessage('The total is 42.'),
                                       make_review_message()])
    override_agent_model(model, [add])
    response = client.post('/chat', json={'message': 'Add 20 and 22'})

    assert response.status_code == 200
    response_body = response.json()
    assert response_body['reply'] == 'The total is 42.'
    assert len(response_body['thread_id']) == 32
    assert len(model.prompts) == 3


def test_chat_keeps_a_given_thread_id(client):
    model = ScriptedChatModel(replies=[AIMessage('ok'), make_review_message()])
    override_agent_model(model)
    response = client.post('/chat', json={'message': 'hi', 'thread_id': 'review-2026_09'})
    assert response.json()['thread_id'] == 'review-2026_09'
    assert len(model.prompts) == 2


def test_each_turn_starts_a_fresh_review(client):
    # Turn 1 ends at the review limit with a REVISE pending; turn 2 on the same thread must not inherit it.
    model = ScriptedChatModel(replies=[AIMessage('draft'), make_review_message('REVISE', 'Wrong.')] * MAX_REVIEWS_PER_TURN
                              + [AIMessage('Fresh answer.'), make_review_message()])
    graph = build_graph(model, checkpointer=InMemorySaver())
    app.dependency_overrides[get_agent_graph] = lambda: graph
    for message in ('first', 'second'):
        response = client.post('/chat', json={'message': message, 'thread_id': 't1'})

    assert response.json()['reply'] == 'Fresh answer.'
    assert model.prompts[2 * MAX_REVIEWS_PER_TURN][-1].content == 'second'


def test_invalid_review_output_returns_a_clear_upstream_error(client):
    model = ScriptedChatModel(replies=[AIMessage('unreviewed draft'), AIMessage('invalid'), AIMessage('still invalid')])
    override_agent_model(model)

    response = client.post('/chat', json={'message': 'hi'})

    assert response.status_code == 502
    assert response.json()['detail'] == {'code': 'review_output_invalid'}
    assert 'reply' not in response.json()
    assert len(model.prompts) == 3


@pytest.mark.parametrize('request_payload', [{'message': ''}, {'message': 'hi', 'thread_id': 'bad id!'}, {}])
def test_invalid_requests_are_rejected(client, request_payload):
    override_agent_model(ScriptedChatModel(replies=[AIMessage('unused')]))
    assert client.post('/chat', json=request_payload).status_code == 422


@pytest.mark.parametrize('api_key', [None, '  '])
def test_chat_needs_an_api_key(api_key):
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, llm_api_key=api_key)
    try:
        response = TestClient(app).post('/chat', json={'message': 'hi'})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()['detail'] == {'code': 'missing_api_key'}


def test_builds_chat_model_from_settings():
    settings = Settings(_env_file=None, llm_api_key='test-key', llm_base_url='https://example.invalid/v1')

    model = build_chat_model(settings)

    assert model.model_name == settings.llm_model


def test_step_limit_returns_a_clear_error(offline_settings):
    settings = offline_settings.model_copy(update={'max_steps': 4})
    app.dependency_overrides[get_settings] = lambda: settings
    override_agent_model(ScriptedChatModel(replies=[make_tool_call_message('add', {'a': 1, 'b': 1})]), [add])
    try:
        response = TestClient(app).post('/chat', json={'message': 'loop'})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json()['detail'] == {'code': 'step_limit_reached'}
