import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.agent.graph import build_graph
from app.config import Settings, get_settings
from app.main import app, get_agent
from tests.conftest import ScriptedChatModel, add, tool_call


@pytest.fixture
def client(offline_settings):
    app.dependency_overrides[get_settings] = lambda: offline_settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_model(model: ScriptedChatModel, tools=()) -> None:
    app.dependency_overrides[get_agent] = lambda: build_graph(model, tools)


def test_health(client):
    assert client.get('/health').json() == {'status': 'ok'}


def test_chat_returns_the_final_reply_and_a_thread_id(client):
    use_model(ScriptedChatModel(replies=[tool_call('add', {'a': 20, 'b': 22}), AIMessage('The total is 42.')]),
              [add])
    response = client.post('/chat', json={'message': 'Add 20 and 22'})

    assert response.status_code == 200
    body = response.json()
    assert body['reply'] == 'The total is 42.'
    assert len(body['thread_id']) == 32


def test_chat_keeps_a_given_thread_id(client):
    use_model(ScriptedChatModel(replies=[AIMessage('ok')]))
    response = client.post('/chat', json={'message': 'hi', 'thread_id': 'review-2026_09'})
    assert response.json()['thread_id'] == 'review-2026_09'


@pytest.mark.parametrize('payload', [{'message': ''}, {'message': 'hi', 'thread_id': 'bad id!'}, {}])
def test_invalid_requests_are_rejected(client, payload):
    use_model(ScriptedChatModel(replies=[AIMessage('unused')]))
    assert client.post('/chat', json=payload).status_code == 422


def test_live_calls_are_off_by_default(client):
    response = client.post('/chat', json={'message': 'hi'})
    assert response.status_code == 503
    assert response.json()['detail'] == {'code': 'live_calls_disabled'}


def test_live_calls_need_an_api_key():
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, allow_live=True, llm_api_key='  ')
    try:
        response = TestClient(app).post('/chat', json={'message': 'hi'})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert response.json()['detail'] == {'code': 'missing_api_key'}


def test_step_limit_returns_a_clear_error(offline_settings):
    settings = offline_settings.model_copy(update={'max_steps': 4})
    app.dependency_overrides[get_settings] = lambda: settings
    use_model(ScriptedChatModel(replies=[tool_call('add', {'a': 1, 'b': 1})]), [add])
    try:
        response = TestClient(app).post('/chat', json={'message': 'loop'})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json()['detail'] == {'code': 'step_limit_reached'}
