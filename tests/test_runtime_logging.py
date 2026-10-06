"""Exercise real HTTP/graph paths and verify diagnostic events never include payloads."""
import asyncio
import io
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from openai import APIError, APITimeoutError

from app.agent.graph import MAX_REVIEWS_PER_TURN, build_graph
from app.config import get_settings
from app.main import app, get_agent_graph
from app.runtime_logging import JsonFormatter, logger, request_id_context, thread_id_context
from tests.conftest import ScriptedChatModel, add, make_review_message, make_tool_call_message

SECRET = 'sensitive-document-and-api-key'


@pytest.fixture
def events(offline_settings):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    app.dependency_overrides[get_settings] = lambda: offline_settings
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    app.dependency_overrides.clear()
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def use_model(replies, tools=()):
    graph = build_graph(ScriptedChatModel(replies=replies), tools)
    app.dependency_overrides[get_agent_graph] = lambda: graph


def test_success_correlates_http_model_tool_and_review_without_payloads(events):
    use_model([make_tool_call_message('add', {'a': 1, 'b': 2}),
               AIMessage(SECRET, usage_metadata={'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6}),
               make_review_message()], [add])
    with TestClient(app) as client:
        response = client.post('/chat', json={'message': SECRET, 'thread_id': 'thread-a'})
    assert response.status_code == 200
    records = events()
    run_records = [record for record in records if record['request_id']]
    assert {record['request_id'] for record in run_records} == {response.headers['X-Request-ID']}
    graph_records = [record for record in run_records if not record['event'].startswith('http.')]
    assert {record['thread_id'] for record in graph_records} == {'thread-a'}
    names = [record['event'] for record in records]
    assert names.count('llm.completed') == 3
    assert names.count('tool.completed') == 1
    assert {'review.completed', 'chat.completed', 'http.completed', 'service.started', 'service.stopped'} <= set(names)
    assert any(record.get('total_tokens') == 6 for record in records)
    assert all(record['duration_ms'] >= 0 for record in records if 'duration_ms' in record)
    assert SECRET not in json.dumps(records)


@pytest.mark.parametrize('kind,code,status', [
    ('timeout', 'upstream_timeout', 504), ('provider', 'upstream_error', 502),
    ('review', 'review_output_invalid', 502), ('steps', 'step_limit_reached', 500),
    ('key', 'missing_api_key', 503),
])
def test_failures_keep_api_contract_and_log_error_codes_without_error_bodies(events, kind, code, status):
    request = httpx.Request('POST', f'https://example.invalid/?token={SECRET}')
    if kind == 'timeout':
        use_model([APITimeoutError(request=request)])
    elif kind == 'provider':
        use_model([APIError(SECRET, request=request, body={'secret': SECRET})])
    elif kind == 'review':
        use_model([AIMessage(SECRET), AIMessage(SECRET), AIMessage(SECRET)])
    elif kind == 'steps':
        use_model([make_tool_call_message('add', {'a': 1, 'b': 2})], [add])
    with TestClient(app) as client:
        response = client.post('/chat', json={'message': SECRET, 'thread_id': 'failure-thread'})
    assert response.status_code == status
    assert response.json()['detail'] == {'code': code}
    assert any(record.get('error_code') == code for record in events())
    assert any(record['event'] == 'http.completed' and record['status_code'] == status for record in events())
    assert SECRET not in json.dumps(events())


def test_review_retries_and_limit_are_distinguishable(events):
    use_model([AIMessage(SECRET), AIMessage('invalid'), make_review_message('REVISE', SECRET)]
              + [AIMessage(SECRET), make_review_message('REVISE', SECRET)] * (MAX_REVIEWS_PER_TURN - 1))
    with TestClient(app) as client:
        assert client.post('/chat', json={'message': SECRET}).status_code == 200
    assert sum(record['event'] == 'review.format_retry' for record in events()) == 1
    limit = next(record for record in events() if record['event'] == 'review.limit_reached')
    assert limit['level'] == 'WARNING'
    assert limit['review_count'] == MAX_REVIEWS_PER_TURN
    assert SECRET not in json.dumps(events())


def test_handled_tool_failure_is_not_logged_as_success(events):
    use_model([make_tool_call_message('add', {'a': SECRET, 'b': 2}), AIMessage('Cannot add.'),
               make_review_message()], [add])
    with TestClient(app) as client:
        assert client.post('/chat', json={'message': SECRET}).status_code == 200
    assert any(record['event'] == 'tool.failed' for record in events())
    assert not any(record['event'] == 'tool.completed' for record in events())
    assert SECRET not in json.dumps(events())


def test_unexpected_error_has_stack_locations_and_no_exception_message(events):
    use_model([RuntimeError(SECRET)])
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post('/chat', json={'message': SECRET}).status_code == 500
    failure = next(record for record in events() if record['event'] == 'http.failed')
    assert failure['error_type'] == 'RuntimeError'
    assert failure['stack']
    assert SECRET not in json.dumps(events())


def test_validation_and_unmatched_paths_do_not_log_payload_or_raw_url(events):
    use_model([AIMessage('unused')])
    with TestClient(app) as client:
        assert client.post('/chat', json={'message': ''}).status_code == 422
        assert client.get(f'/{SECRET}?token={SECRET}').status_code == 404
    assert SECRET not in json.dumps(events())
    assert {record['status_code'] for record in events() if record['event'] == 'http.completed'} == {422, 404}


@pytest.mark.anyio
async def test_concurrent_requests_keep_their_own_correlation_ids(events):
    ready = asyncio.Event()

    class ConcurrentGraph:
        async def ainvoke(self, state, config):
            if config['configurable']['thread_id'] == 'first':
                await ready.wait()
            else:
                ready.set()
            return {'answer': 'ok', 'iteration': 1}

    app.dependency_overrides[get_agent_graph] = ConcurrentGraph
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        responses = await asyncio.gather(*[
            client.post('/chat', json={'message': 'hi', 'thread_id': thread}) for thread in ('first', 'second')
        ])
    identifiers = [response.headers['X-Request-ID'] for response in responses]
    assert len(set(identifiers)) == 2
    for thread, request_id in zip(('first', 'second'), identifiers):
        records = [record for record in events() if record['request_id'] == request_id
                   and record['event'].startswith('chat.')]
        assert {record['event'] for record in records} == {'chat.started', 'chat.completed'}
        assert {record['thread_id'] for record in records} == {thread}
    assert request_id_context.get() is None
    assert thread_id_context.get() is None


@pytest.mark.anyio
async def test_cancelled_request_is_visible_and_does_not_leave_context(events):
    entered = asyncio.Event()

    class WaitingGraph:
        async def ainvoke(self, state, config):
            entered.set()
            await asyncio.Future()

    app.dependency_overrides[get_agent_graph] = WaitingGraph
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        task = asyncio.create_task(client.post('/chat', json={'message': 'hi'}))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert any(record['event'] == 'http.cancelled' for record in events())
    assert any(record['event'] == 'http.completed' and record['status_code'] == 499 for record in events())
    assert request_id_context.get() is None
    assert thread_id_context.get() is None
