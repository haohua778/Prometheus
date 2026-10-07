import asyncio
import io
import json
import logging
import os
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from uuid import uuid4

import httpx
import pytest
from langchain_core.tools import tool

from app.runtime_logging import (JsonFormatter, RuntimeLogCallback, configure_logging, log_event,
                                 logger, request_id_context, thread_id_context)


@pytest.fixture
def events():
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield lambda: [json.loads(line) for line in output.getvalue().splitlines()]
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def available_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


@contextmanager
def running_service(*, use_configured_model=False, **settings):
    directory = Path(__file__).resolve().parents[1] / '.runtime' / f'logging-test-{uuid4().hex}'
    directory.mkdir(parents=True)
    port = available_port()
    env = dict(os.environ, PROMETHEUS_LOG_LEVEL='INFO', PYTHONIOENCODING='utf-8')
    if not use_configured_model:
        env['LLM_API_KEY'] = ''
    env.update(settings)
    with (directory / 'stdout.log').open('w+', encoding='utf-8') as output, \
            (directory / 'stderr.log').open('w+', encoding='utf-8') as errors:
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
             '--port', str(port), '--no-access-log'],
            cwd=directory.parents[1], stdout=output, stderr=errors, env=env)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=180) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        pytest.fail('Uvicorn exited before startup; inspect .runtime logging-test files')
                    try:
                        if client.get('/health').status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    sleep(0.1)
                else:
                    pytest.fail('Uvicorn startup timed out')

                def records():
                    output.seek(0)
                    return [json.loads(line) for line in output.read().splitlines()]

                yield client, records
        finally:
            process.terminate()
            process.wait(timeout=10)


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def divide_numbers(a: int, b: int) -> float:
    """Divide two integers."""
    return a / b


def test_http_routes_errors_and_privacy():
    with running_service(LLM_API_KEY='validation-only-key') as (client, events):
        for path in ('/health', '/docs', '/openapi.json'):
            response = client.get(path)
            assert response.status_code == 200
            assert any(record['event'] == 'http.completed' and record['route'] == path
                       and record['request_id'] == response.headers['X-Request-ID'] for record in events())
        missing = client.get('/PRIVATE_PATH?token=PRIVATE_QUERY')
        assert missing.status_code == 404
        assert any(record['route'] == '<unmatched>' and record['status_code'] == 404
                   for record in events() if record['event'] == 'http.completed')
        assert client.post('/chat', json={'message': ''}).status_code == 422
        assert client.get('/chat').status_code == 405
        content = json.dumps(events())
        assert 'PRIVATE_PATH' not in content
        assert 'PRIVATE_QUERY' not in content
        assert 'validation-only-key' not in content


def test_missing_key_logs_correlated_error():
    with running_service() as (client, events):
        response = client.post('/chat', json={'message': 'PRIVATE_PAYLOAD'})
        assert response.status_code == 503
        assert response.json()['detail'] == {'code': 'missing_api_key'}
        assert any(record.get('error_code') == 'missing_api_key'
                   and record['request_id'] == response.headers['X-Request-ID'] for record in events())
        assert 'PRIVATE_PAYLOAD' not in json.dumps(events())


def test_real_connection_failure_preserves_api_error_and_code_locations():
    with running_service(LLM_API_KEY='connection-test-key', LLM_TIMEOUT_SECONDS='10',
                         LLM_BASE_URL=f'http://127.0.0.1:{available_port()}/v1') as (client, events):
        response = client.post('/chat', json={'message': 'PRIVATE_PAYLOAD', 'thread_id': 'error-thread'})
        assert response.status_code == 502
        assert response.json()['detail'] == {'code': 'upstream_error'}
        failure = next(record for record in events() if record['event'] == 'llm.failed')
        assert failure['thread_id'] == 'error-thread'
        assert failure['stack']
        assert failure['duration_ms'] >= 0
        assert 'connection-test-key' not in json.dumps(events())
        assert 'PRIVATE_PAYLOAD' not in json.dumps(events())


@pytest.mark.anyio
async def test_real_tool_success_and_failure(events):
    callback = RuntimeLogCallback()
    assert await add_numbers.ainvoke({'a': 2, 'b': 3}, config={'callbacks': [callback]}) == 5
    with pytest.raises(ZeroDivisionError):
        await divide_numbers.ainvoke({'a': 1, 'b': 0}, config={'callbacks': [callback]})
    completed = next(record for record in events() if record['event'] == 'tool.completed')
    failure = next(record for record in events() if record['event'] == 'tool.failed')
    assert completed['tool_name'] == 'add_numbers'
    assert failure['tool_name'] == 'divide_numbers'
    assert failure['error_type'] == 'ZeroDivisionError'
    assert any(frame['function'] == 'divide_numbers' for frame in failure['stack'])
    assert callback.started == {}
    assert callback.tool_names == {}


@pytest.mark.anyio
async def test_concurrent_real_tools_keep_context_isolated(events):
    async def invoke(identifier):
        request_token = request_id_context.set(identifier)
        thread_token = thread_id_context.set(f'thread-{identifier}')
        try:
            return await add_numbers.ainvoke({'a': 2, 'b': 3}, config={'callbacks': [RuntimeLogCallback()]})
        finally:
            thread_id_context.reset(thread_token)
            request_id_context.reset(request_token)

    assert await asyncio.gather(invoke('first'), invoke('second')) == [5, 5]
    for identifier in ('first', 'second'):
        records = [record for record in events() if record['request_id'] == identifier]
        assert [record['event'] for record in records] == ['tool.started', 'tool.completed']
        assert {record['thread_id'] for record in records} == {f'thread-{identifier}'}
        assert len({record['span_id'] for record in records}) == 1
    assert request_id_context.get() is None
    assert thread_id_context.get() is None


@pytest.mark.anyio
async def test_cancelled_real_tool_releases_callback_state(events):
    entered = asyncio.Event()

    @tool
    async def wait_seconds(seconds: float) -> str:
        """Wait for the requested number of seconds."""
        entered.set()
        await asyncio.sleep(seconds)
        return 'done'

    callback = RuntimeLogCallback()
    task = asyncio.create_task(wait_seconds.ainvoke({'seconds': 30}, config={'callbacks': [callback]}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert callback.started
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        callback.finish_pending()
    assert any(record['event'] == 'tool.failed' and record['outcome'] == 'interrupted'
               for record in events())
    assert callback.started == {}
    assert callback.tool_names == {}


def test_configuration_is_idempotent_and_respects_level(events):
    configure_logging('WARNING')
    configure_logging('WARNING')
    log_event('filtered.info')
    log_event('visible.warning', level=logging.WARNING)
    assert [record['event'] for record in events()] == ['visible.warning']


@pytest.mark.skipif(os.environ.get('PROMETHEUS_LIVE_LOG_TEST') != '1', reason='Real model calls require opt-in')
def test_live_model_and_reflection_logs():
    with running_service(use_configured_model=True) as (client, events):
        response = client.post('/chat', json={'message': 'Return exactly OK. LOG_PRIVACY_CHECK',
                                              'thread_id': 'live-log-check'})
        assert response.status_code == 200
        assert response.json()['reply']
        records = events()
        assert sum(record['event'] == 'llm.completed' for record in records) >= 2
        assert any(record['event'] in ('review.completed', 'review.limit_reached') for record in records)
        assert any(record['event'] == 'chat.completed' and record['thread_id'] == 'live-log-check'
                   for record in records)
        assert 'LOG_PRIVACY_CHECK' not in json.dumps(records)
