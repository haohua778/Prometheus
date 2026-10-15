import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from app.agent.coding import CheckRequest, make_check_tool, run_check
from app.agent.coding.agent import service_slots
from app.agent.coding.graph import parse_finding
from app.agent.coding.sandbox import (
    EXPIRY_LABEL,
    IMAGE,
    OWNER,
    OWNER_LABEL,
    RUNTIME_ROOT,
    Sandbox,
    cleanup_expired,
    close_safely,
    require_docker,
    validate_files,
)
from app.agent.coding.schemas import CheckFailure, CheckResult, Execution
from app.config import PROJECT_ROOT
from app.runtime_logging import check_request_id_context, request_id_context
from sandbox.inspect_data import inspect


@contextmanager
def data_files():
    directory = RUNTIME_ROOT / f'test-{uuid4().hex}'
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def request_for(file_id='sales', file_format='csv', timeout=60, claim='Revenue grew by 30%.'):
    return CheckRequest(request_id=uuid4().hex, flag_id='flag-1', claim=claim,
                        question='Compute the 2024 to 2025 revenue growth. Values are in millions, same scope.',
                        data_refs=[{'file_id': file_id, 'format': file_format}], timeout_s=timeout)


def sandbox_for(request, files):
    return Sandbox(request, validate_files(request, files), asyncio.get_running_loop().time() + request.timeout_s)


def check_tool_graph(tool):
    builder = StateGraph(MessagesState)
    builder.add_node('tools', ToolNode([tool], handle_tool_errors=False))
    builder.add_edge(START, 'tools')
    builder.add_edge('tools', END)
    return builder.compile(checkpointer=False)


@pytest.mark.parametrize('changes', [
    {'request_id': ''}, {'claim': '  '}, {'timeout_s': 0}, {'timeout_s': 61},
    {'timeout_s': '60'}, {'data_refs': [{'file_id': '../sales', 'format': 'csv'}]},
    {'data_refs': [{'file_id': 'sales', 'format': 'xlsx'}]}, {'unexpected': True},
    {'data_refs': [{'file_id': 'sales', 'format': 'csv'}] * 2},
    {'data_refs': [{'file_id': str(index), 'format': 'csv'} for index in range(6)]},
])
def test_contract_rejects_invalid_requests(changes):
    values = request_for().model_dump()
    values.update(changes)
    with pytest.raises(ValidationError):
        CheckRequest.model_validate(values)


def test_contract_requires_claim_and_roundtrips_json():
    request = request_for()
    assert CheckRequest.model_validate_json(request.model_dump_json()) == request
    values = request.model_dump()
    del values['claim']
    with pytest.raises(ValidationError):
        CheckRequest.model_validate(values)


@pytest.mark.parametrize('status,verdict,error_code', [
    ('completed', 'inconclusive', None),
    ('completed', 'supports', None),
    ('completed', 'refutes', 'model_error'),
    ('incomplete', 'supports', 'budget_exhausted'),
    ('failed', 'supports', 'environment_error'),
    ('failed', 'inconclusive', None),
    ('incomplete', 'inconclusive', 'model_error'),
    ('failed', 'inconclusive', 'timeout'),
])
def test_result_rejects_contradictory_states(status, verdict, error_code):
    with pytest.raises(ValidationError):
        CheckResult(request_id='check-1', flag_id='flag-1', status=status, verdict=verdict,
                    explanation='Validation input.', evidence=[], executions=[], runtime_ms=0.0,
                    error_code=error_code)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0])
def test_execution_requires_finite_nonnegative_duration(value):
    with pytest.raises(ValidationError):
        Execution(code='', stdout='', stderr='', exit_code=None, runtime_ms=value)


@pytest.mark.parametrize('text', [
    '{}',
    '```json\n{"verdict":"inconclusive","explanation":"Missing data","evidence":[]}\n```',
    '{"verdict":"supports","explanation":"Unsupported claim","evidence":[]}',
    '{"verdict":"inconclusive","explanation":"Missing data","evidence":[],"extra":true}',
    '{"verdict":"inconclusive","explanation":"Missing data","evidence":[{"file_id":"private","location":"row 1","observed_value":1}]}',
    'x' * 16385,
])
def test_final_finding_rejects_invalid_or_unsupported_content(text):
    with pytest.raises(CheckFailure) as error:
        parse_finding(text, inspected_files=set(), executions=[])
    assert error.value.code == 'result_invalid'


def test_inconclusive_finding_needs_no_execution():
    finding = parse_finding('{"verdict":"inconclusive","explanation":"Missing data","evidence":[]}',
                            inspected_files=set(), executions=[])
    assert finding.verdict == 'inconclusive'


def test_file_authorization_and_size():
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n2025,120\n', encoding='utf-8')
        assert validate_files(request_for(), {'sales': path})['sales'] == path.absolute()
        with pytest.raises(ValueError, match='Unauthorized'):
            validate_files(request_for(), {})
        with pytest.raises(ValueError, match='regular file'):
            validate_files(request_for(), {'sales': directory})
        with path.open('wb') as stream:
            stream.truncate(10 * 1024 * 1024 + 1)
        with pytest.raises(ValueError, match='10 MiB'):
            validate_files(request_for(), {'sales': path})


def test_symbolic_input_rejected():
    with data_files() as directory:
        target = directory / 'sales.csv'
        target.write_text('a\n1\n', encoding='utf-8')
        link = directory / 'link.csv'
        link.symlink_to(target)
        with pytest.raises(ValueError, match='symbolic'):
            validate_files(request_for(), {'sales': link})


@pytest.mark.parametrize('file_format,content', [
    ('csv', 'year,revenue\n2024,100\n2025,120\n'),
    ('json', '[{"year":2024,"revenue":100},{"year":2025,"revenue":120}]'),
])
def test_real_data_parser(file_format, content):
    with data_files() as directory:
        path = directory / f'data.{file_format}'
        path.write_text(content, encoding='utf-8')
        result = inspect(path, file_format)
        assert result['fields'] == ['year', 'revenue']
        assert result['record_count'] == 2
        assert len(result['samples']) == 2


@pytest.mark.parametrize('file_format,content', [
    ('json', '{}'), ('json', '[1]'), ('json', '[{"a":NaN}]'),
    ('csv', 'a,a\n1,2\n'), ('csv', 'a,b\n1\n'),
])
def test_real_data_parser_rejects_invalid_records(file_format, content):
    with data_files() as directory:
        path = directory / 'invalid'
        path.write_text(content, encoding='utf-8')
        with pytest.raises(ValueError):
            inspect(path, file_format)


@pytest.mark.parametrize('content,expected_code', [
    pytest.param('{}', 2, id='not-record-array'),
    pytest.param('[{"value":"' + 'x' * 17000 + '"}]', 3, id='preview-too-large'),
    pytest.param('[' * 20000 + '0' + ']' * 20000, 2, id='excessive-nesting'),
])
def test_inspection_script_has_explicit_failure_codes(content, expected_code):
    with data_files() as directory:
        path = directory / 'data.json'
        path.write_text(content, encoding='utf-8')
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'sandbox' / 'inspect_data.py'),
                                 str(path), 'json'], capture_output=True, timeout=5, check=False)
        assert result.returncode == expected_code
        assert len(result.stdout) < 16384
        assert json.loads(result.stdout)['error']


@pytest.mark.parametrize('output_bytes,expected_code', [(16384, 0), (16385, 3)])
def test_inspection_limit_includes_the_output_newline(output_bytes, expected_code):
    with data_files() as directory:
        path = directory / 'data.json'
        path.write_text('[{"x":""}]', encoding='utf-8')
        base_bytes = len(json.dumps(inspect(path, 'json'), ensure_ascii=False).encode('utf-8')) + 1
        path.write_text(json.dumps([{'x': 'v' * (output_bytes - base_bytes)}]), encoding='utf-8')
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / 'sandbox' / 'inspect_data.py'),
                                 str(path), 'json'], capture_output=True, timeout=5, check=False)
        assert result.returncode == expected_code
        assert len(result.stdout) <= 16384


def test_concurrent_check_ids_and_parent_context():
    async def scenario():
        parent_token = request_id_context.set('parent-request')
        try:
            first, second = request_for(), request_for()
            first.data_refs, second.data_refs = [], []
            results = await asyncio.gather(run_check(first, authorized_files={}),
                                           run_check(second, authorized_files={}))
            assert [result.request_id for result in results] == [first.request_id, second.request_id]
            assert all(result.status == 'incomplete' for result in results)
            assert results[0].executions is not results[1].executions
            assert request_id_context.get() == 'parent-request'
            assert check_request_id_context.get() is None
        finally:
            request_id_context.reset(parent_token)
    asyncio.run(scenario())


def test_queue_timeout_and_cancellation_release_no_foreign_slots():
    async def scenario():
        slots = service_slots()
        await slots.acquire()
        await slots.acquire()
        request = request_for(timeout=1)
        request.data_refs = []
        try:
            result = await run_check(request, authorized_files={})
            assert result.error_code == 'timeout' and result.status == 'incomplete'
            pending = asyncio.create_task(run_check(request, authorized_files={}))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert slots.locked()
        finally:
            slots.release()
            slots.release()
        result = await run_check(request, authorized_files={})
        assert result.error_code is None
    asyncio.run(scenario())


def test_staging_directory_cleanup_without_container():
    async def scenario():
        request = request_for()
        request.data_refs = []
        sandbox = sandbox_for(request, {})
        sandbox.directory.mkdir(parents=True)
        (sandbox.directory / 'input.csv').write_text('a\n1\n', encoding='utf-8')
        await close_safely(sandbox)
        assert not sandbox.directory.exists()
    asyncio.run(scenario())


def test_cleanup_rejects_a_dangling_staging_link():
    async def scenario(target):
        request = request_for()
        request.data_refs = []
        sandbox = sandbox_for(request, {})
        sandbox.directory.symlink_to(target, target_is_directory=True)
        try:
            with pytest.raises(CheckFailure) as error:
                await close_safely(sandbox)
            assert error.value.code == 'cleanup_failed'
        finally:
            sandbox.directory.unlink()
        assert not target.exists()
    with data_files() as directory:
        asyncio.run(scenario(directory / 'nonexistent'))


def test_queued_request_is_an_independent_snapshot():
    async def scenario():
        slots = service_slots()
        await slots.acquire()
        await slots.acquire()
        request = request_for()
        request.data_refs = []
        original_id = request.request_id
        pending = asyncio.create_task(run_check(request, authorized_files={}))
        await asyncio.sleep(0)
        request.request_id = 'changed-after-submission'
        slots.release()
        slots.release()
        result = await pending
        assert result.request_id == original_id
    asyncio.run(scenario())


def test_request_is_revalidated_at_the_public_boundary():
    request = request_for()
    request.data_refs = []
    request.timeout_s = 0
    with pytest.raises(ValidationError):
        asyncio.run(run_check(request, authorized_files={}))


def test_unknown_tool_file_is_an_explicit_data_failure():
    async def scenario():
        request = request_for()
        request.data_refs = []
        sandbox = sandbox_for(request, {})
        with pytest.raises(CheckFailure) as error:
            await sandbox.inspect_data('unavailable')
        assert error.value.code == 'data_invalid'
        assert not sandbox.created
    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['deleted', 'oversized'])
def test_input_changes_before_staging_are_explicit_failures(change):
    async def scenario(path):
        sandbox = sandbox_for(request_for(), {'sales': path})
        if change == 'deleted':
            path.unlink()
        else:
            with path.open('wb') as stream:
                stream.truncate(10 * 1024 * 1024 + 1)
        try:
            with pytest.raises(CheckFailure) as error:
                sandbox.stage_inputs()
            assert error.value.code == 'data_invalid'
            assert not sandbox.created
        finally:
            await close_safely(sandbox)
        assert not sandbox.directory.exists()
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n', encoding='utf-8')
        asyncio.run(scenario(path))


def test_staged_copy_is_independent_of_original_file():
    async def scenario(path):
        sandbox = sandbox_for(request_for(), {'sales': path})
        try:
            sandbox.stage_inputs()
            path.write_text('changed-after-staging', encoding='utf-8')
            assert (sandbox.directory / '0.csv').read_text(encoding='utf-8') == 'year,revenue\n2024,100\n'
        finally:
            await close_safely(sandbox)
        assert not sandbox.directory.exists()
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n', encoding='utf-8')
        asyncio.run(scenario(path))


docker_test = pytest.mark.skipif(os.environ.get('PROMETHEUS_DOCKER_TEST') != '1',
                                 reason='Explicitly enable real Docker tests with PROMETHEUS_DOCKER_TEST=1.')
live_test = pytest.mark.skipif(os.environ.get('PROMETHEUS_LIVE_CODING_TEST') != '1',
                               reason='Explicitly enable billed model tests with PROMETHEUS_LIVE_CODING_TEST=1.')


@pytest.fixture(scope='module')
def live_loop():
    # langchain-openai caches one async HTTP client per process, and its pooled connections belong
    # to the event loop that opened them; billed tests share one loop instead of asyncio.run().
    with asyncio.Runner() as runner:
        yield runner


async def assert_removed(sandbox):
    names = await require_docker('ps', '-a', '--format', '{{.Names}}')
    assert sandbox.name not in names.splitlines()
    assert not sandbox.directory.exists()


@docker_test
def test_real_sandbox_calculation_repair_and_filesystem():
    async def scenario(path):
        sandbox = sandbox_for(request_for(), {'sales': path})
        try:
            preview = await sandbox.inspect_data('sales')
            assert preview['record_count'] == 2
            settings = json.loads(await require_docker('inspect', sandbox.name))[0]
            assert settings['HostConfig']['ReadonlyRootfs']
            assert settings['HostConfig']['NetworkMode'] == 'none'
            assert settings['Config']['User'] == '10001:10001'
            assert settings['HostConfig']['Memory'] == 512 * 1024 * 1024
            assert settings['HostConfig']['MemorySwap'] == settings['HostConfig']['Memory']
            assert settings['HostConfig']['PidsLimit'] == 64
            failure = await sandbox.run_code('print(1 / 0)')
            assert failure.exit_code != 0 and 'ZeroDivisionError' in failure.stderr
            success = await sandbox.run_code(
                "import pandas as pd\nfrom decimal import Decimal\nfrom pathlib import Path\n"
                "data = pd.read_csv('/input/0.csv')\n"
                "print((Decimal(str(data.revenue.iloc[1])) - Decimal(str(data.revenue.iloc[0]))) / Decimal(str(data.revenue.iloc[0])) * 100)\n"
                "Path('/work/note').write_text('task-only')\n"
                "try:\n Path('/input/0.csv').write_text('modified')\n"
                "except OSError:\n print('input-read-only')\n"
                "try:\n Path('/root/forbidden').read_text()\n"
                "except OSError:\n print('host-unavailable')\n")
            assert success.exit_code == 0
            assert success.stdout.splitlines() == ['20.0', 'input-read-only', 'host-unavailable']
            fresh = await sandbox.run_code("from pathlib import Path\nprint('data' in globals())\nprint(Path('/work/note').read_text())")
            assert fresh.stdout.splitlines() == ['False', 'task-only']
            with pytest.raises(CheckFailure, match='three'):
                await sandbox.run_code('print(1)')
        finally:
            await close_safely(sandbox)
        await assert_removed(sandbox)
        assert '2024,100' in path.read_text(encoding='utf-8')
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n2025,120\n', encoding='utf-8')
        asyncio.run(scenario(path))


@docker_test
@pytest.mark.parametrize('mode', ['timeout', 'cancel', 'output'])
def test_real_sandbox_failure_cleanup(mode):
    async def scenario():
        request = request_for()
        request.data_refs = []
        sandbox = sandbox_for(request, {})
        try:
            await sandbox.start()
            if mode == 'timeout':
                sandbox.deadline = asyncio.get_running_loop().time() + 0.5
                with pytest.raises(CheckFailure) as error:
                    await sandbox.run_code('while True: pass')
                assert error.value.code == 'timeout'
            elif mode == 'output':
                with pytest.raises(CheckFailure) as error:
                    await sandbox.run_code("print('x' * 100000)")
                assert error.value.code == 'output_limit'
                assert sandbox.executions[0].output_truncated
                assert len(sandbox.executions[0].stdout.encode()) == 32768
            else:
                running = asyncio.create_task(sandbox.run_code('while True: pass'))
                await asyncio.sleep(0.3)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
        finally:
            await close_safely(sandbox)
        await assert_removed(sandbox)
    asyncio.run(scenario())


@docker_test
def test_real_concurrent_containers_and_json():
    async def scenario(paths):
        async def worker(path, expected):
            sandbox = sandbox_for(request_for(file_format='json'), {'sales': path})
            try:
                result = await sandbox.run_code(
                    "import json\nfrom pathlib import Path\n"
                    "print(sum(row['value'] for row in json.loads(Path('/input/0.json').read_text())))\n"
                    "print(Path('/work/note').exists())\nPath('/work/note').write_text('private')")
                assert result.exit_code == 0
                assert result.stdout.splitlines() == [str(expected), 'False']
                return sandbox.name
            finally:
                await close_safely(sandbox)
                await assert_removed(sandbox)
        names = await asyncio.gather(worker(paths[0], 3), worker(paths[1], 30))
        assert names[0] != names[1]
    with data_files() as directory:
        paths = [directory / 'a.json', directory / 'b.json']
        paths[0].write_text('[{"value":1},{"value":2}]', encoding='utf-8')
        paths[1].write_text('[{"value":10},{"value":20}]', encoding='utf-8')
        asyncio.run(scenario(paths))


@docker_test
def test_real_expired_container_cleanup():
    async def scenario():
        name = f'pierre-check-{uuid4().hex}'
        directory = RUNTIME_ROOT / name.removeprefix('pierre-check-')
        directory.mkdir(parents=True)
        await require_docker('create', '--name', name, '--label', f'{OWNER_LABEL}={OWNER}',
                             '--label', f'{EXPIRY_LABEL}={time.time() - 1}', IMAGE)
        try:
            await cleanup_expired()
            assert name not in (await require_docker('ps', '-a', '--format', '{{.Names}}')).splitlines()
            assert not directory.exists()
        finally:
            if name in (await require_docker('ps', '-a', '--format', '{{.Names}}')).splitlines():
                await require_docker('rm', '-f', name)
            if directory.exists():
                shutil.rmtree(directory)
    asyncio.run(scenario())


@live_test
@pytest.mark.parametrize('claim,expected', [
    ('Revenue grew by 20% from 2024 to 2025.', 'supports'),
    ('Revenue grew by 30% from 2024 to 2025.', 'refutes'),
])
def test_real_model_checks(claim, expected, live_loop):
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n2025,120\n', encoding='utf-8')
        request = request_for(claim=claim)
        result = live_loop.run(run_check(request, authorized_files={'sales': path}))
        assert result.status == 'completed', result.model_dump_json()
        assert result.verdict == expected
        assert result.evidence and result.evidence[0].file_id == 'sales'
        assert any(execution.exit_code == 0 for execution in result.executions)
        assert result.request_id == request.request_id


def test_missing_files_returns_without_execution():
    request = request_for()
    request.data_refs = []
    result = asyncio.run(run_check(request, authorized_files={}))
    assert result.status == 'incomplete', result.model_dump_json()
    assert result.verdict == 'inconclusive'
    assert not result.executions


def test_check_tool_returns_result_through_real_tool_node():
    request = request_for()
    request.data_refs = []
    tool = make_check_tool(authorized_files={})
    schema = convert_to_openai_tool(tool)['function']['parameters']
    assert set(schema['properties']) == set(CheckRequest.model_fields)
    assert tool.args_schema.model_json_schema()['additionalProperties'] is False
    call = {'name': 'check_claim', 'args': request.model_dump(mode='json'), 'id': 'check-call'}
    output = asyncio.run(check_tool_graph(tool).ainvoke(
        {'messages': [AIMessage(content='', tool_calls=[call])]}))
    message = output['messages'][-1]
    assert message.tool_call_id == 'check-call'
    assert message.status == 'success'
    result = CheckResult.model_validate_json(message.content)
    assert result.request_id == request.request_id
    assert result.flag_id == request.flag_id
    assert result.status == 'incomplete' and result.verdict == 'inconclusive'
    assert not result.executions


@pytest.mark.parametrize('changes', [
    {'timeout_s': 61}, {'claim': ''}, {'authorized_files': {'private': 'secret.csv'}},
])
def test_check_tool_rejects_invalid_or_injected_arguments(changes):
    arguments = request_for().model_dump(mode='json')
    arguments.update(changes)
    with pytest.raises(ValidationError):
        asyncio.run(make_check_tool(authorized_files={}).ainvoke(arguments))


def test_concurrent_calls_to_one_check_tool_keep_request_results_separate():
    async def scenario():
        tool = make_check_tool(authorized_files={})
        requests = [request_for(), request_for()]
        for index, request in enumerate(requests):
            request.flag_id = f'flag-{index}'
            request.data_refs = []
        outputs = await asyncio.gather(*(tool.ainvoke(request.model_dump(mode='json')) for request in requests))
        results = [CheckResult.model_validate(output) for output in outputs]
        assert [(item.request_id, item.flag_id) for item in results] == [
            (item.request_id, item.flag_id) for item in requests]
        assert all(item.status == 'incomplete' and not item.executions for item in results)
    asyncio.run(scenario())


def test_check_tools_keep_authorization_separate_and_snapshot_the_mapping():
    async def scenario(path):
        files = {}
        first = make_check_tool(authorized_files=files)
        files['sales'] = path
        second = make_check_tool(authorized_files=files)
        files.clear()
        arguments = request_for(timeout=1).model_dump(mode='json')
        with pytest.raises(ValueError, match='Unauthorized'):
            await first.ainvoke(arguments)
        # An authorized call reaches the real queue; no model or container is needed.
        slots = service_slots()
        await slots.acquire()
        await slots.acquire()
        try:
            result = CheckResult.model_validate(await second.ainvoke(arguments))
            assert result.status == 'incomplete' and result.error_code == 'timeout'
            assert not result.executions
            assert slots.locked()
        finally:
            slots.release()
            slots.release()

    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n2025,120\n', encoding='utf-8')
        asyncio.run(scenario(path))


def test_check_tool_cancellation_propagates_through_tool_node():
    async def scenario():
        slots = service_slots()
        await slots.acquire()
        await slots.acquire()
        request = request_for()
        request.data_refs = []
        graph = check_tool_graph(make_check_tool(authorized_files={}))
        call = {'name': 'check_claim', 'args': request.model_dump(mode='json'), 'id': 'cancel-call'}
        pending = asyncio.create_task(graph.ainvoke({'messages': [AIMessage(content='', tool_calls=[call])]}))
        try:
            # Cancel only after run_check is actually waiting for an occupied slot.
            async with asyncio.timeout(2):
                while not slots._waiters:
                    await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert slots.locked()
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            slots.release()
            slots.release()
        output = await graph.ainvoke({'messages': [AIMessage(content='', tool_calls=[call])]})
        result = CheckResult.model_validate_json(output['messages'][-1].content)
        assert result.status == 'incomplete' and result.error_code is None
    asyncio.run(scenario())


@live_test
def test_real_model_check_through_tool_node(live_loop):
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2024,100\n2025,120\n', encoding='utf-8')
        request = request_for(claim='Revenue grew by 20% from 2024 to 2025.')
        tool = make_check_tool(authorized_files={'sales': path})
        call = {'name': tool.name, 'args': request.model_dump(mode='json'), 'id': 'live-check'}
        output = live_loop.run(check_tool_graph(tool).ainvoke(
            {'messages': [AIMessage(content='', tool_calls=[call])]}))
        result = CheckResult.model_validate_json(output['messages'][-1].content)
        assert result.status == 'completed', result.model_dump_json()
        assert result.verdict == 'supports'
        assert result.evidence and result.evidence[0].file_id == 'sales'
        assert result.request_id == request.request_id and result.flag_id == request.flag_id
        assert any(item.exit_code == 0 and not item.output_truncated for item in result.executions)


@live_test
def test_real_model_missing_period(live_loop):
    with data_files() as directory:
        path = directory / 'sales.csv'
        path.write_text('year,revenue\n2025,120\n', encoding='utf-8')
        result = live_loop.run(run_check(request_for(), authorized_files={'sales': path}))
        assert result.status == 'incomplete', result.model_dump_json()
        assert result.verdict == 'inconclusive'
