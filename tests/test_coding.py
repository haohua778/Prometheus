import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from app.agent.coding import CheckRequest, make_check_tool, run_check
from app.agent.coding.agent import service_slots
from app.agent.coding.docker_backend import (
    EXPIRY_LABEL,
    NAME_PREFIX,
    OWNER,
    OWNER_LABEL,
    DockerBackend,
    cleanup_expired,
    require_docker,
)
from app.agent.coding.graph import (
    REMINDER,
    InvalidToolCall,
    build_check_graph,
    validate_finding,
)
from app.agent.coding.limits import (
    INSPECTION_BYTES,
    INSPECTION_SAMPLES,
    MAX_CONCURRENT_CHECKS,
    MAX_FILE_BYTES,
    MAX_MODEL_CALLS,
    OUTPUT_BYTES,
)
from app.agent.coding.sandbox import Sandbox, close_safely
from app.agent.coding.schemas import CheckFailure, CheckResult, Execution
from app.agent.coding.staging import stage_inputs, validate_files
from app.agent.coding.tools import INSPECT_DATA, RUN_CODE, SUBMIT_FINDING
from app.config import PROJECT_ROOT, Settings, get_settings
from app.llm import build_chat_model
from app.runtime_logging import check_request_id_context, request_id_context
from sandbox.inspect_data import inspect
from tests.conftest import ScriptedChatModel, make_tool_call_message

SALES_CSV = 'year,revenue\n2024,100\n2025,120\n'
INCONCLUSIVE = {'verdict': 'inconclusive', 'explanation': 'Missing data', 'evidence': []}


@pytest.fixture
def settings(tmp_path):
    # The key is never sent: these tests either finish before a model call or use ScriptedChatModel.
    return Settings(_env_file=None, llm_api_key='offline-unused-key', coding_staging_dir=tmp_path / 'staging')


@pytest.fixture
def model(settings):
    return build_chat_model(settings)


@pytest.fixture
def sales_path(tmp_path):
    path = tmp_path / 'sales.csv'
    path.write_text(SALES_CSV, encoding='utf-8')
    return path


def request_for(file_id='sales', file_format='csv', timeout=60, claim='Revenue grew by 30%.', data_refs=None):
    refs = [{'file_id': file_id, 'format': file_format}] if data_refs is None else data_refs
    return CheckRequest(request_id=uuid4().hex, flag_id='flag-1', claim=claim,
                        question='Compute the 2024 to 2025 revenue growth. Values are in millions, same scope.',
                        data_refs=refs, timeout_s=timeout)


def sandbox_for(request, files, settings):
    return Sandbox(request, validate_files(request, files), asyncio.get_running_loop().time() + request.timeout_s,
                   DockerBackend(settings.coding_image), settings.coding_staging_dir)


def check_tool_graph(tool):
    builder = StateGraph(MessagesState)
    builder.add_node('tools', ToolNode([tool], handle_tool_errors=False))
    builder.add_edge(START, 'tools')
    builder.add_edge('tools', END)
    return builder.compile(checkpointer=False)


def run_inspection(path):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / 'sandbox' / 'inspect_data.py'), str(path), 'json',
                           str(INSPECTION_BYTES), str(INSPECTION_SAMPLES)],
                          capture_output=True, timeout=5, check=False)


@asynccontextmanager
async def occupied_slots():
    slots = service_slots()
    for _ in range(MAX_CONCURRENT_CHECKS):
        await slots.acquire()
    try:
        yield slots
    finally:
        for _ in range(MAX_CONCURRENT_CHECKS):
            slots.release()


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


@pytest.mark.parametrize('arguments', [
    {},
    {'verdict': 'supports', 'explanation': 'Unsupported claim', 'evidence': []},
    {**INCONCLUSIVE, 'extra': True},
    {**INCONCLUSIVE, 'evidence': [{'file_id': 'private', 'location': 'row 1', 'observed_value': 1}]},
    {**INCONCLUSIVE, 'explanation': 'x' * 16385},
])
def test_final_finding_rejects_invalid_or_unsupported_content(arguments):
    with pytest.raises((ValidationError, InvalidToolCall)):
        validate_finding(arguments, inspected_files=set(), executions=[])


def test_inconclusive_finding_needs_no_execution():
    finding = validate_finding(INCONCLUSIVE, inspected_files=set(), executions=[])
    assert finding.verdict == 'inconclusive'


def test_file_authorization_and_size(tmp_path, sales_path):
    assert validate_files(request_for(), {'sales': sales_path})['sales'] == sales_path.absolute()
    with pytest.raises(ValueError, match='Unauthorized'):
        validate_files(request_for(), {})
    with pytest.raises(ValueError, match='regular file'):
        validate_files(request_for(), {'sales': tmp_path})
    with sales_path.open('wb') as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match='10 MiB'):
        validate_files(request_for(), {'sales': sales_path})


def test_symbolic_input_rejected(tmp_path, sales_path):
    link = tmp_path / 'link.csv'
    link.symlink_to(sales_path)
    with pytest.raises(ValueError, match='symbolic'):
        validate_files(request_for(), {'sales': link})


@pytest.mark.parametrize('file_format,content', [
    ('csv', SALES_CSV),
    ('json', '[{"year":2024,"revenue":100},{"year":2025,"revenue":120}]'),
])
def test_real_data_parser(tmp_path, file_format, content):
    path = tmp_path / f'data.{file_format}'
    path.write_text(content, encoding='utf-8')
    result = inspect(path, file_format, INSPECTION_SAMPLES)
    assert result['fields'] == ['year', 'revenue']
    assert result['record_count'] == 2
    assert len(result['samples']) == 2


@pytest.mark.parametrize('file_format,content', [
    ('json', '{}'), ('json', '[1]'), ('json', '[{"a":NaN}]'),
    ('csv', 'a,a\n1,2\n'), ('csv', 'a,b\n1\n'),
])
def test_real_data_parser_rejects_invalid_records(tmp_path, file_format, content):
    path = tmp_path / 'invalid'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        inspect(path, file_format, INSPECTION_SAMPLES)


@pytest.mark.parametrize('content,expected_code', [
    pytest.param('{}', 2, id='not-record-array'),
    pytest.param('[{"value":"' + 'x' * 17000 + '"}]', 3, id='preview-too-large'),
    pytest.param('[' * 20000 + '0' + ']' * 20000, 2, id='excessive-nesting'),
])
def test_inspection_script_has_explicit_failure_codes(tmp_path, content, expected_code):
    path = tmp_path / 'data.json'
    path.write_text(content, encoding='utf-8')
    result = run_inspection(path)
    assert result.returncode == expected_code
    assert len(result.stdout) < INSPECTION_BYTES
    assert json.loads(result.stdout)['error']


@pytest.mark.parametrize('extra_bytes,expected_code', [(0, 0), (1, 3)])
def test_inspection_limit_includes_the_output_newline(tmp_path, extra_bytes, expected_code):
    path = tmp_path / 'data.json'
    path.write_text('[{"x":""}]', encoding='utf-8')
    base_bytes = len(json.dumps(inspect(path, 'json', INSPECTION_SAMPLES), ensure_ascii=False).encode('utf-8')) + 1
    path.write_text(json.dumps([{'x': 'v' * (INSPECTION_BYTES + extra_bytes - base_bytes)}]), encoding='utf-8')
    result = run_inspection(path)
    assert result.returncode == expected_code
    assert len(result.stdout) <= INSPECTION_BYTES


def test_tools_and_prompt_share_the_configured_limits(settings):
    async def scenario():
        model = ScriptedChatModel(replies=[make_tool_call_message(SUBMIT_FINDING, INCONCLUSIVE)])
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        await build_check_graph(model, sandbox).ainvoke({'messages': [HumanMessage('check')], 'model_calls': 0,
                                                         'result': None})
        assert model.bound_tool_names == [INSPECT_DATA, RUN_CODE, SUBMIT_FINDING]
        assert f'{MAX_MODEL_CALLS} model calls' in model.prompts[0][0].text
    asyncio.run(scenario())


def test_graph_returns_correctable_errors_to_the_model(settings):
    async def scenario():
        model = ScriptedChatModel(replies=[
            AIMessage('The claim looks fine.'),
            AIMessage(content='', tool_calls=[
                {'name': 'unknown_tool', 'args': {}, 'id': 'unknown'},
                {'name': SUBMIT_FINDING, 'args': INCONCLUSIVE, 'id': 'mixed-submit'}],
                invalid_tool_calls=[{'name': RUN_CODE, 'args': '{', 'id': 'broken', 'error': None}]),
            make_tool_call_message(SUBMIT_FINDING, {'verdict': 'supports', 'explanation': 'No run', 'evidence': []},
                                   'unsupported'),
            make_tool_call_message(SUBMIT_FINDING, INCONCLUSIVE, 'final'),
        ])
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        output = await build_check_graph(model, sandbox).ainvoke(
            {'messages': [HumanMessage('check')], 'model_calls': 0, 'result': None})
        assert output['result'].verdict == 'inconclusive'
        assert output['model_calls'] == 4
        assert isinstance(output['messages'][2], HumanMessage) and output['messages'][2].text == REMINDER
        errors = {message.tool_call_id: message.text for message in output['messages']
                  if isinstance(message, ToolMessage) and message.status == 'error'}
        assert set(errors) == {'broken', 'unknown', 'mixed-submit', 'unsupported'}
        assert 'not valid JSON' in errors['broken'] and 'alone' in errors['mixed-submit']
        assert 'successful' in errors['unsupported']
        assert not sandbox.started
    asyncio.run(scenario())


def test_graph_stops_when_the_model_never_submits(settings):
    async def scenario():
        model = ScriptedChatModel(replies=[AIMessage('Still thinking.')])
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        with pytest.raises(CheckFailure) as error:
            await build_check_graph(model, sandbox).ainvoke(
                {'messages': [HumanMessage('check')], 'model_calls': 0, 'result': None},
                config={'recursion_limit': 100})
        assert error.value.code == 'budget_exhausted'
        assert len(model.prompts) == MAX_MODEL_CALLS
    asyncio.run(scenario())


def test_concurrent_check_ids_and_parent_context(model, settings):
    async def scenario():
        parent_token = request_id_context.set('parent-request')
        try:
            first, second = request_for(data_refs=[]), request_for(data_refs=[])
            results = await asyncio.gather(run_check(first, authorized_files={}, model=model, settings=settings),
                                           run_check(second, authorized_files={}, model=model, settings=settings))
            assert [result.request_id for result in results] == [first.request_id, second.request_id]
            assert all(result.status == 'incomplete' for result in results)
            assert results[0].executions is not results[1].executions
            assert request_id_context.get() == 'parent-request'
            assert check_request_id_context.get() is None
        finally:
            request_id_context.reset(parent_token)
    asyncio.run(scenario())


def test_queue_timeout_and_cancellation_release_no_foreign_slots(model, settings):
    async def scenario():
        request = request_for(timeout=1, data_refs=[])
        async with occupied_slots() as slots:
            result = await run_check(request, authorized_files={}, model=model, settings=settings)
            assert result.error_code == 'timeout' and result.status == 'incomplete'
            pending = asyncio.create_task(run_check(request, authorized_files={}, model=model, settings=settings))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert slots.locked()
        result = await run_check(request, authorized_files={}, model=model, settings=settings)
        assert result.error_code is None
    asyncio.run(scenario())


def test_staging_directory_cleanup_without_container(settings):
    async def scenario():
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        sandbox.directory.mkdir(parents=True)
        (sandbox.directory / 'input.csv').write_text('a\n1\n', encoding='utf-8')
        await close_safely(sandbox)
        assert not sandbox.directory.exists()
    asyncio.run(scenario())


def test_cleanup_rejects_a_dangling_staging_link(tmp_path, settings):
    async def scenario(target):
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        settings.coding_staging_dir.mkdir(parents=True)
        sandbox.directory.symlink_to(target, target_is_directory=True)
        try:
            with pytest.raises(CheckFailure) as error:
                await close_safely(sandbox)
            assert error.value.code == 'cleanup_failed'
        finally:
            sandbox.directory.unlink()
        assert not target.exists()
    asyncio.run(scenario(tmp_path / 'nonexistent'))


def test_queued_request_is_an_independent_snapshot(model, settings):
    async def scenario():
        slots = service_slots()
        for _ in range(MAX_CONCURRENT_CHECKS):
            await slots.acquire()
        request = request_for(data_refs=[])
        original_id = request.request_id
        pending = asyncio.create_task(run_check(request, authorized_files={}, model=model, settings=settings))
        await asyncio.sleep(0)
        request.request_id = 'changed-after-submission'
        for _ in range(MAX_CONCURRENT_CHECKS):
            slots.release()
        result = await pending
        assert result.request_id == original_id
    asyncio.run(scenario())


def test_request_is_revalidated_at_the_public_boundary(model, settings):
    request = request_for(data_refs=[])
    request.timeout_s = 0
    with pytest.raises(ValidationError):
        asyncio.run(run_check(request, authorized_files={}, model=model, settings=settings))


def test_unknown_tool_file_is_an_explicit_data_failure(settings):
    async def scenario():
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        with pytest.raises(CheckFailure) as error:
            await sandbox.inspect_data('unavailable')
        assert error.value.code == 'data_invalid'
        assert not sandbox.started
    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['deleted', 'oversized'])
def test_input_changes_before_staging_are_explicit_failures(sales_path, settings, change):
    async def scenario():
        sandbox = sandbox_for(request_for(), {'sales': sales_path}, settings)
        if change == 'deleted':
            sales_path.unlink()
        else:
            with sales_path.open('wb') as stream:
                stream.truncate(MAX_FILE_BYTES + 1)
        try:
            with pytest.raises(CheckFailure) as error:
                await sandbox.start()
            assert error.value.code == 'data_invalid'
            assert not sandbox.started
        finally:
            await close_safely(sandbox)
        assert not sandbox.directory.exists()
    asyncio.run(scenario())


def test_staged_copy_is_independent_of_original_file(sales_path, settings):
    async def scenario():
        sandbox = sandbox_for(request_for(), {'sales': sales_path}, settings)
        try:
            stage_inputs(sandbox.request, sandbox.files, sandbox.directory)
            sales_path.write_text('changed-after-staging', encoding='utf-8')
            assert (sandbox.directory / '0.csv').read_text(encoding='utf-8') == SALES_CSV
        finally:
            await close_safely(sandbox)
        assert not sandbox.directory.exists()
    asyncio.run(scenario())


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


@pytest.fixture
def live_settings(tmp_path):
    return get_settings().model_copy(update={'coding_staging_dir': tmp_path / 'staging'})


async def container_names():
    return (await require_docker('ps', '-a', '--format', '{{.Names}}')).splitlines()


async def assert_removed(sandbox, name):
    assert name not in await container_names()
    assert not sandbox.directory.exists()


@docker_test
def test_real_sandbox_calculation_repair_and_filesystem(sales_path, settings):
    async def scenario():
        sandbox = sandbox_for(request_for(), {'sales': sales_path}, settings)
        name = None
        try:
            preview = await sandbox.inspect_data('sales')
            assert preview['record_count'] == 2
            name = sandbox.backend.name
            container = json.loads(await require_docker('inspect', name))[0]
            assert container['HostConfig']['ReadonlyRootfs']
            assert container['HostConfig']['NetworkMode'] == 'none'
            assert container['Config']['User'] == '10001:10001'
            assert container['HostConfig']['Memory'] == 512 * 1024 * 1024
            assert container['HostConfig']['MemorySwap'] == container['HostConfig']['Memory']
            assert container['HostConfig']['PidsLimit'] == 64
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
            with pytest.raises(CheckFailure) as error:
                await sandbox.run_code('print(1)')
            assert error.value.code == 'budget_exhausted'
        finally:
            await close_safely(sandbox)
        await assert_removed(sandbox, name)
        assert '2024,100' in sales_path.read_text(encoding='utf-8')
    asyncio.run(scenario())


@docker_test
@pytest.mark.parametrize('mode', ['timeout', 'cancel', 'output'])
def test_real_sandbox_failure_cleanup(settings, mode):
    async def scenario():
        sandbox = sandbox_for(request_for(data_refs=[]), {}, settings)
        try:
            await sandbox.start()
            name = sandbox.backend.name
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
                assert len(sandbox.executions[0].stdout.encode()) == OUTPUT_BYTES
            else:
                running = asyncio.create_task(sandbox.run_code('while True: pass'))
                await asyncio.sleep(0.3)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
        finally:
            await close_safely(sandbox)
        await assert_removed(sandbox, name)
    asyncio.run(scenario())


@docker_test
def test_real_concurrent_containers_and_json(tmp_path, settings):
    async def scenario(paths):
        async def worker(path, expected):
            sandbox = sandbox_for(request_for(file_format='json'), {'sales': path}, settings)
            try:
                result = await sandbox.run_code(
                    "import json\nfrom pathlib import Path\n"
                    "print(sum(row['value'] for row in json.loads(Path('/input/0.json').read_text())))\n"
                    "print(Path('/work/note').exists())\nPath('/work/note').write_text('private')")
                name = sandbox.backend.name
                assert result.exit_code == 0
                assert result.stdout.splitlines() == [str(expected), 'False']
                return name
            finally:
                await close_safely(sandbox)
        names = await asyncio.gather(worker(paths[0], 3), worker(paths[1], 30))
        assert names[0] != names[1]
        assert not set(names) & set(await container_names())
    paths = [tmp_path / 'a.json', tmp_path / 'b.json']
    paths[0].write_text('[{"value":1},{"value":2}]', encoding='utf-8')
    paths[1].write_text('[{"value":10},{"value":20}]', encoding='utf-8')
    asyncio.run(scenario(paths))


@docker_test
def test_real_expired_container_cleanup(settings):
    async def scenario():
        identifier = uuid4().hex
        name = f'{NAME_PREFIX}{identifier}'
        directory = settings.coding_staging_dir / identifier
        directory.mkdir(parents=True)
        await require_docker('create', '--name', name, '--label', f'{OWNER_LABEL}={OWNER}',
                             '--label', f'{EXPIRY_LABEL}={time.time() - 1}', settings.coding_image)
        try:
            await cleanup_expired(settings.coding_staging_dir)
            assert name not in await container_names()
            assert not directory.exists()
        finally:
            if name in await container_names():
                await require_docker('rm', '-f', name)
    asyncio.run(scenario())


@live_test
@pytest.mark.parametrize('claim,expected', [
    ('Revenue grew by 20% from 2024 to 2025.', 'supports'),
    ('Revenue grew by 30% from 2024 to 2025.', 'refutes'),
])
def test_real_model_checks(sales_path, live_settings, claim, expected, live_loop):
    request = request_for(claim=claim)
    result = live_loop.run(run_check(request, authorized_files={'sales': sales_path},
                                     model=build_chat_model(live_settings), settings=live_settings))
    assert result.status == 'completed', result.model_dump_json()
    assert result.verdict == expected
    assert result.evidence and result.evidence[0].file_id == 'sales'
    assert any(execution.exit_code == 0 for execution in result.executions)
    assert result.request_id == request.request_id


def test_missing_files_returns_without_execution(model, settings):
    result = asyncio.run(run_check(request_for(data_refs=[]), authorized_files={}, model=model, settings=settings))
    assert result.status == 'incomplete', result.model_dump_json()
    assert result.verdict == 'inconclusive'
    assert not result.executions


def test_check_tool_returns_result_through_real_tool_node(model, settings):
    request = request_for(data_refs=[])
    tool = make_check_tool(authorized_files={}, model=model, settings=settings)
    schema = convert_to_openai_tool(tool)['function']['parameters']
    assert set(schema['properties']) == set(CheckRequest.model_fields)
    assert tool.args_schema.model_json_schema()['additionalProperties'] is False
    output = asyncio.run(check_tool_graph(tool).ainvoke(
        {'messages': [make_tool_call_message('check_claim', request.model_dump(mode='json'), 'check-call')]}))
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
def test_check_tool_rejects_invalid_or_injected_arguments(model, settings, changes):
    arguments = request_for().model_dump(mode='json')
    arguments.update(changes)
    with pytest.raises(ValidationError):
        asyncio.run(make_check_tool(authorized_files={}, model=model, settings=settings).ainvoke(arguments))


def test_concurrent_calls_to_one_check_tool_keep_request_results_separate(model, settings):
    async def scenario():
        tool = make_check_tool(authorized_files={}, model=model, settings=settings)
        requests = [request_for(data_refs=[]), request_for(data_refs=[])]
        for index, request in enumerate(requests):
            request.flag_id = f'flag-{index}'
        outputs = await asyncio.gather(*(tool.ainvoke(request.model_dump(mode='json')) for request in requests))
        results = [CheckResult.model_validate(output) for output in outputs]
        assert [(item.request_id, item.flag_id) for item in results] == [
            (item.request_id, item.flag_id) for item in requests]
        assert all(item.status == 'incomplete' and not item.executions for item in results)
    asyncio.run(scenario())


def test_check_tools_keep_authorization_separate_and_snapshot_the_mapping(sales_path, model, settings):
    async def scenario():
        files = {}
        first = make_check_tool(authorized_files=files, model=model, settings=settings)
        files['sales'] = sales_path
        second = make_check_tool(authorized_files=files, model=model, settings=settings)
        files.clear()
        arguments = request_for(timeout=1).model_dump(mode='json')
        with pytest.raises(ValueError, match='Unauthorized'):
            await first.ainvoke(arguments)
        # An authorized call reaches the real queue; no model or container is needed.
        async with occupied_slots() as slots:
            result = CheckResult.model_validate(await second.ainvoke(arguments))
            assert result.status == 'incomplete' and result.error_code == 'timeout'
            assert not result.executions
            assert slots.locked()
    asyncio.run(scenario())


def test_check_tool_cancellation_propagates_through_tool_node(model, settings):
    async def scenario():
        request = request_for(data_refs=[])
        graph = check_tool_graph(make_check_tool(authorized_files={}, model=model, settings=settings))
        call_message = make_tool_call_message('check_claim', request.model_dump(mode='json'), 'cancel-call')
        async with occupied_slots() as slots:
            pending = asyncio.create_task(graph.ainvoke({'messages': [call_message]}))
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
        output = await graph.ainvoke({'messages': [call_message]})
        result = CheckResult.model_validate_json(output['messages'][-1].content)
        assert result.status == 'incomplete' and result.error_code is None
    asyncio.run(scenario())


@live_test
def test_real_model_check_through_tool_node(sales_path, live_settings, live_loop):
    request = request_for(claim='Revenue grew by 20% from 2024 to 2025.')
    tool = make_check_tool(authorized_files={'sales': sales_path}, model=build_chat_model(live_settings),
                           settings=live_settings)
    output = live_loop.run(check_tool_graph(tool).ainvoke(
        {'messages': [make_tool_call_message(tool.name, request.model_dump(mode='json'), 'live-check')]}))
    result = CheckResult.model_validate_json(output['messages'][-1].content)
    assert result.status == 'completed', result.model_dump_json()
    assert result.verdict == 'supports'
    assert result.evidence and result.evidence[0].file_id == 'sales'
    assert result.request_id == request.request_id and result.flag_id == request.flag_id
    assert any(item.exit_code == 0 and not item.output_truncated for item in result.executions)


@live_test
def test_real_model_missing_period(tmp_path, live_settings, live_loop):
    path = tmp_path / 'sales.csv'
    path.write_text('year,revenue\n2025,120\n', encoding='utf-8')
    result = live_loop.run(run_check(request_for(), authorized_files={'sales': path},
                                     model=build_chat_model(live_settings), settings=live_settings))
    assert result.status == 'incomplete', result.model_dump_json()
    assert result.verdict == 'inconclusive'
