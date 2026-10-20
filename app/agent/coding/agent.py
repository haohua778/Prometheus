import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from openai import APIError

from app.agent.coding.docker_backend import DockerBackend
from app.agent.coding.graph import CheckState, build_check_graph
from app.agent.coding.limits import MAX_CONCURRENT_CHECKS, MAX_GRAPH_STEPS
from app.agent.coding.sandbox import Sandbox, close_safely
from app.agent.coding.schemas import (
    CheckFailure,
    CheckRequest,
    CheckResult,
    Finding,
    failure_status,
)
from app.agent.coding.staging import validate_files
from app.config import Settings
from app.runtime_logging import (
    RuntimeLogCallback,
    check_request_id_context,
    exception_fields,
    log_event,
    request_id_context,
)

LOG_LEVELS = {'completed': logging.INFO, 'incomplete': logging.WARNING, 'failed': logging.ERROR}
# Queuing belongs to the service process; task data never lives in this shared object.
_slots: asyncio.Semaphore | None = None
_slots_loop = None


def service_slots() -> asyncio.Semaphore:
    global _slots, _slots_loop
    loop = asyncio.get_running_loop()
    if _slots_loop is not loop:
        _slots = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
        _slots_loop = loop
    return _slots


async def investigate(model: BaseChatModel, sandbox: Sandbox, callbacks: RuntimeLogCallback) -> Finding:
    if not sandbox.paths:
        return Finding(verdict='inconclusive', explanation='No supporting data files were supplied.', evidence=[])
    state: CheckState = {'messages': [HumanMessage(sandbox.request.model_dump_json())], 'model_calls': 0, 'result': None}
    try:
        output = await build_check_graph(model, sandbox).ainvoke(
            state, config={'recursion_limit': MAX_GRAPH_STEPS, 'callbacks': [callbacks]})
    except GraphRecursionError as error:
        raise CheckFailure('budget_exhausted', 'The graph step budget was exhausted.') from error
    except APIError as error:
        log_event('check.model_failed', level=logging.ERROR, **exception_fields(error))
        raise CheckFailure('model_error', 'The configured model is unavailable.') from error
    return output['result']


async def run_check(request: CheckRequest, *, authorized_files: Mapping[str, Path],
                    model: BaseChatModel, settings: Settings) -> CheckResult:
    started = perf_counter()
    # Validate and copy the request before any await; the caller may reuse its own object.
    request = CheckRequest.model_validate(request.model_dump())
    files = validate_files(request, authorized_files)
    deadline = asyncio.get_running_loop().time() + request.timeout_s
    sandbox = Sandbox(request, files, deadline, DockerBackend(settings.coding_image), settings.coding_staging_dir)
    request_token = request_id_context.set(request_id_context.get() or request.request_id)
    check_token = check_request_id_context.set(request.request_id)
    callbacks = RuntimeLogCallback()
    failure = None
    slots = service_slots()
    acquired = False
    log_event('check.started', flag_id=request.flag_id, file_count=len(files))
    try:
        try:
            async with asyncio.timeout_at(deadline):
                await slots.acquire()
                acquired = True
                finding = await investigate(model, sandbox, callbacks)
        except TimeoutError:
            failure = CheckFailure('timeout', 'The check exceeded its time budget.')
        except CheckFailure as error:
            failure = error
        finally:
            try:
                await close_safely(sandbox)
            except CheckFailure as error:
                failure = error
                log_event('check.cleanup_failed', level=logging.ERROR, **exception_fields(error))
            finally:
                if acquired:
                    slots.release()
        if failure is not None:
            finding = Finding(verdict='inconclusive', explanation=str(failure), evidence=[])
            status = failure_status(failure.code)
        else:
            status = 'incomplete' if finding.verdict == 'inconclusive' else 'completed'
        result = CheckResult(**finding.model_dump(), request_id=request.request_id, flag_id=request.flag_id,
                             status=status, executions=sandbox.executions,
                             runtime_ms=round((perf_counter() - started) * 1000, 2),
                             error_code=failure.code if failure is not None else None)
        log_event(f'check.{status}', level=LOG_LEVELS[status], verdict=result.verdict, error_code=result.error_code,
                  execution_count=len(sandbox.executions), duration_ms=result.runtime_ms)
        return result
    except asyncio.CancelledError:
        log_event('check.cancelled', level=logging.WARNING)
        raise
    except Exception as error:
        log_event('check.failed', level=logging.ERROR, **exception_fields(error))
        raise
    finally:
        callbacks.finish_pending()
        check_request_id_context.reset(check_token)
        request_id_context.reset(request_token)
