import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from openai import APIError

from app.agent.coding.graph import CheckContext, CheckState, build_check_graph
from app.agent.coding.sandbox import Sandbox, close_safely, validate_files
from app.agent.coding.schemas import (
    CheckFailure,
    CheckRequest,
    CheckResult,
    Finding,
    failure_status,
)
from app.agent.coding.tools import make_tools
from app.config import get_settings
from app.llm import MissingAPIKeyError, build_chat_model
from app.runtime_logging import (
    RuntimeLogCallback,
    check_request_id_context,
    exception_fields,
    log_event,
    request_id_context,
)

# Queuing belongs to the service process; task data never lives in this shared object.
_slots: asyncio.Semaphore | None = None
_slots_loop = None


def service_slots() -> asyncio.Semaphore:
    global _slots, _slots_loop
    loop = asyncio.get_running_loop()
    if _slots_loop is not loop:
        _slots = asyncio.Semaphore(2)
        _slots_loop = loop
    return _slots


async def run_check(request: CheckRequest, *, authorized_files: Mapping[str, Path]) -> CheckResult:
    started = perf_counter()
    # Validate and copy the request before any await; the caller may reuse its own object.
    request = CheckRequest.model_validate(request.model_dump())
    files = validate_files(request, authorized_files)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + request.timeout_s
    sandbox = Sandbox(request, files, deadline)
    request_token = request_id_context.set(request_id_context.get() or request.request_id)
    check_token = check_request_id_context.set(request.request_id)
    callbacks = RuntimeLogCallback()
    finding = Finding(verdict='inconclusive', explanation='The check did not complete.', evidence=[])
    status = 'failed'
    error_code = None
    slots = service_slots()
    acquired = False
    log_event('check.started', flag_id=request.flag_id, file_count=len(files))
    try:
        try:
            async with asyncio.timeout_at(deadline):
                await slots.acquire()
                acquired = True
                if not files:
                    finding = Finding(verdict='inconclusive', explanation='No supporting data files were supplied.', evidence=[])
                    status = 'incomplete'
                else:
                    tools = make_tools(sandbox)
                    model = build_chat_model(get_settings())
                    state: CheckState = {
                        'request': request,
                        'messages': [HumanMessage(request.model_dump_json())],
                        'model_calls': 0,
                        'executions': [],
                        'result': None,
                    }
                    graph = build_check_graph(model, tools)
                    output = await graph.ainvoke(state, context=CheckContext(sandbox, tools),
                                                config={'recursion_limit': 20, 'callbacks': [callbacks]})
                    finding = output['result']
                    status = 'incomplete' if finding.verdict == 'inconclusive' else 'completed'
        except TimeoutError:
            error_code = 'timeout'
            status = 'incomplete'
            finding = Finding(verdict='inconclusive', explanation='The check exceeded its time budget.', evidence=[])
        except GraphRecursionError:
            error_code = 'budget_exhausted'
            status = 'incomplete'
            finding = Finding(verdict='inconclusive', explanation='The graph step budget was exhausted.', evidence=[])
        except CheckFailure as error:
            error_code = error.code
            status = failure_status(error.code)
            finding = Finding(verdict='inconclusive', explanation=str(error), evidence=[])
        except (APIError, MissingAPIKeyError) as error:
            error_code = 'model_error'
            finding = Finding(verdict='inconclusive', explanation='The configured model is unavailable.', evidence=[])
            log_event('check.model_failed', level=logging.ERROR, **exception_fields(error))
        finally:
            try:
                await close_safely(sandbox)
            except CheckFailure as error:
                error_code = 'cleanup_failed'
                status = 'failed'
                finding = Finding(verdict='inconclusive', explanation=str(error), evidence=[])
                log_event('check.cleanup_failed', level=logging.ERROR, **exception_fields(error))
            finally:
                if acquired:
                    slots.release()
        result = CheckResult(**finding.model_dump(), request_id=request.request_id, flag_id=request.flag_id,
                             status=status, executions=sandbox.executions,
                             runtime_ms=round((perf_counter() - started) * 1000, 2), error_code=error_code)
        log_event('check.completed' if status == 'completed' else 'check.incomplete' if status == 'incomplete' else 'check.failed',
                  level=logging.INFO if status == 'completed' else logging.WARNING if status == 'incomplete' else logging.ERROR,
                  verdict=result.verdict, error_code=error_code, execution_count=len(sandbox.executions),
                  duration_ms=result.runtime_ms)
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
