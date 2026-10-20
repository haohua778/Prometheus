import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from app.agent.coding.limits import INSPECTION_BYTES, MAX_EXECUTIONS
from app.agent.coding.schemas import CheckFailure, CheckRequest, Execution
from app.agent.coding.staging import remove_staging, stage_inputs, staged_name
from app.runtime_logging import exception_fields, log_event


class ExecutionBackend(Protocol):
    # The place that actually runs code; the Docker backend is one implementation.
    input_root: str
    work_root: str

    async def start(self, input_dir: Path, timeout: float) -> None: ...

    async def inspect(self, path: str, file_format: str, timeout: float) -> tuple[int, str]: ...

    async def run_code(self, code: str, execution: Execution, timeout: float) -> None: ...

    async def close(self) -> None: ...


class Sandbox:
    # One check's inputs, deadline and execution budget; the backend only runs what this class allows.
    def __init__(self, request: CheckRequest, files: Mapping[str, Path], deadline: float,
                 backend: ExecutionBackend, staging_root: Path):
        self.request = request
        self.files = dict(files)
        self.deadline = deadline
        self.backend = backend
        self.staging_root = staging_root
        self.directory = staging_root / uuid4().hex
        self.formats = {ref.file_id: ref.format for ref in request.data_refs}
        self.paths = {ref.file_id: f'{backend.input_root}/{staged_name(index, ref.format)}'
                      for index, ref in enumerate(request.data_refs)}
        self.executions: list[Execution] = []
        self.inspected: set[str] = set()
        self.staging: asyncio.Task | None = None
        self.started = False

    def remaining(self) -> float:
        """Return the seconds left before the deadline; raises timeout once it has passed."""
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise CheckFailure('timeout', 'The check exceeded its time budget.')
        return remaining

    async def start(self) -> None:
        """Copy the inputs and start the backend on first use; later calls do nothing."""
        if self.started:
            return
        self.remaining()
        # Cancelling stops only the wait: the copy thread keeps running, so close() waits for it.
        self.staging = asyncio.create_task(asyncio.to_thread(stage_inputs, self.request, self.files, self.directory))
        await asyncio.shield(self.staging)
        await self.backend.start(self.directory, timeout=self.remaining())
        self.started = True

    async def inspect_data(self, file_id: str) -> dict:
        """Check one authorized file's format, mark it inspected and return its path, fields and samples."""
        if file_id not in self.paths:
            raise CheckFailure('data_invalid', 'The requested file ID is not authorized for this task.')
        await self.start()
        code, stdout = await self.backend.inspect(self.paths[file_id], self.formats[file_id], timeout=self.remaining())
        if code == 2:
            raise CheckFailure('data_invalid', 'Input does not match the supported CSV/JSON record format.')
        if code == 3:
            raise CheckFailure('output_limit', f'Inspection output exceeded {INSPECTION_BYTES // 1024} KiB.')
        if code != 0:
            raise CheckFailure('environment_error', 'Data inspection process failed.')
        self.inspected.add(file_id)
        return {'file_id': file_id, 'path': self.paths[file_id], **json.loads(stdout)}

    async def run_code(self, code: str) -> Execution:
        """Run code in a fresh Python process; the execution is recorded even when it fails or times out."""
        if len(self.executions) >= MAX_EXECUTIONS:
            raise CheckFailure('budget_exhausted', f'At most {MAX_EXECUTIONS} code executions are allowed.')
        # Validate referenced input formats before executing generated code.
        for file_id in self.paths:
            if file_id not in self.inspected:
                await self.inspect_data(file_id)
        await self.start()
        execution = Execution(code=code, stdout='', stderr='', exit_code=None, runtime_ms=0.0)
        self.executions.append(execution)
        started = perf_counter()
        try:
            await self.backend.run_code(code, execution, timeout=self.remaining())
            return execution
        finally:
            execution.runtime_ms = round((perf_counter() - started) * 1000, 2)
            log_event('code.completed' if execution.exit_code == 0 else 'code.failed',
                      level=logging.INFO if execution.exit_code == 0 else logging.WARNING,
                      exit_code=execution.exit_code, duration_ms=execution.runtime_ms,
                      output_truncated=execution.output_truncated)

    async def close(self) -> None:
        """Release the backend and the task directory; safe when nothing was started."""
        if self.staging is not None:
            # start() already reported any copy failure; here we only wait for the thread to finish.
            await asyncio.gather(self.staging, return_exceptions=True)
        await self.backend.close()
        await asyncio.to_thread(remove_staging, self.directory, self.staging_root)


async def close_safely(sandbox: Sandbox) -> None:
    """Finish cleanup even if the caller is cancelled meanwhile, then re-raise that cancellation."""
    # Cancellation must not strand a running container.
    cleanup = asyncio.create_task(sandbox.close())
    cancelled = False
    while not cleanup.done():
        try:
            # A cancelled wait() neither cancels cleanup nor raises its exception.
            await asyncio.wait([cleanup])
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        # Cancellation still propagates when cleanup fails; the failure is logged.
        if cleanup.exception() is not None:
            log_event('check.cleanup_failed', level=logging.ERROR, **exception_fields(cleanup.exception()))
        raise asyncio.CancelledError
    cleanup.result()
