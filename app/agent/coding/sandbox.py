import asyncio
import json
import logging
import shutil
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.agent.coding.schemas import CheckFailure, CheckRequest, Execution
from app.config import PROJECT_ROOT
from app.runtime_logging import log_event

IMAGE = 'pierre-coding:local'
OWNER_LABEL = 'pierre.coding.owner'
OWNER = 'prometheus-local-v1'
EXPIRY_LABEL = 'pierre.coding.expires'
RUNTIME_ROOT = PROJECT_ROOT / '.runtime' / 'coding'
MAX_FILE_BYTES = 10 * 1024 * 1024
OUTPUT_BYTES = 32768
_initialized = False
_initialization_lock: asyncio.Lock | None = None
_initialization_loop = None


def validate_files(request: CheckRequest, authorized_files: Mapping[str, Path]) -> dict[str, Path]:
    selected = {}
    for ref in request.data_refs:
        if ref.file_id not in authorized_files:
            raise ValueError(f'Unauthorized file ID: {ref.file_id}')
        path = Path(authorized_files[ref.file_id]).absolute()
        if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
            raise ValueError('Input files must not use symbolic links or junctions.')
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError('Each input must be a regular file of at most 10 MiB.')
        selected[ref.file_id] = path
    return selected


async def docker_command(*args: str, timeout: float = 10, input_bytes: bytes | None = None,
                         capture: Execution | None = None) -> tuple[int, str, str]:
    process = None
    buffers = [bytearray(), bytearray()]

    async def read_stream(stream, buffer):
        while chunk := await stream.read(8192):
            available = OUTPUT_BYTES - len(buffer)
            buffer.extend(chunk[:available])
            if len(chunk) > available:
                if capture is not None:
                    capture.output_truncated = True
                raise CheckFailure('output_limit', 'Execution output exceeded 32 KiB per stream.')

    async def write_input():
        if input_bytes is not None:
            try:
                process.stdin.write(input_bytes)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                # A program may exit before reading all its input.
                pass
            finally:
                process.stdin.close()

    tasks = []
    try:
        async with asyncio.timeout(timeout):
            process = await asyncio.create_subprocess_exec(
                'docker', *args, stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            tasks = [asyncio.create_task(read_stream(process.stdout, buffers[0])),
                     asyncio.create_task(read_stream(process.stderr, buffers[1])),
                     asyncio.create_task(write_input())]
            await asyncio.gather(*tasks)
            await process.wait()
    except TimeoutError as error:
        raise CheckFailure('timeout', 'Docker operation exceeded the available time.') from error
    except OSError as error:
        raise CheckFailure('environment_error', 'Docker CLI could not be started.') from error
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if process is not None and process.returncode is None:
            process.kill()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if process is not None:
            await process.wait()
        if capture is not None:
            capture.stdout = buffers[0].decode('utf-8', errors='replace')
            capture.stderr = buffers[1].decode('utf-8', errors='replace')
            capture.exit_code = process.returncode if process is not None else None
    return process.returncode, buffers[0].decode('utf-8', errors='replace'), buffers[1].decode('utf-8', errors='replace')


async def require_docker(*args: str, timeout: float = 10) -> str:
    code, stdout, _ = await docker_command(*args, timeout=timeout)
    if code:
        raise CheckFailure('environment_error', 'Docker operation failed. Check Docker Desktop and the sandbox image.')
    return stdout.strip()


async def cleanup_expired() -> None:
    identifiers = await require_docker('ps', '-aq', '--filter', f'label={OWNER_LABEL}={OWNER}')
    for identifier in identifiers.splitlines():
        raw = await require_docker('inspect', identifier)
        container = json.loads(raw)[0]
        expiry = container['Config']['Labels'].get(EXPIRY_LABEL)
        if expiry is not None and float(expiry) <= time.time():
            await require_docker('rm', '-f', identifier)
            log_event('sandbox.expired_removed', container_id=identifier)
            # Only remove a directory derived from our generated container name.
            name = container['Name'].removeprefix('/').removeprefix('pierre-check-')
            if len(name) == 32 and all(character in '0123456789abcdef' for character in name):
                remove_staging(RUNTIME_ROOT / name)


def remove_staging(directory: Path) -> None:
    try:
        # Check links before exists(): a dangling link must never look like successful cleanup.
        if directory.parent.resolve() != RUNTIME_ROOT.resolve() or directory.is_symlink() or directory.is_junction():
            raise OSError('Unsafe task directory.')
        if directory.exists():
            shutil.rmtree(directory)
    except OSError as error:
        log_event('sandbox.cleanup_failed', level=logging.ERROR)
        raise CheckFailure('cleanup_failed', 'Could not safely remove the task directory.') from error


async def initialize_coding_runtime() -> None:
    global _initialized, _initialization_lock, _initialization_loop
    loop = asyncio.get_running_loop()
    if _initialization_loop is not loop:
        _initialization_lock = asyncio.Lock()
        _initialization_loop = loop
    async with _initialization_lock:
        if not _initialized:
            await cleanup_expired()
            _initialized = True


class Sandbox:
    def __init__(self, request: CheckRequest, files: Mapping[str, Path], deadline: float):
        self.request = request
        self.files = dict(files)
        self.deadline = deadline
        self.identifier = uuid4().hex
        self.name = f'pierre-check-{self.identifier}'
        self.directory = RUNTIME_ROOT / self.identifier
        self.paths = {ref.file_id: f'/input/{index}.{ref.format}' for index, ref in enumerate(request.data_refs)}
        self.executions: list[Execution] = []
        self.inspected: set[str] = set()
        self.created = False
        self.creation_attempted = False

    def remaining(self) -> float:
        remaining = self.deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise CheckFailure('timeout', 'The check exceeded its time budget.')
        return remaining

    def stage_inputs(self) -> None:
        try:
            selected = validate_files(self.request, self.files)
        except (OSError, ValueError) as error:
            raise CheckFailure('data_invalid', 'An authorized input changed or became unavailable before execution.') from error
        try:
            RUNTIME_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.mkdir(mode=0o755, exist_ok=False)
            self.directory.chmod(0o755)
        except OSError as error:
            raise CheckFailure('environment_error', 'The task staging directory could not be created.') from error
        for ref in self.request.data_refs:
            try:
                with selected[ref.file_id].open('rb') as stream:
                    content = stream.read(MAX_FILE_BYTES + 1)
            except OSError as error:
                raise CheckFailure('data_invalid', 'An authorized input could not be read.') from error
            if len(content) > MAX_FILE_BYTES:
                raise CheckFailure('data_invalid', 'An authorized input grew beyond 10 MiB while staging.')
            try:
                destination = self.directory / Path(self.paths[ref.file_id]).name
                destination.write_bytes(content)
                destination.chmod(0o644)
            except OSError as error:
                raise CheckFailure('environment_error', 'An input copy could not be staged.') from error

    async def start(self) -> None:
        if self.created:
            return
        self.remaining()
        await initialize_coding_runtime()
        image = await require_docker('image', 'inspect', IMAGE, '--format', '{{.Id}}', timeout=self.remaining())
        self.stage_inputs()
        self.remaining()
        self.creation_attempted = True
        await require_docker(
            'create', '--name', self.name, '--pull=never', '--network=none', '--read-only',
            '--user=10001:10001', '--cap-drop=ALL', '--security-opt=no-new-privileges', '--init',
            '--cpus=1', '--memory=512m', '--memory-swap=512m', '--pids-limit=64',
            '--tmpfs', '/work:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001,mode=700',
            '--mount', f'type=bind,source={self.directory.resolve()},target=/input,readonly',
            '--label', f'{OWNER_LABEL}={OWNER}',
            '--label', f'{EXPIRY_LABEL}={time.time() + self.remaining()}',
            '--log-driver=none', image, timeout=self.remaining())
        await require_docker('start', self.name, timeout=self.remaining())
        self.created = True
        log_event('sandbox.created', check_request_id=self.request.request_id, container_id=self.name, image_id=image)

    async def inspect_data(self, file_id: str) -> dict:
        if file_id not in self.paths:
            raise CheckFailure('data_invalid', 'The requested file ID is not authorized for this task.')
        await self.start()
        ref = next(ref for ref in self.request.data_refs if ref.file_id == file_id)
        code, stdout, _ = await docker_command(
            'exec', self.name, 'python', '/opt/inspect_data.py', self.paths[file_id], ref.format,
            timeout=self.remaining())
        if code == 2:
            raise CheckFailure('data_invalid', 'Input does not match the supported CSV/JSON record format.')
        if code == 3:
            raise CheckFailure('output_limit', 'Inspection output exceeded 16 KiB.')
        if code != 0:
            raise CheckFailure('environment_error', 'Data inspection process failed.')
        self.inspected.add(file_id)
        return {'file_id': file_id, 'path': self.paths[file_id], **json.loads(stdout)}

    async def run_code(self, code: str) -> Execution:
        if len(self.executions) >= 3:
            raise CheckFailure('budget_exhausted', 'At most three code executions are allowed.')
        # Validate referenced input formats before executing generated code.
        for file_id in self.paths:
            if file_id not in self.inspected:
                await self.inspect_data(file_id)
        await self.start()
        execution = Execution(code=code, stdout='', stderr='', exit_code=None, runtime_ms=0.0)
        self.executions.append(execution)
        started = perf_counter()
        try:
            await docker_command('exec', '-i', self.name, 'python', '-', timeout=self.remaining(),
                                 input_bytes=code.encode('utf-8'), capture=execution)
            if execution.exit_code != 0:
                running = await require_docker('inspect', self.name, '--format', '{{.State.Running}}',
                                               timeout=self.remaining())
                if running != 'true':
                    raise CheckFailure('environment_error', 'The task container stopped unexpectedly.')
            if execution.exit_code in (125, 126, 127, 137):
                raise CheckFailure('environment_error', 'The sandbox execution was unavailable or terminated.')
            return execution
        finally:
            execution.runtime_ms = round((perf_counter() - started) * 1000, 2)
            log_event('code.completed' if execution.exit_code == 0 else 'code.failed',
                      level=logging.INFO if execution.exit_code == 0 else logging.WARNING,
                      check_request_id=self.request.request_id, exit_code=execution.exit_code,
                      duration_ms=execution.runtime_ms, output_truncated=execution.output_truncated)

    async def close(self) -> None:
        if self.creation_attempted:
            code, _, stderr = await docker_command('rm', '-f', self.name)
            if code and 'No such container' not in stderr:
                raise CheckFailure('cleanup_failed', 'Could not remove the task container.')
            self.creation_attempted = False
            self.created = False
            log_event('sandbox.removed', check_request_id=self.request.request_id, container_id=self.name)
        remove_staging(self.directory)


async def close_safely(sandbox: Sandbox) -> None:
    # Cancellation must not strand a running container.
    cleanup = asyncio.create_task(sandbox.close())
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    cleanup.result()
    if cancelled:
        raise asyncio.CancelledError
