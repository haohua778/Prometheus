import asyncio
import time
from pathlib import Path

from app.agent.coding.limits import INSPECTION_BYTES, INSPECTION_SAMPLES, OUTPUT_BYTES
from app.agent.coding.schemas import CheckFailure, Execution
from app.agent.coding.staging import remove_staging
from app.runtime_logging import log_event

NAME_PREFIX = 'pierre-check-'
OWNER_LABEL = 'pierre.coding.owner'
OWNER = 'prometheus-local-v1'
EXPIRY_LABEL = 'pierre.coding.expires'
# Exit codes from docker exec itself or a killed container, never from user code.
UNAVAILABLE_EXIT_CODES = (125, 126, 127, 137)
_initialized = False
_initialization_lock: asyncio.Lock | None = None
_initialization_loop = None


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
                raise CheckFailure('output_limit', f'Execution output exceeded {OUTPUT_BYTES // 1024} KiB per stream.')

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
        raise CheckFailure('environment_error', 'Docker operation failed. Check the Docker engine and the sandbox image.')
    return stdout.strip()


async def cleanup_expired(staging_root: Path) -> None:
    listing = await require_docker('ps', '-a', '--filter', f'label={OWNER_LABEL}={OWNER}',
                                   '--format', '{{.Names}}\t{{.Label "' + EXPIRY_LABEL + '"}}')
    rows = [line.split('\t') for line in listing.splitlines()]
    expired = [name for name, expiry in rows if expiry and float(expiry) <= time.time()]
    if not expired:
        return
    await require_docker('rm', '-f', *expired)
    for name in expired:
        log_event('sandbox.expired_removed', container_id=name)
        # Only remove a directory derived from our generated container name.
        identifier = name.removeprefix(NAME_PREFIX)
        if len(identifier) == 32 and all(character in '0123456789abcdef' for character in identifier):
            await asyncio.to_thread(remove_staging, staging_root / identifier, staging_root)


async def initialize(staging_root: Path) -> None:
    # Reclaim containers left by a crashed process once per process, before the first task.
    global _initialized, _initialization_lock, _initialization_loop
    loop = asyncio.get_running_loop()
    if _initialization_loop is not loop:
        _initialization_lock = asyncio.Lock()
        _initialization_loop = loop
    async with _initialization_lock:
        if not _initialized:
            await cleanup_expired(staging_root)
            _initialized = True


class DockerBackend:
    input_root = '/input'
    work_root = '/work'

    def __init__(self, image: str):
        self.image = image
        # Set before `docker run`, so close() also removes a partly created container.
        self.name: str | None = None

    async def start(self, input_dir: Path, timeout: float) -> None:
        await initialize(input_dir.parent)
        self.name = f'{NAME_PREFIX}{input_dir.name}'
        await require_docker(
            'run', '--detach', '--name', self.name, '--pull=never', '--network=none', '--read-only',
            '--user=10001:10001', '--cap-drop=ALL', '--security-opt=no-new-privileges', '--init',
            '--cpus=1', '--memory=512m', '--memory-swap=512m', '--pids-limit=64',
            '--tmpfs', f'{self.work_root}:rw,noexec,nosuid,nodev,size=64m,uid=10001,gid=10001,mode=700',
            '--mount', f'type=bind,source={input_dir.resolve()},target={self.input_root},readonly',
            '--label', f'{OWNER_LABEL}={OWNER}',
            '--label', f'{EXPIRY_LABEL}={time.time() + timeout}',
            '--log-driver=none', self.image, timeout=timeout)
        log_event('sandbox.created', container_id=self.name, image=self.image)

    async def inspect(self, path: str, file_format: str, timeout: float) -> tuple[int, str]:
        code, stdout, _ = await docker_command(
            'exec', self.name, 'python', '/opt/inspect_data.py', path, file_format,
            str(INSPECTION_BYTES), str(INSPECTION_SAMPLES), timeout=timeout)
        return code, stdout

    async def run_code(self, code: str, execution: Execution, timeout: float) -> None:
        await docker_command('exec', '-i', self.name, 'python', '-', timeout=timeout,
                             input_bytes=code.encode('utf-8'), capture=execution)
        if execution.exit_code in UNAVAILABLE_EXIT_CODES:
            raise CheckFailure('environment_error', 'The sandbox execution was unavailable or terminated.')

    async def close(self) -> None:
        if self.name is None:
            return
        code, _, stderr = await docker_command('rm', '-f', self.name)
        if code and 'No such container' not in stderr:
            raise CheckFailure('cleanup_failed', 'Could not remove the task container.')
        log_event('sandbox.removed', container_id=self.name)
        self.name = None
