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
# A task removes its own container at its deadline; waiting this long avoids removing it concurrently.
EXPIRY_GRACE_S = 60
_cleanup_lock: asyncio.Lock | None = None
_cleanup_loop = None


async def docker_command(*args: str, timeout: float = 10, input_bytes: bytes | None = None,
                         capture: Execution | None = None) -> tuple[int, str, str]:
    """Run a docker CLI command with a timeout and a per-stream output limit.

    With capture, output and exit code are written into that Execution even when the command
    times out or exceeds the limit. Returns (exit code, stdout, stderr).
    """
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
    """Run a docker command that must succeed and return its stdout; a failure is environment_error."""
    code, stdout, _ = await docker_command(*args, timeout=timeout)
    if code:
        raise CheckFailure('environment_error', 'Docker operation failed. Check the Docker engine and the sandbox image.')
    return stdout.strip()


async def cleanup_expired(staging_root: Path) -> None:
    """Remove expired task containers left by a crashed process, with their staging directories.

    Runs before every new container, so leftovers that had not expired when the service restarted
    are reclaimed by a later task.
    """
    global _cleanup_lock, _cleanup_loop
    loop = asyncio.get_running_loop()
    if _cleanup_loop is not loop:
        _cleanup_lock = asyncio.Lock()
        _cleanup_loop = loop
    async with _cleanup_lock:
        listing = await require_docker('ps', '-a', '--filter', f'label={OWNER_LABEL}={OWNER}',
                                       '--format', '{{.Names}}\t{{.Label "' + EXPIRY_LABEL + '"}}')
        rows = [line.split('\t') for line in listing.splitlines()]
        expired = [name for name, expiry in rows if expiry and float(expiry) + EXPIRY_GRACE_S <= time.time()]
        if not expired:
            return
        await require_docker('rm', '-f', *expired)
        for name in expired:
            log_event('sandbox.expired_removed', container_id=name)
            # Only remove a directory derived from our generated container name.
            identifier = name.removeprefix(NAME_PREFIX)
            if len(identifier) == 32 and all(character in '0123456789abcdef' for character in identifier):
                await asyncio.to_thread(remove_staging, staging_root / identifier, staging_root)


class DockerBackend:
    # ExecutionBackend on the local Docker CLI: one restricted container per check.
    input_root = '/input'
    work_root = '/work'

    def __init__(self, image: str):
        self.image = image
        # Set before `docker run`, so close() also removes a partly created container.
        self.name: str | None = None

    async def start(self, input_dir: Path, timeout: float) -> None:
        """Create the task container: no network, read-only filesystem and inputs, limited resources."""
        await cleanup_expired(input_dir.parent)
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
        """Run the inspection script; returns its exit code (see sandbox/inspect_data.py) and JSON output."""
        code, stdout, _ = await docker_command(
            'exec', self.name, 'python', '/opt/inspect_data.py', path, file_format,
            str(INSPECTION_BYTES), str(INSPECTION_SAMPLES), timeout=timeout)
        return code, stdout

    async def run_code(self, code: str, execution: Execution, timeout: float) -> None:
        """Pipe code to a fresh python process in the container and record the result in execution."""
        await docker_command('exec', '-i', self.name, 'python', '-', timeout=timeout,
                             input_bytes=code.encode('utf-8'), capture=execution)
        # docker exec passes the program's exit code through; only a stopped container is an environment error.
        if execution.exit_code != 0 and not await self.running():
            raise CheckFailure('environment_error', 'The sandbox execution was unavailable or terminated.')

    async def running(self) -> bool:
        """Return whether the task container still exists and is running."""
        code, stdout, _ = await docker_command('inspect', '--format', '{{.State.Running}}', self.name)
        return code == 0 and stdout.strip() == 'true'

    async def close(self) -> None:
        """Remove the task container; one that is already gone counts as removed."""
        if self.name is None:
            return
        code, _, stderr = await docker_command('rm', '-f', self.name)
        if code and 'No such container' not in stderr:
            raise CheckFailure('cleanup_failed', 'Could not remove the task container.')
        log_event('sandbox.removed', container_id=self.name)
        self.name = None
