import logging
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path

from app.agent.coding.limits import MAX_FILE_BYTES
from app.agent.coding.schemas import CheckFailure, CheckRequest
from app.runtime_logging import log_event


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
            raise ValueError(f'Each input must be a regular file of at most {MAX_FILE_BYTES // (1024 * 1024)} MiB.')
        selected[ref.file_id] = path
    return selected


def staged_name(index: int, file_format: str) -> str:
    return f'{index}.{file_format}'


def stage_inputs(request: CheckRequest, authorized_files: Mapping[str, Path], directory: Path) -> None:
    # Copy a snapshot so later changes to the originals cannot reach the running check.
    try:
        selected = validate_files(request, authorized_files)
    except (OSError, ValueError) as error:
        raise CheckFailure('data_invalid', 'An authorized input changed or became unavailable before execution.') from error
    try:
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.mkdir(mode=0o755, exist_ok=False)
        directory.chmod(0o755)
    except OSError as error:
        raise CheckFailure('environment_error', 'The task staging directory could not be created.') from error
    for index, ref in enumerate(request.data_refs):
        try:
            with selected[ref.file_id].open('rb') as stream:
                content = stream.read(MAX_FILE_BYTES + 1)
        except OSError as error:
            raise CheckFailure('data_invalid', 'An authorized input could not be read.') from error
        if len(content) > MAX_FILE_BYTES:
            raise CheckFailure('data_invalid', 'An authorized input grew beyond the size limit while staging.')
        try:
            destination = directory / staged_name(index, ref.format)
            destination.write_bytes(content)
            destination.chmod(0o644)
        except OSError as error:
            raise CheckFailure('environment_error', 'An input copy could not be staged.') from error


def remove_staging(directory: Path, root: Path) -> None:
    try:
        # Check links before exists(): a dangling link must never look like successful cleanup.
        if directory.parent.resolve() != root.resolve() or directory.is_symlink() or directory.is_junction():
            raise OSError('Unsafe task directory.')
        if directory.exists():
            shutil.rmtree(directory)
    except OSError as error:
        log_event('sandbox.cleanup_failed', level=logging.ERROR)
        raise CheckFailure('cleanup_failed', 'Could not safely remove the task directory.') from error
