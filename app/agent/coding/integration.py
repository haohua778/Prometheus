from collections.abc import Mapping
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool

from app.agent.coding.agent import run_check
from app.agent.coding.schemas import CheckRequest
from app.config import Settings


def make_check_tool(*, authorized_files: Mapping[str, Path], model: BaseChatModel,
                    settings: Settings) -> StructuredTool:
    # Capture only this review's authorization; callers may reuse their mapping.
    files = dict(authorized_files)

    async def check_claim(**fields) -> dict:
        result = await run_check(CheckRequest(**fields), authorized_files=files, model=model, settings=settings)
        return result.model_dump(mode='json')

    return StructuredTool.from_function(
        coroutine=check_claim, name='check_claim', args_schema=CheckRequest,
        description='Delegate a numerical claim check against authorized CSV/JSON attachments. '
                    'Provide a unique request_id, the existing flag_id, the exact claim, known units, '
                    'periods and scope in question, and relevant file IDs in data_refs. '
                    'Returns status, verdict about the claim, evidence and execution records. '
                    'incomplete/failed or inconclusive means the claim was not verified. '
                    'Does not create or approve flags.')
