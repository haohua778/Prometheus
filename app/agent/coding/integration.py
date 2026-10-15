from collections.abc import Mapping
from pathlib import Path

from langchain_core.tools import StructuredTool

from app.agent.coding.agent import run_check
from app.agent.coding.schemas import CheckRequest, DataRef


def make_check_tool(*, authorized_files: Mapping[str, Path]) -> StructuredTool:
    # Capture only this review's authorization; callers may reuse their mapping.
    files = dict(authorized_files)

    async def check_claim(request_id: str, flag_id: str, claim: str, question: str,
                          data_refs: list[DataRef], timeout_s: int = 60) -> dict:
        request = CheckRequest(request_id=request_id, flag_id=flag_id, claim=claim,
                               question=question, data_refs=data_refs, timeout_s=timeout_s)
        result = await run_check(request, authorized_files=files)
        return result.model_dump(mode='json')

    return StructuredTool.from_function(
        coroutine=check_claim, name='check_claim', args_schema=CheckRequest,
        description='Delegate a numerical claim check against authorized CSV/JSON attachments. '
                    'Provide a unique request_id, the existing flag_id, the exact claim, known units, '
                    'periods and scope in question, and relevant file IDs in data_refs. '
                    'Returns status, verdict about the claim, evidence and execution records. '
                    'incomplete/failed or inconclusive means the claim was not verified. '
                    'Does not create or approve flags.')
