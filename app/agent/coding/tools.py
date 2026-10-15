from langchain_core.tools import StructuredTool

from app.agent.coding.sandbox import Sandbox
from app.agent.coding.schemas import InspectArgs, RunArgs


def make_tools(sandbox: Sandbox) -> list[StructuredTool]:
    async def inspect_data(file_id: str) -> dict:
        return await sandbox.inspect_data(file_id)

    async def run_code(code: str) -> dict:
        return (await sandbox.run_code(code)).model_dump(mode='json')

    return [
        StructuredTool.from_function(
            coroutine=inspect_data, name='inspect_data', args_schema=InspectArgs,
            description='Inspect an authorized CSV or JSON attachment. Returns its container path, fields, record count and up to five sample records.'),
        StructuredTool.from_function(
            coroutine=run_code, name='run_code', args_schema=RunArgs,
            description='Execute complete Python code in the task sandbox. Read authorized inputs under /input; write temporary files only under /work. Each call starts a fresh Python process. At most three calls.'),
    ]
