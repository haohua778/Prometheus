from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from app.agent.coding.limits import INSPECTION_SAMPLES, MAX_EXECUTIONS
from app.agent.coding.schemas import Finding, InspectArgs, RunArgs

# The model sees only these definitions; graph.py runs each call against the task sandbox.
INSPECT_DATA = 'inspect_data'
RUN_CODE = 'run_code'
SUBMIT_FINDING = 'submit_finding'


def definition(name: str, description: str, schema: type[BaseModel]) -> dict:
    tool = convert_to_openai_tool(schema)
    tool['function'].update(name=name, description=description)
    return tool


def tool_definitions(work_root: str) -> list[dict]:
    return [
        definition(INSPECT_DATA, 'Inspect an authorized CSV or JSON attachment. Returns its sandbox path, fields, '
                   f'record count and up to {INSPECTION_SAMPLES} sample records.', InspectArgs),
        definition(RUN_CODE, 'Execute complete Python code in the task sandbox. Read authorized inputs from the '
                   f'paths returned by {INSPECT_DATA}; write temporary files only under {work_root}. Each call '
                   f'starts a fresh Python process. At most {MAX_EXECUTIONS} calls.', RunArgs),
        definition(SUBMIT_FINDING, 'Return the final result of the check. Call it alone, once the evidence is '
                   'complete; an invalid result comes back as an error to correct.', Finding),
    ]
