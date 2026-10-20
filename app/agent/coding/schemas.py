from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from app.agent.coding.limits import MAX_CODE_CHARS, MAX_FILES, MAX_TIMEOUT_S

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FileId = Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')]
ErrorCode = Literal['data_invalid', 'timeout', 'budget_exhausted', 'output_limit',
                    'environment_error', 'model_error', 'cleanup_failed']


def failure_status(code: ErrorCode) -> Literal['incomplete', 'failed']:
    return 'incomplete' if code in ('timeout', 'budget_exhausted', 'output_limit', 'data_invalid') else 'failed'


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class DataRef(Contract):
    file_id: FileId
    format: Literal['csv', 'json']


class CheckRequest(Contract):
    request_id: Text
    flag_id: Text
    claim: Text
    question: Text
    data_refs: list[DataRef] = Field(max_length=MAX_FILES)
    timeout_s: int = Field(default=MAX_TIMEOUT_S, ge=1, le=MAX_TIMEOUT_S)

    @model_validator(mode='after')
    def unique_files(self):
        if len({ref.file_id for ref in self.data_refs}) != len(self.data_refs):
            raise ValueError('Duplicate file IDs are not allowed.')
        return self


class Evidence(Contract):
    file_id: FileId
    location: Text
    observed_value: JsonValue


class Execution(Contract):
    code: str
    stdout: str
    stderr: str
    exit_code: int | None
    runtime_ms: float = Field(ge=0)
    output_truncated: bool = False


class Finding(Contract):
    verdict: Literal['supports', 'refutes', 'inconclusive']
    explanation: Text
    evidence: list[Evidence]


class CheckResult(Finding):
    request_id: Text
    flag_id: Text
    status: Literal['completed', 'incomplete', 'failed']
    executions: list[Execution]
    runtime_ms: float = Field(ge=0)
    error_code: ErrorCode | None = None

    @model_validator(mode='after')
    def consistent_outcome(self):
        if self.status == 'completed':
            if self.verdict == 'inconclusive' or self.error_code is not None or not self.evidence:
                raise ValueError('Completed checks require a supported verdict, evidence and no error.')
            if not any(item.exit_code == 0 and not item.output_truncated for item in self.executions):
                raise ValueError('Completed checks require a successful, non-truncated execution.')
        elif self.verdict != 'inconclusive':
            raise ValueError('Incomplete or failed checks must be inconclusive.')
        if self.status == 'failed' and self.error_code is None:
            raise ValueError('Failed checks require an error code.')
        if self.error_code is not None and self.status != failure_status(self.error_code):
            raise ValueError('The status does not match the error category.')
        return self


class CheckFailure(RuntimeError):
    def __init__(self, code: ErrorCode, message: str):
        super().__init__(message)
        self.code = code


class InspectArgs(Contract):
    file_id: FileId


class RunArgs(Contract):
    code: Text = Field(max_length=MAX_CODE_CHARS)
