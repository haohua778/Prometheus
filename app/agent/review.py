"""Review of a draft reply: JSON evidence in, a validated verdict out.

The reviewer model receives the thread history and the draft as one JSON document and must answer
with one JSON object. The answer is validated locally; a malformed answer is asked for again once
with the validation error, after which the review fails instead of passing the draft.
"""
import json
import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.agent.prompts import REVIEW_FORMAT_RETRY_PROMPT, REVIEW_PROMPT
from app.runtime_logging import log_event

# Message classes to the role names used in the review input.
_ROLES = {'human': 'user', 'ai': 'assistant', 'tool': 'tool', 'system': 'system'}


class ReviewFormatError(RuntimeError):
    """The reviewer model did not return a valid review, even after one retry."""


class ReviewResult(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid')

    verdict: Literal['PASS', 'REVISE']
    feedback: str  # may contain Markdown; empty on PASS

    @model_validator(mode='after')
    def _feedback_matches_verdict(self) -> 'ReviewResult':
        feedback = self.feedback.strip()
        if self.verdict == 'REVISE' and not feedback:
            raise ValueError('REVISE requires non-empty feedback')
        if self.verdict == 'PASS' and feedback:
            raise ValueError('PASS must not carry feedback')
        self.feedback = feedback
        return self


def _message_to_review_json(message: AnyMessage) -> dict[str, Any]:
    """Keep what the reviewer needs to follow the conversation; drop usage and response metadata."""
    entry: dict[str, Any] = {'role': _ROLES[message.type], 'content': message.content}
    if getattr(message, 'name', None):
        entry['name'] = message.name
    if isinstance(message, AIMessage) and message.tool_calls:
        entry['tool_calls'] = list(message.tool_calls)  # id, name and args pair each call with its result
    if message.type == 'tool':
        entry['tool_call_id'] = message.tool_call_id
    return entry


def build_review_input(messages: list[AnyMessage], draft: AIMessage) -> str:
    review_input = {
        'messages': [_message_to_review_json(message) for message in messages],
        'draft': {'role': 'assistant', 'content': draft.content},
    }
    # Standard JSON only: a NaN or infinite tool argument raises ValueError here rather than reaching the model.
    return json.dumps(review_input, ensure_ascii=False, allow_nan=False)


async def review_draft(model: BaseChatModel, messages: list[AnyMessage], draft: AIMessage) -> ReviewResult:
    """Ask the model to review the draft; retry once on a malformed answer, then fail.

    Only format errors are retried. Provider and network errors propagate unchanged.
    """
    review_prompt = [SystemMessage(REVIEW_PROMPT), HumanMessage(build_review_input(messages, draft))]
    first_answer = await model.ainvoke(review_prompt)
    try:
        return ReviewResult.model_validate_json(first_answer.text)
    except ValidationError as first_error:
        log_event('review.format_retry', level=logging.WARNING, attempt=1)
        # The invalid answer goes back as plain text so stray tool calls or metadata are not replayed.
        retry_prompt = [*review_prompt, AIMessage(first_answer.text),
                        HumanMessage(REVIEW_FORMAT_RETRY_PROMPT.format(error=first_error))]
        second_answer = await model.ainvoke(retry_prompt)
        try:
            return ReviewResult.model_validate_json(second_answer.text)
        except ValidationError as second_error:
            log_event('review.format_failed', level=logging.ERROR, attempt=2)
            raise ReviewFormatError('reviewer returned an invalid review twice') from second_error
