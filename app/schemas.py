"""HTTP request and response fields, including runtime validation rules."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    thread_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,64}$')


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
