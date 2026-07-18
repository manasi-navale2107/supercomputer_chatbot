from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class MessageRequest(BaseModel):
    client_id: UUID
    conversation_id: int | None = Field(default=None, ge=1)
    content: str = Field(min_length=1, max_length=4_000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message content cannot be empty.")
        return cleaned


class MessageResponse(BaseModel):
    conversation_id: int
    title: str | None = None
    user_message: dict[str, Any]
    assistant_message: dict[str, Any]
    route: str | None = None
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "success"


class DeleteConversationResponse(BaseModel):
    deleted: bool
    conversation_id: int

