from __future__ import annotations

from typing import (
    Any,
    Literal,
)
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from src.config import (
    DEFAULT_LLM_PROVIDER,
)


PreferredLanguage = Literal[
    "auto",
    "en-IN",
    "hi-IN",
    "mr-IN",
]

LlmProvider = Literal[
    "ollama",
    "groq",
]


def _clean_message_content(
    value: str,
) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise ValueError(
            "Message content cannot "
            "be empty."
        )

    return cleaned


class MessageRequest(BaseModel):
    client_id: UUID

    conversation_id: int | None = Field(
        default=None,
        ge=1,
    )

    content: str = Field(
        min_length=1,
        max_length=4_000,
    )

    preferred_language: (
        PreferredLanguage
    ) = "auto"

    llm_provider: LlmProvider = (
        DEFAULT_LLM_PROVIDER
    )

    image_base64: str | None = Field(
        default=None,
        max_length=14_000_000,
    )

    image_mime_type: Literal[
        "image/jpeg",
        "image/png",
        "image/webp",
    ] | None = None

    @field_validator("content")
    @classmethod
    def strip_content(
        cls,
        value: str,
    ) -> str:
        return _clean_message_content(
            value
        )

    @model_validator(mode="after")
    def validate_image_payload(
        self,
    ) -> "MessageRequest":
        has_image_data = bool(
            self.image_base64
        )

        has_image_type = bool(
            self.image_mime_type
        )

        if (
            has_image_data
            != has_image_type
        ):
            raise ValueError(
                "Image data and image type "
                "must be provided together."
            )

        return self


class EditMessageRequest(BaseModel):
    """
    Request body used when editing an
    existing user query.

    Image messages are not editable because
    uploaded image bytes are not stored in
    the chat database.
    """

    client_id: UUID

    content: str = Field(
        min_length=1,
        max_length=4_000,
    )

    preferred_language: (
        PreferredLanguage
    ) = "auto"

    llm_provider: LlmProvider = (
        DEFAULT_LLM_PROVIDER
    )

    @field_validator("content")
    @classmethod
    def strip_content(
        cls,
        value: str,
    ) -> str:
        return _clean_message_content(
            value
        )


class MessageResponse(BaseModel):
    conversation_id: int

    title: str | None = None

    user_message: dict[
        str,
        Any,
    ]

    assistant_message: dict[
        str,
        Any,
    ]

    route: str | None = None

    sql: str | None = None

    columns: list[str] = Field(
        default_factory=list
    )

    rows: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    citations: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    status: str = "success"


class DeleteConversationResponse(
    BaseModel
):
    deleted: bool
    conversation_id: int