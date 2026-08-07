from __future__ import annotations

import base64
import binascii
import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api_schemas import (
    DeleteConversationResponse,
    EditMessageRequest,
    MessageRequest,
    MessageResponse,
)
from src.chat_store import (
    ConversationNotFoundError,
    MessageEditError,
    MessageNotFoundError,
    add_message,
    create_conversation,
    delete_conversation,
    edit_user_message,
    get_context_before_message,
    get_conversation_messages,
    get_conversations,
    get_recent_context,
    init_chat_tables,
    require_conversation,
    verify_chat_tables,
)
from src.config import (
    APP_DEBUG,
    LIVE_UPDATE_ENABLED,
    PROJECT_ROOT,
)
from src.graph import (
    _AGENT_GRAPH,
    run_agent,
)
from src.guardrail_service import (
    guard_output,
    sanitize_stream_chunk,
    validate_input,
)
from src.language_service import (
    localized_message,
    normalize_language,
)
from src.live_update_service import (
    get_live_update_status,
    start_live_update_service,
    stop_live_update_service,
)
from src.mysql_store import verify_mysql_ready
from src.qdrant_store import verify_qdrant_ready
from src.vision_service import analyze_image


logger = logging.getLogger(__name__)


def _sse_event(
    event_name: str,
    **data: Any,
) -> str:
    payload = {
        "type": event_name,
        "event": event_name,
        **data,
    }

    encoded = json.dumps(
        payload,
        default=str,
        ensure_ascii=False,
    )

    return f"data: {encoded}\n\n"


def _stream_message_text(
    message: Any,
) -> str:
    content = getattr(
        message,
        "content",
        message,
    )

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif (
                isinstance(item, dict)
                and item.get("text")
            ):
                parts.append(
                    str(item["text"])
                )

        return "".join(parts)

    return str(content)


class _ThinkStreamFilter:
    """
    Remove Qwen <think>...</think>
    content from streamed output.
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._inside_think = False

    def feed(
        self,
        text: str,
    ) -> str:
        self._buffer += text

        visible_parts: list[str] = []

        while self._buffer:
            lowered = self._buffer.lower()

            if self._inside_think:
                close_index = lowered.find(
                    self.CLOSE_TAG
                )

                if close_index < 0:
                    keep_length = (
                        len(self.CLOSE_TAG)
                        - 1
                    )

                    if (
                        len(self._buffer)
                        > keep_length
                    ):
                        self._buffer = (
                            self._buffer[
                                -keep_length:
                            ]
                        )

                    return "".join(
                        visible_parts
                    )

                self._buffer = (
                    self._buffer[
                        close_index
                        + len(
                            self.CLOSE_TAG
                        ):
                    ]
                )

                self._inside_think = False
                continue

            open_index = lowered.find(
                self.OPEN_TAG
            )

            if open_index >= 0:
                visible_parts.append(
                    self._buffer[
                        :open_index
                    ]
                )

                self._buffer = (
                    self._buffer[
                        open_index
                        + len(
                            self.OPEN_TAG
                        ):
                    ]
                )

                self._inside_think = True
                continue

            keep_length = (
                len(self.OPEN_TAG)
                - 1
            )

            if (
                len(self._buffer)
                <= keep_length
            ):
                return "".join(
                    visible_parts
                )

            visible_parts.append(
                self._buffer[
                    :-keep_length
                ]
            )

            self._buffer = (
                self._buffer[
                    -keep_length:
                ]
            )

            return "".join(
                visible_parts
            )

        return "".join(
            visible_parts
        )

    def flush(self) -> str:
        if self._inside_think:
            self._buffer = ""
            return ""

        remaining = self._buffer
        self._buffer = ""

        return remaining


def _assistant_metadata(
    state: dict[str, Any],
) -> dict[str, Any]:
    decision = state.get(
        "decision",
        {},
    )

    evidence = state.get(
        "evidence",
        {},
    )

    metadata: dict[str, Any] = {
        "route": decision.get(
            "route"
        ),
        "intent": decision.get(
            "intent"
        ),
        "route_reason": decision.get(
            "reason"
        ),
        "response_language": (
            decision.get(
                "response_language"
            )
        ),
        "llm_provider": state.get(
            "llm_provider",
            "ollama",
        ),
        "llm_calls": state.get(
            "llm_calls",
            0,
        ),
        "citations": evidence.get(
            "citations",
            {},
        ),
    }

    if (
        decision.get("route")
        == "structured"
    ):
        metadata.update(
            {
                "selected_tables": (
                    decision.get(
                        "selected_tables",
                        [],
                    )
                ),
                "row_count": (
                    evidence.get(
                        "row_count",
                        0,
                    )
                ),
                "columns": (
                    evidence.get(
                        "columns",
                        [],
                    )
                ),
                "truncated": (
                    evidence.get(
                        "truncated",
                        False,
                    )
                ),
                "repaired": (
                    evidence.get(
                        "repaired",
                        False,
                    )
                ),
            }
        )

    elif (
        decision.get("route")
        == "semantic"
    ):
        metadata[
            "retrieval_query"
        ] = evidence.get(
            "retrieval_query"
        )

        metadata["sources"] = sorted(
            {
                document.get(
                    "source",
                    "unknown",
                )
                for document
                in evidence.get(
                    "documents",
                    [],
                )
            }
        )

        metadata[
            "document_count"
        ] = evidence.get(
            "document_count",
            0,
        )

    return metadata


def _response_language(
    state: dict[str, Any],
    fallback_language: str,
) -> str:
    language = str(
        state.get(
            "decision",
            {},
        ).get(
            "response_language"
        )
        or fallback_language
        or "English"
    )

    return normalize_language(
        language
    )


def _apply_output_guard(
    state: dict[str, Any],
    fallback_language: str,
) -> dict[str, Any]:
    answer = str(
        state.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:
        answer = (
            "I could not generate an "
            "evidence-grounded answer."
        )

    decision = state.get(
        "decision",
        {},
    )

    evidence = state.get(
        "evidence",
        {},
    )

    state["answer"] = guard_output(
        answer,
        _response_language(
            state,
            fallback_language,
        ),
        str(
            decision.get("route")
            or "unknown"
        ),
        evidence,
    )

    return state


def _save_assistant_message(
    conversation_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    evidence = state.get(
        "evidence",
        {},
    )

    answer = str(
        state.get(
            "answer",
            "",
        )
    ).strip()

    if not answer:
        answer = (
            "I could not generate an "
            "evidence-grounded answer."
        )

    return add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        route=(
            state.get(
                "decision",
                {},
            ).get("route")
        ),
        sql=evidence.get("sql"),
        metadata=_assistant_metadata(
            state
        ),
    )


def _public_failure_message(
    error: Exception,
    preferred_language: str = "English",
) -> str:
    if APP_DEBUG:
        return (
            "The request failed: "
            f"{error}"
        )

    return localized_message(
        "unexpected_error",
        normalize_language(
            preferred_language
        ),
    )


def _input_guardrail_state(
    request: (
        MessageRequest
        | EditMessageRequest
    ),
    guardrail_result: Any,
) -> dict[str, Any]:
    language = normalize_language(
        request.preferred_language
    )

    public_message = str(
        getattr(
            guardrail_result,
            "public_message",
            "",
        )
        or localized_message(
            "unexpected_error",
            language,
        )
    ).strip()

    category = str(
        getattr(
            guardrail_result,
            "category",
            "blocked_input",
        )
        or "blocked_input"
    )

    return {
        "question":
            request.content,
        "preferred_language":
            request.preferred_language,
        "llm_provider":
            request.llm_provider,
        "decision": {
            "route":
                "guardrail",
            "intent":
                "blocked_input",
            "reason":
                category,
            "response_language":
                language,
        },
        "evidence": {
            "success":
                True,
            "route":
                "guardrail",
            "citations":
                {},
        },
        "answer":
            public_message,
        "llm_calls":
            0,
        "error":
            None,
    }


def _run_image_request(
    request: MessageRequest,
) -> dict[str, Any]:
    encoded_image = (
        request.image_base64
    )

    if not encoded_image:
        raise ValueError(
            "Image data is required."
        )

    try:
        image_bytes = (
            base64.b64decode(
                encoded_image,
                validate=True,
            )
        )

    except (
        binascii.Error,
        ValueError,
    ) as error:
        raise ValueError(
            "The uploaded image "
            "data is invalid."
        ) from error

    if (
        request.preferred_language
        == "auto"
    ):
        language_instruction = (
            "the same language as "
            "the user's question"
        )

        response_language = "auto"

    else:
        language_instruction = (
            normalize_language(
                request.preferred_language
            )
        )

        response_language = (
            language_instruction
        )

    answer = analyze_image(
        image_bytes=image_bytes,
        mime_type=(
            request.image_mime_type
            or (
                "application/"
                "octet-stream"
            )
        ),
        question=request.content,
        response_language=(
            language_instruction
        ),
    )

    citations = {
        "route": "image",
        "datasets": [],
        "documents": [
            {
                "title": (
                    "Uploaded image"
                ),
                "source": (
                    "User upload"
                ),
            }
        ],
        "sql": None,
        "row_count": 0,
    }

    return {
        "question":
            request.content,
        "preferred_language":
            request.preferred_language,
        "llm_provider":
            "ollama",
        "decision": {
            "route":
                "image",
            "intent":
                "image_understanding",
            "reason": (
                "The request contains "
                "an uploaded image."
            ),
            "response_language":
                response_language,
        },
        "evidence": {
            "success":
                True,
            "route":
                "image",
            "citations":
                citations,
        },
        "answer":
            answer,
        "llm_calls":
            1,
        "error":
            None,
    }


def _run_request(
    request: MessageRequest,
    context: str,
) -> dict[str, Any]:
    guardrail_result = (
        validate_input(
            request.content,
            request.preferred_language,
        )
    )

    if not guardrail_result.allowed:
        return _input_guardrail_state(
            request,
            guardrail_result,
        )

    if request.image_base64:
        state = _run_image_request(
            request
        )

    else:
        state = run_agent(
            request.content,
            chat_history=context,
            preferred_language=(
                request.preferred_language
            ),
            llm_provider=(
                request.llm_provider
            ),
        )

    return _apply_output_guard(
        state,
        request.preferred_language,
    )


def _prepare_conversation(
    request: MessageRequest,
) -> tuple[
    int,
    str | None,
    str,
]:
    client_id = str(
        request.client_id
    )

    if (
        request.conversation_id
        is None
    ):
        (
            conversation_id,
            title,
        ) = create_conversation(
            client_id,
            request.content,
        )

        context = ""

    else:
        conversation_id = (
            request.conversation_id
        )

        title = None

        try:
            require_conversation(
                conversation_id,
                client_id,
            )

        except (
            ConversationNotFoundError
        ) as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error

        context = get_recent_context(
            conversation_id
        )

    return (
        conversation_id,
        title,
        context,
    )


def _user_message_metadata(
    request: MessageRequest,
) -> dict[str, Any]:
    return {
        "has_image": bool(
            request.image_base64
        ),
        "image_mime_type": (
            request.image_mime_type
        ),
    }


def _message_response(
    conversation_id: int,
    title: str | None,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
    state: dict[str, Any],
) -> MessageResponse:
    evidence = state.get(
        "evidence",
        {},
    )

    route = (
        state.get(
            "decision",
            {},
        ).get("route")
    )

    status = (
        "success"
        if evidence.get(
            "success",
            True,
        )
        else "error"
    )

    return MessageResponse(
        conversation_id=conversation_id,
        title=title,
        user_message=user_message,
        assistant_message=(
            assistant_message
        ),
        route=route,
        sql=evidence.get("sql"),
        columns=evidence.get(
            "columns",
            [],
        ),
        rows=evidence.get(
            "rows",
            [],
        ),
        citations=evidence.get(
            "citations",
            {},
        ),
        status=status,
    )


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    live_service_started = False

    try:
        init_chat_tables()

        readiness_checks = (
            (
                "MySQL",
                verify_mysql_ready,
            ),
            (
                "Qdrant",
                verify_qdrant_ready,
            ),
            (
                "Chat tables",
                verify_chat_tables,
            ),
        )

        for (
            service_name,
            readiness_check,
        ) in readiness_checks:
            try:
                readiness_check()

            except Exception as error:
                logger.warning(
                    "%s readiness check "
                    "did not pass during "
                    "startup: %s",
                    service_name,
                    error,
                )

        if LIVE_UPDATE_ENABLED:
            start_live_update_service()

            live_service_started = True

            logger.info(
                "Live dataset update "
                "service started in "
                "the background."
            )

        yield

    finally:
        if live_service_started:
            stop_live_update_service()

            logger.info(
                "Live dataset update "
                "service stopped."
            )


app = FastAPI(
    title=(
        "Supercomputer "
        "Analytics Agent"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(
        directory=str(
            PROJECT_ROOT
            / "static"
        )
    ),
    name="static",
)


@app.get(
    "/",
    include_in_schema=False,
)
def home():
    return FileResponse(
        PROJECT_ROOT
        / "templates"
        / "index.html"
    )


@app.get("/health")
def health():
    verify_mysql_ready()
    verify_qdrant_ready()
    verify_chat_tables()

    return {
        "status": "healthy",
    }


@app.get(
    "/data-update/status"
)
def data_update_status():
    return get_live_update_status()


@app.get("/conversations")
def conversations(
    client_id: UUID,
    page: int = Query(
        1,
        ge=1,
    ),
    limit: int = Query(
        20,
        ge=1,
        le=100,
    ),
):
    return get_conversations(
        str(client_id),
        page=page,
        limit=limit,
    )


@app.get(
    "/conversations/"
    "{conversation_id}/messages"
)
def conversation_messages(
    conversation_id: int,
    client_id: UUID,
    page: int = Query(
        1,
        ge=1,
    ),
    limit: int = Query(
        50,
        ge=1,
        le=100,
    ),
):
    try:
        return get_conversation_messages(
            conversation_id=(
                conversation_id
            ),
            client_id=str(
                client_id
            ),
            page=page,
            limit=limit,
        )

    except (
        ConversationNotFoundError
    ) as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.delete(
    "/conversations/"
    "{conversation_id}",
    response_model=(
        DeleteConversationResponse
    ),
)
def remove_conversation(
    conversation_id: int,
    client_id: UUID,
):
    try:
        deleted = delete_conversation(
            conversation_id,
            str(client_id),
        )

        return (
            DeleteConversationResponse(
                deleted=deleted,
                conversation_id=(
                    conversation_id
                ),
            )
        )

    except (
        ConversationNotFoundError
    ) as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.post(
    "/messages",
    response_model=MessageResponse,
)
def send_message(
    request: MessageRequest,
):
    (
        conversation_id,
        title,
        context,
    ) = _prepare_conversation(
        request
    )

    user_message = add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.content,
        metadata=(
            _user_message_metadata(
                request
            )
        ),
    )

    try:
        state = _run_request(
            request,
            context,
        )

        assistant_message = (
            _save_assistant_message(
                conversation_id,
                state,
            )
        )

        return _message_response(
            conversation_id,
            title,
            user_message,
            assistant_message,
            state,
        )

    except Exception as error:
        logger.exception(
            "Synchronous request "
            "failed for "
            "conversation %s",
            conversation_id,
        )

        public_message = (
            _public_failure_message(
                error,
                request
                .preferred_language,
            )
        )

        assistant_message = add_message(
            conversation_id=(
                conversation_id
            ),
            role="assistant",
            content=public_message,
            route="error",
            metadata={
                "status":
                    "error",
                "llm_provider": (
                    request
                    .llm_provider
                ),
            },
        )

        return MessageResponse(
            conversation_id=(
                conversation_id
            ),
            title=title,
            user_message=user_message,
            assistant_message=(
                assistant_message
            ),
            route="error",
            status="error",
        )


@app.patch(
    "/messages/{message_id}",
    response_model=MessageResponse,
)
def edit_message(
    message_id: int,
    request: EditMessageRequest,
):
    try:
        edit_result = (
            edit_user_message(
                message_id=message_id,
                client_id=str(
                    request.client_id
                ),
                new_content=(
                    request.content
                ),
            )
        )

        conversation_id = int(
            edit_result[
                "conversation_id"
            ]
        )

        title = edit_result.get(
            "title"
        )

        user_message = edit_result[
            "message"
        ]

        context = (
            get_context_before_message(
                message_id=message_id,
                client_id=str(
                    request.client_id
                ),
            )
        )

        guardrail_result = (
            validate_input(
                request.content,
                request
                .preferred_language,
            )
        )

        if guardrail_result.allowed:
            state = run_agent(
                request.content,
                chat_history=context,
                preferred_language=(
                    request
                    .preferred_language
                ),
                llm_provider=(
                    request
                    .llm_provider
                ),
            )

            state = (
                _apply_output_guard(
                    state,
                    request
                    .preferred_language,
                )
            )

        else:
            state = (
                _input_guardrail_state(
                    request,
                    guardrail_result,
                )
            )

        assistant_message = (
            _save_assistant_message(
                conversation_id,
                state,
            )
        )

        return _message_response(
            conversation_id,
            title,
            user_message,
            assistant_message,
            state,
        )

    except (
        MessageNotFoundError
    ) as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except MessageEditError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except (
        ConversationNotFoundError
    ) as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Message edit failed "
            "for message %s",
            message_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                _public_failure_message(
                    error,
                    request
                    .preferred_language,
                )
            ),
        ) from error


@app.post(
    "/messages/stream"
)
def send_message_stream(
    request: MessageRequest,
):
    (
        conversation_id,
        title,
        context,
    ) = _prepare_conversation(
        request
    )

    user_message = add_message(
        conversation_id=conversation_id,
        role="user",
        content=request.content,
        metadata=(
            _user_message_metadata(
                request
            )
        ),
    )

    def event_generator():
        state: dict[str, Any] = {
            "question":
                request.content,
            "chat_history":
                context,
            "preferred_language": (
                request
                .preferred_language
            ),
            "llm_provider": (
                request.llm_provider
            ),
            "decision": {},
            "table_selection": {},
            "schema": "",
            "relationships": "",
            "generated_sql": "",
            "evidence": {},
            "answer": "",
            "llm_calls": 0,
            "error": None,
        }

        final_state = dict(
            state
        )

        think_filter = (
            _ThinkStreamFilter()
        )

        answer_was_streamed = False

        try:
            guardrail_result = (
                validate_input(
                    request.content,
                    request
                    .preferred_language,
                )
            )

            if not guardrail_result.allowed:
                final_state = (
                    _input_guardrail_state(
                        request,
                        guardrail_result,
                    )
                )

                answer = str(
                    final_state["answer"]
                )

                yield _sse_event(
                    "route",
                    route="guardrail",
                )

                yield _sse_event(
                    "step",
                    message=(
                        "🛡️ Input guardrail "
                        "blocked this request."
                    ),
                )

                yield _sse_event(
                    "answer_chunk",
                    chunk=answer,
                )

                assistant_message = (
                    _save_assistant_message(
                        conversation_id,
                        final_state,
                    )
                )

                yield _sse_event(
                    "done",
                    answer=answer,
                    route="guardrail",
                    sql=None,
                    columns=[],
                    rows=[],
                    citations={},
                    response_language=(
                        _response_language(
                            final_state,
                            request
                            .preferred_language,
                        )
                    ),
                    llm_provider=(
                        request.llm_provider
                    ),
                    status="success",
                )

                yield _sse_event(
                    "saved",
                    conversation_id=(
                        conversation_id
                    ),
                    title=title,
                    user_message=(
                        user_message
                    ),
                    assistant_message=(
                        assistant_message
                    ),
                )

                return

            if request.image_base64:
                yield _sse_event(
                    "step",
                    message=(
                        "🖼️ Checking and "
                        "analyzing the "
                        "uploaded image..."
                    ),
                )

                final_state = (
                    _run_image_request(
                        request
                    )
                )

                final_state = (
                    _apply_output_guard(
                        final_state,
                        request
                        .preferred_language,
                    )
                )

                answer = str(
                    final_state.get(
                        "answer",
                        "",
                    )
                ).strip()

                safe_chunk = (
                    sanitize_stream_chunk(
                        answer
                    )
                )

                if safe_chunk:
                    yield _sse_event(
                        "answer_chunk",
                        chunk=safe_chunk,
                    )

                assistant_message = (
                    _save_assistant_message(
                        conversation_id,
                        final_state,
                    )
                )

                evidence = (
                    final_state.get(
                        "evidence",
                        {},
                    )
                )

                yield _sse_event(
                    "done",
                    answer=answer,
                    route="image",
                    sql=None,
                    columns=[],
                    rows=[],
                    citations=(
                        evidence.get(
                            "citations",
                            {},
                        )
                    ),
                    response_language=(
                        _response_language(
                            final_state,
                            request
                            .preferred_language,
                        )
                    ),
                    llm_provider=(
                        "ollama"
                    ),
                    status="success",
                )

                yield _sse_event(
                    "saved",
                    conversation_id=(
                        conversation_id
                    ),
                    title=title,
                    user_message=(
                        user_message
                    ),
                    assistant_message=(
                        assistant_message
                    ),
                )

                return

            yield _sse_event(
                "step",
                message=(
                    "🧭 Understanding "
                    "and routing your "
                    "question..."
                ),
            )

            graph_stream = (
                _AGENT_GRAPH.stream(
                    state,
                    stream_mode=[
                        "updates",
                        "messages",
                    ],
                )
            )

            for (
                stream_type,
                stream_payload,
            ) in graph_stream:
                if (
                    stream_type
                    == "messages"
                ):
                    (
                        message_chunk,
                        metadata,
                    ) = stream_payload

                    if (
                        metadata.get(
                            "langgraph_node"
                        )
                        != "answer"
                    ):
                        continue

                    raw_chunk = (
                        _stream_message_text(
                            message_chunk
                        )
                    )

                    visible_chunk = (
                        think_filter.feed(
                            raw_chunk
                        )
                    )

                    safe_chunk = (
                        sanitize_stream_chunk(
                            visible_chunk
                        )
                    )

                    if safe_chunk:
                        answer_was_streamed = True

                        yield _sse_event(
                            "answer_chunk",
                            chunk=safe_chunk,
                        )

                    continue

                if (
                    stream_type
                    != "updates"
                ):
                    continue

                update = stream_payload

                for (
                    node_name,
                    node_output,
                ) in update.items():
                    if not isinstance(
                        node_output,
                        dict,
                    ):
                        continue

                    final_state.update(
                        node_output
                    )

                    if (
                        node_name
                        == "router"
                    ):
                        decision = (
                            node_output.get(
                                "decision",
                                {},
                            )
                        )

                        route = (
                            decision.get(
                                "route",
                                "unknown",
                            )
                        )

                        yield _sse_event(
                            "route",
                            route=route,
                        )

                        if (
                            route
                            == "structured"
                        ):
                            step_message = (
                                "📊 Structured "
                                "route selected."
                            )

                        elif (
                            route
                            == "semantic"
                        ):
                            step_message = (
                                "🔎 Semantic "
                                "route selected."
                            )

                        else:
                            step_message = (
                                "💬 Preparing "
                                "a direct "
                                "response..."
                            )

                        yield _sse_event(
                            "step",
                            message=(
                                step_message
                            ),
                        )

                    elif (
                        node_name
                        == "table_selector"
                    ):
                        tables = (
                            node_output
                            .get(
                                "decision",
                                {},
                            )
                            .get(
                                "selected_tables",
                                [],
                            )
                        )

                        table_text = (
                            ", ".join(tables)
                            if tables
                            else "none"
                        )

                        yield _sse_event(
                            "step",
                            message=(
                                "📖 Relevant "
                                "datasets: "
                                f"{table_text}"
                            ),
                        )

                        yield _sse_event(
                            "step",
                            message=(
                                "💻 Generating "
                                "a safe MySQL "
                                "query..."
                            ),
                        )

                    elif (
                        node_name
                        == "structured"
                    ):
                        evidence = (
                            node_output.get(
                                "evidence",
                                {},
                            )
                        )

                        sql = evidence.get(
                            "sql"
                        )

                        if sql:
                            yield _sse_event(
                                "sql",
                                sql=sql,
                            )

                        if evidence.get(
                            "success"
                        ):
                            if evidence.get(
                                "repaired"
                            ):
                                yield _sse_event(
                                    "step",
                                    message=(
                                        "🛠️ SQL was "
                                        "corrected "
                                        "and retried "
                                        "safely."
                                    ),
                                )

                            yield _sse_event(
                                "step",
                                message=(
                                    "✅ MySQL query "
                                    "completed. "
                                    "Rows: "
                                    f"{evidence.get('row_count', 0)}"
                                ),
                            )

                            yield _sse_event(
                                "step",
                                message=(
                                    "💬 Streaming "
                                    "the final "
                                    "answer..."
                                ),
                            )

                        else:
                            yield _sse_event(
                                "step",
                                message=(
                                    "⚠️ The MySQL "
                                    "query could "
                                    "not be "
                                    "completed "
                                    "safely."
                                ),
                            )

                    elif (
                        node_name
                        == "semantic"
                    ):
                        evidence = (
                            node_output.get(
                                "evidence",
                                {},
                            )
                        )

                        if evidence.get(
                            "success"
                        ):
                            yield _sse_event(
                                "step",
                                message=(
                                    "✅ Qdrant "
                                    "retrieval "
                                    "completed. "
                                    "Documents: "
                                    f"{evidence.get('document_count', 0)}"
                                ),
                            )

                            yield _sse_event(
                                "step",
                                message=(
                                    "💬 Streaming "
                                    "the final "
                                    "answer..."
                                ),
                            )

                        else:
                            yield _sse_event(
                                "step",
                                message=(
                                    "⚠️ Qdrant "
                                    "retrieval "
                                    "could not "
                                    "be completed."
                                ),
                            )

            remaining_chunk = (
                sanitize_stream_chunk(
                    think_filter.flush()
                )
            )

            if remaining_chunk:
                answer_was_streamed = True

                yield _sse_event(
                    "answer_chunk",
                    chunk=remaining_chunk,
                )

            final_state = (
                _apply_output_guard(
                    final_state,
                    request
                    .preferred_language,
                )
            )

            answer = str(
                final_state.get(
                    "answer",
                    "",
                )
            ).strip()

            if not answer_was_streamed:
                yield _sse_event(
                    "answer_chunk",
                    chunk=answer,
                )

            assistant_message = (
                _save_assistant_message(
                    conversation_id,
                    final_state,
                )
            )

            evidence = (
                final_state.get(
                    "evidence",
                    {},
                )
            )

            route = (
                final_state.get(
                    "decision",
                    {},
                ).get("route")
            )

            yield _sse_event(
                "done",
                answer=answer,
                route=route,
                sql=evidence.get("sql"),
                columns=evidence.get(
                    "columns",
                    [],
                ),
                rows=evidence.get(
                    "rows",
                    [],
                ),
                citations=evidence.get(
                    "citations",
                    {},
                ),
                response_language=(
                    _response_language(
                        final_state,
                        request
                        .preferred_language,
                    )
                ),
                llm_provider=(
                    final_state.get(
                        "llm_provider",
                        request
                        .llm_provider,
                    )
                ),
                status=(
                    "success"
                    if evidence.get(
                        "success",
                        True,
                    )
                    else "error"
                ),
            )

            yield _sse_event(
                "saved",
                conversation_id=(
                    conversation_id
                ),
                title=title,
                user_message=(
                    user_message
                ),
                assistant_message=(
                    assistant_message
                ),
            )

        except Exception as error:
            logger.exception(
                "Streaming request "
                "failed for "
                "conversation %s",
                conversation_id,
            )

            public_message = (
                _public_failure_message(
                    error,
                    request
                    .preferred_language,
                )
            )

            try:
                assistant_message = (
                    add_message(
                        conversation_id=(
                            conversation_id
                        ),
                        role="assistant",
                        content=(
                            public_message
                        ),
                        route="error",
                        metadata={
                            "status":
                                "error",
                            "llm_provider": (
                                request
                                .llm_provider
                            ),
                        },
                    )
                )

            except Exception:
                assistant_message = None

            yield _sse_event(
                "error",
                message=public_message,
            )

            yield _sse_event(
                "done",
                answer=public_message,
                route="error",
                sql=None,
                columns=[],
                rows=[],
                citations={},
                response_language=(
                    request
                    .preferred_language
                ),
                llm_provider=(
                    request.llm_provider
                ),
                status="error",
            )

            yield _sse_event(
                "saved",
                conversation_id=(
                    conversation_id
                ),
                title=title,
                user_message=(
                    user_message
                ),
                assistant_message=(
                    assistant_message
                ),
            )

    return StreamingResponse(
        event_generator(),
        media_type=(
            "text/event-stream"
        ),
        headers={
            "Cache-Control": (
                "no-cache, "
                "no-transform"
            ),
            "Connection":
                "keep-alive",
            "X-Accel-Buffering":
                "no",
        },
    )