from __future__ import annotations

import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, Iterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api_schemas import (
    DeleteConversationResponse,
    MessageRequest,
    MessageResponse,
)
from src.chat_store import (
    ConversationNotFoundError,
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    get_conversations,
    get_recent_context,
    require_conversation,
    verify_chat_tables,
)
from src.config import APP_DEBUG, PROJECT_ROOT
from src.graph import _AGENT_GRAPH, run_agent
from src.mysql_store import verify_mysql_ready
from src.qdrant_store import verify_qdrant_ready


logger = logging.getLogger(__name__)


def _sse_event(
    event_type: str,
    **data: Any,
) -> str:
    payload = {
        "type": event_type,
        **data,
    }

    return (
        f"data: "
        f"{json.dumps(payload, default=str, ensure_ascii=False)}"
        f"\n\n"
    )


def _assistant_metadata(
    state: dict[str, Any],
) -> dict[str, Any]:
    decision = state.get("decision", {})
    evidence = state.get("evidence", {})

    metadata: dict[str, Any] = {
        "route": decision.get("route"),
        "intent": decision.get("intent"),
        "route_reason": decision.get("reason"),
        "llm_calls": state.get("llm_calls", 0),
    }

    if decision.get("route") == "structured":
        metadata.update(
            {
                "selected_tables": decision.get(
                    "selected_tables",
                    [],
                ),
                "row_count": evidence.get(
                    "row_count",
                    0,
                ),
                "columns": evidence.get(
                    "columns",
                    [],
                ),
                "truncated": evidence.get(
                    "truncated",
                    False,
                ),
                "repaired": evidence.get(
                    "repaired",
                    False,
                ),
            }
        )

    elif decision.get("route") == "semantic":
        metadata["retrieval_query"] = evidence.get(
            "retrieval_query"
        )

        metadata["sources"] = sorted(
            {
                document.get(
                    "source",
                    "unknown",
                )
                for document in evidence.get(
                    "documents",
                    [],
                )
            }
        )

        metadata["document_count"] = evidence.get(
            "document_count",
            0,
        )

    return metadata


def _save_assistant_message(
    conversation_id: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    evidence = state.get("evidence", {})
    answer = state.get(
        "answer",
        "",
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
        route=state.get(
            "decision",
            {},
        ).get("route"),
        sql=evidence.get("sql"),
        metadata=_assistant_metadata(state),
    )


def _answer_chunks(
    answer: str,
) -> Iterator[str]:
    for chunk in re.findall(
        r"\S+\s*",
        answer,
    ):
        yield chunk


def _public_failure_message(
    error: Exception,
) -> str:
    if APP_DEBUG:
        return f"The request failed: {error}"

    return (
        "I could not complete the request because "
        "an unexpected error occurred."
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    verify_mysql_ready()
    verify_qdrant_ready()
    verify_chat_tables()

    yield


app = FastAPI(
    title="Supercomputer Analytics Agent",
    version="3.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            PROJECT_ROOT / "static"
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
    "/conversations/{conversation_id}/messages"
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
            conversation_id=conversation_id,
            client_id=str(client_id),
            page=page,
            limit=limit,
        )

    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


@app.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
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

        return DeleteConversationResponse(
            deleted=deleted,
            conversation_id=conversation_id,
        )

    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error


def _prepare_conversation(
    request: MessageRequest,
) -> tuple[int, str | None, str]:
    client_id = str(request.client_id)

    if request.conversation_id is None:
        conversation_id, title = create_conversation(
            client_id,
            request.content,
        )

        context = ""

    else:
        conversation_id = request.conversation_id
        title = None

        try:
            require_conversation(
                conversation_id,
                client_id,
            )

        except ConversationNotFoundError as error:
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


@app.post(
    "/messages",
    response_model=MessageResponse,
)
def send_message(
    request: MessageRequest,
):
    conversation_id, title, context = (
        _prepare_conversation(request)
    )

    user_message = add_message(
        conversation_id,
        "user",
        request.content,
    )

    try:
        state = run_agent(
            request.content,
            chat_history=context,
        )

        assistant_message = _save_assistant_message(
            conversation_id,
            state,
        )

        evidence = state.get(
            "evidence",
            {},
        )

        route = state.get(
            "decision",
            {},
        ).get("route")

        status = (
            "success"
            if evidence.get("success", True)
            else "error"
        )

    except Exception as error:
        logger.exception(
            "Synchronous request failed for "
            "conversation %s",
            conversation_id,
        )

        public_message = _public_failure_message(
            error
        )

        assistant_message = add_message(
            conversation_id,
            "assistant",
            public_message,
            route="error",
            metadata={
                "status": "error",
            },
        )

        evidence = {}
        route = "error"
        status = "error"

    return MessageResponse(
        conversation_id=conversation_id,
        title=title,
        user_message=user_message,
        assistant_message=assistant_message,
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
        status=status,
    )


@app.post("/messages/stream")
def send_message_stream(
    request: MessageRequest,
):
    conversation_id, title, context = (
        _prepare_conversation(request)
    )

    user_message = add_message(
        conversation_id,
        "user",
        request.content,
    )

    def event_generator():
        state: dict[str, Any] = {
            "question": request.content,
            "chat_history": context,

            "decision": {},

            "schema": "",
            "relationships": "",
            "generated_sql": "",

            "evidence": {},
            "answer": "",

            "llm_calls": 0,
            "error": None,
        }

        final_state = dict(state)

        try:
            yield _sse_event(
                "step",
                message=(
                    "🧭 Understanding and routing "
                    "your question..."
                ),
            )

            for update in _AGENT_GRAPH.stream(
                state,
                stream_mode="updates",
            ):
                for node_name, node_output in (
                    update.items()
                ):
                    if not isinstance(
                        node_output,
                        dict,
                    ):
                        continue

                    final_state.update(
                        node_output
                    )

                    if node_name == "router":
                        decision = node_output.get(
                            "decision",
                            {},
                        )

                        route = decision.get(
                            "route",
                            "unknown",
                        )

                        yield _sse_event(
                            "route",
                            route=route,
                        )

                        if route == "structured":
                            tables = decision.get(
                                "selected_tables",
                                [],
                            )

                            table_text = (
                                ", ".join(tables)
                                or "relevant tables"
                            )

                            yield _sse_event(
                                "step",
                                message=(
                                    "📊 Structured route "
                                    "selected."
                                ),
                            )

                            yield _sse_event(
                                "step",
                                message=(
                                    "📖 Reading schema for: "
                                    f"{table_text}..."
                                ),
                            )

                            yield _sse_event(
                                "step",
                                message=(
                                    "💻 Generating a read-only "
                                    "MySQL query..."
                                ),
                            )

                        elif route == "semantic":
                            yield _sse_event(
                                "step",
                                message=(
                                    "🔎 Semantic route selected: "
                                    "searching Qdrant..."
                                ),
                            )

                        else:
                            yield _sse_event(
                                "step",
                                message=(
                                    "💬 Preparing a direct "
                                    "response..."
                                ),
                            )
                    elif node_name == "table_selector":
                        tables = (
                            node_output
                            .get("decision", {})
                            .get("selected_tables", [])
                        )

                        yield _sse_event(
                            "step",
                            message=(
                                "📖 Relevant datasets selected: "
                                + (
                                    ", ".join(tables)
                                    if tables
                                    else "none"
                                )
                            ),
                        )

                    elif node_name == "structured":
                        evidence = node_output.get(
                            "evidence",
                            {},
                        )

                        sql = evidence.get("sql")

                        if sql:
                            yield _sse_event(
                                "sql",
                                sql=sql,
                            )

                        if evidence.get("success"):
                            if evidence.get("repaired"):
                                yield _sse_event(
                                    "step",
                                    message=(
                                        "🛠️ SQL was corrected "
                                        "and safely retried once."
                                    ),
                                )

                            yield _sse_event(
                                "step",
                                message=(
                                    "✅ MySQL query executed. "
                                    "Rows returned: "
                                    f"{evidence.get('row_count', 0)}"
                                ),
                            )

                        else:
                            yield _sse_event(
                                "step",
                                message=(
                                    "⚠️ The MySQL query could "
                                    "not be completed safely."
                                ),
                            )

                    elif node_name == "semantic":
                        evidence = node_output.get(
                            "evidence",
                            {},
                        )

                        if evidence.get("success"):
                            message = (
                                "✅ Qdrant retrieval completed. "
                                "Documents found: "
                                f"{evidence.get('document_count', 0)}"
                            )

                        else:
                            message = (
                                "⚠️ Qdrant retrieval could "
                                "not be completed."
                            )

                        yield _sse_event(
                            "step",
                            message=message,
                        )

                    elif node_name == "answer":
                        yield _sse_event(
                            "step",
                            message=(
                                "💬 Converting evidence into "
                                "a natural-language answer..."
                            ),
                        )

            assistant_message = (
                _save_assistant_message(
                    conversation_id,
                    final_state,
                )
            )

            answer = assistant_message["content"]

            for chunk in _answer_chunks(answer):
                yield _sse_event(
                    "answer_chunk",
                    chunk=chunk,
                )

            evidence = final_state.get(
                "evidence",
                {},
            )

            route = final_state.get(
                "decision",
                {},
            ).get("route")

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
                conversation_id=conversation_id,
                title=title,
                user_message=user_message,
                assistant_message=assistant_message,
            )

        except Exception as error:
            logger.exception(
                "Streaming request failed for "
                "conversation %s",
                conversation_id,
            )

            public_message = (
                _public_failure_message(error)
            )

            try:
                assistant_message = add_message(
                    conversation_id,
                    "assistant",
                    public_message,
                    route="error",
                    metadata={
                        "status": "error",
                    },
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
                status="error",
            )

            yield _sse_event(
                "saved",
                conversation_id=conversation_id,
                title=title,
                user_message=user_message,
                assistant_message=assistant_message,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )