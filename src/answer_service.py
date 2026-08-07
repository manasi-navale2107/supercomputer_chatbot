from __future__ import annotations

import json
import re
from typing import Any

from src.llm import get_llm
from src.prompts import ANSWER_PROMPT
import logging
logger = logging.getLogger(__name__)

from src.language_service import (
    localized_message,
    normalize_language,
)


MAX_ROWS_PER_QUERY = 30
MAX_DOCUMENTS = 10
MAX_FIELD_LENGTH = 500
MAX_EVIDENCE_CHARS = 20_000

SOURCE_FIELDS = {
    "source",
    "source_dataset",
    "source_datasets",
    "dataset",
    "table",
}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        text = content

    elif isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))

        text = "\n".join(parts)

    else:
        text = str(content)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


def _clean_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return round(value, 6)

    text = str(value).strip()

    if not text:
        return None

    if text.casefold() in {
        "none",
        "null",
        "nan",
        "unknown",
        "not available",
        "n/a",
    }:
        return None

    if len(text) > MAX_FIELD_LENGTH:
        return text[:MAX_FIELD_LENGTH] + "..."

    return text


def _clean_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        value = _clean_value(row)
        return {"value": value} if value is not None else {}

    cleaned: dict[str, Any] = {}

    for key, value in row.items():
        clean_value = _clean_value(value)

        if clean_value is not None:
            cleaned[str(key)] = clean_value

    return cleaned


def _row_identity(row: dict[str, Any]) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _deduplicate_rows(
    rows: list[Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw_row in rows:
        row = _clean_row(raw_row)

        if not row:
            continue

        identity = _row_identity(row)

        if identity in seen:
            continue

        seen.add(identity)
        output.append(row)

    return output


def _structured_rows(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Collect rows from every executed query.

    query_results is the primary source. Top-level rows are used only
    as a fallback.
    """
    collected: list[Any] = []
    query_results = evidence.get("query_results")

    if isinstance(query_results, list):
        for result in query_results:
            if not isinstance(result, dict):
                continue

            rows = result.get("rows")

            if isinstance(rows, list):
                collected.extend(rows)

    if not collected:
        rows = evidence.get("rows")

        if isinstance(rows, list):
            collected.extend(rows)

    return _deduplicate_rows(collected)


def _has_structured_rows(
    evidence: dict[str, Any],
) -> bool:
    return bool(_structured_rows(evidence))


def _single_verified_value(
    evidence: dict[str, Any],
) -> Any | None:
    """
    Detect a verified single-value result.

    This works generically for results such as:
    {"manufacturer": "Intel"}
    {"vendor": "Intel"}

    Source metadata is ignored. If every result contains only one data
    value and all values agree, that value is returned without another
    LLM call.
    """
    rows = _structured_rows(evidence)

    if not rows:
        return None

    values: list[Any] = []

    for row in rows:
        row_values = [
            value
            for key, value in row.items()
            if key.casefold() not in SOURCE_FIELDS
        ]

        if not row_values:
            continue

        if len(row_values) != 1:
            return None

        values.append(row_values[0])

    if not values:
        return None

    unique_values = {
        str(value).strip().casefold()
        for value in values
    }

    if len(unique_values) != 1:
        return None

    return values[0]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.6f}".rstrip("0").rstrip(".")

    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"

    return str(value).strip()


def _compact_query_results(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    query_results = evidence.get("query_results")

    if not isinstance(query_results, list):
        query_results = []

    compact_results: list[dict[str, Any]] = []

    for index, result in enumerate(query_results, start=1):
        if not isinstance(result, dict):
            continue

        raw_rows = result.get("rows")

        if not isinstance(raw_rows, list):
            raw_rows = []

        rows = _deduplicate_rows(raw_rows)

        compact_results.append(
            {
                "query_number": index,

                # structured_route.py returns source_tables.
                "source_tables": result.get(
                    "source_tables",
                    [],
                ),

                "columns": result.get("columns", []),
                "row_count": result.get(
                    "row_count",
                    len(raw_rows),
                ),
                "matching_rows_exist": bool(rows),
                "rows": rows[:MAX_ROWS_PER_QUERY],
                "rows_truncated": (
                    len(rows) > MAX_ROWS_PER_QUERY
                ),
            }
        )

    if compact_results:
        return compact_results

    rows = _structured_rows(evidence)

    return [
        {
            "query_number": 1,
            "source_tables": evidence.get(
                "selected_tables",
                [],
            ),
            "columns": evidence.get("columns", []),
            "row_count": evidence.get(
                "row_count",
                len(rows),
            ),
            "matching_rows_exist": bool(rows),
            "rows": rows[:MAX_ROWS_PER_QUERY],
            "rows_truncated": (
                len(rows) > MAX_ROWS_PER_QUERY
            ),
        }
    ]


def _semantic_documents(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_documents = (
        evidence.get("documents")
        or evidence.get("matches")
        or evidence.get("results")
        or []
    )

    if not isinstance(raw_documents, list):
        return []

    documents: list[dict[str, Any]] = []

    for document in raw_documents[:MAX_DOCUMENTS]:
        if not isinstance(document, dict):
            continue

        payload = document.get("payload")

        if not isinstance(payload, dict):
            payload = {}

        source = (
            document.get("source")
            or document.get("table")
            or payload.get("source")
            or payload.get("table")
            or "unknown"
        )

        content = (
            document.get("content")
            or document.get("text")
            or document.get("page_content")
            or payload.get("content")
            or payload.get("text")
            or payload.get("page_content")
        )

        clean_content = _clean_value(content)

        if clean_content is None:
            continue

        documents.append(
            {
                "source": source,
                "score": document.get("score"),
                "content": clean_content,
            }
        )

    return documents


def _build_evidence_text(
    evidence: dict[str, Any],
) -> str:
    route = str(
        evidence.get("route", "")
    ).casefold()

    if route == "structured" or evidence.get("query_results"):
        compact_evidence = {
            "route": "structured",
            "matching_rows_exist": _has_structured_rows(
                evidence
            ),
            "total_row_count": evidence.get(
                "row_count",
                len(_structured_rows(evidence)),
            ),
            "selected_tables": evidence.get(
                "selected_tables",
                [],
            ),
            "query_results": _compact_query_results(
                evidence
            ),
        }

    else:
        documents = _semantic_documents(evidence)

        compact_evidence = {
            "route": "semantic",
            "matching_documents_exist": bool(documents),
            "documents": documents,
        }

    return json.dumps(
        compact_evidence,
        ensure_ascii=False,
        indent=2,
        default=str,
    )[:MAX_EVIDENCE_CHARS]


def _clean_answer(answer: str) -> str:
    answer = str(answer).strip()

    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove only surrounding code fences.
    answer = re.sub(
        r"^\s*```(?:markdown|text)?\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"\s*```\s*$",
        "",
        answer,
    )

    answer = re.sub(
        r"^\s*(final answer|answer|response)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


def generate_answer(
    question: str,
    evidence: dict[str, Any],
    context: str = "",
    response_language: str = "English",
    llm_provider: str | None = None,
) -> str:
    evidence_text = (
        _build_evidence_text(
            evidence
        )
    )

    prompt = ANSWER_PROMPT.format(
        question=question.strip(),
        context=(
            context.strip()
            or (
                "No previous conversation "
                "context."
            )
        ),
        evidence=evidence_text,
        response_language=(
            response_language.strip()
            or "English"
        ),
    )

    response = (
        get_llm(
            llm_provider
        ).invoke(
            prompt
        )
    )

    content = getattr(
        response,
        "content",
        response,
    )

    answer = _clean_answer(
        str(content)
    )

    if not answer:
        raise ValueError(
            "The answer model returned "
            "an empty response."
        )

    return answer


def generate_evidence_answer(
    state: dict[str, Any],
) -> dict[str, Any]:
    evidence = (
        state.get("evidence")
        or {}
    )

    decision = (
        state.get("decision")
        or {}
    )

    route = str(
        decision.get("route")
        or "unknown"
    ).strip().lower()

    response_language = (
        normalize_language(
            decision.get(
                "response_language"
            )
        )
    )

    question = str(
        state.get("question")
        or ""
    ).strip()

    context = str(
        state.get("chat_history")
        or state.get("context")
        or ""
    ).strip()

    current_calls = int(
        state.get(
            "llm_calls",
            0,
        )
    )

    citations = evidence.get(
        "citations",
        {},
    )

    if not evidence.get(
        "success",
        False,
    ):
        message_key = (
            "structured_failed"
            if route == "structured"
            else "semantic_failed"
        )

        answer = localized_message(
            message_key,
            response_language,
        )

        return {
            "answer": answer,
            "citations": citations,
            "llm_calls": current_calls,
            "error": str(
                evidence.get("error")
                or answer
            ),
        }

    if route == "structured":
        if not _has_structured_rows(
            evidence
        ):
            return {
                "answer": localized_message(
                    "no_records",
                    response_language,
                ),
                "citations": citations,
                "llm_calls": current_calls,
                "error": None,
            }

        single_value = (
            _single_verified_value(
                evidence
            )
        )

        if single_value is not None:
            return {
                "answer": localized_message(
                    "verified_answer",
                    response_language,
                    value=_format_value(
                        single_value
                    ),
                ),
                "citations": citations,
                "llm_calls": current_calls,
                "error": None,
            }

    if route == "semantic":
        documents = (
            _semantic_documents(
                evidence
            )
        )

        if not documents:
            return {
                "answer": localized_message(
                    "no_information",
                    response_language,
                ),
                "citations": citations,
                "llm_calls": current_calls,
                "error": None,
            }

    try:
        answer = generate_answer(
            question=question,
            evidence=evidence,
            context=context,
            response_language=(
                response_language
            ),
            llm_provider=state.get(
                "llm_provider"
            ),
        )

        return {
            "answer": answer,
            "citations": citations,
            "llm_calls":
                current_calls + 1,
            "error": None,
        }

    except Exception as error:
        logger.exception(
            "Final answer generation failed. "
            "Route=%s, Question=%s",
            route,
            question,
        )

        return {
            "answer": localized_message(
                "answer_failed",
                response_language,
            ),
            "citations": citations,
            "llm_calls": current_calls,
            "error": str(error),
        }