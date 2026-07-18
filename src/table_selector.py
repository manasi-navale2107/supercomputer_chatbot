from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.config import (
    MAX_CHAT_HISTORY_CHARS,
)
from src.llm import get_json_llm
from src.models import TableSelection
from src.mysql_store import (
    get_table_catalog_context,
)
from src.prompts import (
    TABLE_SELECTOR_PROMPT,
)


def _message_text(
    message: Any,
) -> str:
    content = getattr(
        message,
        "content",
        message,
    )

    if isinstance(content, str):
        return content.strip()

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

        return "\n".join(parts).strip()

    return str(content).strip()


def _parse_selection(
    message: Any,
) -> TableSelection:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        _message_text(message),
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    ).strip()

    text = (
        text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            "Table selector did not return "
            "a JSON object."
        )

    payload = json.loads(
        text[start : end + 1]
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Table selector response must "
            "be a JSON object."
        )

    return TableSelection.model_validate(
        payload
    )


def select_relevant_tables(
    question: str,
    chat_history: str = "",
) -> tuple[TableSelection, int]:
    """
    Evaluate every live table using one dedicated LLM call.
    """

    prompt = TABLE_SELECTOR_PROMPT.format(
        question=question.strip(),
        chat_history=(
            chat_history[
                -MAX_CHAT_HISTORY_CHARS:
            ]
            or "No previous conversation."
        ),
        table_catalog=(
            get_table_catalog_context()
        ),
    )

    last_error: Exception | None = None

    for attempt in range(2):
        current_prompt = prompt

        if attempt:
            current_prompt += (
                "\n\nThe previous response was invalid. "
                "Evaluate every live table and return "
                "only the required JSON object."
            )

        try:
            response = (
                get_json_llm().invoke(
                    current_prompt
                )
            )

            selection = _parse_selection(
                response
            )

            return selection, attempt + 1

        except (
            ValueError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            last_error = error

    raise RuntimeError(
        "The table selector returned invalid "
        f"output twice: {last_error}"
    )