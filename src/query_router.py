from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.config import MAX_CHAT_HISTORY_CHARS
from src.llm import get_json_llm
from src.models import RouteDecision
from src.prompts import ROUTER_PROMPT


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))

        return "\n".join(parts).strip()

    return str(content).strip()


def _parse_router_response(
    message: Any,
) -> RouteDecision:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        _message_text(message),
        flags=re.DOTALL | re.IGNORECASE,
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
            "Router response did not contain a JSON object."
        )

    payload = json.loads(
        text[start : end + 1]
    )

    if not isinstance(payload, dict):
        raise ValueError(
            "Router response must be a JSON object."
        )

    return RouteDecision.model_validate(payload)


def route_question(
    question: str,
    chat_history: str = "",
) -> tuple[RouteDecision, int]:
    """
    Decide only structured, semantic, or direct route.

    Table selection is not performed here.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    prompt = ROUTER_PROMPT.format(
        chat_history=(
            chat_history[-MAX_CHAT_HISTORY_CHARS:]
            or "No previous conversation."
        ),
        question=cleaned_question,
    )

    last_error: Exception | None = None

    for attempt in range(2):
        current_prompt = prompt

        if attempt:
            current_prompt += (
                "\n\nYour previous output was invalid. "
                "Return only the required JSON object."
            )

        try:
            response = get_json_llm().invoke(
                current_prompt
            )

            decision = _parse_router_response(
                response
            )

            return decision, attempt + 1

        except (
            ValueError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            last_error = error

    raise RuntimeError(
        "The query router returned invalid output twice: "
        f"{last_error}"
    )