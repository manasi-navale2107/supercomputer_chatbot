from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from src.config import MAX_CHAT_HISTORY_CHARS
from src.llm import get_json_llm
from src.models import RouteDecision
from src.prompts import ROUTER_PROMPT
from src.language_service import (
    normalize_language,
)

_LANGUAGE_NAMES = {
    "auto": "Automatically detect the language of the current question",
    "en-IN": "English",
    "hi-IN": "Hindi",
    "mr-IN": "Marathi",
}


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
    preferred_language: str = "auto",
    llm_provider: str | None = None,
) -> tuple[RouteDecision, int]:
    """
    Select the route and response language
    using one LLM call.
    """

    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:
        raise ValueError(
            "Question cannot be empty."
        )

    language_instruction = (
        _LANGUAGE_NAMES.get(
            preferred_language,
            _LANGUAGE_NAMES["auto"],
        )
    )

    prompt = ROUTER_PROMPT.format(
        chat_history=(
            chat_history[
                -MAX_CHAT_HISTORY_CHARS:
            ]
            or "No previous conversation."
        ),
        question=cleaned_question,
        preferred_language=(
            language_instruction
        ),
    )

    last_error: Exception | None = None

    for attempt in range(2):
        current_prompt = prompt

        if attempt:
            current_prompt += (
                "\n\nThe previous output was invalid. "
                "Return only the exact JSON object "
                "required by the output contract."
            )

        try:
            response = (
                get_json_llm(
                    llm_provider
                ).invoke(
                    current_prompt
                )
            )

            decision = (
                _parse_router_response(
                    response
                )
            )

            # Language selected in the UI has
            # priority over automatic detection.
            if preferred_language != "auto":
                decision.response_language = (
                    normalize_language(
                        language_instruction
                    )
                )

            else:
                decision.response_language = (
                    normalize_language(
                        decision.response_language
                    )
                )

            return (
                decision,
                attempt + 1,
            )

        except (
            ValueError,
            json.JSONDecodeError,
            ValidationError,
        ) as error:
            last_error = error

    raise RuntimeError(
        "The query router returned invalid "
        f"output twice: {last_error}"
    )