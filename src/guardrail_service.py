from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MAX_SAFE_OUTPUT_CHARS = 20_000


@dataclass(
    frozen=True,
    slots=True,
)
class InputGuardrailResult:
    allowed: bool
    category: str
    public_message: str | None = None


_CONTROL_CHARACTERS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


_PROMPT_INJECTION_PATTERNS = (
    re.compile(
        (
            r"\b("
            r"ignore|disregard|forget|"
            r"override|bypass"
            r")\b"
            r".{0,100}"
            r"\b("
            r"previous|above|system|"
            r"developer|hidden"
            r")\b"
            r".{0,50}"
            r"\b("
            r"instruction|instructions|"
            r"prompt|rules"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),

    re.compile(
        (
            r"\b("
            r"jailbreak|developer mode|"
            r"dan mode|unrestricted mode"
            r")\b"
        ),
        flags=re.IGNORECASE,
    ),

    re.compile(
        (
            r"\b("
            r"act as|pretend to be"
            r")\b"
            r".{0,80}"
            r"\b("
            r"without restrictions|"
            r"without rules|"
            r"no safety"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),
)


_SECRET_REQUEST_PATTERNS = (
    re.compile(
        (
            r"\b("
            r"show|reveal|print|display|"
            r"give|expose|return|leak"
            r")\b"
            r".{0,100}"
            r"\b("
            r"api[ _-]?key|password|"
            r"credential|credentials|"
            r"secret|token|"
            r"environment variable|"
            r"\.env|system prompt|"
            r"developer message"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),
)


_DATABASE_MUTATION_PATTERNS = (
    re.compile(
        (
            r"\b("
            r"execute|run|perform|apply|"
            r"issue|submit"
            r")\b"
            r".{0,100}"
            r"\b("
            r"insert|update|delete|drop|"
            r"alter|truncate|create|"
            r"grant|revoke"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),

    re.compile(
        (
            r"\b("
            r"delete|remove|erase|destroy"
            r")\b"
            r".{0,80}"
            r"\b("
            r"database|table|records?|"
            r"rows?|collection|index"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),
)


_INTERNAL_OUTPUT_PATTERNS = (
    re.compile(
        r"<think>.*?</think>",
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),

    re.compile(
        (
            r"\b("
            r"here is|the following is|"
            r"my"
            r")\b"
            r".{0,50}"
            r"\b("
            r"system prompt|"
            r"developer message|"
            r"hidden instruction"
            r")\b"
        ),
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    ),
)


_SECRET_VALUE_PATTERN = re.compile(
    (
        r"\b("
        r"GROQ_API_KEY|"
        r"QDRANT_API_KEY|"
        r"MYSQL_[A-Z0-9_]*PASSWORD|"
        r"OLLAMA_API_KEY|"
        r"API_KEY|"
        r"ACCESS_TOKEN"
        r")"
        r"\s*[:=]\s*"
        r"([^\s,;]+)"
    ),
    flags=re.IGNORECASE,
)


def _language_name(
    preferred_language: str,
    text: str = "",
) -> str:
    normalized = (
        preferred_language
        .strip()
        .lower()
    )

    if (
        normalized
        in {
            "mr",
            "mr-in",
            "marathi",
            "मराठी",
        }
        or "marathi"
        in normalized
    ):
        return "marathi"

    if (
        normalized
        in {
            "hi",
            "hi-in",
            "hindi",
            "हिन्दी",
            "हिंदी",
        }
        or "hindi"
        in normalized
    ):
        return "hindi"

    if (
        normalized
        in {
            "en",
            "en-in",
            "english",
        }
        or "english"
        in normalized
    ):
        return "english"

    marathi_words = re.compile(
        (
            r"(मला|आहे|आहेत|नाही|"
            r"बद्दल|सांग|सांगा|"
            r"करायचं|यामध्ये|"
            r"त्यामुळे|कशासाठी)"
        )
    )

    hindi_words = re.compile(
        (
            r"(क्या|है|हैं|नहीं|"
            r"बताओ|बताइए|इसके|"
            r"बारे|यह|इसमें)"
        )
    )

    if marathi_words.search(text):
        return "marathi"

    if hindi_words.search(text):
        return "hindi"

    return "english"


def _input_block_message(
    preferred_language: str,
    content: str,
) -> str:
    language = _language_name(
        preferred_language,
        content,
    )

    if language == "marathi":
        return (
            "मी अशा सूचनांवर प्रक्रिया करू शकत "
            "नाही ज्यामध्ये सुरक्षा नियम टाळणे, "
            "गुप्त माहिती मिळवणे किंवा डेटाबेसमध्ये "
            "असुरक्षित बदल करणे अपेक्षित आहे. "
            "कृपया सुपरकॉम्प्युटर किंवा HPC संबंधित "
            "सुरक्षित प्रश्न विचारा."
        )

    if language == "hindi":
        return (
            "मैं ऐसे निर्देशों पर प्रक्रिया नहीं "
            "कर सकती जिनमें सुरक्षा नियमों को "
            "दरकिनार करना, गुप्त जानकारी प्राप्त "
            "करना या डेटाबेस में असुरक्षित बदलाव "
            "करना शामिल हो। कृपया सुपरकंप्यूटर या "
            "HPC से संबंधित सुरक्षित प्रश्न पूछें।"
        )

    return (
        "I cannot process instructions that attempt "
        "to bypass safety rules, obtain confidential "
        "information, or make unsafe database changes. "
        "Please ask a safe question related to "
        "supercomputers or HPC."
    )


def _safe_output_fallback(
    preferred_language: str,
) -> str:
    language = _language_name(
        preferred_language
    )

    if language == "marathi":
        return (
            "सुरक्षा तपासणीमुळे हा प्रतिसाद "
            "दाखवता आला नाही. कृपया प्रश्न "
            "दुसऱ्या पद्धतीने विचारा."
        )

    if language == "hindi":
        return (
            "सुरक्षा जाँच के कारण यह उत्तर "
            "दिखाया नहीं जा सका। कृपया प्रश्न "
            "को दूसरे तरीके से पूछें।"
        )

    return (
        "The response could not be displayed because "
        "it did not pass the output safety check. "
        "Please rephrase the question."
    )


def validate_input(
    content: str,
    preferred_language: str = "auto",
) -> InputGuardrailResult:
    cleaned = content.strip()

    if not cleaned:
        return InputGuardrailResult(
            allowed=False,
            category="empty_input",
            public_message=(
                "Message content cannot "
                "be empty."
            ),
        )

    if _CONTROL_CHARACTERS.search(
        cleaned
    ):
        return InputGuardrailResult(
            allowed=False,
            category="invalid_characters",
            public_message=(
                _input_block_message(
                    preferred_language,
                    cleaned,
                )
            ),
        )

    for pattern in (
        _PROMPT_INJECTION_PATTERNS
    ):
        if pattern.search(cleaned):
            return InputGuardrailResult(
                allowed=False,
                category=(
                    "prompt_injection"
                ),
                public_message=(
                    _input_block_message(
                        preferred_language,
                        cleaned,
                    )
                ),
            )

    for pattern in (
        _SECRET_REQUEST_PATTERNS
    ):
        if pattern.search(cleaned):
            return InputGuardrailResult(
                allowed=False,
                category=(
                    "secret_extraction"
                ),
                public_message=(
                    _input_block_message(
                        preferred_language,
                        cleaned,
                    )
                ),
            )

    for pattern in (
        _DATABASE_MUTATION_PATTERNS
    ):
        if pattern.search(cleaned):
            return InputGuardrailResult(
                allowed=False,
                category=(
                    "unsafe_database_request"
                ),
                public_message=(
                    _input_block_message(
                        preferred_language,
                        cleaned,
                    )
                ),
            )

    return InputGuardrailResult(
        allowed=True,
        category="allowed",
        public_message=None,
    )


def guard_output(
    answer: str,
    preferred_language: str = "English",
    route: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> str:
    del route
    del evidence

    cleaned = str(
        answer or ""
    ).strip()

    if not cleaned:
        return _safe_output_fallback(
            preferred_language
        )

    cleaned = _CONTROL_CHARACTERS.sub(
        "",
        cleaned,
    )

    cleaned = _INTERNAL_OUTPUT_PATTERNS[
        0
    ].sub(
        "",
        cleaned,
    ).strip()

    for leak_pattern in (
        _INTERNAL_OUTPUT_PATTERNS[1:]
    ):
        if leak_pattern.search(cleaned):
            return _safe_output_fallback(
                preferred_language
            )

    cleaned = _SECRET_VALUE_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}=[REDACTED]"
        ),
        cleaned,
    )

    if (
        len(cleaned)
        > MAX_SAFE_OUTPUT_CHARS
    ):
        cleaned = (
            cleaned[
                :MAX_SAFE_OUTPUT_CHARS
            ].rstrip()
            + "\n\n[Response truncated safely.]"
        )

    if not cleaned:
        return _safe_output_fallback(
            preferred_language
        )

    return cleaned


def sanitize_stream_chunk(
    chunk: str,
) -> str:
    cleaned = str(
        chunk or ""
    )

    cleaned = _CONTROL_CHARACTERS.sub(
        "",
        cleaned,
    )

    cleaned = _SECRET_VALUE_PATTERN.sub(
        lambda match: (
            f"{match.group(1)}=[REDACTED]"
        ),
        cleaned,
    )

    return cleaned