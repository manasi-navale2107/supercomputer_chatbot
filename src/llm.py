from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import (
    DEFAULT_LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MAX_RETRIES,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
    LLM_PROVIDERS,
    OLLAMA_BASE_URL,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
)


def normalize_llm_provider(
    provider: str | None = None,
) -> str:
    normalized = (
        provider
        or DEFAULT_LLM_PROVIDER
    ).strip().lower()

    if (
        normalized
        not in LLM_PROVIDERS
    ):
        raise ValueError(
            "LLM provider must be "
            "'ollama' or 'groq'."
        )

    if (
        normalized == "groq"
        and not GROQ_API_KEY
    ):
        raise RuntimeError(
            "GROQ_API_KEY is required "
            "when Groq is selected."
        )

    return normalized


def get_model_name(
    provider: str | None = None,
) -> str:
    normalized = (
        normalize_llm_provider(
            provider
        )
    )

    if normalized == "groq":
        return GROQ_MODEL

    return OLLAMA_MODEL


@lru_cache(maxsize=4)
def _get_llm(
    provider: str,
) -> Any:
    if provider == "groq":
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
            timeout=(
                GROQ_TIMEOUT_SECONDS
            ),
            max_retries=(
                GROQ_MAX_RETRIES
            ),
        )

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=(
            OLLAMA_CONTEXT_WINDOW
        ),
        keep_alive=(
            OLLAMA_KEEP_ALIVE
        ),
    )


def get_llm(
    provider: str | None = None,
) -> Any:
    normalized = (
        normalize_llm_provider(
            provider
        )
    )

    return _get_llm(
        normalized
    )


@lru_cache(maxsize=4)
def _get_json_llm(
    provider: str,
) -> Any:
    if provider == "groq":
        """
        Do not pass response_format=json_object
        to Groq here.

        Groq rejects requests when JSON mode is
        enabled but a particular prompt does not
        explicitly contain the word JSON.

        Router and table-selector prompts already
        request JSON, and their parsing functions
        validate the returned object.
        """

        return _get_llm(
            "groq"
        )

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=(
            OLLAMA_CONTEXT_WINDOW
        ),
        keep_alive=(
            OLLAMA_KEEP_ALIVE
        ),
        format="json",
    )


def get_json_llm(
    provider: str | None = None,
) -> Any:
    normalized = (
        normalize_llm_provider(
            provider
        )
    )

    return _get_json_llm(
        normalized
    )