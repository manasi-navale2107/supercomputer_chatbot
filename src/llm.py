from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=OLLAMA_CONTEXT_WINDOW,
        keep_alive=OLLAMA_KEEP_ALIVE,
    )


@lru_cache(maxsize=1)
def get_json_llm() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_ctx=OLLAMA_CONTEXT_WINDOW,
        keep_alive=OLLAMA_KEEP_ALIVE,
        format="json",
    )

