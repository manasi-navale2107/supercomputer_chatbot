from __future__ import annotations

from typing import Any

from src.hybrid_retriever import semantic_search


def run_semantic_route(
    state: dict[str, Any],
) -> dict[str, Any]:
    decision = state.get("decision", {})

    retrieval_query = str(
        decision.get("retrieval_query")
        or decision.get("standalone_question")
        or state["question"]
    ).strip()

    try:
        evidence = semantic_search(
            retrieval_query
        )

        evidence.update(
            {
                "route": "semantic",
                "retrieval_query": retrieval_query,
            }
        )

        error = None

    except Exception as retrieval_error:
        evidence = {
            "route": "semantic",
            "success": False,
            "documents": [],
            "document_count": 0,
            "retrieval_query": retrieval_query,
            "error": (
                "Semantic retrieval could not "
                "be completed."
            ),
            "retrieval_details": str(
                retrieval_error
            ),
        }

        error = str(retrieval_error)

    return {
        "evidence": evidence,
        "llm_calls": int(
            state.get("llm_calls", 0)
        ),
        "error": error,
    }