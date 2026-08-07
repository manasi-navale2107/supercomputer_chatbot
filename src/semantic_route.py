from __future__ import annotations

from typing import Any

from src.hybrid_retriever import semantic_search


def _build_semantic_citations(
    documents: list[dict[str, Any]],
    retrieval_query: str,
) -> dict[str, Any]:
    datasets: list[str] = []
    citation_documents: list[
        dict[str, Any]
    ] = []

    seen_documents: set[
        tuple[Any, ...]
    ] = set()

    for document in documents:
        source = str(
            document.get("source")
            or "unknown"
        )

        table = str(
            document.get("table")
            or source
            or "unknown"
        )

        row_index = document.get(
            "row_index"
        )

        document_type = str(
            document.get(
                "document_type"
            )
            or "unknown"
        )

        score = document.get(
            "score"
        )

        retrieval_type = document.get(
            "retrieval_type"
        )

        identity = (
            source,
            table,
            row_index,
            document_type,
        )

        if identity in seen_documents:
            continue

        seen_documents.add(
            identity
        )

        if (
            table
            and table != "unknown"
            and table not in datasets
        ):
            datasets.append(
                table
            )

        title = source

        if (
            row_index is not None
            and row_index != -1
        ):
            title = (
                f"{source} — record "
                f"{row_index}"
            )

        citation_documents.append(
            {
                "title": title,
                "source": source,
                "table": table,
                "row_index": row_index,
                "document_type":
                    document_type,
                "score": score,
                "retrieval_type":
                    retrieval_type,
            }
        )

    return {
        "route": "semantic",
        "datasets": datasets,
        "documents":
            citation_documents,
        "sql": None,
        "row_count": len(documents),
        "retrieval_query":
            retrieval_query,
    }


def run_semantic_route(
    state: dict[str, Any],
) -> dict[str, Any]:
    decision = state.get(
        "decision",
        {},
    )

    retrieval_query = str(
        decision.get(
            "retrieval_query"
        )
        or decision.get(
            "standalone_question"
        )
        or state["question"]
    ).strip()

    try:
        evidence = semantic_search(
            retrieval_query
        )

        documents = evidence.get(
            "documents",
            [],
        )

        if not isinstance(
            documents,
            list,
        ):
            documents = []

        citations = (
            _build_semantic_citations(
                documents=documents,
                retrieval_query=(
                    retrieval_query
                ),
            )
        )

        evidence.update(
            {
                "route": "semantic",
                "retrieval_query":
                    retrieval_query,
                "citations": citations,
            }
        )

        error = None

    except Exception as retrieval_error:
        citations = {
            "route": "semantic",
            "datasets": [],
            "documents": [],
            "sql": None,
            "row_count": 0,
            "retrieval_query":
                retrieval_query,
        }

        evidence = {
            "route": "semantic",
            "success": False,
            "documents": [],
            "document_count": 0,
            "retrieval_query":
                retrieval_query,
            "citations": citations,
            "error": (
                "Semantic retrieval could not "
                "be completed."
            ),
            "retrieval_details": str(
                retrieval_error
            ),
        }

        error = str(
            retrieval_error
        )

    return {
        "evidence": evidence,
        "llm_calls": int(
            state.get(
                "llm_calls",
                0,
            )
        ),
        "error": error,
    }