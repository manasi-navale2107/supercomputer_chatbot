from __future__ import annotations

import re
from typing import Any

from qdrant_client import models

from src.config import (
    QDRANT_COLLECTION,
    QDRANT_TOP_K,
)
from src.qdrant_store import (
    get_embeddings,
    get_qdrant_client,
    verify_qdrant_ready,
)


def _document_from_point(
    point: Any,
    retrieval_type: str,
) -> dict[str, Any]:
    payload = dict(
        point.payload or {}
    )

    score = getattr(
        point,
        "score",
        None,
    )

    return {
        "content": str(
            payload.get(
                "content",
                "",
            )
        )[:2_500],
        "source": payload.get(
            "source",
            "unknown",
        ),
        "table": payload.get(
            "table",
            "unknown",
        ),
        "row_id": payload.get(
            "row_id"
        ),
        "row_index": payload.get(
            "row_index"
        ),
        "document_type": payload.get(
            "document_type",
            "unknown",
        ),
        "score": (
            round(
                float(score),
                6,
            )
            if score is not None
            else None
        ),
        "retrieval_type":
            retrieval_type,
    }


def _document_key(
    document: dict[str, Any],
) -> tuple[Any, ...]:
    row_id = document.get(
        "row_id"
    )

    if row_id:
        return (
            document.get(
                "source"
            ),
            row_id,
            document.get(
                "document_type"
            ),
        )

    return (
        document.get(
            "source"
        ),
        document.get(
            "row_index"
        ),
        document.get(
            "document_type"
        ),
    )


def _normalise_query(
    query: str,
) -> str:
    """
    Normalize whitespace and case.

    KAIROS, Kairos and kairos
    should match the same token.
    """

    cleaned_query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    return (
        cleaned_query.casefold()
    )


def ensure_content_text_index() -> None:
    client = get_qdrant_client()

    collection = (
        client.get_collection(
            QDRANT_COLLECTION
        )
    )

    payload_schema = (
        getattr(
            collection,
            "payload_schema",
            {},
        )
        or {}
    )

    if "content" in payload_schema:
        return

    client.create_payload_index(
        collection_name=(
            QDRANT_COLLECTION
        ),
        field_name="content",
        field_schema=(
            models.TextIndexParams(
                type=(
                    models
                    .TextIndexType
                    .TEXT
                ),
                tokenizer=(
                    models
                    .TokenizerType
                    .WORD
                ),
                lowercase=True,
            )
        ),
        wait=True,
    )


def _manifest_condition():
    """
    Return a condition that identifies
    the internal Qdrant manifest point.
    """

    return models.FieldCondition(
        key="kind",
        match=models.MatchValue(
            value="manifest",
        ),
    )


def _full_text_search(
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    client = get_qdrant_client()

    search_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="content",
                match=models.MatchText(
                    text=(
                        _normalise_query(
                            query
                        )
                    ),
                ),
            )
        ],
        must_not=[
            _manifest_condition()
        ],
    )

    points, _ = client.scroll(
        collection_name=(
            QDRANT_COLLECTION
        ),
        scroll_filter=search_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    documents: list[
        dict[str, Any]
    ] = []

    for point in points:
        payload = dict(
            point.payload or {}
        )

        content = str(
            payload.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        documents.append(
            _document_from_point(
                point,
                "full_text",
            )
        )

    return documents


def _vector_search(
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_vector = (
        get_embeddings()
        .embed_query(
            query
        )
    )

    response = (
        get_qdrant_client()
        .query_points(
            collection_name=(
                QDRANT_COLLECTION
            ),
            query=query_vector,
            query_filter=models.Filter(
                must_not=[
                    _manifest_condition()
                ],
            ),
            limit=(
                limit + 4
            ),
            with_payload=True,
            with_vectors=False,
        )
    )

    documents: list[
        dict[str, Any]
    ] = []

    for point in response.points:
        payload = dict(
            point.payload or {}
        )

        if (
            payload.get("kind")
            == "manifest"
        ):
            continue

        content = str(
            payload.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        documents.append(
            _document_from_point(
                point,
                "vector",
            )
        )

        if len(documents) >= limit:
            break

    return documents


def _merge_documents(
    text_documents: list[
        dict[str, Any]
    ],
    vector_documents: list[
        dict[str, Any]
    ],
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[Any, ...]
    ] = set()

    for document in [
        *text_documents,
        *vector_documents,
    ]:
        content = str(
            document.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        key = _document_key(
            document
        )

        if key in seen:
            continue

        merged.append(
            document
        )

        seen.add(
            key
        )

        if len(merged) >= limit:
            break

    return merged


def _diversify_by_dataset(
    documents: list[
        dict[str, Any]
    ],
    limit: int,
) -> list[dict[str, Any]]:
    buckets: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for document in documents:
        dataset = str(
            document.get("table")
            or document.get("source")
            or "unknown"
        )

        buckets.setdefault(
            dataset,
            [],
        ).append(
            document
        )

    selected: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[Any, ...]
    ] = set()

    # Select one exact match from
    # each represented dataset.
    for bucket in buckets.values():
        document = bucket[0]

        key = _document_key(
            document
        )

        if key in seen:
            continue

        selected.append(
            document
        )

        seen.add(
            key
        )

        if len(selected) >= limit:
            return selected

    # Fill remaining positions in
    # their retrieval order.
    for document in documents:
        key = _document_key(
            document
        )

        if key in seen:
            continue

        selected.append(
            document
        )

        seen.add(
            key
        )

        if len(selected) >= limit:
            break

    return selected


def semantic_search(
    query: str,
    k: int = QDRANT_TOP_K,
) -> dict[str, Any]:
    query = re.sub(
        r"\s+",
        " ",
        query,
    ).strip()

    if not query:
        raise ValueError(
            "Semantic search query "
            "cannot be empty."
        )

    verify_qdrant_ready()
    ensure_content_text_index()

    # No global fingerprint filter.
    # Stable unchanged documents must
    # remain searchable after updates.
    text_candidates = (
        _full_text_search(
            query=query,
            limit=max(
                k * 16,
                128,
            ),
        )
    )

    represented_datasets = {
        str(
            document.get("table")
            or document.get("source")
            or "unknown"
        )
        for document
        in text_candidates
    }

    text_limit = min(
        k,
        max(
            1,
            k // 2,
            len(
                represented_datasets
            ),
        ),
    )

    text_documents = (
        _diversify_by_dataset(
            documents=(
                text_candidates
            ),
            limit=text_limit,
        )
    )

    vector_documents = (
        _vector_search(
            query=query,
            limit=max(
                k * 2,
                16,
            ),
        )
    )

    documents = (
        _merge_documents(
            text_documents=(
                text_documents
            ),
            vector_documents=(
                vector_documents
            ),
            limit=k,
        )
    )

    return {
        "success":
            True,
        "query":
            query,
        "normalized_text_query": (
            _normalise_query(
                query
            )
        ),
        "search_type": (
            "case_insensitive_"
            "qdrant_full_text_"
            "and_dense_vector"
        ),
        "documents":
            documents,
        "document_count":
            len(documents),
    }