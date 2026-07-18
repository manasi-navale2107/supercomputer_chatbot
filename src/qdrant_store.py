from __future__ import annotations

import json
import math
import uuid
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

from src.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_LOCAL_PATH,
    QDRANT_PREFER_GRPC,
    QDRANT_TIMEOUT_SECONDS,
    QDRANT_TOP_K,
    QDRANT_UPLOAD_BATCH_SIZE,
    QDRANT_URL,
)
from src.data_loader import dataset_fingerprint, load_datasets
from src.mysql_store import TABLE_DESCRIPTIONS


DOMAIN_KNOWLEDGE = {
    "green500": (
        "Green500 ranks supercomputers by energy efficiency. A smaller "
        "Green500 rank is better, and a larger GFLOPS-per-watt value indicates "
        "better energy efficiency."
    ),
    "top500": (
        "TOP500 ranks supercomputers by measured LINPACK performance. A smaller "
        "TOP500 rank is better."
    ),
    "rmax": (
        "Rmax is the measured LINPACK performance achieved by a supercomputer. "
        "In these datasets rmax_tflops is expressed in TFLOPS, and a higher "
        "value means higher measured performance."
    ),
    "rpeak": (
        "Rpeak is theoretical peak performance. It is different from Rmax, "
        "which is measured LINPACK performance."
    ),
    "power": (
        "power_kw represents power consumption in kilowatts. Lower power means "
        "less electricity consumption, but Green500 rank is primarily based on "
        "energy efficiency rather than power alone."
    ),
    "energy_efficiency": (
        "energy_efficiency_gflops_watt measures computational work per watt in "
        "GFLOPS/W. A higher value means better energy efficiency."
    ),
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBEDDING_BATCH_SIZE,
        },
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=QDRANT_PREFER_GRPC,
            timeout=QDRANT_TIMEOUT_SECONDS,
        )
    QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(
        path=str(QDRANT_LOCAL_PATH),
        timeout=QDRANT_TIMEOUT_SECONDS,
    )


def _safe_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _display_name(column: str) -> str:
    return column.replace("_", " ").strip().title()


def _row_document(table_name: str, row_index: int, row: pd.Series) -> str:
    lines = [
        f"Dataset: {table_name}",
        f"Dataset purpose: {TABLE_DESCRIPTIONS[table_name]}",
        f"Record index: {row_index}",
    ]
    for column, raw_value in row.items():
        value = _safe_value(raw_value)
        if value is None or str(value).strip() == "":
            continue
        rendered = str(value)
        if len(rendered) > 1_500:
            rendered = rendered[:1_500] + "..."
        lines.append(f"{_display_name(column)}: {rendered}")
    return "\n".join(lines)[:8_000]


def _iter_documents() -> Iterable[tuple[str, dict[str, Any]]]:
    datasets = load_datasets()

    for table_name, dataframe in datasets.items():
        schema_text = (
            f"Dataset: {table_name}\n"
            f"Purpose: {TABLE_DESCRIPTIONS[table_name]}\n"
            f"Columns: {', '.join(dataframe.columns)}\n"
            f"Number of records: {len(dataframe)}"
        )
        yield schema_text, {
            "kind": "document",
            "document_type": "dataset_schema",
            "source": f"{table_name}.csv",
            "table": table_name,
            "row_index": -1,
        }

        for row_index, row in dataframe.iterrows():
            yield _row_document(table_name, int(row_index), row), {
                "kind": "document",
                "document_type": "csv_row",
                "source": f"{table_name}.csv",
                "table": table_name,
                "row_index": int(row_index),
            }

    for term, description in DOMAIN_KNOWLEDGE.items():
        yield f"Term: {term}\nDefinition: {description}", {
            "kind": "document",
            "document_type": "domain_glossary",
            "source": "domain_glossary",
            "table": "domain_glossary",
            "row_index": -1,
        }


def _stored_manifest(client: QdrantClient) -> dict[str, Any] | None:
    if not client.collection_exists(QDRANT_COLLECTION):
        return None
    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kind",
                    match=models.MatchValue(value="manifest"),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    return dict(points[0].payload or {})


def build_qdrant_index(force: bool = False) -> dict[str, Any]:
    client = get_qdrant_client()
    embeddings = get_embeddings()
    fingerprint = dataset_fingerprint()
    manifest = _stored_manifest(client)

    if (
        not force
        and manifest
        and manifest.get("fingerprint") == fingerprint
        and manifest.get("embedding_model") == EMBEDDING_MODEL
    ):
        return {
            "status": "unchanged",
            "fingerprint": fingerprint,
            "collection": QDRANT_COLLECTION,
            "document_count": int(manifest.get("document_count", 0)),
        }

    probe_vector = embeddings.embed_query("supercomputer vector dimension probe")
    vector_size = len(probe_vector)

    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    documents = list(_iter_documents())
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start : start + EMBEDDING_BATCH_SIZE]
        texts = [text for text, _ in batch]
        vectors = embeddings.embed_documents(texts)
        points: list[models.PointStruct] = []
        for offset, ((text, metadata), vector) in enumerate(zip(batch, vectors)):
            absolute_index = start + offset
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{fingerprint}:{EMBEDDING_MODEL}:{absolute_index}",
                )
            )
            payload = dict(metadata)
            payload["content"] = text
            payload["fingerprint"] = fingerprint
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )
        client.upload_points(
            collection_name=QDRANT_COLLECTION,
            points=points,
            batch_size=QDRANT_UPLOAD_BATCH_SIZE,
            max_retries=3,
            wait=True,
        )

    manifest_payload = {
        "kind": "manifest",
        "fingerprint": fingerprint,
        "embedding_model": EMBEDDING_MODEL,
        "document_count": len(documents),
    }
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        wait=True,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"manifest:{QDRANT_COLLECTION}")),
                vector=[0.0] * vector_size,
                payload=manifest_payload,
            )
        ],
    )

    return {
        "status": "indexed",
        "fingerprint": fingerprint,
        "collection": QDRANT_COLLECTION,
        "document_count": len(documents),
    }


def verify_qdrant_ready() -> None:
    client = get_qdrant_client()
    manifest = _stored_manifest(client)
    if not manifest:
        raise RuntimeError(
            "The Qdrant collection is not initialized. Run: python -m src.bootstrap"
        )
    current = dataset_fingerprint()
    if manifest.get("fingerprint") != current:
        raise RuntimeError(
            "The Qdrant index is stale because the CSV files changed. "
            "Run: python -m src.bootstrap"
        )
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            "The configured embedding model differs from the indexed model. "
            "Run: python -m src.bootstrap --force"
        )


def semantic_search(question: str, k: int = QDRANT_TOP_K) -> dict[str, Any]:
    verify_qdrant_ready()
    query_vector = get_embeddings().embed_query(question)
    response = get_qdrant_client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kind",
                    match=models.MatchValue(value="document"),
                )
            ]
        ),
        limit=k,
        with_payload=True,
        with_vectors=False,
    )

    documents: list[dict[str, Any]] = []
    for point in response.points:
        payload = dict(point.payload or {})
        documents.append(
            {
                "content": str(payload.get("content", ""))[:2_500],
                "source": payload.get("source", "unknown"),
                "table": payload.get("table", "unknown"),
                "row_index": payload.get("row_index"),
                "document_type": payload.get("document_type", "unknown"),
                "score": round(float(point.score), 6),
            }
        )

    return {
        "success": True,
        "query": question,
        "documents": documents,
        "document_count": len(documents),
    }


def evidence_to_text(
    evidence: dict[str, Any],
) -> str:
    compact = dict(evidence)

    if compact.get("route") == "structured":
        rows = list(
            compact.get("rows", [])
        )

        compact["rows"] = rows[:50]

        if len(rows) > 50:
            compact[
                "rows_omitted_from_answer_prompt"
            ] = len(rows) - 50

        compact["query_results"] = [
            {
                "source_tables":
                    result.get(
                        "source_tables",
                        [],
                    ),
                "row_count":
                    result.get(
                        "row_count",
                        0,
                    ),
                "truncated":
                    result.get(
                        "truncated",
                        False,
                    ),
                "columns":
                    result.get(
                        "columns",
                        [],
                    ),
                "rows": list(
                    result.get(
                        "rows",
                        [],
                    )
                )[:12],
            }
            for result
            in compact.get(
                "query_results",
                [],
            )
        ]

    elif compact.get("route") == "semantic":
        compact["documents"] = [
            {
                **document,
                "content": str(
                    document.get(
                        "content",
                        "",
                    )
                )[:2500],
            }
            for document
            in compact.get(
                "documents",
                [],
            )[:8]
        ]

    rendered = json.dumps(
        compact,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return rendered[:30000]