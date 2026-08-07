from __future__ import annotations

import json
import logging
import math
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)
from qdrant_client import (
    QdrantClient,
    models,
)

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
from src.data_loader import (
    INTERNAL_ROW_HASH_COLUMN,
    INTERNAL_ROW_ID_COLUMN,
    build_dataset_change_plan,
    file_fingerprints,
    load_datasets,
)
from src.mysql_store import (
    TABLE_DESCRIPTIONS,
)


logger = logging.getLogger(
    __name__
)

INDEX_VERSION = 2


DOMAIN_KNOWLEDGE = {
    "green500": (
        "Green500 ranks supercomputers by "
        "energy efficiency. A smaller Green500 "
        "rank is better, and a larger "
        "GFLOPS-per-watt value indicates better "
        "energy efficiency."
    ),
    "top500": (
        "TOP500 ranks supercomputers by measured "
        "LINPACK performance. A smaller TOP500 "
        "rank is better."
    ),
    "rmax": (
        "Rmax is measured LINPACK performance "
        "achieved by a supercomputer. In these "
        "datasets rmax_tflops is expressed in "
        "TFLOPS, and a higher value means higher "
        "performance."
    ),
    "rpeak": (
        "Rpeak is theoretical peak performance. "
        "It is different from Rmax, which is "
        "measured LINPACK performance."
    ),
    "power": (
        "power_kw represents power consumption "
        "in kilowatts. Lower power means less "
        "electricity consumption, but Green500 "
        "rank is primarily based on efficiency."
    ),
    "energy_efficiency": (
        "energy_efficiency_gflops_watt measures "
        "computational work per watt in GFLOPS/W. "
        "A higher value means better efficiency."
    ),
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size":
                EMBEDDING_BATCH_SIZE,
        },
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if QDRANT_URL:
        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=(
                QDRANT_PREFER_GRPC
            ),
            timeout=(
                QDRANT_TIMEOUT_SECONDS
            ),
        )

    QDRANT_LOCAL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(
            QDRANT_LOCAL_PATH
        ),
        timeout=QDRANT_TIMEOUT_SECONDS,
    )


def _safe_value(
    value: Any,
) -> Any:
    if (
        value is None
        or value is pd.NA
    ):
        return None

    if (
        isinstance(
            value,
            (
                float,
                np.floating,
            ),
        )
        and math.isnan(
            float(value)
        )
    ):
        return None

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    try:
        if pd.isna(value):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(value)


def _display_name(
    column: str,
) -> str:
    return (
        column.replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )


def _dataset_description(
    table_name: str,
) -> str:
    return TABLE_DESCRIPTIONS.get(
        table_name,
        (
            "Dataset loaded from "
            f"{table_name}.csv."
        ),
    )


def _stable_point_id(
    document_type: str,
    logical_id: str,
) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            (
                f"{QDRANT_COLLECTION}:"
                f"{EMBEDDING_MODEL}:"
                f"{document_type}:"
                f"{logical_id}"
            ),
        )
    )


def _row_document(
    table_name: str,
    row_index: int,
    row: pd.Series,
) -> tuple[
    str,
    dict[str, Any],
]:
    row_id = str(
        row[
            INTERNAL_ROW_ID_COLUMN
        ]
    )

    row_hash = str(
        row[
            INTERNAL_ROW_HASH_COLUMN
        ]
    )

    lines = [
        f"Dataset: {table_name}",
        (
            "Dataset purpose: "
            f"{_dataset_description(table_name)}"
        ),
        f"Record ID: {row_id}",
    ]

    for column, raw_value in row.items():
        if column in {
            INTERNAL_ROW_ID_COLUMN,
            INTERNAL_ROW_HASH_COLUMN,
        }:
            continue

        value = _safe_value(
            raw_value
        )

        if (
            value is None
            or not str(value).strip()
        ):
            continue

        rendered = str(value)

        if len(rendered) > 1_500:
            rendered = (
                rendered[:1_500]
                + "..."
            )

        lines.append(
            f"{_display_name(column)}: "
            f"{rendered}"
        )

    content = "\n".join(
        lines
    )[:8_000]

    payload = {
        "kind": "document",
        "document_type": "csv_row",
        "source": f"{table_name}.csv",
        "table": table_name,
        "row_id": row_id,
        "row_hash": row_hash,
        "row_index": row_index,
        "content": content,
    }

    return content, payload


def _schema_document(
    table_name: str,
    dataframe: pd.DataFrame,
) -> tuple[
    str,
    dict[str, Any],
]:
    visible_columns = [
        column
        for column
        in dataframe.columns
        if column not in {
            INTERNAL_ROW_ID_COLUMN,
            INTERNAL_ROW_HASH_COLUMN,
        }
    ]

    content = (
        f"Dataset: {table_name}\n"
        "Purpose: "
        f"{_dataset_description(table_name)}\n"
        "Columns: "
        f"{', '.join(visible_columns)}\n"
        "Number of records: "
        f"{len(dataframe)}"
    )

    payload = {
        "kind": "document",
        "document_type":
            "dataset_schema",
        "source": f"{table_name}.csv",
        "table": table_name,
        "row_id":
            f"schema:{table_name}",
        "row_index": -1,
        "content": content,
    }

    return content, payload


def _glossary_document(
    term: str,
    description: str,
) -> tuple[
    str,
    dict[str, Any],
]:
    content = (
        f"Term: {term}\n"
        f"Definition: {description}"
    )

    payload = {
        "kind": "document",
        "document_type":
            "domain_glossary",
        "source": "domain_glossary",
        "table": "domain_glossary",
        "row_id":
            f"glossary:{term}",
        "row_index": -1,
        "content": content,
    }

    return content, payload


def _stored_manifest(
    client: QdrantClient,
) -> dict[str, Any] | None:
    if not client.collection_exists(
        QDRANT_COLLECTION
    ):
        return None

    points, _ = client.scroll(
        collection_name=(
            QDRANT_COLLECTION
        ),
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kind",
                    match=(
                        models.MatchValue(
                            value="manifest"
                        )
                    ),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    return dict(
        points[0].payload or {}
    )


def get_active_qdrant_fingerprint() -> str:
    manifest = _stored_manifest(
        get_qdrant_client()
    )

    if not manifest:
        raise RuntimeError(
            "The Qdrant manifest "
            "is missing."
        )

    fingerprint = str(
        manifest.get(
            "fingerprint",
            "",
        )
    ).strip()

    if not fingerprint:
        raise RuntimeError(
            "The Qdrant manifest does "
            "not contain a fingerprint."
        )

    return fingerprint


def _existing_row_points(
    client: QdrantClient,
    table_name: str,
) -> dict[str, str]:
    mapping: dict[
        str,
        str,
    ] = {}

    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=(
                QDRANT_COLLECTION
            ),
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="kind",
                        match=(
                            models.MatchValue(
                                value="document"
                            )
                        ),
                    ),
                    models.FieldCondition(
                        key="document_type",
                        match=(
                            models.MatchValue(
                                value="csv_row"
                            )
                        ),
                    ),
                    models.FieldCondition(
                        key="table",
                        match=(
                            models.MatchValue(
                                value=table_name
                            )
                        ),
                    ),
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = dict(
                point.payload or {}
            )

            row_id = str(
                payload.get(
                    "row_id",
                    "",
                )
            ).strip()

            if row_id:
                mapping[row_id] = str(
                    point.id
                )

        if offset is None:
            break

    return mapping


def _document_needs_upsert(
    client: QdrantClient,
    point_id: str,
    content: str,
) -> bool:
    points = client.retrieve(
        collection_name=(
            QDRANT_COLLECTION
        ),
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return True

    payload = dict(
        points[0].payload or {}
    )

    return (
        str(
            payload.get(
                "content",
                "",
            )
        )
        != content
    )


def _delete_point_ids(
    client: QdrantClient,
    point_ids: list[str],
) -> None:
    if not point_ids:
        return

    client.delete(
        collection_name=(
            QDRANT_COLLECTION
        ),
        points_selector=(
            models.PointIdsList(
                points=point_ids
            )
        ),
        wait=True,
    )


def _delete_table_documents(
    client: QdrantClient,
    table_name: str,
) -> None:
    client.delete(
        collection_name=(
            QDRANT_COLLECTION
        ),
        points_selector=(
            models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kind",
                            match=(
                                models.MatchValue(
                                    value="document"
                                )
                            ),
                        ),
                        models.FieldCondition(
                            key="table",
                            match=(
                                models.MatchValue(
                                    value=(
                                        table_name
                                    )
                                )
                            ),
                        ),
                    ]
                )
            )
        ),
        wait=True,
    )


def _upload_documents(
    client: QdrantClient,
    documents: list[
        tuple[
            str,
            str,
            dict[str, Any],
        ]
    ],
) -> int:
    if not documents:
        return 0

    embeddings = get_embeddings()

    for start in range(
        0,
        len(documents),
        EMBEDDING_BATCH_SIZE,
    ):
        batch = documents[
            start:
            start
            + EMBEDDING_BATCH_SIZE
        ]

        texts = [
            content
            for _, content, _
            in batch
        ]

        vectors = (
            embeddings
            .embed_documents(
                texts
            )
        )

        points = [
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for (
                point_id,
                _,
                payload,
            ), vector
            in zip(
                batch,
                vectors,
            )
        ]

        client.upload_points(
            collection_name=(
                QDRANT_COLLECTION
            ),
            points=points,
            batch_size=(
                QDRANT_UPLOAD_BATCH_SIZE
            ),
            max_retries=3,
            wait=True,
        )

    return len(documents)


def _create_collection(
    client: QdrantClient,
) -> int:
    probe_vector = (
        get_embeddings()
        .embed_query(
            "supercomputer vector "
            "dimension probe"
        )
    )

    vector_size = len(
        probe_vector
    )

    client.create_collection(
        collection_name=(
            QDRANT_COLLECTION
        ),
        vectors_config=(
            models.VectorParams(
                size=vector_size,
                distance=(
                    models.Distance.COSINE
                ),
            )
        ),
    )

    return vector_size


def _count_documents(
    client: QdrantClient,
) -> int:
    response = client.count(
        collection_name=(
            QDRANT_COLLECTION
        ),
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kind",
                    match=(
                        models.MatchValue(
                            value="document"
                        )
                    ),
                )
            ]
        ),
        exact=True,
    )

    return int(
        response.count
    )


def build_qdrant_index(
    force: bool = False,
    change_plan: (
        dict[str, Any] | None
    ) = None,
) -> dict[str, Any]:
    plan = (
        dict(change_plan)
        if change_plan is not None
        else build_dataset_change_plan()
    )

    client = get_qdrant_client()

    manifest = _stored_manifest(
        client
    )

    collection_exists = (
        client.collection_exists(
            QDRANT_COLLECTION
        )
    )

    current_fingerprints = dict(
        plan["current_fingerprints"]
    )

    current_tables = sorted(
        current_fingerprints
    )

    fingerprint = str(
        plan["fingerprint"]
    )

    requires_recreate = bool(
        force
        or not collection_exists
        or manifest is None
        or manifest.get(
            "embedding_model"
        ) != EMBEDDING_MODEL
        or int(
            manifest.get(
                "index_version",
                0,
            )
        ) != INDEX_VERSION
    )

    if (
        not requires_recreate
        and not plan["has_changes"]
        and manifest.get(
            "fingerprint"
        ) == fingerprint
    ):
        return {
            "status": "unchanged",
            "fingerprint":
                fingerprint,
            "collection":
                QDRANT_COLLECTION,
            "document_count": int(
                manifest.get(
                    "document_count",
                    0,
                )
            ),
            "embedded_documents": 0,
            "deleted_documents": 0,
            "collection_recreated":
                False,
        }

    if (
        requires_recreate
        and collection_exists
    ):
        client.delete_collection(
            QDRANT_COLLECTION
        )

        collection_exists = False

    if not collection_exists:
        vector_size = (
            _create_collection(
                client
            )
        )

    else:
        vector_size = int(
            manifest.get(
                "vector_size",
                0,
            )
        )

        if vector_size <= 0:
            vector_size = len(
                get_embeddings()
                .embed_query(
                    "vector dimension probe"
                )
            )

    tables_to_load = (
        current_tables
        if requires_recreate
        else list(
            plan["tables_to_load"]
        )
    )

    deleted_tables = (
        []
        if requires_recreate
        else list(
            plan["deleted_tables"]
        )
    )

    datasets = load_datasets(
        include_row_metadata=True,
        table_names=tables_to_load,
    )

    documents_to_embed: list[
        tuple[
            str,
            str,
            dict[str, Any],
        ]
    ] = []

    deleted_documents = 0

    for table_name in deleted_tables:
        existing = (
            _existing_row_points(
                client,
                table_name,
            )
        )

        deleted_documents += len(
            existing
        )

        _delete_table_documents(
            client,
            table_name,
        )

    for table_name in tables_to_load:
        dataframe = datasets[
            table_name
        ]

        existing_rows = (
            {}
            if requires_recreate
            else _existing_row_points(
                client,
                table_name,
            )
        )

        current_row_ids = set(
            dataframe[
                INTERNAL_ROW_ID_COLUMN
            ].astype(str)
        )

        stale_point_ids = [
            point_id
            for row_id, point_id
            in existing_rows.items()
            if row_id
            not in current_row_ids
        ]

        _delete_point_ids(
            client,
            stale_point_ids,
        )

        deleted_documents += len(
            stale_point_ids
        )

        (
            schema_content,
            schema_payload,
        ) = _schema_document(
            table_name,
            dataframe,
        )

        schema_point_id = (
            _stable_point_id(
                "dataset_schema",
                table_name,
            )
        )

        if (
            requires_recreate
            or _document_needs_upsert(
                client,
                schema_point_id,
                schema_content,
            )
        ):
            documents_to_embed.append(
                (
                    schema_point_id,
                    schema_content,
                    schema_payload,
                )
            )

        for row_index, (_, row) in enumerate(
            dataframe.iterrows()
        ):
            row_id = str(
                row[
                    INTERNAL_ROW_ID_COLUMN
                ]
            )

            if row_id in existing_rows:
                continue

            content, payload = (
                _row_document(
                    table_name,
                    row_index,
                    row,
                )
            )

            documents_to_embed.append(
                (
                    _stable_point_id(
                        "csv_row",
                        row_id,
                    ),
                    content,
                    payload,
                )
            )

    for term, description in (
        DOMAIN_KNOWLEDGE.items()
    ):
        content, payload = (
            _glossary_document(
                term,
                description,
            )
        )

        point_id = _stable_point_id(
            "domain_glossary",
            term,
        )

        if (
            requires_recreate
            or _document_needs_upsert(
                client,
                point_id,
                content,
            )
        ):
            documents_to_embed.append(
                (
                    point_id,
                    content,
                    payload,
                )
            )

    embedded_documents = (
        _upload_documents(
            client,
            documents_to_embed,
        )
    )

    if (
        current_fingerprints
        != file_fingerprints()
    ):
        raise RuntimeError(
            "CSV files changed during "
            "Qdrant synchronization. "
            "The update will be retried."
        )

    document_count = (
        _count_documents(
            client
        )
    )

    manifest_payload = {
        "kind": "manifest",
        "index_version":
            INDEX_VERSION,
        "fingerprint":
            fingerprint,
        "embedding_model":
            EMBEDDING_MODEL,
        "vector_size":
            vector_size,
        "document_count":
            document_count,
        "file_fingerprints":
            current_fingerprints,
    }

    client.upsert(
        collection_name=(
            QDRANT_COLLECTION
        ),
        wait=True,
        points=[
            models.PointStruct(
                id=_stable_point_id(
                    "manifest",
                    QDRANT_COLLECTION,
                ),
                vector=(
                    [0.0]
                    * vector_size
                ),
                payload=(
                    manifest_payload
                ),
            )
        ],
    )

    return {
        "status": "synchronized",
        "fingerprint":
            fingerprint,
        "collection":
            QDRANT_COLLECTION,
        "document_count":
            document_count,
        "embedded_documents":
            embedded_documents,
        "deleted_documents":
            deleted_documents,
        "collection_recreated":
            requires_recreate,
    }


def verify_qdrant_ready() -> None:
    client = get_qdrant_client()

    manifest = _stored_manifest(
        client
    )

    if not manifest:
        raise RuntimeError(
            "The Qdrant collection is not "
            "initialized. Run: "
            "python -m src.bootstrap"
        )

    if (
        manifest.get(
            "embedding_model"
        )
        != EMBEDDING_MODEL
    ):
        raise RuntimeError(
            "The configured embedding model "
            "differs from the indexed model. "
            "Run: python -m "
            "src.bootstrap --force"
        )


def semantic_search(
    question: str,
    k: int = QDRANT_TOP_K,
) -> dict[str, Any]:
    verify_qdrant_ready()

    query_vector = (
        get_embeddings()
        .embed_query(
            question
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
                must=[
                    models.FieldCondition(
                        key="kind",
                        match=(
                            models.MatchValue(
                                value="document"
                            )
                        ),
                    )
                ]
            ),
            limit=k,
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

        documents.append(
            {
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
                "document_type":
                    payload.get(
                        "document_type",
                        "unknown",
                    ),
                "score": round(
                    float(point.score),
                    6,
                ),
            }
        )

    return {
        "success": True,
        "query": question,
        "documents": documents,
        "document_count":
            len(documents),
    }


def evidence_to_text(
    evidence: dict[str, Any],
) -> str:
    compact = dict(
        evidence
    )

    if (
        compact.get("route")
        == "structured"
    ):
        rows = list(
            compact.get(
                "rows",
                [],
            )
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

    elif (
        compact.get("route")
        == "semantic"
    ):
        compact["documents"] = [
            {
                **document,
                "content": str(
                    document.get(
                        "content",
                        "",
                    )
                )[:2_500],
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

    return rendered[:30_000]