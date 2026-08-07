from __future__ import annotations

import hashlib
import json
import math
from datetime import (
    date,
    datetime,
    timezone,
)
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import (
    DATASET_MANIFEST_PATH,
    discover_csv_paths,
)


INTERNAL_ROW_ID_COLUMN = "_row_id"
INTERNAL_ROW_HASH_COLUMN = "_row_hash"


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_identifier(
    value: str,
) -> str:
    value = value.strip().lower()

    value = pd.Series(
        [value]
    ).str.replace(
        r"[^a-z0-9_]+",
        "_",
        regex=True,
    ).iloc[0]

    while "__" in value:
        value = value.replace(
            "__",
            "_",
        )

    value = value.strip("_")

    if not value:
        value = "unnamed_column"

    if value[0].isdigit():
        value = f"col_{value}"

    if len(value) > 64:
        suffix = hashlib.sha1(
            value.encode("utf-8")
        ).hexdigest()[:8]

        value = (
            f"{value[:55]}_{suffix}"
        )

    return value


def clean_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    cleaned = dataframe.copy()

    cleaned.columns = [
        normalize_identifier(
            str(column)
        )
        for column in dataframe.columns
    ]

    duplicates = (
        cleaned.columns[
            cleaned.columns.duplicated()
        ].tolist()
    )

    if duplicates:
        raise ValueError(
            "Duplicate columns remain after "
            "normalization: "
            + ", ".join(
                sorted(
                    set(duplicates)
                )
            )
        )

    protected_columns = {
        INTERNAL_ROW_ID_COLUMN,
        INTERNAL_ROW_HASH_COLUMN,
    }

    conflicting_columns = (
        protected_columns
        & set(cleaned.columns)
    )

    if conflicting_columns:
        raise ValueError(
            "CSV files cannot use the reserved "
            "internal columns: "
            + ", ".join(
                sorted(
                    conflicting_columns
                )
            )
        )

    return cleaned


def validate_csv_files() -> list[Path]:
    """
    Return a fresh runtime scan of data/*.csv.

    The function name is retained for compatibility
    with existing modules.
    """

    return list(
        discover_csv_paths()
    )


def get_dataset_table_names() -> tuple[str, ...]:
    """
    Return the current dataset names.

    This function performs a fresh folder scan,
    so newly added CSV files are included.
    """

    return tuple(
        path.stem.strip().lower()
        for path in discover_csv_paths()
    )


def _canonical_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        numeric_value = float(value)

        if math.isnan(numeric_value):
            return None

        if math.isinf(numeric_value):
            return str(numeric_value)

        return numeric_value

    if not isinstance(
        value,
        (
            str,
            bytes,
        ),
    ):
        try:
            if pd.isna(value):
                return None

        except (
            TypeError,
            ValueError,
        ):
            pass

    if isinstance(value, np.generic):
        return _canonical_value(
            value.item()
        )

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, bytes):
        return value.hex()

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

    return str(value)


def _canonical_row_json(
    table_name: str,
    columns: list[str],
    values: tuple[Any, ...],
) -> str:
    payload = {
        "table": table_name,
        "values": {
            column: _canonical_value(value)
            for column, value
            in zip(columns, values)
        },
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def add_row_metadata(
    table_name: str,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add stable row identifiers.

    Unchanged row:
        keeps the same ID.

    New row:
        receives a new ID.

    Modified row:
        old ID disappears and a new ID is created.

    Exact duplicate rows:
        receive separate occurrence-based IDs.
    """

    if (
        INTERNAL_ROW_ID_COLUMN
        in dataframe.columns
        or INTERNAL_ROW_HASH_COLUMN
        in dataframe.columns
    ):
        raise ValueError(
            "The dataframe already contains "
            "internal row metadata."
        )

    source_columns = list(
        dataframe.columns
    )

    row_hashes: list[str] = []

    for values in dataframe.itertuples(
        index=False,
        name=None,
    ):
        canonical_row = (
            _canonical_row_json(
                table_name=table_name,
                columns=source_columns,
                values=values,
            )
        )

        row_hashes.append(
            hashlib.sha256(
                canonical_row.encode("utf-8")
            ).hexdigest()
        )

    occurrences: dict[str, int] = {}
    row_ids: list[str] = []

    for row_hash in row_hashes:
        occurrence = occurrences.get(
            row_hash,
            0,
        )

        occurrences[row_hash] = (
            occurrence + 1
        )

        row_id_source = (
            f"{table_name}:"
            f"{row_hash}:"
            f"{occurrence}"
        )

        row_ids.append(
            hashlib.sha256(
                row_id_source.encode("utf-8")
            ).hexdigest()
        )

    result = dataframe.copy()

    result.insert(
        0,
        INTERNAL_ROW_HASH_COLUMN,
        row_hashes,
    )

    result.insert(
        0,
        INTERNAL_ROW_ID_COLUMN,
        row_ids,
    )

    return result


def load_datasets(
    include_row_metadata: bool = False,
    table_names: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all datasets or only selected changed datasets.

    Passing table_names prevents unchanged CSV files from
    being loaded unnecessarily.
    """

    selected_tables = (
        {
            str(name).strip().lower()
            for name in table_names
        }
        if table_names is not None
        else None
    )

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    discovered_tables: set[str] = set()

    for csv_path in discover_csv_paths():
        table_name = (
            csv_path.stem
            .strip()
            .lower()
        )

        discovered_tables.add(
            table_name
        )

        if (
            selected_tables is not None
            and table_name
            not in selected_tables
        ):
            continue

        dataframe = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        dataframe = clean_columns(
            dataframe
        )

        if include_row_metadata:
            dataframe = add_row_metadata(
                table_name=table_name,
                dataframe=dataframe,
            )

        datasets[table_name] = (
            dataframe
        )

    if selected_tables is not None:
        unknown_tables = (
            selected_tables
            - discovered_tables
        )

        if unknown_tables:
            raise ValueError(
                "Requested datasets were not found: "
                + ", ".join(
                    sorted(unknown_tables)
                )
            )

    return datasets


def file_fingerprints() -> dict[str, str]:
    """
    Calculate one fresh SHA-256 fingerprint
    for every current CSV file.
    """

    fingerprints: dict[str, str] = {}

    for path in discover_csv_paths():
        digest = hashlib.sha256()

        with path.open("rb") as csv_file:
            for block in iter(
                lambda: csv_file.read(
                    1024 * 1024
                ),
                b"",
            ):
                digest.update(block)

        fingerprints[
            path.stem.strip().lower()
        ] = digest.hexdigest()

    return fingerprints


def dataset_fingerprint(
    fingerprints: dict[str, str] | None = None,
) -> str:
    """
    Calculate a combined fresh dataset fingerprint.

    This function intentionally does not use lru_cache.
    """

    current_fingerprints = (
        fingerprints
        if fingerprints is not None
        else file_fingerprints()
    )

    digest = hashlib.sha256()

    for table_name in sorted(
        current_fingerprints
    ):
        digest.update(
            table_name.encode("utf-8")
        )

        digest.update(
            current_fingerprints[
                table_name
            ].encode("utf-8")
        )

    return digest.hexdigest()


def load_dataset_manifest() -> dict[str, Any]:
    """
    Load the fingerprints saved after the last
    successful MySQL and Qdrant synchronization.
    """

    if not DATASET_MANIFEST_PATH.is_file():
        return {
            "version": 1,
            "file_fingerprints": {},
        }

    try:
        payload = json.loads(
            DATASET_MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "The dataset synchronization manifest "
            "could not be loaded."
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "The dataset synchronization manifest "
            "must contain a JSON object."
        )

    fingerprints = payload.get(
        "file_fingerprints",
        {},
    )

    if not isinstance(
        fingerprints,
        dict,
    ):
        raise RuntimeError(
            "The dataset manifest contains "
            "invalid file fingerprints."
        )

    return payload


def build_dataset_change_plan(
    previous_fingerprints: (
        dict[str, str] | None
    ) = None,
    current_fingerprints: (
        dict[str, str] | None
    ) = None,
) -> dict[str, Any]:
    """
    Identify new, changed, deleted and unchanged CSV files.
    """

    previous = (
        previous_fingerprints
        if previous_fingerprints is not None
        else dict(
            load_dataset_manifest().get(
                "file_fingerprints",
                {},
            )
        )
    )

    current = (
        current_fingerprints
        if current_fingerprints is not None
        else file_fingerprints()
    )

    previous_names = set(
        previous
    )

    current_names = set(
        current
    )

    new_tables = sorted(
        current_names
        - previous_names
    )

    deleted_tables = sorted(
        previous_names
        - current_names
    )

    changed_tables = sorted(
        table_name
        for table_name
        in current_names
        & previous_names
        if (
            current[table_name]
            != previous[table_name]
        )
    )

    unchanged_tables = sorted(
        table_name
        for table_name
        in current_names
        & previous_names
        if (
            current[table_name]
            == previous[table_name]
        )
    )

    tables_to_load = sorted(
        set(new_tables)
        | set(changed_tables)
    )

    return {
        "new_tables": new_tables,
        "changed_tables": changed_tables,
        "deleted_tables": deleted_tables,
        "unchanged_tables":
            unchanged_tables,
        "tables_to_load":
            tables_to_load,
        "previous_fingerprints":
            previous,
        "current_fingerprints":
            current,
        "fingerprint":
            dataset_fingerprint(
                current
            ),
        "has_changes": bool(
            new_tables
            or changed_tables
            or deleted_tables
        ),
    }


def save_dataset_manifest(
    file_fingerprints_value: dict[str, str],
    synchronization_result: dict[str, Any],
) -> None:
    """
    Save the manifest only after both MySQL
    and Qdrant synchronization succeed.
    """

    DATASET_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "updated_at": _utc_now(),
        "fingerprint":
            dataset_fingerprint(
                file_fingerprints_value
            ),
        "file_fingerprints":
            file_fingerprints_value,
        "last_result":
            synchronization_result,
    }

    temporary_path = Path(
        str(DATASET_MANIFEST_PATH)
        + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        DATASET_MANIFEST_PATH
    )