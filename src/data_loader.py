from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import CSV_FILES, DATA_DIR


def normalize_identifier(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "unnamed_column"
    if value[0].isdigit():
        value = f"col_{value}"
    if len(value) > 64:
        suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        value = f"{value[:55]}_{suffix}"
    return value


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [normalize_identifier(str(column)) for column in df.columns]
    duplicates = cleaned.columns[cleaned.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(
            "Duplicate columns remain after normalization: "
            + ", ".join(sorted(set(duplicates)))
        )
    return cleaned


def validate_csv_files() -> list[Path]:
    paths = [DATA_DIR / filename for filename in CSV_FILES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required CSV files in the data folder: "
            + ", ".join(missing)
        )
    return paths


def load_datasets() -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for csv_path in validate_csv_files():
        dataframe = pd.read_csv(csv_path, low_memory=False)
        datasets[csv_path.stem] = clean_columns(dataframe)
    return datasets


@lru_cache(maxsize=1)
def dataset_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in validate_csv_files():
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as csv_file:
            for block in iter(lambda: csv_file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
