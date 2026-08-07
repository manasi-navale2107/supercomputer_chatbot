from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

load_dotenv(
    PROJECT_ROOT / ".env"
)

DATA_DIR = (
    PROJECT_ROOT / "data"
)

RUNTIME_DIR = (
    PROJECT_ROOT / "runtime"
)

RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Environment helpers
# =========================================================

def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return (
        value.strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def env_int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    raw = os.getenv(
        name,
        str(default),
    )

    try:
        value = int(raw)

    except ValueError as error:
        raise ValueError(
            f"{name} must be an integer."
        ) from error

    if value < minimum:
        raise ValueError(
            f"{name} must be at least "
            f"{minimum}."
        )

    return value


def validate_identifier(
    value: str,
    setting_name: str,
) -> str:
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        value,
    ):
        raise ValueError(
            f"{setting_name} must contain only "
            "letters, numbers, and underscores "
            "and cannot start with a number."
        )

    return value


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


# =========================================================
# Dynamic CSV dataset discovery
# =========================================================

CSV_DISCOVERY_PATTERN = os.getenv(
    "CSV_DISCOVERY_PATTERN",
    "*.csv",
).strip()

if not CSV_DISCOVERY_PATTERN:
    raise ValueError(
        "CSV_DISCOVERY_PATTERN cannot be empty."
    )

if (
    "/" in CSV_DISCOVERY_PATTERN
    or "\\" in CSV_DISCOVERY_PATTERN
):
    raise ValueError(
        "CSV_DISCOVERY_PATTERN must be "
        "a filename pattern, not a path."
    )


def discover_csv_paths() -> tuple[Path, ...]:
    """
    Discover every CSV file currently available
    inside the data directory.

    This function scans the folder every time it is called,
    therefore new CSV files can be detected at runtime.
    """

    csv_paths = tuple(
        sorted(
            (
                path
                for path in DATA_DIR.glob(
                    CSV_DISCOVERY_PATTERN
                )
                if path.is_file()
            ),
            key=lambda path: (
                path.name.lower()
            ),
        )
    )

    if not csv_paths:
        raise FileNotFoundError(
            "No CSV files were found in "
            f"{DATA_DIR}."
        )

    table_names: list[str] = []

    for path in csv_paths:
        table_name = (
            path.stem.strip().lower()
        )

        validate_identifier(
            table_name,
            (
                "CSV table name for "
                f"{path.name}"
            ),
        )

        if table_name in table_names:
            raise ValueError(
                "Multiple CSV files resolve to "
                f"the same table name: {table_name}"
            )

        table_names.append(
            table_name
        )

    return csv_paths


def discover_csv_files() -> tuple[str, ...]:
    """
    Return the current CSV filenames.

    Example:
        country_stats.csv
        processor_stats.csv
    """

    return tuple(
        path.name
        for path in discover_csv_paths()
    )


def discover_table_names() -> tuple[str, ...]:
    """
    Return the current runtime table names.

    Example:
        country_stats
        processor_stats
    """

    return tuple(
        path.stem.strip().lower()
        for path in discover_csv_paths()
    )


# Compatibility snapshot for existing modules.
#
# New live-update modules will call discover_csv_files()
# or discover_table_names() during every update cycle.
CSV_FILES = discover_csv_files()
TABLE_NAMES = discover_table_names()


# =========================================================
# Synchronization manifest paths
# =========================================================

DATASET_MANIFEST_PATH = (
    resolve_project_path(
        os.getenv(
            "DATASET_MANIFEST_PATH",
            str(
                RUNTIME_DIR
                / "dataset_sync_manifest.json"
            ),
        )
    )
)

LIVE_UPDATE_MANIFEST_PATH = (
    resolve_project_path(
        os.getenv(
            "LIVE_UPDATE_MANIFEST_PATH",
            str(
                RUNTIME_DIR
                / "live_update_manifest.json"
            ),
        )
    )
)


# =========================================================
# Ollama
# =========================================================

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)

OLLAMA_VISION_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL",
    "gemma3:4b",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE",
    "30m",
)

OLLAMA_CONTEXT_WINDOW = env_int(
    "OLLAMA_CONTEXT_WINDOW",
    8192,
)


# =========================================================
# LLM provider selection
# =========================================================

LLM_PROVIDERS = (
    "ollama",
    "groq",
)

DEFAULT_LLM_PROVIDER = os.getenv(
    "DEFAULT_LLM_PROVIDER",
    "ollama",
).strip().lower()

if (
    DEFAULT_LLM_PROVIDER
    not in LLM_PROVIDERS
):
    raise ValueError(
        "DEFAULT_LLM_PROVIDER must be "
        "'ollama' or 'groq'."
    )


# =========================================================
# Groq
# =========================================================

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
).strip()

GROQ_TIMEOUT_SECONDS = env_int(
    "GROQ_TIMEOUT_SECONDS",
    60,
)

GROQ_MAX_RETRIES = env_int(
    "GROQ_MAX_RETRIES",
    2,
    minimum=0,
)


# =========================================================
# MySQL
# =========================================================

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "127.0.0.1",
)

MYSQL_PORT = env_int(
    "MYSQL_PORT",
    3306,
)

MYSQL_DATABASE = validate_identifier(
    os.getenv(
        "MYSQL_DATABASE",
        "supercomputer_analytics",
    ),
    "MYSQL_DATABASE",
)

MYSQL_INGEST_USER = os.getenv(
    "MYSQL_INGEST_USER",
    "root",
)

MYSQL_INGEST_PASSWORD = os.getenv(
    "MYSQL_INGEST_PASSWORD",
    "",
)

MYSQL_QUERY_USER = os.getenv(
    "MYSQL_QUERY_USER",
    MYSQL_INGEST_USER,
)

MYSQL_QUERY_PASSWORD = os.getenv(
    "MYSQL_QUERY_PASSWORD",
    MYSQL_INGEST_PASSWORD,
)

MYSQL_CHAT_USER = os.getenv(
    "MYSQL_CHAT_USER",
    MYSQL_INGEST_USER,
)

MYSQL_CHAT_PASSWORD = os.getenv(
    "MYSQL_CHAT_PASSWORD",
    MYSQL_INGEST_PASSWORD,
)

MYSQL_POOL_SIZE = env_int(
    "MYSQL_POOL_SIZE",
    5,
)

MYSQL_CONNECT_TIMEOUT = env_int(
    "MYSQL_CONNECT_TIMEOUT",
    10,
)

MYSQL_QUERY_TIMEOUT_MS = env_int(
    "MYSQL_QUERY_TIMEOUT_MS",
    10_000,
)

MYSQL_INSERT_BATCH_SIZE = env_int(
    "MYSQL_INSERT_BATCH_SIZE",
    500,
)

SQL_RESULT_ROW_LIMIT = env_int(
    "SQL_RESULT_ROW_LIMIT",
    200,
)


# =========================================================
# Qdrant
# =========================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "",
).strip()

QDRANT_API_KEY = (
    os.getenv(
        "QDRANT_API_KEY",
        "",
    ).strip()
    or None
)

_qdrant_local_path = Path(
    os.getenv(
        "QDRANT_LOCAL_PATH",
        str(
            RUNTIME_DIR / "qdrant"
        ),
    )
)

QDRANT_LOCAL_PATH = (
    _qdrant_local_path
    if _qdrant_local_path.is_absolute()
    else PROJECT_ROOT
    / _qdrant_local_path
)

QDRANT_COLLECTION = validate_identifier(
    os.getenv(
        "QDRANT_COLLECTION",
        "supercomputer_knowledge",
    ),
    "QDRANT_COLLECTION",
)

QDRANT_PREFER_GRPC = env_bool(
    "QDRANT_PREFER_GRPC",
    False,
)

QDRANT_TIMEOUT_SECONDS = env_int(
    "QDRANT_TIMEOUT_SECONDS",
    30,
)

QDRANT_TOP_K = env_int(
    "QDRANT_TOP_K",
    8,
)

QDRANT_UPLOAD_BATCH_SIZE = env_int(
    "QDRANT_UPLOAD_BATCH_SIZE",
    64,
)


# =========================================================
# Embeddings
# =========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    ),
)

EMBEDDING_BATCH_SIZE = env_int(
    "EMBEDDING_BATCH_SIZE",
    64,
)


# =========================================================
# Application
# =========================================================

APP_DEBUG = env_bool(
    "APP_DEBUG",
    False,
)

MAX_CHAT_HISTORY_CHARS = env_int(
    "MAX_CHAT_HISTORY_CHARS",
    6_000,
)

CHAT_CONTEXT_MESSAGE_LIMIT = env_int(
    "CHAT_CONTEXT_MESSAGE_LIMIT",
    10,
)

CHAT_TITLE_MAX_LENGTH = env_int(
    "CHAT_TITLE_MAX_LENGTH",
    80,
)


# =========================================================
# Live ETL and vector synchronization
# =========================================================

LIVE_UPDATE_ENABLED = env_bool(
    "LIVE_UPDATE_ENABLED",
    True,
)

LIVE_UPDATE_POLL_SECONDS = env_int(
    "LIVE_UPDATE_POLL_SECONDS",
    10,
)

LIVE_UPDATE_DEBOUNCE_SECONDS = env_int(
    "LIVE_UPDATE_DEBOUNCE_SECONDS",
    2,
)