from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Iterable

import mysql.connector
import numpy as np
import pandas as pd
from mysql.connector import pooling

from src.config import (
    MYSQL_CONNECT_TIMEOUT,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_INGEST_PASSWORD,
    MYSQL_INGEST_USER,
    MYSQL_INSERT_BATCH_SIZE,
    MYSQL_POOL_SIZE,
    MYSQL_PORT,
    MYSQL_QUERY_PASSWORD,
    MYSQL_QUERY_TIMEOUT_MS,
    MYSQL_QUERY_USER,
    SQL_RESULT_ROW_LIMIT,
    TABLE_NAMES,
)
from src.data_loader import dataset_fingerprint, load_datasets
from src.sql_safety import validate_and_limit_sql


MANIFEST_TABLE = "_dataset_ingestion_manifest"

COMMON_INDEX_COLUMNS = (
    "system_name",
    "country",
    "manufacturer",
    "list_year",
    "list_month",
    "green500_rank",
    "top500_rank",
    "rmax_tflops",
    "energy_efficiency_gflops_watt",
)

TABLE_DESCRIPTIONS = {
    table_name: f"Dataset loaded from {table_name}.csv."
    for table_name in TABLE_NAMES
}


def quote_identifier(identifier: str) -> str:
    if "`" in identifier:
        raise ValueError("Invalid MySQL identifier.")

    return f"`{identifier}`"


def _server_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_INGEST_USER,
        password=MYSQL_INGEST_PASSWORD,
        connection_timeout=MYSQL_CONNECT_TIMEOUT,
        autocommit=False,
    )


def _ingest_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_INGEST_USER,
        password=MYSQL_INGEST_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=MYSQL_CONNECT_TIMEOUT,
        autocommit=False,
        charset="utf8mb4",
        use_unicode=True,
    )


def get_admin_connection():
    """
    Return a write-capable MySQL connection.

    This connection should be used only during bootstrap and ingestion.
    """

    return _ingest_connection()


@lru_cache(maxsize=1)
def get_query_pool() -> pooling.MySQLConnectionPool:
    pool_hash = hashlib.sha1(
        (
            f"{MYSQL_HOST}:"
            f"{MYSQL_PORT}:"
            f"{MYSQL_DATABASE}:"
            f"{MYSQL_QUERY_USER}"
        ).encode()
    ).hexdigest()[:10]

    return pooling.MySQLConnectionPool(
        pool_name=f"scqa_{pool_hash}",
        pool_size=MYSQL_POOL_SIZE,
        pool_reset_session=True,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_QUERY_USER,
        password=MYSQL_QUERY_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=MYSQL_CONNECT_TIMEOUT,
        autocommit=False,
        charset="utf8mb4",
        use_unicode=True,
    )


def ensure_database_exists() -> None:
    connection = _server_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS "
            f"{quote_identifier(MYSQL_DATABASE)} "
            "CHARACTER SET utf8mb4 "
            "COLLATE utf8mb4_0900_ai_ci"
        )

        connection.commit()
        cursor.close()

    finally:
        connection.close()


def _mysql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype):
        return "TINYINT(1)"

    if pd.api.types.is_integer_dtype(series.dtype):
        return "BIGINT"

    if pd.api.types.is_float_dtype(series.dtype):
        return "DOUBLE"

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "DATETIME"

    non_null = series.dropna().astype(str)

    max_length = (
        int(non_null.str.len().max())
        if not non_null.empty
        else 1
    )

    if max_length <= 255:
        return f"VARCHAR({max(1, max_length)})"

    if max_length <= 65_535:
        return "TEXT"

    if max_length <= 16_777_215:
        return "MEDIUMTEXT"

    return "LONGTEXT"


def _python_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return None

    if value is pd.NA:
        return None

    if not isinstance(value, (str, bytes)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if isinstance(
        value,
        (
            datetime,
            date,
            str,
            int,
            float,
            bool,
            bytes,
        ),
    ):
        return value

    return str(value)


def _batched_rows(
    dataframe: pd.DataFrame,
) -> Iterable[list[tuple[Any, ...]]]:
    columns = list(dataframe.columns)
    batch: list[tuple[Any, ...]] = []

    for row in dataframe.itertuples(
        index=False,
        name=None,
    ):
        batch.append(
            tuple(
                _python_value(value)
                for value in row[: len(columns)]
            )
        )

        if len(batch) >= MYSQL_INSERT_BATCH_SIZE:
            yield batch
            batch = []

    if batch:
        yield batch


def _table_exists(
    cursor,
    table_name: str,
) -> bool:
    cursor.execute(
        "SELECT COUNT(*) "
        "FROM information_schema.tables "
        "WHERE table_schema = %s "
        "AND table_name = %s",
        (
            MYSQL_DATABASE,
            table_name,
        ),
    )

    return bool(cursor.fetchone()[0])


def _create_and_fill_staging_table(
    connection,
    table_name: str,
    dataframe: pd.DataFrame,
) -> str:
    if len(dataframe.columns) == 0:
        raise ValueError(
            f"{table_name}.csv does not contain any columns."
        )

    staging_table = f"_stg_{table_name}"
    cursor = connection.cursor()

    cursor.execute(
        f"DROP TABLE IF EXISTS "
        f"{quote_identifier(staging_table)}"
    )

    column_definitions = [
        (
            f"{quote_identifier(column)} "
            f"{_mysql_type(dataframe[column])} NULL"
        )
        for column in dataframe.columns
    ]

    cursor.execute(
        f"CREATE TABLE {quote_identifier(staging_table)} "
        f"({', '.join(column_definitions)}) "
        "ENGINE=InnoDB "
        "DEFAULT CHARSET=utf8mb4"
    )

    if not dataframe.empty:
        column_sql = ", ".join(
            quote_identifier(column)
            for column in dataframe.columns
        )

        placeholders = ", ".join(
            ["%s"] * len(dataframe.columns)
        )

        insert_sql = (
            f"INSERT INTO {quote_identifier(staging_table)} "
            f"({column_sql}) "
            f"VALUES ({placeholders})"
        )

        for batch in _batched_rows(dataframe):
            cursor.executemany(
                insert_sql,
                batch,
            )
            connection.commit()

    for column in COMMON_INDEX_COLUMNS:
        if column not in dataframe.columns:
            continue

        mysql_type = _mysql_type(
            dataframe[column]
        )

        index_name = f"idx_{column}"[:64]
        indexed_column = quote_identifier(column)

        if "TEXT" in mysql_type:
            indexed_column = (
                f"{indexed_column}(191)"
            )

        cursor.execute(
            f"CREATE INDEX {quote_identifier(index_name)} "
            f"ON {quote_identifier(staging_table)} "
            f"({indexed_column})"
        )

    cursor.execute(
        f"SELECT COUNT(*) "
        f"FROM {quote_identifier(staging_table)}"
    )

    imported_count = int(
        cursor.fetchone()[0]
    )

    expected_count = len(dataframe)

    if imported_count != expected_count:
        raise RuntimeError(
            f"Row-count validation failed for {table_name}: "
            f"expected {expected_count}, "
            f"imported {imported_count}."
        )

    cursor.close()

    return staging_table


def _ensure_manifest_table(cursor) -> None:
    cursor.execute(
        f"CREATE TABLE IF NOT EXISTS "
        f"{quote_identifier(MANIFEST_TABLE)} ("
        "manifest_id TINYINT PRIMARY KEY, "
        "fingerprint CHAR(64) NOT NULL, "
        "updated_at TIMESTAMP NOT NULL "
        "DEFAULT CURRENT_TIMESTAMP "
        "ON UPDATE CURRENT_TIMESTAMP"
        ") ENGINE=InnoDB"
    )


def _current_manifest(
    cursor,
) -> str | None:
    if not _table_exists(
        cursor,
        MANIFEST_TABLE,
    ):
        return None

    cursor.execute(
        f"SELECT fingerprint "
        f"FROM {quote_identifier(MANIFEST_TABLE)} "
        "WHERE manifest_id = 1"
    )

    row = cursor.fetchone()

    return row[0] if row else None


def _clear_mysql_caches() -> None:
    _get_schema_context_cached.cache_clear()
    get_table_catalog_context.cache_clear()
    get_query_pool.cache_clear()


def ingest_csvs_to_mysql(
    force: bool = False,
) -> dict[str, Any]:
    datasets = load_datasets()
    fingerprint = dataset_fingerprint()

    ensure_database_exists()

    connection = _ingest_connection()
    staging_tables: list[str] = []

    try:
        cursor = connection.cursor()

        _ensure_manifest_table(cursor)

        all_tables_exist = all(
            _table_exists(cursor, table_name)
            for table_name in TABLE_NAMES
        )

        existing_fingerprint = (
            _current_manifest(cursor)
        )

        if (
            not force
            and all_tables_exist
            and existing_fingerprint == fingerprint
        ):
            cursor.close()

            return {
                "status": "unchanged",
                "fingerprint": fingerprint,
                "tables": {
                    table_name: len(
                        datasets[table_name]
                    )
                    for table_name in TABLE_NAMES
                },
            }

        for table_name in TABLE_NAMES:
            staging_table = (
                _create_and_fill_staging_table(
                    connection=connection,
                    table_name=table_name,
                    dataframe=datasets[table_name],
                )
            )

            staging_tables.append(
                staging_table
            )

        rename_parts: list[str] = []
        backup_tables: list[str] = []

        for table_name, staging_table in zip(
            TABLE_NAMES,
            staging_tables,
        ):
            backup_table = (
                f"_old_{table_name}"
            )

            cursor.execute(
                f"DROP TABLE IF EXISTS "
                f"{quote_identifier(backup_table)}"
            )

            if _table_exists(
                cursor,
                table_name,
            ):
                rename_parts.append(
                    f"{quote_identifier(table_name)} "
                    f"TO {quote_identifier(backup_table)}"
                )

                backup_tables.append(
                    backup_table
                )

            rename_parts.append(
                f"{quote_identifier(staging_table)} "
                f"TO {quote_identifier(table_name)}"
            )

        cursor.execute(
            "RENAME TABLE "
            + ", ".join(rename_parts)
        )

        cursor.execute(
            f"INSERT INTO "
            f"{quote_identifier(MANIFEST_TABLE)} "
            "(manifest_id, fingerprint) "
            "VALUES (1, %s) "
            "ON DUPLICATE KEY UPDATE "
            "fingerprint = VALUES(fingerprint), "
            "updated_at = CURRENT_TIMESTAMP",
            (fingerprint,),
        )

        for backup_table in backup_tables:
            cursor.execute(
                f"DROP TABLE "
                f"{quote_identifier(backup_table)}"
            )

        connection.commit()
        cursor.close()

        _clear_mysql_caches()

        return {
            "status": "imported",
            "fingerprint": fingerprint,
            "tables": {
                table_name: len(
                    datasets[table_name]
                )
                for table_name in TABLE_NAMES
            },
        }

    except Exception:
        connection.rollback()

        cleanup_cursor = connection.cursor()

        for staging_table in staging_tables:
            try:
                cleanup_cursor.execute(
                    f"DROP TABLE IF EXISTS "
                    f"{quote_identifier(staging_table)}"
                )
            except Exception:
                pass

        connection.commit()
        cleanup_cursor.close()

        raise

    finally:
        connection.close()


def verify_mysql_ready() -> None:
    connection = (
        get_query_pool().get_connection()
    )

    try:
        cursor = connection.cursor()

        missing_tables = [
            table_name
            for table_name in TABLE_NAMES
            if not _table_exists(
                cursor,
                table_name,
            )
        ]

        cursor.close()

        if missing_tables:
            raise RuntimeError(
                "MySQL is missing required tables: "
                + ", ".join(missing_tables)
            )

    finally:
        connection.close()


def _json_safe(value: Any) -> Any:
    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(value, Decimal):
        return str(value)

    return value


def execute_select(
    sql: str,
) -> dict[str, Any]:
    safe_sql = validate_and_limit_sql(sql)

    connection = (
        get_query_pool().get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True,
            buffered=True,
        )

        cursor.execute(
            "SET SESSION MAX_EXECUTION_TIME = %s",
            (MYSQL_QUERY_TIMEOUT_MS,),
        )

        cursor.execute(
            "START TRANSACTION READ ONLY"
        )

        cursor.execute(safe_sql)

        rows = cursor.fetchall()

        truncated = (
            len(rows)
            > SQL_RESULT_ROW_LIMIT
        )

        rows = rows[
            :SQL_RESULT_ROW_LIMIT
        ]

        columns = (
            list(rows[0].keys())
            if rows
            else list(
                cursor.column_names or []
            )
        )

        return {
            "success": True,
            "sql": safe_sql,
            "columns": columns,
            "rows": [
                {
                    key: _json_safe(value)
                    for key, value in row.items()
                }
                for row in rows
            ],
            "row_count": len(rows),
            "truncated": truncated,
        }

    finally:
        try:
            connection.rollback()
        finally:
            if cursor is not None:
                cursor.close()

            connection.close()


def _normalise_table_names(
    table_names: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if table_names is None:
        requested_tables: Iterable[str] = (
            TABLE_NAMES
        )

    elif isinstance(table_names, str):
        requested_tables = (
            table_names,
        )

    else:
        requested_tables = table_names

    selected_tables = tuple(
        dict.fromkeys(
            str(table_name).strip()
            for table_name in requested_tables
            if str(table_name).strip()
        )
    )

    if not selected_tables:
        raise ValueError(
            "At least one table is required "
            "for schema context."
        )

    invalid_tables = sorted(
        set(selected_tables)
        - set(TABLE_NAMES)
    )

    if invalid_tables:
        raise ValueError(
            "Schema requested for unknown tables: "
            + ", ".join(invalid_tables)
        )

    return selected_tables


def get_schema_context(
    table_names: Iterable[str] | str | None = None,
) -> str:
    """
    Return live schema context for the requested MySQL tables.

    If table_names is None, schema for every configured dataset is returned.
    Both a single table name and an iterable of table names are supported.
    """

    selected_tables = (
        _normalise_table_names(
            table_names
        )
    )

    return _get_schema_context_cached(
        selected_tables
    )


@lru_cache(maxsize=32)
def _get_schema_context_cached(
    selected_tables: tuple[str, ...],
) -> str:
    verify_mysql_ready()

    connection = (
        get_query_pool().get_connection()
    )

    try:
        cursor = connection.cursor(
            dictionary=True,
            buffered=True,
        )

        sections: list[str] = []

        for table_name in selected_tables:
            cursor.execute(
                "SELECT "
                "column_name, "
                "column_type, "
                "is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = %s "
                "AND table_name = %s "
                "ORDER BY ordinal_position",
                (
                    MYSQL_DATABASE,
                    table_name,
                ),
            )

            columns = [
                {
                    str(key).lower(): value
                    for key, value in row.items()
                }
                for row in cursor.fetchall()
            ]

            if not columns:
                raise RuntimeError(
                    f"No schema was found for "
                    f"MySQL table {table_name}."
                )

            cursor.execute(
                f"SELECT * "
                f"FROM {quote_identifier(table_name)} "
                "LIMIT 1"
            )

            samples = [
                {
                    str(key).lower(): value
                    for key, value in row.items()
                }
                for row in cursor.fetchall()
            ]

            compact_samples = [
                {
                    key: (
                        str(
                            _json_safe(value)
                        )[:160]
                        if value is not None
                        else None
                    )
                    for key, value
                    in sample.items()
                }
                for sample in samples
            ]

            column_text = "\n".join(
                (
                    f"- {item['column_name']}: "
                    f"{item['column_type']} "
                    f"(nullable: "
                    f"{item['is_nullable']})"
                )
                for item in columns
            )

            sample_text = json.dumps(
                compact_samples,
                indent=2,
                ensure_ascii=False,
                default=str,
            )[:2_500]

            sections.append(
                f"TABLE: {table_name}\n"
                f"PURPOSE: "
                f"{TABLE_DESCRIPTIONS[table_name]}\n"
                f"COLUMNS:\n"
                f"{column_text}\n"
                f"SAMPLE ROWS:\n"
                f"{sample_text}"
            )

        cursor.close()

        return "\n\n".join(sections)

    finally:
        connection.close()


@lru_cache(maxsize=1)
def get_table_catalog_context() -> str:
    """
    Return a compact live MySQL catalog.

    The query router uses this catalog to dynamically determine which
    datasets are relevant. No question-to-table mapping is hard-coded here.
    """

    verify_mysql_ready()

    connection = (
        get_query_pool().get_connection()
    )

    try:
        cursor = connection.cursor(
            dictionary=True,
            buffered=True,
        )

        sections: list[str] = []

        for table_name in TABLE_NAMES:
            cursor.execute(
                "SELECT "
                "column_name, "
                "column_type "
                "FROM information_schema.columns "
                "WHERE table_schema = %s "
                "AND table_name = %s "
                "ORDER BY ordinal_position",
                (
                    MYSQL_DATABASE,
                    table_name,
                ),
            )

            columns = [
                {
                    str(key).lower(): value
                    for key, value in row.items()
                }
                for row in cursor.fetchall()
            ]

            cursor.execute(
                f"SELECT COUNT(*) AS row_count "
                f"FROM {quote_identifier(table_name)}"
            )

            raw_count_row = (
                cursor.fetchone()
                or {}
            )

            count_row = {
                str(key).lower(): value
                for key, value
                in raw_count_row.items()
            }

            row_count = int(
                count_row.get(
                    "row_count",
                    0,
                )
            )

            column_text = ", ".join(
                (
                    f"{item['column_name']} "
                    f"({item['column_type']})"
                )
                for item in columns
            )

            sections.append(
                f"TABLE: {table_name}\n"
                f"PURPOSE: "
                f"{TABLE_DESCRIPTIONS[table_name]}\n"
                f"ROW COUNT: {row_count}\n"
                f"COLUMNS: {column_text}"
            )

        cursor.close()

        return "\n\n".join(sections)

    finally:
        connection.close()


def get_table_overview() -> dict[str, int]:
    verify_mysql_ready()

    connection = (
        get_query_pool().get_connection()
    )

    try:
        cursor = connection.cursor()
        table_counts: dict[str, int] = {}

        for table_name in TABLE_NAMES:
            cursor.execute(
                f"SELECT COUNT(*) "
                f"FROM {quote_identifier(table_name)}"
            )

            table_counts[table_name] = int(
                cursor.fetchone()[0]
            )

        cursor.close()

        return table_counts

    finally:
        connection.close()