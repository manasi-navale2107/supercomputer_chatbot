from __future__ import annotations

import json
import logging
import re
from typing import Any

import mysql.connector
from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from src.config import SQL_RESULT_ROW_LIMIT
from src.llm import get_json_llm
from src.mysql_store import (
    execute_select,
    get_schema_context,
)
from src.prompts import (
    SQL_GENERATOR_PROMPT,
    SQL_REPAIR_PROMPT,
)
from src.relationship_context import (
    get_relationship_context,
)
from src.sql_safety import UnsafeSQLError


logger = logging.getLogger(__name__)


def _message_text(
    message: Any,
) -> str:
    content = getattr(
        message,
        "content",
        message,
    )

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            (
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
            )
            for item in content
        ).strip()

    return str(content).strip()


def _query_plan(
    message: Any,
) -> list[str]:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        _message_text(message),
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    text = re.sub(
        r"```(?:json|sql)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError(
            "SQL model did not return "
            "a JSON object."
        )

    payload = json.loads(
        text[start : end + 1]
    )

    raw_queries = (
        payload.get("queries")
        if isinstance(payload, dict)
        else None
    )

    if (
        raw_queries is None
        and isinstance(payload, dict)
        and payload.get("sql")
    ):
        raw_queries = [
            {"sql": payload["sql"]}
        ]

    if (
        not isinstance(raw_queries, list)
        or not raw_queries
    ):
        raise ValueError(
            "SQL model must return a "
            "non-empty queries list."
        )

    if len(raw_queries) > 32:
        raise ValueError(
            "SQL query plan is too large."
        )

    queries: list[str] = []

    for item in raw_queries:
        if isinstance(item, str):
            sql = item.strip()

        elif isinstance(item, dict):
            sql = str(
                item.get(
                    "sql",
                    "",
                )
            ).strip()

        else:
            sql = ""

        if not sql:
            raise ValueError(
                "Every query-plan item "
                "must contain SQL."
            )

        if sql not in queries:
            queries.append(sql)

    return queries


def _referenced_tables(
    sql: str,
) -> set[str]:
    statement = parse_one(
        sql,
        read="mysql",
    )

    cte_names = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(
            exp.CTE
        )
        if cte.alias_or_name
    }

    return {
        table.name.lower()
        for table in statement.find_all(
            exp.Table
        )
        if (
            table.name
            and table.name.lower()
            not in cte_names
        )
    }

def _validate_plan(
    queries: list[str],
    selected_tables: list[str],
) -> list[set[str]]:
    """
    Validate that generated queries use only
    the tables allowed by the table selector.

    selected_tables are candidate tables.
    Every selected table is not required to
    appear in the final SQL plan.
    """

    allowed_tables = {
        table_name.strip().lower()
        for table_name in selected_tables
        if table_name.strip()
    }

    if not allowed_tables:
        raise ValueError(
            "At least one selected table "
            "is required."
        )

    if not queries:
        raise ValueError(
            "At least one SQL query "
            "is required."
        )

    references: list[set[str]] = []

    for sql in queries:
        query_tables = _referenced_tables(
            sql
        )

        if not query_tables:
            raise UnsafeSQLError(
                "Every query must read from "
                "a selected dataset table."
            )

        unknown_tables = (
            query_tables
            - allowed_tables
        )

        if unknown_tables:
            raise UnsafeSQLError(
                "SQL used tables not selected "
                "by the table selector: "
                + ", ".join(
                    sorted(
                        unknown_tables
                    )
                )
            )

        references.append(
            query_tables
        )

    return references


def _balanced_rows(
    query_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets = [
        list(result["rows"])
        for result in query_results
    ]

    positions = [0] * len(buckets)
    output: list[dict[str, Any]] = []

    while (
        len(output)
        < SQL_RESULT_ROW_LIMIT
    ):
        added = False

        for index, bucket in enumerate(
            buckets
        ):
            position = positions[index]

            if position >= len(bucket):
                continue

            output.append(
                bucket[position]
            )

            positions[index] += 1
            added = True

            if (
                len(output)
                >= SQL_RESULT_ROW_LIMIT
            ):
                break

        if not added:
            break

    return output


def _display_sql(
    queries: list[str],
) -> str:
    return "\n\n".join(
        f"-- Query {index}\n{sql}"
        for index, sql in enumerate(
            queries,
            start=1,
        )
    )


def _execute_plan(
    queries: list[str],
    selected_tables: list[str],
) -> dict[str, Any]:
    references = _validate_plan(
        queries,
        selected_tables,
    )

    query_results: list[
        dict[str, Any]
    ] = []

    for sql, query_tables in zip(
        queries,
        references,
    ):
        result = execute_select(
            sql
        )

        source_tables = sorted(
            query_tables
        )

        labelled_rows = []

        for row in result.get(
            "rows",
            [],
        ):
            labelled_row = dict(row)

            if len(source_tables) == 1:
                labelled_row.setdefault(
                    "source_dataset",
                    source_tables[0],
                )

            else:
                labelled_row.setdefault(
                    "source_datasets",
                    source_tables,
                )

            labelled_rows.append(
                labelled_row
            )

        query_results.append(
            {
                "source_tables": source_tables,
                "sql": result.get(
                    "sql",
                    sql,
                ),
                "columns": result.get(
                    "columns",
                    [],
                ),
                "rows": labelled_rows,
                "row_count": result.get(
                    "row_count",
                    len(labelled_rows),
                ),
                "truncated": bool(
                    result.get(
                        "truncated",
                        False,
                    )
                ),
            }
        )

    rows = _balanced_rows(
        query_results
    )

    total_rows = sum(
        int(result["row_count"])
        for result in query_results
    )

    columns = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row
        )
    )

    executed_queries = [
        str(result["sql"])
        for result in query_results
    ]

    used_tables = sorted(
        {
            table_name
            for reference in references
            for table_name in reference
        }
    )

    citations = {
    "route": "structured",
    "datasets": used_tables,
    "documents": [],
    "sql": _display_sql(executed_queries),
    "row_count": total_rows,
    "retrieval_query": None,
}

    return {
        "route": "structured",
        "success": True,
        "sql": _display_sql(
            executed_queries
        ),
        "queries": executed_queries,
        "query_results": query_results,
        "query_count": len(
            query_results
        ),
        "selected_tables": selected_tables,
        "used_tables": used_tables,
        "columns": columns,
        "rows": rows,
        "row_count": total_rows,
        "truncated": (
            any(
                result["truncated"]
                for result in query_results
            )
            or total_rows > len(rows)
        ),

        "citations": citations,
    }


def _invoke_sql(
    prompt: str,
    llm_provider: str | None = None,
) -> list[str]:
    response = (
        get_json_llm(
            llm_provider
        ).invoke(
            prompt
        )
    )

    return _query_plan(
        response
    )


def _repairable(
    error: Exception,
) -> bool:
    if isinstance(
        error,
        (
            ValueError,
            UnsafeSQLError,
            ParseError,
        ),
    ):
        return True

    return (
        isinstance(
            error,
            mysql.connector.Error,
        )
        and error.errno in {
            1052,
            1054,
            1055,
            1064,
            1066,
            1111,
            1140,
            1146,
            1241,
            1242,
            1582,
        }
    )


def _failure(
    selected_tables: list[str],
    queries: list[str],
    initial_error: Exception,
    repair_error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "route": "structured",
        "success": False,
        "sql": (
            _display_sql(queries)
            if queries
            else None
        ),
        "queries": queries,
        "query_results": [],
        "selected_tables": selected_tables,
        "used_tables": [],
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "repaired": (
            repair_error is not None
        ),
        "error": (
            "The structured query plan "
            "could not be completed safely."
        ),
        "generation_or_execution_error": str(
            initial_error
        ),
        "repair_error": (
            str(repair_error)
            if repair_error
            else None
        ),
    }


def run_structured_route(
    state: dict[str, Any],
) -> dict[str, Any]:
    decision = state.get(
        "decision",
        {},
    )

    llm_provider = state.get(
        "llm_provider"
    )

    selected_tables = list(
        decision.get(
            "selected_tables",
            [],
        )
    )

    if not selected_tables:
        raise ValueError(
            "The table selector returned "
            "no relevant dataset."
        )

    question = str(
        decision.get("standalone_question")
        or state["question"]
    ).strip()

    chat_history = str(
        state.get(
            "chat_history",
            "",
        )
    )

    schema = get_schema_context(
        tuple(selected_tables)
    )

    relationships = (
        get_relationship_context(
            selected_tables
        )
    )

    calls = int(
        state.get(
            "llm_calls",
            0,
        )
    )

    queries: list[str] = []

    try:
        calls += 1

        queries = _invoke_sql(
            SQL_GENERATOR_PROMPT.format(
                question=question,
                chat_history=(
                    chat_history
                    or "No previous conversation."
                ),
                selected_tables=", ".join(
                    selected_tables
                ),
                schema=schema,
                relationships=relationships,
            ),
            llm_provider=llm_provider,
        )

        evidence = _execute_plan(
            queries,
            selected_tables,
        )

        evidence["repaired"] = False

        return {
            "schema": schema,
            "relationships": relationships,
            "generated_sql": evidence["sql"],
            "evidence": evidence,
            "llm_calls": calls,
            "error": None,
        }

    except Exception as initial_error:
        logger.warning(
            "Initial SQL plan failed: %s",
            initial_error,
        )

        if not _repairable(
            initial_error
        ):
            evidence = _failure(
                selected_tables,
                queries,
                initial_error,
            )

            return {
                "schema": schema,
                "relationships": relationships,
                "generated_sql": (
                    evidence.get("sql")
                    or ""
                ),
                "evidence": evidence,
                "llm_calls": calls,
                "error": str(
                    initial_error
                ),
            }

        try:
            calls += 1

            repaired_queries = _invoke_sql(
                SQL_REPAIR_PROMPT.format(
                    question=question,
                    chat_history=(
                        chat_history
                        or "No previous conversation."
                    ),
                    selected_tables=", ".join(
                        selected_tables
                    ),
                    schema=schema,
                    relationships=relationships,
                    sql=json.dumps(
                        {
                            "queries": [
                                {
                                    "sql": sql
                                }
                                for sql
                                in queries
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    error=str(
                        initial_error
                    ),
                ),
                llm_provider=llm_provider,
            )
            evidence = _execute_plan(
                repaired_queries,
                selected_tables,
            )

            evidence.update(
                {
                    "repaired": True,
                    "original_error": str(
                        initial_error
                    ),
                }
            )

            return {
                "schema": schema,
                "relationships": relationships,
                "generated_sql": evidence["sql"],
                "evidence": evidence,
                "llm_calls": calls,
                "error": None,
            }

        except Exception as repair_error:
            logger.warning(
                "Repaired SQL plan failed: %s",
                repair_error,
            )

            evidence = _failure(
                selected_tables,
                queries,
                initial_error,
                repair_error,
            )

            return {
                "schema": schema,
                "relationships": relationships,
                "generated_sql": (
                    evidence.get("sql")
                    or ""
                ),
                "evidence": evidence,
                "llm_calls": calls,
                "error": str(
                    repair_error
                ),
            }