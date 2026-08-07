from __future__ import annotations

import re

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from src.config import (
    MYSQL_DATABASE,
    SQL_RESULT_ROW_LIMIT,
)

from src.data_loader import (
    get_dataset_table_names,
)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL violates the read-only policy."""


_DISALLOWED_NODE_NAMES = (
    "Alter",
    "Analyze",
    "Attach",
    "Command",
    "Commit",
    "Copy",
    "Create",
    "Delete",
    "Detach",
    "Drop",
    "Grant",
    "Insert",
    "Into",
    "LoadData",
    "Lock",
    "Merge",
    "Pragma",
    "Rollback",
    "Set",
    "Transaction",
    "TruncateTable",
    "Update",
    "Use",
)

_DISALLOWED_TYPES = tuple(
    node_type
    for name in _DISALLOWED_NODE_NAMES
    if (node_type := getattr(exp, name, None)) is not None
)

_BLOCKED_FUNCTIONS = {
    "benchmark",
    "connection_id",
    "current_user",
    "database",
    "get_lock",
    "is_free_lock",
    "is_used_lock",
    "load_file",
    "master_pos_wait",
    "name_const",
    "release_all_locks",
    "release_lock",
    "schema",
    "sleep",
    "system_user",
    "sys_exec",
    "sys_eval",
    "user",
    "version",
}

_BLOCKED_TEXT = re.compile(
    r"\b(into\s+(outfile|dumpfile)|load\s+data|procedure\s+analyse)\b|@{1,2}",
    flags=re.IGNORECASE,
)


def clean_sql(sql: str) -> str:
    value = (sql or "").strip()
    value = re.sub(r"^```(?:sql|mysql)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip().rstrip(";").strip()


def validate_and_limit_sql(
    sql: str,
) -> str:
    cleaned = clean_sql(
        sql
    )

    if not cleaned:
        raise UnsafeSQLError(
            "The generated SQL query "
            "is empty."
        )

    if _BLOCKED_TEXT.search(
        cleaned
    ):
        raise UnsafeSQLError(
            "The query contains a "
            "blocked MySQL operation."
        )

    try:
        statements = parse(
            cleaned,
            read="mysql",
        )

    except ParseError as error:
        raise UnsafeSQLError(
            "The generated SQL is "
            f"invalid: {error}"
        ) from error

    if len(statements) != 1:
        raise UnsafeSQLError(
            "Exactly one SQL statement "
            "is allowed."
        )

    statement = statements[0]

    if not isinstance(
        statement,
        exp.Query,
    ):
        raise UnsafeSQLError(
            "Only SELECT or WITH queries "
            "are allowed."
        )

    for node in statement.walk():
        if (
            _DISALLOWED_TYPES
            and isinstance(
                node,
                _DISALLOWED_TYPES,
            )
        ):
            raise UnsafeSQLError(
                "Blocked SQL operation: "
                f"{type(node).__name__}."
            )

    cte_names = {
        cte.alias_or_name.lower()
        for cte
        in statement.find_all(
            exp.CTE
        )
        if cte.alias_or_name
    }

    live_table_names = {
        name.lower()
        for name
        in get_dataset_table_names()
    }

    allowed_tables = (
        live_table_names
        | cte_names
    )

    referenced_tables = {
        table.name.lower()
        for table
        in statement.find_all(
            exp.Table
        )
        if table.name
    }

    unknown_tables = (
        referenced_tables
        - allowed_tables
    )

    if unknown_tables:
        raise UnsafeSQLError(
            "Query references unknown "
            "tables: "
            + ", ".join(
                sorted(
                    unknown_tables
                )
            )
        )

    if not referenced_tables:
        raise UnsafeSQLError(
            "The query must read from "
            "one of the current "
            "dataset tables."
        )

    for table in statement.find_all(
        exp.Table
    ):
        database_name = (
            table.db
            or ""
        ).lower()

        if (
            database_name
            and database_name
            != MYSQL_DATABASE.lower()
        ):
            raise UnsafeSQLError(
                "Cross-database access "
                "is blocked: "
                f"{database_name}."
            )

    for function in statement.find_all(
        exp.Func
    ):
        detected_names: set[str] = set()

        sql_name = (
            function.sql_name()
            or ""
        ).strip().lower()

        if sql_name:
            detected_names.add(
                sql_name
            )

        function_name = str(
            getattr(
                function,
                "name",
                "",
            )
            or ""
        ).strip().lower()

        if function_name:
            detected_names.add(
                function_name
            )

        blocked_names = (
            detected_names
            & _BLOCKED_FUNCTIONS
        )

        if blocked_names:
            blocked_function = sorted(
                blocked_names
            )[0]

            raise UnsafeSQLError(
                "Blocked SQL function: "
                f"{blocked_function}."
            )

    # Add one extra row so the application
    # can detect result truncation.
    hard_limit = (
        SQL_RESULT_ROW_LIMIT
        + 1
    )

    limit_node = (
        statement.args.get(
            "limit"
        )
    )

    should_set_limit = (
        limit_node is None
    )

    if limit_node is not None:
        limit_expression = (
            limit_node.expression
        )

        if not isinstance(
            limit_expression,
            exp.Literal,
        ):
            should_set_limit = True

        else:
            try:
                requested_limit = int(
                    str(
                        limit_expression.this
                    )
                )

            except ValueError:
                should_set_limit = True

            else:
                should_set_limit = (
                    requested_limit
                    > hard_limit
                )

    if should_set_limit:
        statement = statement.limit(
            hard_limit,
            copy=False,
        )

    return statement.sql(
        dialect="mysql"
    )