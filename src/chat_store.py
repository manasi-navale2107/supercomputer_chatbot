from __future__ import annotations

import hashlib
import json
import re
from datetime import (
    date,
    datetime,
)
from functools import lru_cache
from typing import Any

from mysql.connector import pooling

from src.config import (
    CHAT_CONTEXT_MESSAGE_LIMIT,
    CHAT_TITLE_MAX_LENGTH,
    MAX_CHAT_HISTORY_CHARS,
    MYSQL_CHAT_PASSWORD,
    MYSQL_CHAT_USER,
    MYSQL_CONNECT_TIMEOUT,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_POOL_SIZE,
    MYSQL_PORT,
)
from src.mysql_store import (
    get_admin_connection,
    quote_identifier,
)


CONVERSATIONS_TABLE = (
    "chat_conversations"
)

MESSAGES_TABLE = (
    "chat_messages"
)


class ConversationNotFoundError(
    LookupError
):
    pass


class MessageNotFoundError(
    LookupError
):
    pass


class MessageEditError(
    ValueError
):
    pass


@lru_cache(maxsize=1)
def get_chat_pool(
) -> pooling.MySQLConnectionPool:
    pool_hash = hashlib.sha1(
        (
            f"{MYSQL_HOST}:"
            f"{MYSQL_PORT}:"
            f"{MYSQL_DATABASE}:"
            f"{MYSQL_CHAT_USER}"
        ).encode()
    ).hexdigest()[:10]

    return pooling.MySQLConnectionPool(
        pool_name=(
            f"scchat_{pool_hash}"
        ),
        pool_size=MYSQL_POOL_SIZE,
        pool_reset_session=True,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_CHAT_USER,
        password=MYSQL_CHAT_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=(
            MYSQL_CONNECT_TIMEOUT
        ),
        autocommit=False,
        charset="utf8mb4",
        use_unicode=True,
    )


def init_chat_tables() -> None:
    connection = (
        get_admin_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {quote_identifier(CONVERSATIONS_TABLE)}
            (
                id BIGINT UNSIGNED NOT NULL
                    AUTO_INCREMENT,

                client_id CHAR(36) NOT NULL,

                title VARCHAR(200) NOT NULL,

                created_at TIMESTAMP(6) NOT NULL
                    DEFAULT CURRENT_TIMESTAMP(6),

                updated_at TIMESTAMP(6) NOT NULL
                    DEFAULT CURRENT_TIMESTAMP(6)
                    ON UPDATE CURRENT_TIMESTAMP(6),

                PRIMARY KEY (id),

                INDEX
                    idx_chat_conversations_client_updated
                    (client_id, updated_at DESC)
            )
            ENGINE=InnoDB
            DEFAULT CHARSET=utf8mb4
            COLLATE=utf8mb4_0900_ai_ci
            """
        )

        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {quote_identifier(MESSAGES_TABLE)}
            (
                id BIGINT UNSIGNED NOT NULL
                    AUTO_INCREMENT,

                conversation_id BIGINT UNSIGNED
                    NOT NULL,

                role ENUM(
                    'user',
                    'assistant'
                ) NOT NULL,

                content LONGTEXT NOT NULL,

                route VARCHAR(32) NULL,

                sql_query LONGTEXT NULL,

                metadata_json JSON NULL,

                created_at TIMESTAMP(6) NOT NULL
                    DEFAULT CURRENT_TIMESTAMP(6),

                PRIMARY KEY (id),

                INDEX
                    idx_chat_messages_conversation_created
                    (
                        conversation_id,
                        created_at,
                        id
                    ),

                CONSTRAINT
                    fk_chat_messages_conversation
                    FOREIGN KEY (conversation_id)
                    REFERENCES
                    {quote_identifier(CONVERSATIONS_TABLE)}
                    (id)
                    ON DELETE CASCADE
            )
            ENGINE=InnoDB
            DEFAULT CHARSET=utf8mb4
            COLLATE=utf8mb4_0900_ai_ci
            """
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def verify_chat_tables() -> None:
    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            (
                "SELECT table_name "
                "FROM information_schema.tables "
                "WHERE table_schema = %s "
                "AND table_name IN (%s, %s)"
            ),
            (
                MYSQL_DATABASE,
                CONVERSATIONS_TABLE,
                MESSAGES_TABLE,
            ),
        )

        found = {
            row[0]
            for row in cursor.fetchall()
        }

        required = {
            CONVERSATIONS_TABLE,
            MESSAGES_TABLE,
        }

        if found != required:
            raise RuntimeError(
                "Chat tables are not initialized. "
                "Run: python -m src.bootstrap"
            )

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def _clean_title(
    first_message: str,
) -> str:
    title = re.sub(
        r"\s+",
        " ",
        first_message,
    ).strip()

    if (
        len(title)
        > CHAT_TITLE_MAX_LENGTH
    ):
        title = (
            title[
                :CHAT_TITLE_MAX_LENGTH - 1
            ].rstrip()
            + "…"
        )

    return (
        title
        or "New conversation"
    )


def _clean_message_content(
    content: str,
) -> str:
    cleaned = re.sub(
        r"\s+",
        " ",
        str(content),
    ).strip()

    if not cleaned:
        raise MessageEditError(
            "Message content cannot "
            "be empty."
        )

    if len(cleaned) > 4_000:
        raise MessageEditError(
            "Message content cannot be "
            "longer than 4000 characters."
        )

    return cleaned


def _serialize_datetime(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    return value


def _decode_metadata(
    value: Any,
) -> dict[str, Any]:
    if not value:
        return {}

    if isinstance(
        value,
        dict,
    ):
        return value

    try:
        parsed = json.loads(
            value
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except (
        TypeError,
        json.JSONDecodeError,
    ):
        pass

    return {}


def _message_dict(
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": int(
            row["id"]
        ),

        "conversation_id": int(
            row["conversation_id"]
        ),

        "role": row["role"],

        "content": row["content"],

        "route": row.get(
            "route"
        ),

        "sql": row.get(
            "sql_query"
        ),

        "metadata": (
            _decode_metadata(
                row.get(
                    "metadata_json"
                )
            )
        ),

        "created_at": (
            _serialize_datetime(
                row.get(
                    "created_at"
                )
            )
        ),
    }


def create_conversation(
    client_id: str,
    first_message: str,
) -> tuple[int, str]:
    title = _clean_title(
        first_message
    )

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            (
                f"INSERT INTO "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "(client_id, title) "
                "VALUES (%s, %s)"
            ),
            (
                client_id,
                title,
            ),
        )

        conversation_id = int(
            cursor.lastrowid
        )

        connection.commit()

        return (
            conversation_id,
            title,
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def require_conversation(
    conversation_id: int,
    client_id: str,
) -> None:
    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            (
                f"SELECT id FROM "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "WHERE id = %s "
                "AND client_id = %s"
            ),
            (
                conversation_id,
                client_id,
            ),
        )

        if not cursor.fetchone():
            raise (
                ConversationNotFoundError(
                    "Conversation not found."
                )
            )

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    route: str | None = None,
    sql: str | None = None,
    metadata: (
        dict[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    if role not in {
        "user",
        "assistant",
    }:
        raise ValueError(
            "Message role must be "
            "user or assistant."
        )

    cleaned_content = str(
        content
    ).strip()

    if not cleaned_content:
        raise ValueError(
            "Message content cannot "
            "be empty."
        )

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"INSERT INTO "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "("
                "conversation_id, "
                "role, "
                "content, "
                "route, "
                "sql_query, "
                "metadata_json"
                ") "
                "VALUES "
                "(%s, %s, %s, %s, %s, %s)"
            ),
            (
                conversation_id,
                role,
                cleaned_content,
                route,
                sql,
                (
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                        default=str,
                    )
                    if metadata
                    else None
                ),
            ),
        )

        message_id = int(
            cursor.lastrowid
        )

        cursor.execute(
            (
                f"UPDATE "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "SET updated_at = "
                "CURRENT_TIMESTAMP(6) "
                "WHERE id = %s"
            ),
            (
                conversation_id,
            ),
        )

        cursor.execute(
            (
                f"SELECT * FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE id = %s"
            ),
            (
                message_id,
            ),
        )

        row = cursor.fetchone()

        if not row:
            raise RuntimeError(
                "The saved chat message "
                "could not be loaded."
            )

        connection.commit()

        return _message_dict(
            row
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def get_recent_context(
    conversation_id: int,
    message_limit: int = (
        CHAT_CONTEXT_MESSAGE_LIMIT
    ),
    character_limit: int = (
        MAX_CHAT_HISTORY_CHARS
    ),
) -> str:
    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"SELECT role, content FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s "
                "ORDER BY created_at DESC, "
                "id DESC "
                "LIMIT %s"
            ),
            (
                conversation_id,
                message_limit,
            ),
        )

        rows = list(
            reversed(
                cursor.fetchall()
            )
        )

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()

    lines = [
        (
            "User"
            if row["role"] == "user"
            else "Assistant"
        )
        + ": "
        + str(
            row["content"]
        )
        for row in rows
    ]

    context = "\n".join(
        lines
    )

    return context[
        -character_limit:
    ]


def get_context_before_message(
    message_id: int,
    client_id: str,
    message_limit: int = (
        CHAT_CONTEXT_MESSAGE_LIMIT
    ),
    character_limit: int = (
        MAX_CHAT_HISTORY_CHARS
    ),
) -> str:
    """
    Return only the messages that appeared
    before the edited user message.

    The edited message itself is excluded so
    it is not duplicated inside the LLM prompt.
    """

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"SELECT "
                "m.conversation_id "
                f"FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "AS m "
                f"INNER JOIN "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "AS c "
                "ON c.id = m.conversation_id "
                "WHERE m.id = %s "
                "AND c.client_id = %s"
            ),
            (
                message_id,
                client_id,
            ),
        )

        target = cursor.fetchone()

        if not target:
            raise MessageNotFoundError(
                "Message not found."
            )

        conversation_id = int(
            target["conversation_id"]
        )

        cursor.execute(
            (
                f"SELECT role, content FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s "
                "AND id < %s "
                "ORDER BY created_at DESC, "
                "id DESC "
                "LIMIT %s"
            ),
            (
                conversation_id,
                message_id,
                message_limit,
            ),
        )

        rows = list(
            reversed(
                cursor.fetchall()
            )
        )

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()

    lines = [
        (
            "User"
            if row["role"] == "user"
            else "Assistant"
        )
        + ": "
        + str(
            row["content"]
        )
        for row in rows
    ]

    context = "\n".join(
        lines
    )

    return context[
        -character_limit:
    ]


def edit_user_message(
    message_id: int,
    client_id: str,
    new_content: str,
) -> dict[str, Any]:
    """
    Edit one user message transactionally.

    All messages after the edited user message
    are deleted because their answers were based
    on the previous version of the query.
    """

    cleaned_content = (
        _clean_message_content(
            new_content
        )
    )

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"SELECT "
                "m.*, "
                "c.client_id, "
                "c.title "
                f"FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "AS m "
                f"INNER JOIN "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "AS c "
                "ON c.id = m.conversation_id "
                "WHERE m.id = %s "
                "AND c.client_id = %s "
                "FOR UPDATE"
            ),
            (
                message_id,
                client_id,
            ),
        )

        target = cursor.fetchone()

        if not target:
            raise MessageNotFoundError(
                "Message not found."
            )

        if (
            target["role"]
            != "user"
        ):
            raise MessageEditError(
                "Only user messages "
                "can be edited."
            )

        original_metadata = (
            _decode_metadata(
                target.get(
                    "metadata_json"
                )
            )
        )

        if original_metadata.get(
            "has_image",
            False,
        ):
            raise MessageEditError(
                "Image messages cannot be "
                "edited. Please create a new "
                "message and upload the image "
                "again."
            )

        conversation_id = int(
            target[
                "conversation_id"
            ]
        )

        cursor.execute(
            (
                f"DELETE FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s "
                "AND id > %s"
            ),
            (
                conversation_id,
                message_id,
            ),
        )

        deleted_following_messages = (
            int(cursor.rowcount)
        )

        cursor.execute(
            (
                f"UPDATE "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "SET content = %s, "
                "route = NULL, "
                "sql_query = NULL, "
                "metadata_json = NULL "
                "WHERE id = %s "
                "AND conversation_id = %s"
            ),
            (
                cleaned_content,
                message_id,
                conversation_id,
            ),
        )

        cursor.execute(
            (
                f"SELECT id FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s "
                "AND role = 'user' "
                "ORDER BY created_at ASC, "
                "id ASC "
                "LIMIT 1"
            ),
            (
                conversation_id,
            ),
        )

        first_user_row = (
            cursor.fetchone()
        )

        first_user_message_id = (
            int(first_user_row["id"])
            if first_user_row
            else None
        )

        if (
            first_user_message_id
            == message_id
        ):
            updated_title = (
                _clean_title(
                    cleaned_content
                )
            )

            cursor.execute(
                (
                    f"UPDATE "
                    f"{quote_identifier(CONVERSATIONS_TABLE)} "
                    "SET title = %s, "
                    "updated_at = "
                    "CURRENT_TIMESTAMP(6) "
                    "WHERE id = %s"
                ),
                (
                    updated_title,
                    conversation_id,
                ),
            )

        else:
            cursor.execute(
                (
                    f"UPDATE "
                    f"{quote_identifier(CONVERSATIONS_TABLE)} "
                    "SET updated_at = "
                    "CURRENT_TIMESTAMP(6) "
                    "WHERE id = %s"
                ),
                (
                    conversation_id,
                ),
            )

        cursor.execute(
            (
                f"SELECT title FROM "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "WHERE id = %s"
            ),
            (
                conversation_id,
            ),
        )

        conversation_row = (
            cursor.fetchone()
        )

        updated_title = (
            str(
                conversation_row[
                    "title"
                ]
            )
            if conversation_row
            else None
        )

        cursor.execute(
            (
                f"SELECT * FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE id = %s"
            ),
            (
                message_id,
            ),
        )

        updated_message_row = (
            cursor.fetchone()
        )

        if not updated_message_row:
            raise RuntimeError(
                "The edited message could "
                "not be loaded."
            )

        connection.commit()

        return {
            "conversation_id":
                conversation_id,

            "title":
                updated_title,

            "message":
                _message_dict(
                    updated_message_row
                ),

            "deleted_following_messages":
                deleted_following_messages,
        }

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()


def get_conversations(
    client_id: str,
    page: int = 1,
    limit: int = 20,
) -> dict[str, Any]:
    offset = (
        page - 1
    ) * limit

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"SELECT COUNT(*) AS total "
                f"FROM "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "WHERE client_id = %s"
            ),
            (
                client_id,
            ),
        )

        total_row = cursor.fetchone()

        total = int(
            (
                total_row
                or {}
            ).get(
                "total",
                0,
            )
        )

        cursor.execute(
            (
                f"SELECT "
                "id, "
                "title, "
                "created_at, "
                "updated_at "
                f"FROM "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "WHERE client_id = %s "
                "ORDER BY updated_at DESC, "
                "id DESC "
                "LIMIT %s OFFSET %s"
            ),
            (
                client_id,
                limit,
                offset,
            ),
        )

        rows = cursor.fetchall()

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()

    return {
        "conversations": [
            {
                "id": int(
                    row["id"]
                ),

                "title": row[
                    "title"
                ],

                "created_at": (
                    _serialize_datetime(
                        row[
                            "created_at"
                        ]
                    )
                ),

                "updated_at": (
                    _serialize_datetime(
                        row[
                            "updated_at"
                        ]
                    )
                ),
            }
            for row in rows
        ],

        "page": page,
        "limit": limit,
        "total": total,
    }


def get_conversation_messages(
    conversation_id: int,
    client_id: str,
    page: int = 1,
    limit: int = 50,
) -> dict[str, Any]:
    require_conversation(
        conversation_id,
        client_id,
    )

    offset = (
        page - 1
    ) * limit

    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            (
                f"SELECT COUNT(*) AS total "
                f"FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s"
            ),
            (
                conversation_id,
            ),
        )

        total_row = cursor.fetchone()

        total = int(
            (
                total_row
                or {}
            ).get(
                "total",
                0,
            )
        )

        cursor.execute(
            (
                f"SELECT * FROM "
                f"{quote_identifier(MESSAGES_TABLE)} "
                "WHERE conversation_id = %s "
                "ORDER BY created_at DESC, "
                "id DESC "
                "LIMIT %s OFFSET %s"
            ),
            (
                conversation_id,
                limit,
                offset,
            ),
        )

        rows = list(
            reversed(
                cursor.fetchall()
            )
        )

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()

    return {
        "messages": [
            _message_dict(row)
            for row in rows
        ],

        "page": page,
        "limit": limit,
        "total": total,
    }


def delete_conversation(
    conversation_id: int,
    client_id: str,
) -> bool:
    connection = (
        get_chat_pool()
        .get_connection()
    )

    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute(
            (
                f"DELETE FROM "
                f"{quote_identifier(CONVERSATIONS_TABLE)} "
                "WHERE id = %s "
                "AND client_id = %s"
            ),
            (
                conversation_id,
                client_id,
            ),
        )

        deleted = (
            cursor.rowcount
            == 1
        )

        if not deleted:
            raise (
                ConversationNotFoundError(
                    "Conversation not found."
                )
            )

        connection.commit()

        return True

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()