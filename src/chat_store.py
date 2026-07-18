from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
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
from src.mysql_store import get_admin_connection, quote_identifier


CONVERSATIONS_TABLE = "chat_conversations"
MESSAGES_TABLE = "chat_messages"


class ConversationNotFoundError(LookupError):
    pass


@lru_cache(maxsize=1)
def get_chat_pool() -> pooling.MySQLConnectionPool:
    pool_hash = hashlib.sha1(
        f"{MYSQL_HOST}:{MYSQL_PORT}:{MYSQL_DATABASE}:{MYSQL_CHAT_USER}".encode()
    ).hexdigest()[:10]
    return pooling.MySQLConnectionPool(
        pool_name=f"scchat_{pool_hash}",
        pool_size=MYSQL_POOL_SIZE,
        pool_reset_session=True,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_CHAT_USER,
        password=MYSQL_CHAT_PASSWORD,
        database=MYSQL_DATABASE,
        connection_timeout=MYSQL_CONNECT_TIMEOUT,
        autocommit=False,
        charset="utf8mb4",
        use_unicode=True,
    )


def init_chat_tables() -> None:
    connection = get_admin_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quote_identifier(CONVERSATIONS_TABLE)} (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                client_id CHAR(36) NOT NULL,
                title VARCHAR(200) NOT NULL,
                created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                updated_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                    ON UPDATE CURRENT_TIMESTAMP(6),
                PRIMARY KEY (id),
                INDEX idx_chat_conversations_client_updated
                    (client_id, updated_at DESC)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            COLLATE=utf8mb4_0900_ai_ci
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quote_identifier(MESSAGES_TABLE)} (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                conversation_id BIGINT UNSIGNED NOT NULL,
                role ENUM('user', 'assistant') NOT NULL,
                content LONGTEXT NOT NULL,
                route VARCHAR(32) NULL,
                sql_query LONGTEXT NULL,
                metadata_json JSON NULL,
                created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
                PRIMARY KEY (id),
                INDEX idx_chat_messages_conversation_created
                    (conversation_id, created_at, id),
                CONSTRAINT fk_chat_messages_conversation
                    FOREIGN KEY (conversation_id)
                    REFERENCES {quote_identifier(CONVERSATIONS_TABLE)} (id)
                    ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            COLLATE=utf8mb4_0900_ai_ci
            """
        )
        connection.commit()
        cursor.close()
    finally:
        connection.close()


def verify_chat_tables() -> None:
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name IN (%s, %s)",
            (MYSQL_DATABASE, CONVERSATIONS_TABLE, MESSAGES_TABLE),
        )
        found = {row[0] for row in cursor.fetchall()}
        cursor.close()
        required = {CONVERSATIONS_TABLE, MESSAGES_TABLE}
        if found != required:
            raise RuntimeError(
                "Chat tables are not initialized. Run: python -m src.bootstrap"
            )
    finally:
        connection.close()


def _clean_title(first_message: str) -> str:
    title = re.sub(r"\s+", " ", first_message).strip()
    if len(title) > CHAT_TITLE_MAX_LENGTH:
        title = title[: CHAT_TITLE_MAX_LENGTH - 1].rstrip() + "…"
    return title or "New conversation"


def _serialize_datetime(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _decode_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _message_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "conversation_id": int(row["conversation_id"]),
        "role": row["role"],
        "content": row["content"],
        "route": row.get("route"),
        "sql": row.get("sql_query"),
        "metadata": _decode_metadata(row.get("metadata_json")),
        "created_at": _serialize_datetime(row.get("created_at")),
    }


def create_conversation(client_id: str, first_message: str) -> tuple[int, str]:
    title = _clean_title(first_message)
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"INSERT INTO {quote_identifier(CONVERSATIONS_TABLE)} "
            "(client_id, title) VALUES (%s, %s)",
            (client_id, title),
        )
        conversation_id = int(cursor.lastrowid)
        connection.commit()
        cursor.close()
        return conversation_id, title
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def require_conversation(conversation_id: int, client_id: str) -> None:
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT id FROM {quote_identifier(CONVERSATIONS_TABLE)} "
            "WHERE id = %s AND client_id = %s",
            (conversation_id, client_id),
        )
        found = cursor.fetchone()
        cursor.close()
        if not found:
            raise ConversationNotFoundError("Conversation not found.")
    finally:
        connection.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    route: str | None = None,
    sql: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if role not in {"user", "assistant"}:
        raise ValueError("Message role must be user or assistant.")
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            f"INSERT INTO {quote_identifier(MESSAGES_TABLE)} "
            "(conversation_id, role, content, route, sql_query, metadata_json) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (
                conversation_id,
                role,
                content,
                route,
                sql,
                json.dumps(metadata, default=str) if metadata else None,
            ),
        )
        message_id = int(cursor.lastrowid)
        cursor.execute(
            f"UPDATE {quote_identifier(CONVERSATIONS_TABLE)} "
            "SET updated_at = CURRENT_TIMESTAMP(6) WHERE id = %s",
            (conversation_id,),
        )
        cursor.execute(
            f"SELECT * FROM {quote_identifier(MESSAGES_TABLE)} WHERE id = %s",
            (message_id,),
        )
        row = cursor.fetchone()
        connection.commit()
        cursor.close()
        return _message_dict(row)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_recent_context(
    conversation_id: int,
    message_limit: int = CHAT_CONTEXT_MESSAGE_LIMIT,
    character_limit: int = MAX_CHAT_HISTORY_CHARS,
) -> str:
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            f"SELECT role, content FROM {quote_identifier(MESSAGES_TABLE)} "
            "WHERE conversation_id = %s ORDER BY created_at DESC, id DESC LIMIT %s",
            (conversation_id, message_limit),
        )
        rows = list(reversed(cursor.fetchall()))
        cursor.close()
    finally:
        connection.close()

    lines = [
        f"{'User' if row['role'] == 'user' else 'Assistant'}: {row['content']}"
        for row in rows
    ]
    context = "\n".join(lines)
    return context[-character_limit:]


def get_conversations(
    client_id: str,
    page: int = 1,
    limit: int = 20,
) -> dict[str, Any]:
    offset = (page - 1) * limit
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            f"SELECT COUNT(*) AS total FROM {quote_identifier(CONVERSATIONS_TABLE)} "
            "WHERE client_id = %s",
            (client_id,),
        )
        total = int(cursor.fetchone()["total"])
        cursor.execute(
            f"SELECT id, title, created_at, updated_at "
            f"FROM {quote_identifier(CONVERSATIONS_TABLE)} "
            "WHERE client_id = %s ORDER BY updated_at DESC, id DESC LIMIT %s OFFSET %s",
            (client_id, limit, offset),
        )
        rows = cursor.fetchall()
        cursor.close()
    finally:
        connection.close()

    return {
        "conversations": [
            {
                "id": int(row["id"]),
                "title": row["title"],
                "created_at": _serialize_datetime(row["created_at"]),
                "updated_at": _serialize_datetime(row["updated_at"]),
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
    require_conversation(conversation_id, client_id)
    offset = (page - 1) * limit
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            f"SELECT COUNT(*) AS total FROM {quote_identifier(MESSAGES_TABLE)} "
            "WHERE conversation_id = %s",
            (conversation_id,),
        )
        total = int(cursor.fetchone()["total"])
        cursor.execute(
            f"SELECT * FROM {quote_identifier(MESSAGES_TABLE)} "
            "WHERE conversation_id = %s "
            "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
            (conversation_id, limit, offset),
        )
        rows = list(reversed(cursor.fetchall()))
        cursor.close()
    finally:
        connection.close()
    return {
        "messages": [_message_dict(row) for row in rows],
        "page": page,
        "limit": limit,
        "total": total,
    }


def delete_conversation(conversation_id: int, client_id: str) -> bool:
    connection = get_chat_pool().get_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            f"DELETE FROM {quote_identifier(CONVERSATIONS_TABLE)} "
            "WHERE id = %s AND client_id = %s",
            (conversation_id, client_id),
        )
        deleted = cursor.rowcount == 1
        if not deleted:
            connection.rollback()
            raise ConversationNotFoundError("Conversation not found.")
        connection.commit()
        cursor.close()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

