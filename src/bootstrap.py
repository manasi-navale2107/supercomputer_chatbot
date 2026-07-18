from __future__ import annotations

import argparse
import json

from src.chat_store import init_chat_tables
from src.mysql_store import ingest_csvs_to_mysql
from src.qdrant_store import build_qdrant_index


def bootstrap_all(force: bool = False) -> dict:
    mysql_result = ingest_csvs_to_mysql(force=force)
    init_chat_tables()
    qdrant_result = build_qdrant_index(force=force)
    return {
        "mysql": mysql_result,
        "chat_history": {"status": "initialized"},
        "qdrant": qdrant_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import the five CSV files into MySQL and index them in Qdrant."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import and re-index even when the CSV fingerprint is unchanged.",
    )
    args = parser.parse_args()
    print(json.dumps(bootstrap_all(force=args.force), indent=2))


if __name__ == "__main__":
    main()
