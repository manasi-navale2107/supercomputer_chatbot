from __future__ import annotations

import argparse
import json
from typing import Any

from src.chat_store import (
    init_chat_tables,
)
from src.data_loader import (
    build_dataset_change_plan,
    file_fingerprints,
    save_dataset_manifest,
)
from src.mysql_store import (
    ingest_csvs_to_mysql,
)
from src.qdrant_store import (
    build_qdrant_index,
)


def sync_datasets(
    force: bool = False,
) -> dict[str, Any]:
    """
    Synchronize current CSV datasets with
    MySQL and Qdrant.

    The same change plan is passed to both
    storage systems.

    The dataset manifest is saved only after
    both MySQL and Qdrant complete successfully.
    """

    change_plan = (
        build_dataset_change_plan()
    )

    fingerprints_before = dict(
        change_plan[
            "current_fingerprints"
        ]
    )

    mysql_result = (
        ingest_csvs_to_mysql(
            force=force,
            change_plan=change_plan,
        )
    )

    qdrant_result = (
        build_qdrant_index(
            force=force,
            change_plan=change_plan,
        )
    )

    fingerprints_after = (
        file_fingerprints()
    )

    if (
        fingerprints_before
        != fingerprints_after
    ):
        raise RuntimeError(
            "CSV files changed while the "
            "synchronization pipeline was "
            "running. The update will be "
            "retried."
        )

    fingerprint = str(
        change_plan["fingerprint"]
    )

    mysql_fingerprint = str(
        mysql_result.get(
            "fingerprint",
            "",
        )
    )

    qdrant_fingerprint = str(
        qdrant_result.get(
            "fingerprint",
            "",
        )
    )

    if (
        mysql_fingerprint
        != fingerprint
    ):
        raise RuntimeError(
            "MySQL did not synchronize with "
            "the current dataset fingerprint."
        )

    if (
        qdrant_fingerprint
        != fingerprint
    ):
        raise RuntimeError(
            "Qdrant did not synchronize with "
            "the current dataset fingerprint."
        )

    status = (
        "unchanged"
        if (
            mysql_result.get("status")
            == "unchanged"
            and qdrant_result.get("status")
            == "unchanged"
        )
        else "synchronized"
    )

    result: dict[str, Any] = {
        "status": status,
        "fingerprint": fingerprint,
        "change_plan": {
            "new_tables":
                change_plan[
                    "new_tables"
                ],
            "changed_tables":
                change_plan[
                    "changed_tables"
                ],
            "deleted_tables":
                change_plan[
                    "deleted_tables"
                ],
            "unchanged_tables":
                change_plan[
                    "unchanged_tables"
                ],
        },
        "mysql": mysql_result,
        "qdrant": qdrant_result,
    }

    save_dataset_manifest(
        file_fingerprints_value=(
            fingerprints_after
        ),
        synchronization_result=result,
    )

    return result


def bootstrap_all(
    force: bool = False,
) -> dict[str, Any]:
    """
    Initialize or synchronize the complete
    application.

    --force performs the one-time full migration.
    Later runs are incremental.
    """

    synchronization = (
        sync_datasets(
            force=force
        )
    )

    init_chat_tables()

    return {
        "status":
            synchronization[
                "status"
            ],
        "fingerprint":
            synchronization[
                "fingerprint"
            ],
        "change_plan":
            synchronization[
                "change_plan"
            ],
        "mysql":
            synchronization[
                "mysql"
            ],
        "qdrant":
            synchronization[
                "qdrant"
            ],
        "chat_history": {
            "status": "initialized",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize CSV datasets with "
            "MySQL and Qdrant."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Perform a full one-time MySQL "
            "migration and Qdrant rebuild."
        ),
    )

    args = parser.parse_args()

    result = bootstrap_all(
        force=args.force
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()