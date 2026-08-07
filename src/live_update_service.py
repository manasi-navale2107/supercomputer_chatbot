from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.bootstrap import sync_datasets
from src.config import (
    LIVE_UPDATE_DEBOUNCE_SECONDS,
    LIVE_UPDATE_ENABLED,
    LIVE_UPDATE_POLL_SECONDS,
    RUNTIME_DIR,
)
from src.data_loader import dataset_fingerprint


logger = logging.getLogger(__name__)


MANIFEST_PATH = (
    RUNTIME_DIR
    / "live_update_manifest.json"
)


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _current_fingerprint() -> str:
    """
    Calculate the current CSV fingerprint.

    dataset_fingerprint is cached in data_loader,
    therefore its cache must be cleared before every
    live-update check.
    """

    cache_clear = getattr(
        dataset_fingerprint,
        "cache_clear",
        None,
    )

    if callable(cache_clear):
        cache_clear()

    return dataset_fingerprint()


def _load_manifest() -> dict[str, Any]:
    """
    Load the last successful synchronization state.

    The manifest allows the live-update service to remember
    the synchronized fingerprint after application restart.
    """

    if not MANIFEST_PATH.is_file():
        return {}

    try:
        content = MANIFEST_PATH.read_text(
            encoding="utf-8"
        )

        payload = json.loads(content)

        if not isinstance(payload, dict):
            raise ValueError(
                "Live-update manifest must "
                "contain a JSON object."
            )

        return payload

    except Exception:
        logger.exception(
            "The live-update manifest could "
            "not be loaded."
        )

        return {}


def _save_manifest(
    fingerprint: str,
    result: dict[str, Any],
) -> None:
    """
    Persist a successful synchronization atomically.
    """

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "synchronized_fingerprint":
            fingerprint,
        "last_completed_at": _utc_now(),
        "last_result": result,
    }

    temporary_path = Path(
        str(MANIFEST_PATH) + ".tmp"
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
        MANIFEST_PATH
    )


class LiveUpdateService:
    """
    Detect CSV changes and automatically synchronize
    MySQL and Qdrant.

    The service runs in a background thread.

    Only one synchronization operation can run at a time.
    A persistent manifest prevents unnecessary processing
    after application restarts.
    """

    def __init__(self) -> None:
        self._stop_event = (
            threading.Event()
        )

        self._worker: (
            threading.Thread | None
        ) = None

        self._update_lock = (
            threading.Lock()
        )

        self._status_lock = (
            threading.RLock()
        )

        manifest = _load_manifest()

        saved_fingerprint = (
            manifest.get(
                "synchronized_fingerprint"
            )
        )

        self._last_successful_fingerprint: (
            str | None
        ) = (
            str(saved_fingerprint)
            if saved_fingerprint
            else None
        )

        self._status: dict[str, Any] = {
            "enabled": LIVE_UPDATE_ENABLED,
            "service_running": False,
            "state": (
                "idle"
                if LIVE_UPDATE_ENABLED
                else "disabled"
            ),
            "is_updating": False,
            "poll_seconds":
                LIVE_UPDATE_POLL_SECONDS,
            "debounce_seconds":
                LIVE_UPDATE_DEBOUNCE_SECONDS,
            "manifest_path":
                str(MANIFEST_PATH),
            "observed_fingerprint": None,
            "synchronized_fingerprint": (
                self
                ._last_successful_fingerprint
            ),
            "last_checked_at": None,
            "last_started_at": None,
            "last_completed_at": (
                manifest.get(
                    "last_completed_at"
                )
            ),
            "last_error": None,
            "last_reason": None,
            "last_result": (
                manifest.get(
                    "last_result"
                )
            ),
        }

    def _update_status(
        self,
        **changes: Any,
    ) -> None:
        with self._status_lock:
            self._status.update(
                changes
            )

    def get_status(
        self,
    ) -> dict[str, Any]:
        with self._status_lock:
            status = deepcopy(
                self._status
            )

        status["worker_alive"] = bool(
            self._worker
            and self._worker.is_alive()
        )

        return status

    def synchronize(
        self,
        reason: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Run one synchronization operation.

        A non-blocking lock prevents simultaneous updates.
        """

        acquired = (
            self._update_lock.acquire(
                blocking=False
            )
        )

        if not acquired:
            logger.info(
                "Live data synchronization "
                "is already running."
            )

            return {
                "status": "already_running",
                "reason": reason,
            }

        started_at = _utc_now()

        self._update_status(
            state="synchronizing",
            is_updating=True,
            last_started_at=started_at,
            last_reason=reason,
            last_error=None,
        )

        logger.info(
            "Starting live data synchronization. "
            "Reason: %s",
            reason,
        )

        try:
            current_fingerprint = (
                _current_fingerprint()
            )

            self._update_status(
                observed_fingerprint=(
                    current_fingerprint
                )
            )

            if (
                not force
                and current_fingerprint
                == self
                ._last_successful_fingerprint
            ):
                result = {
                    "status": "unchanged",
                    "reason": reason,
                    "fingerprint":
                        current_fingerprint,
                }

                self._update_status(
                    state="idle",
                    is_updating=False,
                    synchronized_fingerprint=(
                        current_fingerprint
                    ),
                    last_completed_at=_utc_now(),
                    last_result=result,
                    last_error=None,
                )

                logger.info(
                    "Datasets are already synchronized. "
                    "No update is required."
                )

                return result

            result = sync_datasets(
                force=force
            )

            synchronized_fingerprint = str(
                result.get(
                    "fingerprint",
                    current_fingerprint,
                )
            )

            _save_manifest(
                fingerprint=(
                    synchronized_fingerprint
                ),
                result=result,
            )

            self._last_successful_fingerprint = (
                synchronized_fingerprint
            )

            self._update_status(
                state="idle",
                is_updating=False,
                observed_fingerprint=(
                    synchronized_fingerprint
                ),
                synchronized_fingerprint=(
                    synchronized_fingerprint
                ),
                last_completed_at=_utc_now(),
                last_error=None,
                last_result=result,
            )

            logger.info(
                "Live data synchronization "
                "completed successfully. "
                "Fingerprint: %s",
                synchronized_fingerprint,
            )

            return result

        except Exception as error:
            completed_at = _utc_now()

            self._update_status(
                state="error",
                is_updating=False,
                last_completed_at=completed_at,
                last_error=str(error),
            )

            logger.exception(
                "Live data synchronization failed. "
                "Reason: %s",
                reason,
            )

            raise

        finally:
            self._update_lock.release()

    def _check_for_changes(
        self,
    ) -> None:
        """
        Check whether the CSV dataset fingerprint changed.

        A debounce period ensures files are no longer being
        copied or modified before synchronization begins.
        """

        observed_fingerprint = (
            _current_fingerprint()
        )

        self._update_status(
            last_checked_at=_utc_now(),
            observed_fingerprint=(
                observed_fingerprint
            ),
        )

        if (
            observed_fingerprint
            == self
            ._last_successful_fingerprint
        ):
            return

        logger.info(
            "Dataset change detected. "
            "Waiting for files to become stable."
        )

        stopped = self._stop_event.wait(
            LIVE_UPDATE_DEBOUNCE_SECONDS
        )

        if stopped:
            return

        stable_fingerprint = (
            _current_fingerprint()
        )

        self._update_status(
            last_checked_at=_utc_now(),
            observed_fingerprint=(
                stable_fingerprint
            ),
        )

        if (
            stable_fingerprint
            != observed_fingerprint
        ):
            logger.info(
                "CSV files are still changing. "
                "Synchronization will be "
                "retried during the next cycle."
            )

            return

        if (
            stable_fingerprint
            == self
            ._last_successful_fingerprint
        ):
            return

        self.synchronize(
            reason="csv_change",
            force=False,
        )

    def _run(
        self,
    ) -> None:
        logger.info(
            "Live update service started. "
            "Polling every %s seconds.",
            LIVE_UPDATE_POLL_SECONDS,
        )

        self._update_status(
            service_running=True,
            state="idle",
        )

        while not self._stop_event.is_set():
            try:
                self._check_for_changes()

            except Exception:
                # Detailed synchronization errors are stored
                # in the status. Keep the worker alive so it
                # can retry during the next polling cycle.
                logger.exception(
                    "Live update check failed. "
                    "It will be retried."
                )

            self._stop_event.wait(
                LIVE_UPDATE_POLL_SECONDS
            )

        self._update_status(
            service_running=False,
            state="stopped",
        )

        logger.info(
            "Live update service stopped."
        )

    def start(
        self,
    ) -> None:
        """
        Start the background worker without blocking
        FastAPI application startup.
        """

        if not LIVE_UPDATE_ENABLED:
            self._update_status(
                state="disabled",
                service_running=False,
            )

            logger.info(
                "Live update service is disabled."
            )

            return

        if (
            self._worker
            and self._worker.is_alive()
        ):
            return

        self._stop_event.clear()

        self._worker = threading.Thread(
            target=self._run,
            name="live-data-update-service",
            daemon=True,
        )

        self._worker.start()

    def stop(
        self,
    ) -> None:
        self._stop_event.set()

        worker = self._worker

        if (
            worker
            and worker.is_alive()
        ):
            worker.join(
                timeout=5
            )

        self._worker = None

        self._update_status(
            service_running=False,
            is_updating=False,
            state=(
                "idle"
                if LIVE_UPDATE_ENABLED
                else "disabled"
            ),
        )


_live_update_service = (
    LiveUpdateService()
)


def run_live_update_once(
    reason: str = "manual",
    force: bool = False,
) -> dict[str, Any]:
    return (
        _live_update_service
        .synchronize(
            reason=reason,
            force=force,
        )
    )


def start_live_update_service() -> None:
    _live_update_service.start()


def stop_live_update_service() -> None:
    _live_update_service.stop()


def get_live_update_status() -> dict[str, Any]:
    return (
        _live_update_service
        .get_status()
    )