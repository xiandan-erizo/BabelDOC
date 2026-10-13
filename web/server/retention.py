"""Automatic cleanup of old translation jobs.

Finished/error jobs keep their uploads and generated outputs on disk
(a large paper can produce ~200MB of PDFs). This module deletes
terminal-state jobs older than ``JOB_RETENTION_DAYS`` days (upload,
output directory and DB row together, mirroring the
``DELETE /api/jobs/{id}`` endpoint). Set the env to ``0`` to disable.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from web.server.db import OUTPUT_BASE_DIR
from web.server.db import Job
from web.server.db import utcnow

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = max(0, int(os.getenv("JOB_RETENTION_DAYS", "30") or 30))
DEFAULT_CLEANUP_INTERVAL = max(
    60.0, float(os.getenv("JOB_CLEANUP_INTERVAL_SECONDS", "21600") or 21600)  # 6h
)


def cleanup_expired_jobs(retention_days: int) -> int:
    """Delete finished/error jobs whose last update is older than the cutoff.

    Never touches queued/running jobs (they may still be active). Returns
    the number of removed jobs.
    """
    if retention_days <= 0:
        return 0
    cutoff = utcnow() - dt.timedelta(days=retention_days)
    removed = 0
    for row in (
        Job.select()
        .where(Job.status.in_(("finished", "error")), Job.updated_at < cutoff)
        .order_by(Job.updated_at)
    ):
        try:
            upload = Path(row.file_path) if row.file_path else None
            if upload and upload.is_file():
                try:
                    upload.unlink()
                except OSError:
                    logger.warning("retention: cannot remove upload %s", upload)
            output_dir = OUTPUT_BASE_DIR / row.job_id
            if output_dir.is_dir():
                shutil.rmtree(output_dir, ignore_errors=True)
            row.delete_instance()
            removed += 1
        except Exception:
            logger.exception("retention: failed to clean up job %s", row.job_id)
    if removed:
        logger.info(
            "retention: removed %d job(s) older than %d day(s)", removed, retention_days
        )
    return removed


def start_retention_thread(
    retention_days: int = DEFAULT_RETENTION_DAYS,
    interval_seconds: float = DEFAULT_CLEANUP_INTERVAL,
) -> threading.Thread | None:
    """Daemon thread that cleans up at startup, then once per interval."""
    if retention_days <= 0:
        logger.info("retention: disabled (JOB_RETENTION_DAYS <= 0)")
        return None

    def _run() -> None:
        while True:
            try:
                cleanup_expired_jobs(retention_days)
            except Exception:
                logger.exception("retention: cleanup pass failed")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_run, name="job-retention", daemon=True)
    thread.start()
    logger.info(
        "retention: cleaning jobs older than %d day(s) every %.0fs",
        retention_days,
        interval_seconds,
    )
    return thread
