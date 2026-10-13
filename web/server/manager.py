"""Worker manager: run translation jobs in short-lived subprocesses.

Design goals (the "don't keep it resident in memory" fix):

- The API server process never imports babeldoc. A translation job is
  executed by a dedicated `python -m web.server.worker <job_id>` subprocess
  which loads the layout ONNX model and BabelDOC assets, does the work,
  writes the result paths to SQLite, then exits. All of that memory is
  returned to the OS as soon as the job finishes.
- At most ``max_workers`` subprocesses run at the same time (default 1),
  so peak memory stays predictable regardless of how many jobs are queued.
- Finished children are reaped (no zombies), crash-exited workers mark the
  job as failed, and workers exceeding the timeout are killed.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from web.server.db import Job
from web.server.db import utcnow

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = max(1, int(os.getenv("MAX_CONCURRENT_WORKERS", "1") or 1))
DEFAULT_WORKER_TIMEOUT_SECONDS = max(
    60, int(os.getenv("WORKER_TIMEOUT_SECONDS", str(30 * 60)) or 30 * 60)
)
# Once a worker has written a terminal state (finished/error) to the DB but
# the process is still alive (lingering non-daemon threads), wait this long
# before killing the process group. The worker itself also arms a hard-exit
# watchdog; this is the second line of defense.
TERMINAL_GRACE_SECONDS = float(os.getenv("WORKER_TERMINAL_GRACE_SECONDS", "10") or 10)


def build_worker_command(job_id: str) -> list[str]:
    return [sys.executable, "-m", "web.server.worker", job_id]


def launch_translate_worker(job_id: str, project_dir: Path) -> subprocess.Popen:
    """Spawn a detached worker process (own session, so it can be killpg'd)."""
    return subprocess.Popen(  # noqa: S603
        build_worker_command(job_id),
        cwd=project_dir,
        start_new_session=True,
    )


def mark_interrupted_jobs() -> int:
    """Mark jobs left in queued/running state by a previous server life."""
    now = utcnow()
    return (
        Job.update(
            status="error",
            error="服务重启导致任务中断，请重新提交",
            updated_at=now,
        )
        .where(Job.status.in_(["queued", "running"]))  # noqa: RUF010
        .execute()
    )


class WorkerManager:
    def __init__(
        self,
        project_dir: Path,
        max_workers: int = DEFAULT_MAX_CONCURRENT,
        worker_timeout_seconds: int = DEFAULT_WORKER_TIMEOUT_SECONDS,
        poll_interval: float = 1.0,
        terminal_grace_seconds: float = TERMINAL_GRACE_SECONDS,
    ) -> None:
        self._project_dir = Path(project_dir)
        self._max_workers = max(1, int(max_workers))
        self._timeout = max(60, int(worker_timeout_seconds))
        self._terminal_grace = max(0.0, float(terminal_grace_seconds))
        self._poll_interval = poll_interval
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        # job_id -> (popen, started_monotonic, terminal_since_monotonic|None)
        self._running: dict[str, tuple[subprocess.Popen, float, float | None]] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="worker-manager", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop dispatching new jobs. Running workers are detached and may
        keep writing progress/results to SQLite, which the next server
        process will pick up; that is intentional."""
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._dispatch_once()
            except Exception:  # pragma: no cover - defensive
                logger.exception("worker manager iteration failed")

    # -- public API ----------------------------------------------------------

    def enqueue(self, job_id: str) -> int:
        with self._lock:
            self._queue.append(job_id)
            return self._queue.index(job_id) + 1

    def dequeue(self, job_id: str) -> bool:
        with self._lock:
            try:
                self._queue.remove(job_id)
                return True
            except ValueError:
                return False

    def position(self, job_id: str) -> int:
        """0 = currently running, 1..n = queue position, -1 = unknown."""
        with self._lock:
            if job_id in self._running:
                return 0
            try:
                return self._queue.index(job_id) + 1
            except ValueError:
                return -1

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {"running": len(self._running), "queued": len(self._queue)}

    @property
    def max_workers(self) -> int:
        return self._max_workers

    # -- internals -----------------------------------------------------------

    def _dispatch_once(self) -> None:
        with self._lock:
            self._reap_locked()
            self._enforce_timeout_locked()
            while len(self._running) < self._max_workers and self._queue:
                job_id = self._queue.popleft()
                try:
                    popen = launch_translate_worker(job_id, self._project_dir)
                except Exception as exc:
                    logger.exception("failed to launch worker for job %s", job_id)
                    self._mark_error(job_id, f"启动翻译进程失败：{exc}")
                    continue
                self._running[job_id] = (popen, time.monotonic(), None)
                self._mark_running(job_id)

    def _reap_locked(self) -> None:
        """Reap finished children; detect crashed workers; kill workers that
        finished their job but keep lingering in memory."""
        for job_id, (popen, _started, terminal_since) in list(self._running.items()):
            rc = popen.poll()
            if rc is not None:
                del self._running[job_id]
                status = self._job_status(job_id)
                if rc == 0:
                    if status == "running":
                        self._mark_error(
                            job_id, "worker 进程已退出但未写入结果，任务可能中途失败"
                        )
                else:
                    if status in ("queued", "running"):
                        reason = f"worker 进程异常退出 (exit code {rc})"
                        if rc == -9:
                            reason = "worker 进程被强制终止（超时或内存占用被清理）"
                        self._mark_error(job_id, reason)
                    logger.warning("worker for job %s exited with code %s", job_id, rc)
                continue

            # process still alive: if the job already reached a terminal state,
            # the worker wrote its results and is only lingering (babeldoc keeps
            # non-daemon threads). Track it and kill after the grace period.
            status = self._job_status(job_id)
            if status in ("finished", "error"):
                if terminal_since is None:
                    self._running[job_id] = (popen, _started, time.monotonic())
                elif time.monotonic() - terminal_since > self._terminal_grace:
                    logger.warning(
                        "worker for job %s finished but still alive after %.0fs, killing",
                        job_id,
                        self._terminal_grace,
                    )
                    self._kill_process_group(popen)

    def _kill_process_group(self, popen: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        except OSError:
            try:
                popen.kill()
            except OSError:
                pass

    def _enforce_timeout_locked(self) -> None:
        now = time.monotonic()
        for job_id, (popen, started, _ts) in list(self._running.items()):
            if now - started <= self._timeout:
                continue
            logger.warning(
                "job %s exceeded worker timeout of %ss, killing", job_id, self._timeout
            )
            self._kill_process_group(popen)
            # the next _reap_locked pass will clean up and persist the error

    # -- DB helpers (peewee auto-connects per thread) ------------------------

    @staticmethod
    def _job_status(job_id: str) -> str | None:
        row = Job.get_or_none(Job.job_id == job_id)
        if row is None:
            return None
        return row.status

    @staticmethod
    def _mark_running(job_id: str) -> None:
        Job.update(status="running", started_at=utcnow(), updated_at=utcnow()).where(
            Job.job_id == job_id
        ).execute()

    @staticmethod
    def _mark_error(job_id: str, error: str) -> None:
        Job.update(status="error", error=error, updated_at=utcnow()).where(
            Job.job_id == job_id
        ).execute()
