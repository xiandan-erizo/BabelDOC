"""Tests for the WorkerManager: queueing, serialization, reaping, timeouts."""

import sys
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture()
def manager_db(tmp_path):
    from web.server import db as web_db

    test_db_path = tmp_path / "babeldoc.db"
    web_db.db.init(str(test_db_path))
    web_db.db.connect(reuse_if_open=True)
    web_db.db.create_tables([web_db.SystemSettings, web_db.Job])
    yield web_db
    web_db.db.drop_tables([web_db.Job, web_db.SystemSettings])
    web_db.db.close()


def make_job(db, job_id: str, status: str = "queued") -> None:
    db.Job.create(
        job_id=job_id,
        status=status,
        progress=0.0,
        stage=None,
        stage_current=0,
        stage_total=0,
        lang_in="en",
        lang_out="zh",
        openai_model="gpt-test",
        qps=4,
        watermark_output_mode="watermarked",
        no_dual=False,
        no_mono=False,
        file_path="fake.pdf",
    )


def patched_command(monkeypatch, script: str, *args: str) -> None:
    """Replace the real worker command with `python -c <script> *args`."""

    from web.server import manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "build_worker_command",
        lambda job_id: [sys.executable, "-c", script, *args, job_id],
    )


def test_enqueue_dequeue_position(manager_db):  # noqa: ARG001
    from web.server.manager import WorkerManager

    m = WorkerManager(Path(), max_workers=1, poll_interval=0.01)
    assert m.enqueue("job-a") == 1
    assert m.enqueue("job-b") == 2
    assert m.position("job-a") == 1
    assert m.position("job-b") == 2
    assert m.position("unknown") == -1
    assert m.dequeue("job-a") is True
    assert m.dequeue("job-a") is False
    assert m.position("job-b") == 1
    assert m.counts() == {"running": 0, "queued": 1}
    assert m.max_workers == 1


def test_manager_serializes_jobs_and_reaps(manager_db, monkeypatch):
    """Jobs run one at a time; exited children are reaped and don't override
    a terminal status the worker itself wrote to the DB."""
    real_db_path = str(manager_db.db.database)
    script = (
        "import sqlite3,time,sys;"  # noqa: S608
        f"time.sleep(0.3);"
        f"con=sqlite3.connect({str(real_db_path)!r});"
        "con.execute('update job set status=?, progress=100.0 where job_id=?',"
        "('finished', sys.argv[1]));"
        "con.commit()"
    )
    patched_command(monkeypatch, script)

    from web.server.manager import WorkerManager

    make_job(manager_db, "job-1")
    make_job(manager_db, "job-2")

    m = WorkerManager(Path(), max_workers=1, poll_interval=0.01)
    m.enqueue("job-1")
    m.enqueue("job-2")

    m._dispatch_once()
    assert manager_db.Job.get(manager_db.Job.job_id == "job-1").status == "running"
    assert manager_db.Job.get(manager_db.Job.job_id == "job-2").status == "queued"
    assert m.counts()["running"] == 1

    # second dispatch while first still running -> nothing new spawns
    m._dispatch_once()
    assert m.counts() == {"running": 1, "queued": 1}

    time.sleep(0.5)
    m._dispatch_once()  # reaps job-1, spawns job-2
    assert manager_db.Job.get(manager_db.Job.job_id == "job-1").status == "finished"
    assert manager_db.Job.get(manager_db.Job.job_id == "job-2").status == "running"

    time.sleep(0.5)
    m._dispatch_once()
    assert manager_db.Job.get(manager_db.Job.job_id == "job-2").status == "finished"
    assert m.counts() == {"running": 0, "queued": 0}


def test_manager_marks_crashed_worker_as_error(manager_db, monkeypatch):
    patched_command(monkeypatch, "import sys; sys.exit(7)")
    make_job(manager_db, "job-crash")

    from web.server.manager import WorkerManager

    m = WorkerManager(Path(), max_workers=1, poll_interval=0.01)
    m.enqueue("job-crash")
    m._dispatch_once()
    time.sleep(0.2)
    m._dispatch_once()

    row = manager_db.Job.get(manager_db.Job.job_id == "job-crash")
    assert row.status == "error"
    assert "7" in row.error


def test_manager_timeout_kills_worker(manager_db, monkeypatch):
    patched_command(monkeypatch, "import time; time.sleep(60)")
    make_job(manager_db, "job-slow")

    from web.server.manager import WorkerManager

    m = WorkerManager(Path(), max_workers=1, poll_interval=0.01)
    m.enqueue("job-slow")
    m._dispatch_once()

    # pretend the worker started long before the timeout window
    with m._lock:
        popen, _started, _ts = m._running["job-slow"]
        m._running["job-slow"] = (popen, time.monotonic() - m._timeout - 10, None)

    m._dispatch_once()  # kills the timed-out worker
    time.sleep(0.2)
    m._dispatch_once()  # reaps the killed worker
    with m._lock:
        assert "job-slow" not in m._running

    row = manager_db.Job.get(manager_db.Job.job_id == "job-slow")
    assert row.status == "error"


def test_manager_kills_lingering_worker_after_terminal_state(manager_db, monkeypatch):
    """A worker that persists `finished` but keeps running (babeldoc keeps
    non-daemon threads alive) must be killed after the grace period."""
    real_db_path = str(manager_db.db.database)
    script = (
        "import sqlite3,time,sys;"  # noqa: S608
        f"con=sqlite3.connect({real_db_path!r});"
        "con.execute('update job set status=? where job_id=?',('finished', sys.argv[1]));"
        "con.commit();"
        "time.sleep(300)"
    )
    patched_command(monkeypatch, script)
    make_job(manager_db, "job-linger")

    from web.server.manager import WorkerManager

    m = WorkerManager(
        Path(), max_workers=1, poll_interval=0.01, terminal_grace_seconds=0.1
    )
    m.enqueue("job-linger")
    m._dispatch_once()
    time.sleep(0.3)

    # worker already wrote `finished`; first pass records terminal_since,
    # a later pass (after grace) kills it, the next pass reaps it
    m._dispatch_once()  # observe terminal state, record terminal_since
    time.sleep(0.2)  # exceed terminal_grace_seconds=0.1
    m._dispatch_once()  # kill the lingering process group
    time.sleep(0.2)
    m._dispatch_once()  # reap the killed worker

    assert (
        manager_db.Job.get(manager_db.Job.job_id == "job-linger").status == "finished"
    )  # terminal state preserved, not overwritten
    with m._lock:
        assert "job-linger" not in m._running  # killed and reaped


def test_mark_interrupted_jobs(manager_db):
    from web.server.manager import mark_interrupted_jobs

    make_job(manager_db, "job-queued", status="queued")
    make_job(manager_db, "job-running", status="running")
    make_job(manager_db, "job-finished", status="finished")
    make_job(manager_db, "job-error", status="error")

    count = mark_interrupted_jobs()
    assert count == 2
    get = lambda jid: manager_db.Job.get(manager_db.Job.job_id == jid)  # noqa: E731
    assert get("job-queued").status == "error"
    assert get("job-running").status == "error"
    assert get("job-finished").status == "finished"
    assert get("job-error").status == "error"
    assert "中断" in (get("job-queued").error or "")


def test_running_job_sets_started_at(manager_db, monkeypatch):
    patched_command(monkeypatch, "import time; time.sleep(60)")
    make_job(manager_db, "job-start")
    from web.server.manager import WorkerManager

    m = WorkerManager(Path(), max_workers=1, poll_interval=0.01)
    m.enqueue("job-start")
    m._dispatch_once()

    row = manager_db.Job.get(manager_db.Job.job_id == "job-start")
    assert row.status == "running"
    assert row.started_at is not None

    with m._lock:
        m._running["job-start"][0].kill()
    m._dispatch_once()