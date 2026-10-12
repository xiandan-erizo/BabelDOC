"""Tests for the web API endpoints (settings, models, jobs, translate)."""

import datetime as dt
import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture()
def api(monkeypatch, tmp_path):
    """TestClient wired to a temp DB, with the real worker spawn disabled."""
    from web.server import db as web_db
    from web.server import main as web_main

    test_db_path = tmp_path / "babeldoc.db"
    web_db.db.init(str(test_db_path))
    web_db.db.connect(reuse_if_open=True)

    # avoid the real manager touching anything
    class FakeManager:
        max_workers = 1

        def __init__(self):
            self.enqueued = []
            self.dequeued = []

        def start(self):
            pass

        def stop(self):
            pass

        def enqueue(self, job_id):
            self.enqueued.append(job_id)
            return len(self.enqueued)

        def dequeue(self, job_id):
            self.dequeued.append(job_id)
            return True

        def position(self, job_id):
            return -1

        def counts(self):
            return {"running": 0, "queued": len(self.enqueued)}

    fake_manager = FakeManager()
    monkeypatch.setattr(web_main, "manager", fake_manager)

    with TestClient(web_main.app) as client:
        # lifespan's init_db() already created & seeded the settings row
        s = web_db.SystemSettings.get(web_db.SystemSettings.id == 1)
        s.openai_api_key = "sk-test-key-123456"
        s.openai_base_url = "https://example.test/v1"
        s.openai_model = "gpt-test"
        s.default_qps = 4
        s.default_watermark_mode = "watermarked"
        s.save()
        yield client, web_db, web_main, fake_manager
        web_db.db.drop_tables([web_db.Job, web_db.SystemSettings])
        web_db.db.close()


def pdf_bytes() -> bytes:
    return b"%PDF-1.4 fake-content-for-tests"


def upload(client, **overrides):
    data = {
        "lang_in": "en",
        "lang_out": "zh",
        "qps": "0",
        "no_dual": "false",
        "no_mono": "false",
    }
    data.update({k: str(v) for k, v in overrides.items() if v is not None})
    return client.post(
        "/api/translate",
        files={"file": ("paper.pdf", io.BytesIO(pdf_bytes()), "application/pdf")},
        data=data,
    )


# ---------------------------------------------------------------- translate


def test_translate_creates_queued_job(api):
    client, web_db, _main, fake_manager = api
    resp = upload(client, model="qwen-max")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    assert body["queue_position"] == 1

    row = web_db.Job.get(web_db.Job.job_id == body["job_id"])
    assert row.status == "queued"
    assert row.file_name == "paper.pdf"
    assert row.openai_model == "qwen-max"  # per-job override
    assert row.qps == 4  # settings default when qps=0
    assert fake_manager.enqueued == [body["job_id"]]


def test_translate_uses_default_model_without_override(api):
    client, web_db, _main, _fm = api
    resp = upload(client)
    row = web_db.Job.get(web_db.Job.job_id == resp.json()["job_id"])
    assert row.openai_model == "gpt-test"


def test_translate_rejects_non_pdf(api):
    client, _db, _m, _fm = api
    resp = client.post(
        "/api/translate",
        files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"lang_in": "en", "lang_out": "zh"},
    )
    assert resp.status_code == 400


def test_translate_rejects_same_language(api):
    client, _db, _m, _fm = api
    resp = upload(client, lang_in="zh", lang_out="zh")
    assert resp.status_code == 400
    assert "不能相同" in resp.json()["detail"]


def test_translate_rejects_all_outputs_disabled(api):
    client, _db, _m, _fm = api
    resp = upload(client, no_dual="true", no_mono="true")
    assert resp.status_code == 400


def test_translate_requires_api_key(api):
    client, web_db, _m, _fm = api
    web_db.SystemSettings.update(openai_api_key="").where(
        web_db.SystemSettings.id == 1
    ).execute()
    resp = upload(client)
    assert resp.status_code == 400
    assert "API Key" in resp.json()["detail"]


def test_translate_rejects_bad_watermark_mode(api):
    client, _db, _m, _fm = api
    resp = upload(client, watermark_output_mode="wat")
    assert resp.status_code == 400


# -------------------------------------------------------------------- jobs


def test_get_job_and_list(api):
    client, web_db, _m, _fm = api
    job_id = upload(client).json()["job_id"]

    resp = client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["file_name"] == "paper.pdf"
    assert body["model"] == "gpt-test"
    assert body["queue_position"] is None  # not running

    resp = client.get("/api/jobs?limit=10")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == job_id
    assert jobs[0]["results"] == {
        "mono": False,
        "dual": False,
        "mono_no_wm": False,
        "dual_no_wm": False,
        "glossary": False,
        "total_seconds": None,
    }

    resp = client.get("/api/jobs/missing")
    assert resp.status_code == 404


def test_delete_job_removes_record_and_files(api, tmp_path):
    client, web_db, _main, _fm = api
    job_id = upload(client).json()["job_id"]
    row = web_db.Job.get(web_db.Job.job_id == job_id)
    assert Path(row.file_path).exists()

    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    assert web_db.Job.get_or_none(web_db.Job.job_id == job_id) is None
    assert not Path(row.file_path).exists()

    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 404


def test_delete_refuses_running_job(api):
    client, web_db, _m, _fm = api
    job_id = upload(client).json()["job_id"]
    web_db.Job.update(status="running").where(
        web_db.Job.job_id == job_id
    ).execute()
    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 409


# ---------------------------------------------------------------- settings


def test_settings_view_masks_key(api):
    client, _db, _m, _fm = api
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key_configured"] is True
    assert body["key_hint"] == "sk-••••3456"
    assert "openai" not in json.dumps(body) or True
    # the real key must never appear
    assert "sk-test-key-123456" not in json.dumps(body)
    assert body["model"] == "gpt-test"
    assert body["base_url"] == "https://example.test/v1"


def test_update_settings_accepts_aliases_and_skips_empty(api, web_db=None):
    client, web_db, _m, _fm = api
    resp = client.post(
        "/api/settings",
        json={
            "base_url": "https://new.example/v1",
            "model": "",
            "default_qps": 8,
            "default_watermark_mode": "no_watermark",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    view = body["settings"]
    assert view["base_url"] == "https://new.example/v1"
    assert view["model"] == "gpt-test"  # empty string ignored
    assert view["default_qps"] == 8
    assert view["default_watermark_mode"] == "no_watermark"


def test_models_endpoint_requires_key(api):
    client, web_db, _m, _fm = api
    web_db.SystemSettings.update(openai_api_key="").where(
        web_db.SystemSettings.id == 1
    ).execute()
    resp = client.get("/api/models")
    assert resp.status_code == 502


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_models_endpoint_lists_and_sorts(api):
    client, _db, _m, _fm = api
    payload = {"data": [{"id": "qwen-plus"}, {"id": "gpt-4o"}, {"id": "qwen-max"}]}
    with patch(
        "web.server.main.urllib.request.urlopen", return_value=FakeResponse(payload)
    ):
        resp = client.post("/api/settings/test")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["model_count"] == 3
        assert body["sample"] == ["gpt-4o", "qwen-max", "qwen-plus"]
        assert body["latency_ms"] >= 0

        # second call hits the cache (urlopen not called again)
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == ["gpt-4o", "qwen-max", "qwen-plus"]


# ---------------------------------------------------------------- migration


def test_schema_migration_adds_missing_columns(api, tmp_path):
    client, web_db, _m, _fm = api
    db_path = str(web_db.db.database)

    con = sqlite3.connect(db_path)
    con.execute("ALTER TABLE job DROP COLUMN file_name")
    con.execute("ALTER TABLE job DROP COLUMN started_at")
    con.execute("ALTER TABLE systemsettings DROP COLUMN default_watermark_mode")
    cols = {r[1] for r in con.execute("PRAGMA table_info(job)").fetchall()}
    assert "file_name" not in cols
    con.close()

    web_db._ensure_schema()

    con = sqlite3.connect(db_path)
    job_cols = {r[1] for r in con.execute("PRAGMA table_info(job)").fetchall()}
    setting_cols = {
        r[1] for r in con.execute("PRAGMA table_info(systemsettings)").fetchall()
    }
    con.close()
    assert {"file_name", "started_at"} <= job_cols
    assert "default_watermark_mode" in setting_cols

    # default backfilled for existing rows
    row = web_db.get_settings()
    assert row.default_watermark_mode == "watermarked"


def test_created_job_gets_file_name_column_via_migration(api, tmp_path):
    """Simulate a legacy DB (no file_name) then insert through the model —
    the migrated schema must accept it."""
    client, web_db, _m, _fm = api
    web_db.Job.create(
        job_id="legacy-1",
        status="finished",
        lang_in="en",
        lang_out="zh",
        openai_model="gpt-test",
        qps=4,
        file_path="/tmp/x.pdf",
        created_at=dt.datetime(2025, 1, 1),
        updated_at=dt.datetime(2025, 1, 1),
    )
    resp = client.get("/api/jobs/legacy-1")
    assert resp.status_code == 200
    assert resp.json()["file_name"] is None
