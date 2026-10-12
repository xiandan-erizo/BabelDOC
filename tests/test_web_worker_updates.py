import datetime as dt
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture()
def worker_db(monkeypatch, tmp_path):
    from web.server import db as web_db

    test_db_path = tmp_path / "babeldoc.db"
    web_db.db.init(str(test_db_path))
    web_db.db.connect(reuse_if_open=True)
    web_db.db.create_tables([web_db.SystemSettings, web_db.Job])
    web_db.SystemSettings.create(
        id=1,
        openai_api_key="sk-test",
        openai_base_url="https://example.test/v1",
        openai_model="gpt-test",
        default_qps=4,
    )
    web_db.Job.create(
        job_id="job-123",
        status="running",
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
        file_path=str(tmp_path / "uploads" / "job-123.pdf"),
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    monkeypatch.setattr(web_db, "DB_PATH", test_db_path)
    yield web_db
    web_db.db.drop_tables([web_db.Job, web_db.SystemSettings])
    web_db.db.close()


def test_worker_persists_progress(worker_db):
    from web.server.worker import persist_progress

    persist_progress(
        "job-123",
        {
            "overall_progress": 42.5,
            "stage": "Translate Paragraphs",
            "stage_current": 17,
            "stage_total": 40,
        },
    )

    job = worker_db.Job.get(worker_db.Job.job_id == "job-123")
    assert job.status == "running"
    assert job.progress == 42.5
    assert job.stage == "Translate Paragraphs"
    assert job.stage_current == 17
    assert job.stage_total == 40


def test_worker_persists_finished_result(worker_db, tmp_path):
    from web.server.worker import persist_finish

    result = SimpleNamespace(
        mono_pdf_path=tmp_path / "mono.pdf",
        dual_pdf_path=tmp_path / "dual.pdf",
        no_watermark_mono_pdf_path=None,
        no_watermark_dual_pdf_path=tmp_path / "dual.no-wm.pdf",
        auto_extracted_glossary_path=tmp_path / "glossary.csv",
        total_seconds=12.34,
    )

    persist_finish("job-123", result)

    job = worker_db.Job.get(worker_db.Job.job_id == "job-123")
    assert job.status == "finished"
    assert job.progress == 100.0
    assert job.mono_pdf == str(tmp_path / "mono.pdf")
    assert job.dual_pdf == str(tmp_path / "dual.pdf")
    assert job.mono_no_wm is None
    assert job.dual_no_wm == str(tmp_path / "dual.no-wm.pdf")
    assert job.glossary_csv == str(tmp_path / "glossary.csv")
    assert job.total_seconds == 12.34


def test_worker_persists_error(worker_db):
    from web.server.worker import update_job_error

    update_job_error("job-123", "boom")

    job = worker_db.Job.get(worker_db.Job.job_id == "job-123")
    assert job.status == "error"
    assert job.error == "boom"
