import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import Body  # type: ignore
from fastapi import FastAPI  # type: ignore
from fastapi import File  # type: ignore
from fastapi import Form  # type: ignore
from fastapi import HTTPException  # type: ignore
from fastapi import UploadFile  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from web.server.db import Job
from web.server.db import get_settings
from web.server.db import init_db
from web.server.db import update_settings

app = FastAPI(title="BabelDOC Web Service")

# Allow simple cross-origin usage during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_BASE_DIR = RUNTIME_DIR / "outputs"

for d in (UPLOAD_DIR, OUTPUT_BASE_DIR):
    d.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def _startup():
    # Initialize database (SQLite) and seed settings from environment if present
    init_db()


def build_worker_command(job_id: str) -> list[str]:
    return [sys.executable, "-m", "web.server.worker", job_id]


def launch_translate_worker(job_id: str) -> subprocess.Popen:
    return subprocess.Popen(  # noqa: S603
        build_worker_command(job_id),
        cwd=PROJECT_DIR,
        start_new_session=True,
    )


@app.post("/api/translate")
async def create_translate_job(
    file: UploadFile = File(...),  # noqa: B008
    lang_in: str = Form("en"),
    lang_out: str = Form("zh"),
    qps: int = Form(0),
    watermark_output_mode: str = Form("watermarked"),
    no_dual: bool = Form(False),
    no_mono: bool = Form(False),
):
    # Be defensive: filename may be None in some clients; also accept content-type
    fname = (file.filename or "").lower()
    if not (
        fname.endswith(".pdf")
        or (getattr(file, "content_type", None) or "").lower() == "application/pdf"
    ):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    # Load sensitive settings from DB (seeded from env on first run)
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="缺少后端配置的 OpenAI API Key")

    job_id = uuid.uuid4().hex
    job_upload_path = UPLOAD_DIR / f"{job_id}.pdf"
    job_output_dir = OUTPUT_BASE_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    job_upload_path.write_bytes(content)

    effective_qps = (
        int(qps)
        if (isinstance(qps, int) and qps > 0)
        else int(getattr(settings, "default_qps", 4) or 4)
    )

    # Persist job snapshot to DB
    Job.create(
        job_id=job_id,
        status="running",
        progress=0.0,
        stage=None,
        stage_current=0,
        stage_total=0,
        lang_in=lang_in,
        lang_out=lang_out,
        openai_model=settings.openai_model,
        qps=effective_qps,
        watermark_output_mode=watermark_output_mode,
        no_dual=no_dual,
        no_mono=no_mono,
        file_path=str(job_upload_path),
    )

    try:
        launch_translate_worker(job_id)
    except Exception as exc:
        Job.update(status="error", error=str(exc)).where(Job.job_id == job_id).execute()
        raise HTTPException(status_code=500, detail="启动翻译 worker 失败") from exc

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    db_job = Job.get_or_none(Job.job_id == job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="任务不存在")
    data = getattr(db_job, "__data__", {})
    status = data.get("status")
    return {
        "status": status,
        "progress": float(data.get("progress") or 0.0),
        "stage": data.get("stage"),
        "stage_current": int(data.get("stage_current") or 0),
        "stage_total": int(data.get("stage_total") or 0),
        "result": (
            {
                "mono_pdf": data.get("mono_pdf"),
                "dual_pdf": data.get("dual_pdf"),
                "mono_no_wm": data.get("mono_no_wm"),
                "dual_no_wm": data.get("dual_no_wm"),
                "glossary_csv": data.get("glossary_csv"),
                "total_seconds": data.get("total_seconds"),
            }
            if status == "finished"
            else None
        ),
        "error": data.get("error"),
    }


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str, kind: str):
    db_job = Job.get_or_none(Job.job_id == job_id)
    if not db_job or db_job.status != "finished":
        raise HTTPException(status_code=404, detail="任务未完成或不存在")
    mapping = {
        "mono": db_job.mono_pdf,
        "dual": db_job.dual_pdf,
        "mono_no_wm": db_job.mono_no_wm,
        "dual_no_wm": db_job.dual_no_wm,
        "glossary": db_job.glossary_csv,
    }
    target = mapping.get(kind)
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="文件不可用")
    return FileResponse(target, filename=Path(target).name)


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    """List recent jobs from DB (safe fields only)."""
    rows = Job.select().order_by(Job.created_at.desc()).limit(max(1, min(limit, 200)))

    def sanitize(row: Job):
        data = getattr(row, "__data__", {})
        created = data.get("created_at")
        return {
            "job_id": data.get("job_id"),
            "status": data.get("status"),
            "progress": float(data.get("progress") or 0.0),
            "created_at": created.isoformat() if created else None,
            "lang_in": data.get("lang_in"),
            "lang_out": data.get("lang_out"),
            "results": {
                "mono": bool(data.get("mono_pdf")),
                "dual": bool(data.get("dual_pdf")),
                "mono_no_wm": bool(data.get("mono_no_wm")),
                "dual_no_wm": bool(data.get("dual_no_wm")),
                "glossary": bool(data.get("glossary_csv")),
            },
        }

    return [sanitize(r) for r in rows]


@app.get("/api/settings")
def get_public_settings():
    """Expose only non-sensitive settings to frontend."""
    s = get_settings()
    return {
        "default_qps": int(getattr(s, "default_qps", 4) or 4),
        # Deliberately do NOT expose API key or base_url/model here
    }


@app.post("/api/settings")
def update_server_settings(payload: dict[str, Any] = Body(...)):  # noqa: B008
    """Update server-only settings. Should be called from admin or ops tools.

    Accepts keys: openai_api_key, openai_base_url, openai_model, default_qps
    """
    _ = update_settings(
        openai_api_key=payload.get("openai_api_key"),
        openai_base_url=payload.get("openai_base_url"),
        openai_model=payload.get("openai_model"),
        default_qps=payload.get("default_qps"),
    )
    s = get_settings()
    updated = getattr(s, "__data__", {}).get("updated_at")
    return {"ok": True, "updated_at": (updated.isoformat() if updated else None)}


# Serve static frontend
STATIC_DIR = BASE_DIR.parent / "web" / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
