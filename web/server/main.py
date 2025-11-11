import asyncio
import os
import uuid
import datetime as dt
from pathlib import Path
from typing import Optional, TypedDict, Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import FileResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore

import babeldoc.format.pdf.high_level as high_level
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator, set_translate_rate_limiter
from babeldoc.docvision.doclayout import DocLayoutModel
from web.server.db import init_db, get_settings, update_settings, Job


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
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_BASE_DIR = RUNTIME_DIR / "outputs"

for d in (UPLOAD_DIR, OUTPUT_BASE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# In-memory job store (simple demo). For production, use a database or Redis.

class JobResult(TypedDict, total=False):
    mono_pdf: Optional[str]
    dual_pdf: Optional[str]
    mono_no_wm: Optional[str]
    dual_no_wm: Optional[str]
    glossary_csv: Optional[str]
    total_seconds: Optional[float]


class JobState(TypedDict):
    status: str
    progress: float
    stage: Optional[str]
    stage_current: int
    stage_total: int
    result: Optional[JobResult]
    error: Optional[str]


jobs: dict[str, JobState] = {}


@app.on_event("startup")
def _startup():
    # Initialize cache folders and optionally warmup font assets
    # Initialize database (SQLite) and seed settings from environment if present
    init_db()
    high_level.init()
    # Do not warmup assets synchronously at startup to avoid blocking service start.
    # Assets will be loaded on demand during translation.


def _parse_wm_mode(mode: str) -> WatermarkOutputMode:
    mode = (mode or "watermarked").lower()
    if mode == "no_watermark":
        return WatermarkOutputMode.NoWatermark
    if mode == "both":
        return WatermarkOutputMode.Both
    return WatermarkOutputMode.Watermarked


@app.post("/api/translate")
async def create_translate_job(
    file: UploadFile = File(...),
    lang_in: str = Form("en"),
    lang_out: str = Form("zh"),
    qps: int = Form(0),
    watermark_output_mode: str = Form("watermarked"),
    no_dual: bool = Form(False),
    no_mono: bool = Form(False),
):
    # Be defensive: filename may be None in some clients; also accept content-type
    fname = (file.filename or "").lower()
    if not (fname.endswith(".pdf") or (getattr(file, "content_type", None) or "").lower() == "application/pdf"):
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

    # Translator & rate limit (from server-side settings)
    translator = OpenAITranslator(
        lang_in=lang_in,
        lang_out=lang_out,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        enable_json_mode_if_requested=True,
    )
    term_extraction_translator = translator
    effective_qps = int(qps) if (isinstance(qps, int) and qps > 0) else int(getattr(settings, "default_qps", 4) or 4)
    set_translate_rate_limiter(effective_qps)

    # Doc layout model
    doc_layout_model = DocLayoutModel.load_onnx()

    wm_mode = _parse_wm_mode(watermark_output_mode)

    config = TranslationConfig(
        translator=translator,
        term_extraction_translator=term_extraction_translator,
        input_file=str(job_upload_path),
        lang_in=lang_in,
        lang_out=lang_out,
        doc_layout_model=doc_layout_model,
        qps=effective_qps,
        output_dir=str(job_output_dir),
        no_dual=no_dual,
        no_mono=no_mono,
        watermark_output_mode=wm_mode,
        report_interval=0.2,
    )

    jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "stage": None,
        "stage_current": 0,
        "stage_total": 0,
        "result": None,
        "error": None,
    }

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

    async def _run():
        try:
            async for event in high_level.async_translate(config):
                et = event.get("type")
                if et == "progress_update":
                    jobs[job_id]["progress"] = float(event.get("overall_progress", 0.0))
                    jobs[job_id]["stage"] = event.get("stage")
                    jobs[job_id]["stage_current"] = event.get("stage_current")
                    jobs[job_id]["stage_total"] = event.get("stage_total")
                    # Persist progress to DB
                    Job.update(
                        progress=jobs[job_id]["progress"],
                        stage=jobs[job_id]["stage"],
                        stage_current=jobs[job_id]["stage_current"],
                        stage_total=jobs[job_id]["stage_total"],
                        updated_at=dt.datetime.utcnow(),
                    ).where(Job.job_id == job_id).execute()
                elif et == "progress_end":
                    jobs[job_id]["progress"] = float(event.get("overall_progress", 100.0))
                    Job.update(progress=jobs[job_id]["progress"], updated_at=dt.datetime.utcnow()).where(Job.job_id == job_id).execute()
                elif et == "finish":
                    res = event.get("translate_result")
                    jobs[job_id]["status"] = "finished"
                    jobs[job_id]["progress"] = 100.0
                    jobs[job_id]["result"] = {
                        "mono_pdf": str(res.mono_pdf_path) if getattr(res, "mono_pdf_path", None) else None,
                        "dual_pdf": str(res.dual_pdf_path) if getattr(res, "dual_pdf_path", None) else None,
                        "mono_no_wm": str(res.no_watermark_mono_pdf_path)
                        if getattr(res, "no_watermark_mono_pdf_path", None)
                        else None,
                        "dual_no_wm": str(res.no_watermark_dual_pdf_path)
                        if getattr(res, "no_watermark_dual_pdf_path", None)
                        else None,
                        "glossary_csv": str(res.auto_extracted_glossary_path)
                        if getattr(res, "auto_extracted_glossary_path", None)
                        else None,
                        "total_seconds": getattr(res, "total_seconds", None),
                    }
                    # Persist final result
                    _res = jobs[job_id].get("result") or {}
                    Job.update(
                        status="finished",
                        progress=100.0,
                        mono_pdf=_res.get("mono_pdf"),
                        dual_pdf=_res.get("dual_pdf"),
                        mono_no_wm=_res.get("mono_no_wm"),
                        dual_no_wm=_res.get("dual_no_wm"),
                        glossary_csv=_res.get("glossary_csv"),
                        total_seconds=_res.get("total_seconds"),
                        updated_at=dt.datetime.utcnow(),
                    ).where(Job.job_id == job_id).execute()
                elif et == "error":
                    jobs[job_id]["status"] = "error"
                    jobs[job_id]["error"] = str(event.get("error"))
                    Job.update(status="error", error=jobs[job_id]["error"], updated_at=dt.datetime.utcnow()).where(Job.job_id == job_id).execute()
                    break
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)
            Job.update(status="error", error=jobs[job_id]["error"], updated_at=dt.datetime.utcnow()).where(Job.job_id == job_id).execute()

    asyncio.create_task(_run())
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        # Fallback to DB if not in memory (e.g., after server restart)
        db_job = Job.get_or_none(Job.job_id == job_id)
        if not db_job:
            raise HTTPException(status_code=404, detail="任务不存在")
        # Map DB fields to response format
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
    return job


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str, kind: str):
    job = jobs.get(job_id)
    if not job or job.get("status") != "finished":
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
    # result is a TypedDict (JobResult); cast the optional value for type safety
    from typing import cast
    res_mem: JobResult = cast(JobResult, job.get("result") or {})
    mapping = {
        "mono": res_mem.get("mono_pdf"),
        "dual": res_mem.get("dual_pdf"),
        "mono_no_wm": res_mem.get("mono_no_wm"),
        "dual_no_wm": res_mem.get("dual_no_wm"),
        "glossary": res_mem.get("glossary_csv"),
    }
    target = mapping.get(kind)
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="文件不可用")
    return FileResponse(target, filename=Path(target).name)


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    """List recent jobs from DB (safe fields only)."""
    rows = (
        Job.select()
        .order_by(Job.created_at.desc())
        .limit(max(1, min(limit, 200)))
    )
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
def update_server_settings(payload: Dict[str, Any] = Body(...)):
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