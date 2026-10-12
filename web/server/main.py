"""BabelDOC Web Service.

This process intentionally does NOT import babeldoc: the API server stays
lightweight (~100MB RSS). Each translation job runs in a short-lived
`web.server.worker` subprocess, so the ONNX layout model and all heavy
dependencies are loaded per job and released with the process on exit.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.server.db import (
    OUTPUT_BASE_DIR,
    UPLOAD_DIR,
    Job,
    get_settings,
    init_db,
    update_settings,
    utcnow,
)
from web.server.manager import (
    WorkerManager,
    build_worker_command as _build_worker_command,
    launch_translate_worker as _launch_worker,
    mark_interrupted_jobs,
)
from web.server.retention import start_retention_thread

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
RUNTIME_DIR = BASE_DIR / "runtime"
STATIC_DIR = BASE_DIR / "static"

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "300"))
MODELS_FETCH_TIMEOUT = float(os.getenv("MODELS_FETCH_TIMEOUT", "10"))

for d in (UPLOAD_DIR, OUTPUT_BASE_DIR):
    d.mkdir(parents=True, exist_ok=True)

manager = WorkerManager(PROJECT_DIR)

# (settings.updated_at, models list, expiry) — invalidated when settings change
_models_cache: dict[str, Any] = {"key": None, "models": [], "expires": 0.0}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    interrupted = mark_interrupted_jobs()
    if interrupted:
        logger.warning("marked %d interrupted job(s) from previous run", interrupted)
    manager.start()
    start_retention_thread()
    yield
    manager.stop()


app = FastAPI(title="BabelDOC Web Service", lifespan=lifespan)

# Allow simple cross-origin usage during development (no credentials needed
# for wildcard origins).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Backwards-compatible helpers (also used by tests)
def build_worker_command(job_id: str) -> list[str]:
    return _build_worker_command(job_id)


def launch_translate_worker(job_id: str) -> Any:
    return _launch_worker(job_id, PROJECT_DIR)


# ---------------------------------------------------------------------------
# Translation jobs
# ---------------------------------------------------------------------------


def _result_view(row: Job) -> dict[str, Any]:
    return {
        "mono": bool(row.mono_pdf),
        "dual": bool(row.dual_pdf),
        "mono_no_wm": bool(row.mono_no_wm),
        "dual_no_wm": bool(row.dual_no_wm),
        "glossary": bool(row.glossary_csv),
        "total_seconds": row.total_seconds,
    }


def _validate_watermark_mode(mode: str) -> str:
    mode = (mode or "").strip() or "watermarked"
    if mode not in {"watermarked", "no_watermark", "both"}:
        raise HTTPException(status_code=400, detail="无效的水印模式")
    return mode


@app.post("/api/translate")
async def create_translate_job(
    file: UploadFile = File(...),  # noqa: B008
    lang_in: str = Form("en"),
    lang_out: str = Form("zh"),
    qps: int = Form(0),
    model: str = Form(""),
    watermark_output_mode: str = Form(""),
    no_dual: bool = Form(False),
    no_mono: bool = Form(False),
):
    # Be defensive: filename may be None in some clients; also accept content-type
    fname = (file.filename or "").lower()
    content_type = (getattr(file, "content_type", None) or "").lower()
    if not (fname.endswith(".pdf") or content_type == "application/pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    # Sensitive settings are read from DB (seeded from env on first run)
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=400,
            detail="尚未配置 OpenAI API Key，请先在「设置」页填写并保存",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB）",
        )

    lang_in = (lang_in or "").strip() or "en"
    lang_out = (lang_out or "").strip() or "zh"
    if lang_in == lang_out:
        raise HTTPException(status_code=400, detail="源语言和目标语言不能相同")
    if no_dual and no_mono:
        raise HTTPException(status_code=400, detail="双语和单语 PDF 不能同时关闭")

    wm_mode = _validate_watermark_mode(
        (watermark_output_mode or "").strip() or (settings.default_watermark_mode or "watermarked")
    )
    job_model = (model or "").strip() or settings.openai_model
    effective_qps = qps if isinstance(qps, int) and qps > 0 else int(
        getattr(settings, "default_qps", 4) or 4
    )

    job_id = uuid.uuid4().hex
    job_upload_path = UPLOAD_DIR / f"{job_id}.pdf"
    job_output_dir = OUTPUT_BASE_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    job_upload_path.write_bytes(content)

    # Persist job snapshot to DB (status starts as queued)
    Job.create(
        job_id=job_id,
        status="queued",
        progress=0.0,
        stage=None,
        stage_current=0,
        stage_total=0,
        lang_in=lang_in,
        lang_out=lang_out,
        openai_model=job_model,
        qps=effective_qps,
        watermark_output_mode=wm_mode,
        no_dual=no_dual,
        no_mono=no_mono,
        file_path=str(job_upload_path),
        file_name=Path(file.filename or f"{job_id}.pdf").name,
    )

    position = manager.enqueue(job_id)
    return {"job_id": job_id, "status": "queued", "queue_position": position}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    row = Job.get_or_none(Job.job_id == job_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    position = manager.position(job_id)
    queue_position = position if (row.status == "queued" and position >= 0) else None
    return {
        "job_id": row.job_id,
        "status": row.status,
        "progress": float(row.progress or 0.0),
        "stage": row.stage,
        "stage_current": int(row.stage_current or 0),
        "stage_total": int(row.stage_total or 0),
        "lang_in": row.lang_in,
        "lang_out": row.lang_out,
        "model": row.openai_model,
        "qps": row.qps,
        "watermark_output_mode": row.watermark_output_mode,
        "file_name": row.file_name,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "queue_position": queue_position,
        "result": _result_view(row) if row.status == "finished" else None,
        "error": row.error,
    }


@app.get("/api/jobs")
def list_jobs(limit: int = 20):
    """List recent jobs from DB (safe fields only)."""
    rows = Job.select().order_by(Job.created_at.desc()).limit(max(1, min(limit, 200)))

    def sanitize(row: Job):
        return {
            "job_id": row.job_id,
            "status": row.status,
            "progress": float(row.progress or 0.0),
            "stage": row.stage,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "lang_in": row.lang_in,
            "lang_out": row.lang_out,
            "model": row.openai_model,
            "file_name": row.file_name,
            "results": _result_view(row),
            "error": row.error,
        }

    return [sanitize(r) for r in rows]


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    row = Job.get_or_none(Job.job_id == job_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    if row.status == "running":
        raise HTTPException(status_code=409, detail="任务正在翻译中，无法删除")
    if row.status == "queued":
        manager.dequeue(job_id)

    upload_path = Path(row.file_path) if row.file_path else None
    if upload_path and upload_path.exists():
        try:
            upload_path.unlink()
        except OSError:
            logger.warning("failed to remove upload %s", upload_path)
    output_dir = OUTPUT_BASE_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    row.delete_instance()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str):
    """Re-queue a failed job with its original parameters and stored upload."""
    row = Job.get_or_none(Job.job_id == job_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务不存在")
    if row.status != "error":
        raise HTTPException(status_code=409, detail="只能重新提交失败的任务")
    upload = Path(row.file_path) if row.file_path else None
    if not upload or not upload.is_file():
        raise HTTPException(
            status_code=404, detail="原始上传文件已被清理，请重新上传 PDF"
        )

    # drop any stale outputs from the failed attempt
    output_dir = OUTPUT_BASE_DIR / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    row.status = "queued"
    row.progress = 0.0
    row.stage = None
    row.stage_current = 0
    row.stage_total = 0
    row.error = None
    row.mono_pdf = None
    row.dual_pdf = None
    row.mono_no_wm = None
    row.dual_no_wm = None
    row.glossary_csv = None
    row.total_seconds = None
    row.started_at = None
    row.updated_at = utcnow()
    row.save()

    position = manager.enqueue(job_id)
    return {"ok": True, "status": "queued", "queue_position": position}


_KIND_SUFFIX = {
    "mono": "单语",
    "dual": "双语",
    "mono_no_wm": "单语-无水印",
    "dual_no_wm": "双语-无水印",
    "glossary": "术语表",
}


def _download_filename(row: Job, kind: str) -> str:
    """Download name based on the original upload name, e.g.
    ``2506.05301v2.双语-无水印.pdf`` (falls back to the job id)."""
    stem = Path(row.file_name).stem if row.file_name else ""
    stem = stem.strip() or row.job_id
    suffix = _KIND_SUFFIX.get(kind, kind)
    ext = ".csv" if kind == "glossary" else ".pdf"
    return f"{stem}.{suffix}{ext}"


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str, kind: str):
    row = Job.get_or_none(Job.job_id == job_id)
    if not row or row.status != "finished":
        raise HTTPException(status_code=404, detail="任务未完成或不存在")
    mapping = {
        "mono": row.mono_pdf,
        "dual": row.dual_pdf,
        "mono_no_wm": row.mono_no_wm,
        "dual_no_wm": row.dual_no_wm,
        "glossary": row.glossary_csv,
    }
    target = mapping.get(kind)
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="文件不可用")
    return FileResponse(target, filename=_download_filename(row, kind))


@app.get("/api/queue")
def queue_status():
    counts = manager.counts()
    return {"max_workers": manager.max_workers, **counts}


# ---------------------------------------------------------------------------
# Settings & model configuration
# ---------------------------------------------------------------------------


def _mask_key(key: str | None) -> str | None:
    if not key:
        return None
    if len(key) <= 8:
        return "••••"
    return f"{key[:3]}••••{key[-4:]}"


def _settings_view() -> dict[str, Any]:
    s = get_settings()
    return {
        "base_url": s.openai_base_url,
        "model": s.openai_model,
        "default_qps": int(getattr(s, "default_qps", 4) or 4),
        "default_watermark_mode": s.default_watermark_mode or "watermarked",
        "key_configured": bool(s.openai_api_key),
        "key_hint": _mask_key(s.openai_api_key),
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


@app.get("/api/settings")
def get_public_settings():
    """Non-sensitive view for the frontend; the API key is masked."""
    return _settings_view()


@app.post("/api/settings")
def update_server_settings(payload: dict[str, Any] = Body(...)):  # noqa: B008
    """Update server settings.

    Accepts keys: openai_api_key (or api_key), openai_base_url (or base_url),
    openai_model (or model), default_qps, default_watermark_mode.
    Empty strings are treated as "leave unchanged" for base_url/model/qps;
    the API key is only updated when a non-empty value is provided.
    """
    api_key = payload.get("openai_api_key") or payload.get("api_key") or None
    base_url = payload.get("openai_base_url") or payload.get("base_url") or None
    model = payload.get("openai_model") or payload.get("model") or None
    _ = update_settings(
        openai_api_key=api_key if api_key else None,
        openai_base_url=base_url,
        openai_model=model,
        default_qps=payload.get("default_qps"),
        default_watermark_mode=payload.get("default_watermark_mode"),
    )
    _models_cache["expires"] = 0.0  # invalidate
    return {"ok": True, "settings": _settings_view()}


def _fetch_models_from_provider(
    base_url: str | None = None, api_key: str | None = None
) -> tuple[list[str], int]:
    """GET {base_url}/models; uses stored credentials unless overridden.

    Returns (sorted model ids, latency in ms). Raises RuntimeError with a
    readable message on failure.
    """
    s = get_settings()
    api_key = api_key or s.openai_api_key
    base = (base_url or s.openai_base_url or "").rstrip("/")
    if not api_key:
        raise RuntimeError("尚未配置 API Key")
    if not base:
        raise RuntimeError("尚未配置 Base URL")
    url = f"{base}/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=MODELS_FETCH_TIMEOUT) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = ""
        raise RuntimeError(f"模型服务返回 HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"无法连接模型服务: {exc}") from exc

    latency_ms = int((time.monotonic() - started) * 1000)
    models: list[str] = []
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for item in data:
            mid = item.get("id") if isinstance(item, dict) else item
            if isinstance(mid, str) and mid:
                models.append(mid)
    elif isinstance(payload, dict) and isinstance(payload.get("models"), list):
        models = [str(m) for m in payload["models"] if m]
    return sorted(set(models)), latency_ms


def _models_cached(force: bool = False) -> tuple[list[str], int]:
    settings = get_settings()
    key = settings.updated_at.isoformat() if settings.updated_at else "na"
    now = time.monotonic()
    if (
        not force
        and _models_cache["key"] == key
        and now < _models_cache["expires"]
    ):
        return _models_cache["models"], 0  # 0 => cache hit
    models, latency = _fetch_models_from_provider()
    _models_cache.update({"key": key, "models": models, "expires": now + MODELS_CACHE_TTL})
    return models, latency


@app.get("/api/models")
def list_models(force: bool = False):
    """List model ids available on the configured provider (cached)."""
    try:
        models, _latency = _models_cached(force=force)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": models, "count": len(models)}


@app.post("/api/settings/test")
def test_settings(payload: dict[str, Any] | None = Body(default=None)):  # noqa: B008
    """Connectivity test: fetch the model list.

    Accepts optional {"base_url", "api_key"} to test values the user just
    typed into the form (before saving); falls back to the stored settings.
    """
    payload = payload or {}
    base_url = (payload.get("base_url") or "").strip() or None
    api_key = (payload.get("api_key") or "").strip() or None
    try:
        models, latency_ms = _fetch_models_from_provider(base_url, api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "model_count": len(models),
        "sample": models[:8],
    }


# Serve static frontend
app.mount(
    "/",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="static",
)
