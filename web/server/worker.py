from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

import babeldoc.format.pdf.high_level as high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.translator.translator import OpenAITranslator
from babeldoc.translator.translator import set_translate_rate_limiter
from web.server.db import Job
from web.server.db import get_settings
from web.server.db import init_db
from web.server.db import utcnow

logger = logging.getLogger(__name__)

EXIT_WATCHDOG_SECONDS = 15.0


def arm_exit_watchdog(rc: int, seconds: float = EXIT_WATCHDOG_SECONDS) -> None:
    """Force-exit this process after `seconds` unless it exits on its own.

    babeldoc leaves non-daemon threads (onnxruntime / joblib executors) that
    can block interpreter shutdown long after the job's terminal state is
    persisted to SQLite. This watchdog guarantees the worker's memory
    (typically 1-4 GB) is returned to the OS promptly, which also unblocks
    the server's job queue for the next document.
    """
    timer = threading.Timer(seconds, lambda: os._exit(rc))
    timer.daemon = True
    timer.start()


def parse_watermark_mode(mode: str) -> WatermarkOutputMode:
    mode = (mode or "watermarked").lower()
    if mode == "no_watermark":
        return WatermarkOutputMode.NoWatermark
    if mode == "both":
        return WatermarkOutputMode.Both
    return WatermarkOutputMode.Watermarked


def update_job_error(job_id: str, error: str) -> None:
    Job.update(
        status="error",
        error=error,
        updated_at=utcnow(),
    ).where(Job.job_id == job_id).execute()


def build_translation_config(job: Job) -> TranslationConfig:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("缺少后端配置的 OpenAI API Key")

    translator = OpenAITranslator(
        lang_in=job.lang_in,
        lang_out=job.lang_out,
        model=job.openai_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        enable_json_mode_if_requested=True,
        # DeepSeek-style thinking switch (supported since BabelDOC v0.6.4);
        # thinking mode interferes with structured translation output.
        thinking="disabled",
    )
    set_translate_rate_limiter(int(job.qps or settings.default_qps or 4))

    output_dir = Path(job.file_path).parent.parent / "outputs" / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return TranslationConfig(
        translator=translator,
        term_extraction_translator=translator,
        input_file=job.file_path,
        lang_in=job.lang_in,
        lang_out=job.lang_out,
        doc_layout_model=DocLayoutModel.load_onnx(),
        qps=int(job.qps or settings.default_qps or 4),
        output_dir=str(output_dir),
        no_dual=bool(job.no_dual),
        no_mono=bool(job.no_mono),
        watermark_output_mode=parse_watermark_mode(job.watermark_output_mode),
        report_interval=0.2,
    )


def persist_finish(job_id: str, translate_result) -> None:
    Job.update(
        status="finished",
        progress=100.0,
        mono_pdf=str(translate_result.mono_pdf_path)
        if getattr(translate_result, "mono_pdf_path", None)
        else None,
        dual_pdf=str(translate_result.dual_pdf_path)
        if getattr(translate_result, "dual_pdf_path", None)
        else None,
        mono_no_wm=str(translate_result.no_watermark_mono_pdf_path)
        if getattr(translate_result, "no_watermark_mono_pdf_path", None)
        else None,
        dual_no_wm=str(translate_result.no_watermark_dual_pdf_path)
        if getattr(translate_result, "no_watermark_dual_pdf_path", None)
        else None,
        glossary_csv=str(translate_result.auto_extracted_glossary_path)
        if getattr(translate_result, "auto_extracted_glossary_path", None)
        else None,
        total_seconds=getattr(translate_result, "total_seconds", None),
        updated_at=utcnow(),
    ).where(Job.job_id == job_id).execute()


def persist_progress(job_id: str, event: dict) -> None:
    Job.update(
        progress=float(event.get("overall_progress", 0.0)),
        stage=event.get("stage"),
        stage_current=int(event.get("stage_current") or 0),
        stage_total=int(event.get("stage_total") or 0),
        updated_at=utcnow(),
    ).where(Job.job_id == job_id).execute()


async def run_translate_job(job_id: str) -> int:
    init_db()
    high_level.init()

    job = Job.get_or_none(Job.job_id == job_id)
    if not job:
        logger.error("Job %s does not exist", job_id)
        return 1

    try:
        config = build_translation_config(job)
        async for event in high_level.async_translate(config):
            event_type = event.get("type")
            if event_type == "progress_update":
                persist_progress(job_id, event)
            elif event_type == "progress_end":
                Job.update(
                    progress=float(event.get("overall_progress", 100.0)),
                    updated_at=utcnow(),
                ).where(Job.job_id == job_id).execute()
            elif event_type == "finish":
                persist_finish(job_id, event.get("translate_result"))
            elif event_type == "error":
                update_job_error(job_id, str(event.get("error")))
                arm_exit_watchdog(1)
                return 1
    except Exception as exc:
        logger.exception("Translation worker failed for job %s", job_id)
        update_job_error(job_id, str(exc))
        arm_exit_watchdog(1)
        return 1

    arm_exit_watchdog(0)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m web.server.worker <job_id>", file=sys.stderr)
        return 2
    return asyncio.run(run_translate_job(args[0]))


if __name__ == "__main__":
    rc = main()
    # babeldoc keeps non-daemon threads alive (onnxruntime session threads,
    # executors) which can block interpreter shutdown long after the job is
    # persisted. Hard-exit so the OS reclaims the worker's memory (~1-2 GB)
    # the moment the job is done.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
