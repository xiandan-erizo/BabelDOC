from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, Optional

from peewee import (
    SqliteDatabase,
    Model,
    IntegerField,
    CharField,
    FloatField,
    BooleanField,
    TextField,
    DateTimeField,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = BASE_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RUNTIME_DIR / "babeldoc.db"

# WAL + busy_timeout: the API server and the worker subprocess both write to
# this SQLite file (the worker persists progress several times per second).
# WAL allows concurrent readers while a writer is active; busy_timeout avoids
# spurious "database is locked" errors under short write contention.
db = SqliteDatabase(
    DB_PATH.as_posix(),
    pragmas={"journal_mode": "wal", "busy_timeout": 10000},
)


def utcnow() -> dt.datetime:
    """Naive UTC timestamp (same format as legacy rows, no deprecation)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class BaseModel(Model):
    class Meta:
        database = db


class SystemSettings(BaseModel):
    id = IntegerField(primary_key=True)
    openai_api_key = TextField(null=True)
    openai_base_url = CharField(default="https://api.openai.com/v1")
    openai_model = CharField(default="gpt-4o-mini")
    default_qps = IntegerField(default=4)
    default_watermark_mode = CharField(default="watermarked")
    updated_at = DateTimeField(default=utcnow)


class Job(BaseModel):
    id = IntegerField(primary_key=True)
    job_id = CharField(unique=True)
    status = CharField(default="queued")
    progress = FloatField(default=0.0)
    stage = CharField(null=True)
    stage_current = IntegerField(default=0)
    stage_total = IntegerField(default=0)

    # input config snapshot
    lang_in = CharField()
    lang_out = CharField()
    openai_model = CharField()
    qps = IntegerField(default=4)
    watermark_output_mode = CharField(default="watermarked")
    no_dual = BooleanField(default=False)
    no_mono = BooleanField(default=False)

    # file paths
    file_path = TextField()  # uploaded PDF
    file_name = CharField(null=True)  # original upload filename (display only)
    mono_pdf = TextField(null=True)
    dual_pdf = TextField(null=True)
    mono_no_wm = TextField(null=True)
    dual_no_wm = TextField(null=True)
    glossary_csv = TextField(null=True)
    total_seconds = FloatField(null=True)
    error = TextField(null=True)

    created_at = DateTimeField(default=utcnow)
    started_at = DateTimeField(null=True)
    updated_at = DateTimeField(default=utcnow)


# ---------------------------------------------------------------------------
# Additive schema migration for databases created by older versions.
# SQLite has no "ADD COLUMN IF NOT EXISTS", so check PRAGMA table_info first.
# ---------------------------------------------------------------------------

_SCHEMA_MIGRATIONS: list[tuple[str, str, str]] = [
    # (table, column, DDL)
    ("job", "file_name", "TEXT"),
    ("job", "started_at", "DATETIME"),
    ("systemsettings", "default_watermark_mode", "VARCHAR(64) DEFAULT 'watermarked'"),
]


def _table_columns(table: str) -> set[str]:
    cursor = db.execute_sql(f"PRAGMA table_info({table})")  # noqa: S608
    return {row[1] for row in cursor.fetchall()}


def _ensure_schema() -> None:
    for table, column, ddl in _SCHEMA_MIGRATIONS:
        if column not in _table_columns(table):
            db.execute_sql(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}')  # noqa: S608


def init_db() -> None:
    """Initialize tables, run additive migrations and seed a settings row.

    - Creates tables if not exist
    - Adds columns introduced by newer versions (idempotent)
    - Seeds SystemSettings(id=1) and imports OPENAI_* from env once if set
    """
    db.connect(reuse_if_open=True)
    db.create_tables([SystemSettings, Job])
    _ensure_schema()

    # Ensure a single settings row exists (id=1)
    settings, created = SystemSettings.get_or_create(id=1)
    if created:
        # seed env variables once
        env_key = os.getenv("OPENAI_API_KEY") or None
        env_base = os.getenv("OPENAI_BASE_URL") or None
        env_model = os.getenv("OPENAI_MODEL") or None
        env_qps = os.getenv("DEFAULT_QPS") or None
        updates: Dict[str, Any] = {}
        if env_key is not None:
            updates["openai_api_key"] = env_key
        if env_base:
            updates["openai_base_url"] = env_base
        if env_model:
            updates["openai_model"] = env_model
        if env_qps:
            try:
                updates["default_qps"] = int(env_qps)
            except Exception:
                pass
        updates["updated_at"] = utcnow()
        if updates:
            SystemSettings.update(**updates).where(SystemSettings.id == 1).execute()


def get_settings() -> SystemSettings:
    return SystemSettings.get(SystemSettings.id == 1)


def update_settings(
    *,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: Optional[str] = None,
    default_qps: Optional[int] = None,
    default_watermark_mode: Optional[str] = None,
) -> SystemSettings:
    updates: Dict[str, Any] = {}
    if openai_api_key is not None:
        updates["openai_api_key"] = openai_api_key
    if openai_base_url is not None and openai_base_url.strip():
        updates["openai_base_url"] = openai_base_url.strip()
    if openai_model is not None and openai_model.strip():
        updates["openai_model"] = openai_model.strip()
    if default_qps is not None:
        try:
            updates["default_qps"] = int(default_qps)
        except Exception:
            pass
    if default_watermark_mode is not None and default_watermark_mode in (
        "watermarked",
        "no_watermark",
        "both",
    ):
        updates["default_watermark_mode"] = default_watermark_mode
    if updates:
        updates["updated_at"] = utcnow()
        SystemSettings.update(**updates).where(SystemSettings.id == 1).execute()
    return get_settings()
