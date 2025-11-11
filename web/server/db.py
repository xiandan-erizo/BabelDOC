from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any

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

db = SqliteDatabase(DB_PATH.as_posix())


class BaseModel(Model):
    class Meta:
        database = db


class SystemSettings(BaseModel):
    id = IntegerField(primary_key=True)
    openai_api_key = TextField(null=True)
    openai_base_url = CharField(default="https://api.openai.com/v1")
    openai_model = CharField(default="gpt-4o-mini")
    default_qps = IntegerField(default=4)
    updated_at = DateTimeField(default=dt.datetime.utcnow)


class Job(BaseModel):
    id = IntegerField(primary_key=True)
    job_id = CharField(unique=True)
    status = CharField(default="running")
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
    mono_pdf = TextField(null=True)
    dual_pdf = TextField(null=True)
    mono_no_wm = TextField(null=True)
    dual_no_wm = TextField(null=True)
    glossary_csv = TextField(null=True)
    total_seconds = FloatField(null=True)
    error = TextField(null=True)

    created_at = DateTimeField(default=dt.datetime.utcnow)
    updated_at = DateTimeField(default=dt.datetime.utcnow)


def init_db() -> None:
    """Initialize tables and seed a settings row.

    - Creates tables if not exist
    - Seeds SystemSettings(id=1) and imports OPENAI_API_KEY from env once if set
    """
    db.connect(reuse_if_open=True)
    db.create_tables([SystemSettings, Job])

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
        updates["updated_at"] = dt.datetime.utcnow()
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
) -> SystemSettings:
    updates: Dict[str, Any] = {}
    if openai_api_key is not None:
        updates["openai_api_key"] = openai_api_key
    if openai_base_url is not None:
        updates["openai_base_url"] = openai_base_url
    if openai_model is not None:
        updates["openai_model"] = openai_model
    if default_qps is not None:
        try:
            updates["default_qps"] = int(default_qps)
        except Exception:
            pass
    if updates:
        updates["updated_at"] = dt.datetime.utcnow()
        SystemSettings.update(**updates).where(SystemSettings.id == 1).execute()
    return get_settings()