# BabelDOC Web UI

自托管的 PDF 翻译服务：FastAPI + SQLite + 原生 HTML/JS 前端，任务由独立子进程执行。

## 架构（内存不常驻）

```
uvicorn (常驻, ~100MB, 不 import babeldoc)
   │
   ├── POST /api/translate → Job 入库 (status=queued) → 入队
   │
WorkerManager (后台线程)
   │  同时最多 MAX_CONCURRENT_WORKERS 个 (默认 1)
   │  - 按顺序 spawn: python -m web.server.worker <job_id>
   │  - 轮询 poll() 回收子进程（无僵尸进程）
   │  - 子进程崩溃/超时 → 任务标记失败
   ▼
worker 子进程 (每次任务新进程, ~几百 MB)
   │  加载布局 ONNX 模型 + babeldoc
   │  进度实时写 SQLite (WAL)
   │  任务结束 → 进程退出 → 内存全部归还系统
   ▼
SQLite (web/runtime/babeldoc.db) ← 前端轮询进度
```

服务重启时，残留的 `queued/running` 任务会自动标记为「已中断」，重新提交即可。

## API

| Method | Path | 说明 |
| --- | --- | --- |
| POST | `/api/translate` | multipart 上传：`file, lang_in, lang_out, qps, model, watermark_output_mode, no_dual, no_mono` |
| GET | `/api/jobs/{id}` | 单任务状态/进度/结果 |
| GET | `/api/jobs?limit=` | 任务列表 |
| DELETE | `/api/jobs/{id}` | 删除任务及输出文件（运行中不可删） |
| GET | `/api/jobs/{id}/download?kind=` | kind: `mono, dual, mono_no_wm, dual_no_wm, glossary` |
| GET | `/api/settings` | 非敏感配置视图（密钥仅显示掩码） |
| POST | `/api/settings` | 更新 base_url / api_key / model / default_qps / default_watermark_mode |
| GET | `/api/models` | 代理拉取模型服务的 `/models` 列表（带缓存） |
| POST | `/api/settings/test` | 连通性测试（可传入未保存的 base_url/api_key） |
| GET | `/api/queue` | 当前运行/排队数量 |

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` / `DEFAULT_QPS` | — | 首次建库时种子化到 SQLite |
| `MAX_CONCURRENT_WORKERS` | `1` | 并发翻译子进程数 |
| `WORKER_TIMEOUT_SECONDS` | `1800` | 单任务超时（超时 kill） |
| `JOB_RETENTION_DAYS` | `30` | 已结束/失败任务（含文件）自动清理的天数，`0` 关闭 |
| `MAX_UPLOAD_BYTES` | `209715200` | 上传大小上限 (200 MB) |

## 本地开发

```bash
python -m venv .venv && .venv/bin/pip install -e . fastapi uvicorn python-multipart peewee
.venv/bin/python -m uvicorn web.server.main:app --port 8000
.venv/bin/python -m pytest tests/ -q
```

Docker 部署见根目录 `docker-compose.yml`（`init: true` 保证孤儿进程回收）。
