/* Shared helpers for the BabelDOC web UI. No external dependencies. */
"use strict";

/** Fetch wrapper that normalizes FastAPI HTTPException errors. */
async function api(path, opts) {
  opts = opts || {};
  const init = { method: opts.method || "GET" };
  if (opts.form) init.body = opts.form;
  if (opts.json !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
  }
  const resp = await fetch(path, init);
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {
    /* non-JSON response */
  }
  if (!resp.ok) {
    const detail =
      data && (typeof data.detail === "string" ? data.detail : data.detail);
    const err = new Error(
      typeof detail === "string" && detail ? detail : `请求失败 (HTTP ${resp.status})`
    );
    err.status = resp.status;
    throw err;
  }
  return data;
}

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtBytes(n) {
  if (!n && n !== 0) return "";
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(2) + " MB";
}

function pad2(n) {
  return String(n).padStart(2, "0");
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(
    d.getHours()
  )}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

function fmtDuration(sec) {
  sec = Number(sec);
  if (!isFinite(sec) || sec <= 0) return "—";
  if (sec < 60) return sec.toFixed(1) + " 秒";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m} 分 ${s} 秒`;
}

const STATUS = {
  queued: { label: "排队中", cls: "badge--queued" },
  running: { label: "翻译中", cls: "badge--running" },
  finished: { label: "已完成", cls: "badge--finished" },
  error: { label: "失败", cls: "badge--error" },
};
function statusMeta(status) {
  return STATUS[status] || { label: status || "未知", cls: "" };
}

/** BabelDOC pipeline stages, in order, with display labels. */
const STAGES = [
  ["DetectScannedFile", "检测扫描文件"],
  ["Parse Page Layout", "解析页面版面"],
  ["Parse Table", "解析表格"],
  ["Parse Paragraphs", "解析段落"],
  ["Parse Formulas and Styles", "解析公式与样式"],
  ["Automatic Term Extraction", "自动提取术语"],
  ["Translate Paragraphs", "翻译段落"],
  ["Typesetting", "排版"],
  ["Add Fonts", "嵌入字体"],
  ["Generate drawing instructions", "生成 PDF"],
  ["Save PDF", "保存 PDF"],
];

const EXTRA_STAGE_LABELS = {
  "Parse PDF and Create Intermediate Representation": "解析 PDF 生成中间表示",
  "Remove Char Descent": "移除字符降部",
  "Add Debug Information": "添加调试信息",
};

function stageInfo(stage) {
  if (!stage) return null;
  const idx = STAGES.findIndex((s) => s[0] === stage);
  if (idx >= 0) return { index: idx, label: STAGES[idx][1], known: true };
  return { index: -1, label: EXTRA_STAGE_LABELS[stage] || stage, known: false };
}

const KIND_LABELS = [
  ["dual", "dual", "双语 PDF"],
  ["mono", "mono", "单语 PDF"],
  ["dual_no_wm", "dual_no_wm", "双语 PDF（无水印）"],
  ["mono_no_wm", "mono_no_wm", "单语 PDF（无水印）"],
  ["glossary", "glossary", "术语表 CSV"],
];

function downloadUrl(jobId, kind) {
  return `/api/jobs/${encodeURIComponent(jobId)}/download?kind=${kind}`;
}

function toast(msg, type) {
  let zone = document.querySelector(".toast-zone");
  if (!zone) {
    zone = document.createElement("div");
    zone.className = "toast-zone";
    document.body.appendChild(zone);
  }
  const el = document.createElement("div");
  el.className = "toast" + (type === "err" ? " toast--err" : type === "ok" ? " toast--ok" : "");
  el.textContent = msg;
  zone.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 350);
  }, 2600);
}

/** Common language options for the translate form. */
const LANGS = [
  ["en", "英语"],
  ["zh", "中文"],
  ["ja", "日语"],
  ["ko", "韩语"],
  ["es", "西班牙语"],
  ["fr", "法语"],
  ["de", "德语"],
  ["ru", "俄语"],
  ["pt", "葡萄牙语"],
  ["it", "意大利语"],
  ["ar", "阿拉伯语"],
  ["hi", "印地语"],
  ["th", "泰语"],
  ["vi", "越南语"],
  ["id", "印尼语"],
  ["tr", "土耳其语"],
  ["nl", "荷兰语"],
  ["pl", "波兰语"],
  ["uk", "乌克兰语"],
];

function fillLangOptions(select, selected) {
  select.innerHTML = LANGS.map(
    ([code, label]) => `<option value="${code}">${label}（${code}）</option>`
  ).join("");
  if (selected) select.value = selected;
}

/**
 * Populate a <select> with model ids.
 * Returns true when the list was successfully fetched.
 */
async function fillModelOptions(select, currentModel, allowCustom) {
  select.innerHTML = "";
  try {
    const data = await api("/api/models");
    const models = (data && data.models) || [];
    const seen = new Set();
    const opts = [];
    if (currentModel && !models.includes(currentModel)) {
      opts.push(`<option value="${escapeHtml(currentModel)}">${escapeHtml(currentModel)}（当前配置）</option>`);
      seen.add(currentModel);
    }
    for (const m of models) {
      if (seen.has(m)) continue;
      seen.add(m);
      opts.push(`<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`);
    }
    select.innerHTML = opts.join("");
    if (currentModel) select.value = currentModel;
    return true;
  } catch (e) {
    if (currentModel) {
      select.innerHTML = `<option value="${escapeHtml(currentModel)}">${escapeHtml(currentModel)}</option>`;
    }
    if (allowCustom) select.insertAdjacentHTML(
      "beforeend",
      '<option value="__custom__">自定义模型…</option>'
    );
    return false;
  }
}
