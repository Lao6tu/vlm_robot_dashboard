/* global state */
let wsConnected = false;
let snapshotFps = null;
const MAX_HISTORY = 50;

const $ = (id) => document.getElementById(id);

const camBadge   = $("cam-status");
const wsBadge    = $("ws-status");
const frameBadge = $("frame-badge");
const mjpeg      = $("mjpeg");

/* ── WebSocket ─────────────────────────────────────────────────────────────── */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/results`);

  ws.onopen = () => {
    wsConnected = true;
    wsBadge.textContent = "WebSocket  ●";
    wsBadge.className = "badge ok";
  };

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    if (data._ping) return;
    renderResult(data);
  };

  ws.onclose = () => {
    wsConnected = false;
    wsBadge.textContent = "WebSocket  ○";
    wsBadge.className = "badge err";
    setTimeout(connectWS, 3000);
  };

  ws.onerror = () => {
    wsBadge.textContent = "WebSocket  ✕";
    wsBadge.className = "badge err";
  };
}

/* ── Render result ─────────────────────────────────────────────────────────── */
function catClass(cat) {
  if (!cat) return "";
  const v = cat.trim().toUpperCase();
  if (v === "GO")      return "go";
  if (v === "CAUTION") return "caution";
  if (v === "NO-GO")   return "nogo";
  return "";
}

function renderResult(data) {
  /* timestamp */
  const ts = data._inferred_at ? new Date(data._inferred_at * 1000) : new Date();
  $("last-inferred").textContent = ts.toLocaleTimeString();

  /* frame badge */
  if (data._frame_count !== undefined) {
    const fpsLabel = snapshotFps !== null ? ` @ ${snapshotFps} FPS` : "";
    frameBadge.textContent = `${data._frame_count} frames${fpsLabel}`;
    frameBadge.classList.remove("hidden");
  }

  /* ── Current card ── */
  const errEl    = $("inf-error");
  const catEl    = $("inf-cat");
  const spaceEl  = $("inf-space");
  const actionEl = $("inf-action");
  const reasonEl = $("inf-reason");

  if (data.error) {
    catEl.textContent    = "ERR";
    catEl.className      = "inf-cat";
    spaceEl.textContent  = "";
    actionEl.textContent = "—";
    reasonEl.textContent = "—";
    errEl.textContent    = "⚠ " + escHtml(data.error);
    errEl.classList.remove("hidden");
  } else {
    errEl.classList.add("hidden");
    const status = data.status || {};
    const cat    = status.cat   || "—";
    const space  = status.free_space != null ? status.free_space + " cm" : "—";
    const cls    = catClass(cat);

    catEl.textContent    = cat;
    catEl.className      = "inf-cat" + (cls ? " " + cls : "");
    spaceEl.textContent  = "Free space: " + space;
    actionEl.textContent = data.action_advice || "—";
    reasonEl.textContent = data.reason        || "—";
  }

  /* ── Append to history ── */
  appendHistory(data, ts);
}

/* ── History log ───────────────────────────────────────────────────────────── */
const histList = $("hist-list");

function appendHistory(data, ts) {
  /* trim old entries */
  while (histList.children.length >= MAX_HISTORY) {
    histList.removeChild(histList.lastChild);
  }

  const row = document.createElement("div");
  row.className = "hist-row";

  let catText = "ERR", cls = "err", action = "", reason = "";
  if (!data.error) {
    const status = data.status || {};
    catText = status.cat || "—";
    cls     = catClass(catText);
    action  = data.action_advice || "";
    reason  = data.reason        || "";
  } else {
    action = data.error;
  }

  const space = (data.status && data.status.free_space != null)
    ? ` · ${data.status.free_space} cm` : "";

  row.innerHTML = `
    <div class="hist-cat ${cls}">${escHtml(catText)}</div>
    <div class="hist-detail">
      <span class="hist-time">${ts.toLocaleTimeString()}${escHtml(space)}</span>
      <span class="hist-action">${escHtml(action)}</span>
      <span class="hist-reason">${escHtml(reason)}</span>
    </div>`;

  histList.prepend(row);
}

function clearHistory() {
  histList.innerHTML = "";
}

/* ── Camera status polling ─────────────────────────────────────────────────── */
function pollStatus() {
  fetch("/api/status")
    .then((r) => r.json())
    .then((d) => {
      if (d.camera_ready) {
        camBadge.textContent = "Camera  ●";
        camBadge.className = "badge ok";
        $("cam-overlay").classList.add("hidden");
      } else {
        camBadge.textContent = "Camera  ○";
        camBadge.className = "badge err";
        $("cam-overlay").classList.remove("hidden");
      }
    })
    .catch(() => {
      camBadge.textContent = "Camera  —";
      camBadge.className = "badge";
    });
}

/* ── MJPEG error / reconnect ───────────────────────────────────────────────── */
mjpeg.addEventListener("error", () => {
  setTimeout(() => {
    mjpeg.src = `/stream.mjpeg?_=${Date.now()}`;
  }, 2000);
});

/* ── Helpers ───────────────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ── Boot ──────────────────────────────────────────────────────────────────── */
fetch("/api/config")
  .then((r) => r.json())
  .then((cfg) => { snapshotFps = cfg.snapshot_fps; })
  .catch(() => {});

connectWS();
pollStatus();
setInterval(pollStatus, 5000);
