/* global state */
let wsConnected = false;
let snapshotFps = null;
const MAX_HISTORY = 50;

const $ = (id) => document.getElementById(id);

const camBadge   = $("cam-status");
const wsBadge    = $("ws-status");
const frameBadge = $("frame-badge");
const mjpeg      = $("mjpeg");
const detCanvas  = $("det-overlay");
const detCtx     = detCanvas ? detCanvas.getContext("2d") : null;
let latestBoxes  = [];

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

/* ── Detections overlay WS ────────────────────────────────────────────────── */
function connectDetectionsWS() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/detections`);

  ws.onmessage = (ev) => {
    let data;
    try { data = JSON.parse(ev.data); } catch { return; }
    latestBoxes = Array.isArray(data.boxes) ? data.boxes : [];
    drawDetections();
  };

  ws.onclose = () => {
    setTimeout(connectDetectionsWS, 1500);
  };

  ws.onerror = () => {
    ws.close();
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
        clearDetections();
      }
    })
    .catch(() => {
      camBadge.textContent = "Camera  —";
      camBadge.className = "badge";
    });
}

function clearDetections() {
  latestBoxes = [];
  drawDetections();
}

function resizeDetectionsCanvas() {
  if (!detCanvas || !mjpeg) return;
  const w = mjpeg.clientWidth || 0;
  const h = mjpeg.clientHeight || 0;
  if (w <= 0 || h <= 0) return;

  // Match the canvas box to the actual displayed MJPEG box (not the whole wrapper).
  detCanvas.style.left = `${mjpeg.offsetLeft}px`;
  detCanvas.style.top = `${mjpeg.offsetTop}px`;
  detCanvas.style.width = `${w}px`;
  detCanvas.style.height = `${h}px`;

  const dpr = window.devicePixelRatio || 1;
  const targetW = Math.max(1, Math.round(w * dpr));
  const targetH = Math.max(1, Math.round(h * dpr));
  if (detCanvas.width !== targetW) detCanvas.width = targetW;
  if (detCanvas.height !== targetH) detCanvas.height = targetH;

  // Draw in CSS pixels while the backing store is scaled for retina displays.
  detCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawDetections() {
  if (!detCanvas || !detCtx) return;
  resizeDetectionsCanvas();

  const w = mjpeg.clientWidth || 0;
  const h = mjpeg.clientHeight || 0;
  if (w <= 0 || h <= 0) return;
  detCtx.clearRect(0, 0, w, h);
  if (!latestBoxes.length) return;

  detCtx.lineWidth = 2;
  detCtx.font = "13px system-ui, sans-serif";

  for (const b of latestBoxes) {
    const x1 = Math.max(0, Math.min(w, b.x1 * w));
    const y1 = Math.max(0, Math.min(h, b.y1 * h));
    const x2 = Math.max(0, Math.min(w, b.x2 * w));
    const y2 = Math.max(0, Math.min(h, b.y2 * h));
    const bw = Math.max(1, x2 - x1);
    const bh = Math.max(1, y2 - y1);

    detCtx.strokeStyle = "#20d36b";
    detCtx.strokeRect(x1, y1, bw, bh);

    const label = `${b.label || "obj"} ${(Number(b.conf) || 0).toFixed(2)}`;
    const tw = detCtx.measureText(label).width;
    const ty = Math.max(14, y1 - 6);
    detCtx.fillStyle = "rgba(32, 211, 107, 0.95)";
    detCtx.fillRect(x1, ty - 14, tw + 10, 16);
    detCtx.fillStyle = "#08140d";
    detCtx.fillText(label, x1 + 5, ty - 2);
  }
}

/* ── MJPEG error / reconnect ───────────────────────────────────────────────── */
mjpeg.addEventListener("error", () => {
  setTimeout(() => {
    mjpeg.src = `/stream.mjpeg?_=${Date.now()}`;
  }, 2000);
});

mjpeg.addEventListener("load", () => {
  resizeDetectionsCanvas();
  drawDetections();
});

window.addEventListener("resize", () => {
  resizeDetectionsCanvas();
  drawDetections();
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
connectDetectionsWS();
pollStatus();
setInterval(pollStatus, 5000);
