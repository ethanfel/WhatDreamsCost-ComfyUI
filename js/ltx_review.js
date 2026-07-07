import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// LTX Review Gate — blocking human-in-the-loop review for the extend retry loop.
// The Python node (LTXReviewGate) blocks the worker, pushes this attempt's decoded frames over the
// websocket ("ltx_review_show"), and waits on /ltx_review_decide. This extension renders the frames
// as a small looping preview inside the node and exposes Pass / Reroll seed / Reload prompt buttons.

const STYLE = `
.ltxrg-wrap { display:flex; flex-direction:column; gap:6px; width:100%; box-sizing:border-box; padding:4px; }
.ltxrg-view { width:100%; background:#111; border-radius:6px; overflow:hidden; display:flex;
  align-items:center; justify-content:center; min-height:160px; }
.ltxrg-view img, .ltxrg-view video { max-width:100%; max-height:340px; display:block; }
.ltxrg-view.empty::after { content:"idle — queue to review"; color:#666; font:12px monospace; }
.ltxrg-status { font:12px monospace; color:#bbb; text-align:center; min-height:14px; }
.ltxrg-btns { display:flex; gap:6px; }
.ltxrg-btns button { flex:1; padding:9px 4px; border:none; border-radius:6px; cursor:pointer;
  font-weight:600; color:#fff; font-size:12px; }
.ltxrg-pass { background:#2e7d32; }
.ltxrg-reroll { background:#1565c0; }
.ltxrg-reload { background:#6a4caf; }
.ltxrg-btns button:disabled { opacity:0.35; cursor:default; }
`;

function ensureStyle() {
  if (document.getElementById("ltxrg-style")) return;
  const s = document.createElement("style");
  s.id = "ltxrg-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

// Keep the DOM widget from collapsing to ~half width on select/relayout (see global ComfyUI notes):
// clamp the container to the node's own width reference via a ResizeObserver.
function keepFullWidth(node, container) {
  try {
    const apply = () => {
      const w = Math.max(0, (node.size?.[0] || 0) - 20);
      if (w > 0) container.style.width = w + "px";
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(container);
    node.__ltxrgRO = ro;
    requestAnimationFrame(apply);
  } catch (e) { /* ResizeObserver unavailable — width:100% fallback still applies */ }
}

function stopMedia(ui) {
  clearInterval(ui.timer);
  ui.timer = null;
  if (ui.video) { try { ui.video.pause(); } catch (e) {} }
  if (ui.audio) { try { ui.audio.pause(); } catch (e) {} }
}

// Prefer a real <video> at the generation fps; play the audio window alongside it. Fall back to the
// base64 frame slideshow (at the correct fps) only when no video was encoded.
function showMedia(ui, d) {
  stopMedia(ui);
  const fps = Math.max(1, parseFloat(d.fps) || 24);

  if (d.video_url) {
    ui.view.classList.remove("empty");
    ui.img.style.display = "none";
    ui.video.style.display = "block";
    ui.video.loop = true;
    ui.video.muted = !!d.audio_url;          // separate audio track -> keep video muted, play the audio el
    ui.video.src = d.video_url;
    if (ui.audio) { ui.audio.src = d.audio_url || ""; ui.audio.loop = true; }
    const play = () => {
      ui.video.play().catch(() => {});
      if (d.audio_url && ui.audio) {
        try { ui.audio.currentTime = ui.video.currentTime || 0; } catch (e) {}
        ui.audio.play().catch(() => {});
      }
    };
    ui.video.onloadeddata = play;
    play();
    return;
  }

  // fallback: frame slideshow at the real fps
  ui.video.style.display = "none";
  ui.img.style.display = "block";
  ui.frames = d.frames || [];
  if (ui.audio) { ui.audio.src = d.audio_url || ""; ui.audio.loop = true; if (d.audio_url) ui.audio.play().catch(() => {}); }
  if (!ui.frames.length) { ui.img.src = ""; ui.view.classList.add("empty"); return; }
  ui.view.classList.remove("empty");
  ui.idx = 0;
  ui.img.src = ui.frames[0];
  if (ui.frames.length > 1) {
    ui.timer = setInterval(() => {
      ui.idx = (ui.idx + 1) % ui.frames.length;
      ui.img.src = ui.frames[ui.idx];
    }, 1000 / fps);
  }
}

function setButtons(ui, enabled) {
  for (const b of ui.buttons) b.disabled = !enabled;
}

async function decide(nodeId, action, ui) {
  setButtons(ui, false);
  ui.status.textContent =
    action === "pass" ? "passing…" : action === "reroll" ? "rerolling (new seed)…" : "reloading prompt…";
  try {
    await api.fetchApi("/ltx_review_decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: String(nodeId), action }),
    });
  } catch (e) {
    ui.status.textContent = "error: " + e;
    setButtons(ui, true);
  }
}

function buildUI(node) {
  ensureStyle();
  const wrap = document.createElement("div");
  wrap.className = "ltxrg-wrap";
  const view = document.createElement("div");
  view.className = "ltxrg-view empty";
  const img = document.createElement("img");
  view.appendChild(img);
  const video = document.createElement("video");
  video.playsInline = true;
  video.controls = true;
  video.style.display = "none";
  view.appendChild(video);
  const audio = document.createElement("audio");
  audio.style.display = "none";
  view.appendChild(audio);
  const status = document.createElement("div");
  status.className = "ltxrg-status";
  status.textContent = "idle";
  const btns = document.createElement("div");
  btns.className = "ltxrg-btns";
  const pass = document.createElement("button");
  pass.className = "ltxrg-pass";
  pass.textContent = "Pass";
  const reroll = document.createElement("button");
  reroll.className = "ltxrg-reroll";
  reroll.textContent = "Reroll seed";
  const reload = document.createElement("button");
  reload.className = "ltxrg-reload";
  reload.textContent = "Reload prompt";
  btns.append(pass, reroll, reload);
  wrap.append(view, status, btns);

  const ui = { wrap, view, img, video, audio, status, buttons: [pass, reroll, reload], frames: [], idx: 0, timer: null };
  // read node.id FRESH at click — it isn't final at onNodeCreated and the backend keys by the display id
  pass.onclick = () => decide(String(node.id), "pass", ui);
  reroll.onclick = () => decide(String(node.id), "reroll", ui);
  reload.onclick = () => decide(String(node.id), "reload", ui);
  setButtons(ui, false);
  return ui;
}

function uiForNodeId(nodeId) {
  const g = app.graph;
  if (!g || !g.getNodeById) return null;
  const node = g.getNodeById(Number(nodeId)) || g.getNodeById(nodeId);
  return node ? node._ltxrgUI : null;
}

app.registerExtension({
  name: "WhatDreamsCost.LTXReviewGate",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LTXReviewGate") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      const ui = buildUI(this);
      this._ltxrgUI = ui;
      if (typeof this.addDOMWidget === "function") {
        const w = this.addDOMWidget("Review", "div", ui.wrap, { serialize: false, hideOnZoom: false });
        w.computeSize = () => [0, 430];
        keepFullWidth(this, ui.wrap);
      }
      requestAnimationFrame(() => {
        if (this.size[0] < 360) this.size[0] = 360;
        if (this.size[1] < 500) this.size[1] = 500;
      });
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        stopMedia(ui);
        if (this.__ltxrgRO) { try { this.__ltxrgRO.disconnect(); } catch (e) {} }
        return onRemoved ? onRemoved.apply(this, arguments) : undefined;
      };
      return r;
    };
  },
});

api.addEventListener("ltx_review_show", (e) => {
  const d = e.detail || {};
  const ui = uiForNodeId(d.node_id);
  if (!ui) return;
  const kind = d.video_url ? "video" : `${(d.frames || []).length} frames`;
  ui.status.textContent = `attempt ${d.attempt} — review (${kind})`;
  showMedia(ui, d);
  setButtons(ui, true);
});

api.addEventListener("ltx_review_done", (e) => {
  const d = e.detail || {};
  const ui = uiForNodeId(d.node_id);
  if (!ui) return;
  stopMedia(ui);
  setButtons(ui, false);
  ui.status.textContent = "decision: " + (d.action || "?");
});
