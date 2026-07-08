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
  pauseMedia(ui);
  if (ui.video) ui.video.onloadeddata = null;
}

function pauseMedia(ui) {
  clearInterval(ui.timer);
  ui.timer = null;
  if (ui.video) { try { ui.video.pause(); } catch (e) {} }
  if (ui.audio) { try { ui.audio.pause(); } catch (e) {} }
}

function playMedia(ui) {
  if (ui.video && ui.video.style.display !== "none" && ui.video.src) {
    ui.video.play().catch(() => {});
  } else if (ui.img && ui.img.style.display !== "none" && ui.frames?.length > 1 && !ui.timer) {
    ui.timer = setInterval(() => {
      ui.idx = (ui.idx + 1) % ui.frames.length;
      ui.img.src = ui.frames[ui.idx];
    }, 1000 / Math.max(1, ui.fps || 24));
  }
  if (ui.audio && ui.audio.src) {
    try {
      if (ui.video && ui.video.style.display !== "none") {
        ui.audio.currentTime = ui.video.currentTime || 0;
      } else {
        ui.audio.currentTime = (ui.idx || 0) / Math.max(1, ui.fps || 24);
      }
    } catch (e) {}
    ui.audio.play().catch(() => {});
  }
}

function setMediaSource(el, url) {
  if (!el) return;
  if (url) {
    el.src = url;
    return;
  }
  el.removeAttribute("src");
  try { el.load(); } catch (e) {}
}

// Prefer a real <video> at the generation fps. Fall back to the base64 frame slideshow
// only when no video was encoded. Playback itself is hover-driven by buildUI().
function showMedia(ui, d) {
  stopMedia(ui);
  const fps = Math.max(1, parseFloat(d.fps) || 24);
  ui.fps = fps;

  if (d.video_url) {
    ui.view.classList.remove("empty");
    ui.img.style.display = "none";
    ui.video.style.display = "block";
    ui.video.loop = true;
    ui.video.muted = !!d.audio_url;          // separate audio track -> keep video muted, play the audio el
    ui.video.src = d.video_url;
    if (ui.audio) { setMediaSource(ui.audio, d.audio_url); ui.audio.loop = true; }
    return;
  }

  // fallback: frame slideshow at the real fps
  ui.video.style.display = "none";
  ui.img.style.display = "block";
  ui.frames = d.frames || [];
  if (ui.audio) { setMediaSource(ui.audio, d.audio_url); ui.audio.loop = true; }
  if (!ui.frames.length) { ui.img.src = ""; ui.view.classList.add("empty"); return; }
  ui.view.classList.remove("empty");
  ui.idx = 0;
  ui.img.src = ui.frames[0];
}

function setButtons(ui, enabled) {
  for (const b of ui.buttons) b.disabled = !enabled;
}

function setPassthroughButtons(ui) {
  ui.passButton.disabled = true;
  ui.rerollButton.disabled = false;
  ui.reloadButton.disabled = true;
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

const UINT64_MASK = (1n << 64n) - 1n;

function widgetByName(node, name) {
  return node?.widgets?.find((w) => w.name === name) || null;
}

function nodeClassName(node) {
  return node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "";
}

function isReviewSeedNode(node) {
  return node?._ltxReviewSeedNode || nodeClassName(node) === "LTXReviewSeed";
}

function nextSeedValue(value) {
  let current = 0n;
  try {
    current = BigInt(String(value ?? 0).trim().split(".")[0] || "0");
  } catch (e) {
    current = 0n;
  }
  const next = (current + 1n) & UINT64_MASK;
  return next <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(next) : next.toString();
}

function findControlledSeedNode(gateNode) {
  const nodes = app.graph?._nodes || [];
  const seeds = nodes.filter(isReviewSeedNode);
  const gateId = String(gateNode?.id ?? "");
  const matching = seeds.filter((node) => String(widgetByName(node, "gate_id")?.value || "").trim() === gateId);
  if (matching.length === 1) return { node: matching[0], reason: "" };
  if (matching.length > 1) return { node: null, reason: "multiple seed nodes match this gate_id" };
  if (seeds.length === 1) return { node: seeds[0], reason: "" };
  if (seeds.length === 0) return { node: null, reason: "no LTX Review Seed node found" };
  return { node: null, reason: "set gate_id on one LTX Review Seed" };
}

function incrementControlledSeed(gateNode, ui) {
  const { node, reason } = findControlledSeedNode(gateNode);
  if (!node) {
    if (ui) ui.status.textContent = reason;
    return false;
  }
  const seedWidget = widgetByName(node, "seed");
  if (!seedWidget) {
    if (ui) ui.status.textContent = "controlled seed node has no seed widget";
    return false;
  }
  const next = nextSeedValue(seedWidget.value);
  seedWidget.value = next;
  if (seedWidget.callback) {
    try { seedWidget.callback(next); } catch (e) {}
  }
  if (app.graph?.setDirtyCanvas) app.graph.setDirtyCanvas(true, true);
  if (app.graph?.change) app.graph.change();
  if (ui) ui.status.textContent = `seed -> ${next}`;
  return true;
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

  const ui = {
    wrap, view, img, video, audio, status,
    buttons: [pass, reroll, reload],
    passButton: pass,
    rerollButton: reroll,
    reloadButton: reload,
    frames: [], idx: 0, timer: null, fps: 24, passthrough: false,
  };
  view.addEventListener("mouseenter", () => playMedia(ui));
  view.addEventListener("mouseleave", () => pauseMedia(ui));
  // read node.id FRESH at click — it isn't final at onNodeCreated and the backend keys by the display id
  pass.onclick = () => decide(String(node.id), "pass", ui);
  reroll.onclick = () => {
    incrementControlledSeed(node, ui);
    if (!ui.passthrough) decide(String(node.id), "reroll", ui);
  };
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
    if (nodeData.name === "LTXReviewSeed") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        this._ltxReviewSeedNode = true;
        return r;
      };
      return;
    }
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
  showMedia(ui, d);
  ui.passthrough = !!d.passthrough;
  if (d.passthrough) {
    setPassthroughButtons(ui);   // preview only — only manual seed control remains active
    ui.status.textContent = `attempt ${d.attempt} — preview (${kind}, passthrough)`;
  } else {
    setButtons(ui, true);
    ui.status.textContent = `attempt ${d.attempt} — review (${kind})`;
  }
});

api.addEventListener("ltx_review_done", (e) => {
  const d = e.detail || {};
  const ui = uiForNodeId(d.node_id);
  if (!ui) return;
  stopMedia(ui);
  setButtons(ui, false);
  ui.status.textContent = "decision: " + (d.action || "?");
});
