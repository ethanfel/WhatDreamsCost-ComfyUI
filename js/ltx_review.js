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
.ltxrg-view img { max-width:100%; max-height:340px; display:block; }
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

function renderFrames(ui) {
  clearInterval(ui.timer);
  if (!ui.frames || !ui.frames.length) {
    ui.img.src = "";
    ui.view.classList.add("empty");
    return;
  }
  ui.view.classList.remove("empty");
  ui.idx = 0;
  ui.img.src = ui.frames[0];
  if (ui.frames.length > 1) {
    ui.timer = setInterval(() => {
      ui.idx = (ui.idx + 1) % ui.frames.length;
      ui.img.src = ui.frames[ui.idx];
    }, 110); // ~9fps looping preview
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

  const ui = { wrap, view, img, status, buttons: [pass, reroll, reload], frames: [], idx: 0, timer: null };
  const nid = String(node.id);
  pass.onclick = () => decide(nid, "pass", ui);
  reroll.onclick = () => decide(nid, "reroll", ui);
  reload.onclick = () => decide(nid, "reload", ui);
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
        clearInterval(ui.timer);
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
  ui.frames = d.frames || [];
  ui.status.textContent = `attempt ${d.attempt} — review (${ui.frames.length} frames)`;
  renderFrames(ui);
  setButtons(ui, true);
});

api.addEventListener("ltx_review_done", (e) => {
  const d = e.detail || {};
  const ui = uiForNodeId(d.node_id);
  if (!ui) return;
  clearInterval(ui.timer);
  setButtons(ui, false);
  ui.status.textContent = "decision: " + (d.action || "?");
});
