import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// LTX Extend Prompt Studio — audition each extension step's audio window and write its prompt.
// Prompt authoring works immediately (N slots from the `total` widget). After one run the node
// broadcasts "ltx_studio_layout" with the exact segment timings + a master-audio URL, upgrading the
// strip to a real waveform timeline with per-segment playback. Prompts persist in the hidden
// `prompts_json` widget as {"1": "...", "2": "..."} (1-based step -> prompt).

const STYLE = `
.ltxst-wrap { display:flex; flex-direction:column; gap:8px; width:100%; box-sizing:border-box; padding:6px; }
.ltxst-timeline { position:relative; width:100%; height:64px; background:#141414; border-radius:6px; overflow:hidden; }
.ltxst-tl-canvas { position:absolute; inset:0; width:100%; height:100%; }
.ltxst-seg { position:absolute; top:0; bottom:0; border-right:1px solid #000; box-sizing:border-box; cursor:pointer; }
.ltxst-seg.sel { background:rgba(80,150,255,0.28); box-shadow:inset 0 0 0 2px #4c9cff; }
.ltxst-seg .ov { position:absolute; top:0; bottom:0; left:0; background:rgba(255,180,60,0.22); }
.ltxst-seg .lbl { position:absolute; top:2px; left:3px; font:10px monospace; color:#ccc; pointer-events:none; }
.ltxst-seg.filled .lbl::after { content:" ●"; color:#4caf50; }
.ltxst-ctrl { display:flex; align-items:center; gap:6px; }
.ltxst-ctrl button { padding:5px 9px; border:none; border-radius:5px; background:#333; color:#fff; cursor:pointer; font-weight:600; }
.ltxst-ctrl .play { background:#2e7d32; }
.ltxst-ctrl .pos { font:12px monospace; color:#bbb; flex:1; text-align:center; }
.ltxst-prompt { width:100%; min-height:70px; resize:vertical; box-sizing:border-box; background:#1b1b1b;
  color:#eee; border:1px solid #333; border-radius:6px; padding:6px; font:13px sans-serif; }
`;

function ensureStyle() {
  if (document.getElementById("ltxst-style")) return;
  const s = document.createElement("style");
  s.id = "ltxst-style";
  s.textContent = STYLE;
  document.head.appendChild(s);
}

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function readPrompts(node) {
  const w = getWidget(node, "prompts_json");
  if (!w || !w.value) return {};
  try { return JSON.parse(w.value) || {}; } catch (e) { return {}; }
}

function writePrompts(node, prompts) {
  const w = getWidget(node, "prompts_json");
  if (w) w.value = JSON.stringify(prompts);
}

// Synthesize equal segments from the `total` widget until the backend sends real timings.
function fallbackSegments(node) {
  const total = Math.max(1, parseInt(getWidget(node, "total")?.value ?? 14, 10) || 14);
  const segs = [];
  for (let i = 0; i < total; i++) segs.push({ index: i + 1, start_s: null, len_s: null, overlap_px: 0 });
  return segs;
}

function decodePeaks(url, buckets = 600) {
  return fetch(url)
    .then((r) => r.arrayBuffer())
    .then((buf) => new (window.AudioContext || window.webkitAudioContext)().decodeAudioData(buf))
    .then((audio) => {
      const ch = audio.getChannelData(0);
      const step = Math.max(1, Math.floor(ch.length / buckets));
      const peaks = [];
      for (let i = 0; i < ch.length; i += step) {
        let m = 0;
        for (let j = i; j < i + step && j < ch.length; j++) m = Math.max(m, Math.abs(ch[j]));
        peaks.push(m);
      }
      return { peaks, duration: audio.duration };
    });
}

function drawTimeline(ui) {
  const cvs = ui.canvas;
  const w = (cvs.width = cvs.clientWidth || 300);
  const h = (cvs.height = cvs.clientHeight || 64);
  const ctx = cvs.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (ui.peaks && ui.peaks.length) {
    ctx.strokeStyle = "#3a6ea5";
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const p = ui.peaks[Math.floor((x / w) * ui.peaks.length)] || 0;
      const y = (h / 2) * p;
      ctx.moveTo(x, h / 2 - y);
      ctx.lineTo(x, h / 2 + y);
    }
    ctx.stroke();
  }
}

function layoutSegments(ui) {
  const strip = ui.strip;
  strip.innerHTML = "";
  const segs = ui.segments;
  const n = segs.length;
  // If we have real timings + audio duration, place by time; else equal widths.
  const dur = ui.duration || (segs[n - 1]?.start_s != null ? segs[n - 1].start_s + segs[n - 1].len_s : n);
  const fps = parseFloat(getWidget(ui.node, "frame_rate")?.value ?? 24) || 24;
  segs.forEach((seg, i) => {
    const el = document.createElement("div");
    el.className = "ltxst-seg" + (i === ui.sel ? " sel" : "");
    if (String(ui.prompts[seg.index] || "").trim()) el.classList.add("filled");
    let left, width;
    if (seg.start_s != null && dur) {
      left = (seg.start_s / dur) * 100;
      width = (seg.len_s / dur) * 100;
    } else {
      left = (i / n) * 100;
      width = (1 / n) * 100;
    }
    el.style.left = left + "%";
    el.style.width = width + "%";
    if (seg.overlap_px && seg.len_s) {
      const ov = document.createElement("div");
      ov.className = "ov";
      ov.style.width = ((seg.overlap_px / fps / seg.len_s) * 100) + "%";
      el.appendChild(ov);
    }
    const lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.textContent = seg.start_s != null ? `${seg.index} · ${seg.start_s.toFixed(1)}s` : `${seg.index}`;
    el.appendChild(lbl);
    el.onclick = () => selectSegment(ui, i);
    strip.appendChild(el);
  });
}

function selectSegment(ui, i) {
  ui.sel = Math.max(0, Math.min(i, ui.segments.length - 1));
  const seg = ui.segments[ui.sel];
  ui.pos.textContent = `step ${seg.index} / ${ui.segments.length}` + (seg.start_s != null ? `  (${seg.start_s.toFixed(1)}–${(seg.start_s + seg.len_s).toFixed(1)}s)` : "");
  ui.prompt.value = ui.prompts[seg.index] || "";
  layoutSegments(ui);
}

function playSelected(ui) {
  const seg = ui.segments[ui.sel];
  if (!ui.audio || seg.start_s == null) {
    ui.pos.textContent = "(run once to load audio for playback)";
    return;
  }
  ui.audio.currentTime = seg.start_s;
  ui.audio.play();
  clearTimeout(ui.stopTimer);
  ui.stopTimer = setTimeout(() => ui.audio.pause(), seg.len_s * 1000);
}

function buildUI(node) {
  ensureStyle();
  const wrap = document.createElement("div");
  wrap.className = "ltxst-wrap";
  const timeline = document.createElement("div");
  timeline.className = "ltxst-timeline";
  const canvas = document.createElement("canvas");
  canvas.className = "ltxst-tl-canvas";
  const strip = document.createElement("div");
  strip.style.cssText = "position:absolute;inset:0;";
  timeline.append(canvas, strip);

  const ctrl = document.createElement("div");
  ctrl.className = "ltxst-ctrl";
  const prev = document.createElement("button"); prev.textContent = "◀";
  const play = document.createElement("button"); play.className = "play"; play.textContent = "▶ play";
  const next = document.createElement("button"); next.textContent = "▶";
  const pos = document.createElement("div"); pos.className = "pos";
  ctrl.append(prev, pos, play, next);

  const prompt = document.createElement("textarea");
  prompt.className = "ltxst-prompt";
  prompt.placeholder = "Prompt for this step…";
  wrap.append(timeline, ctrl, prompt);

  const ui = {
    node, wrap, canvas, strip, pos, prompt,
    segments: fallbackSegments(node), sel: 0,
    prompts: readPrompts(node), audio: null, peaks: null, duration: 0, stopTimer: null,
  };
  prev.onclick = () => selectSegment(ui, ui.sel - 1);
  next.onclick = () => selectSegment(ui, ui.sel + 1);
  play.onclick = () => playSelected(ui);
  prompt.oninput = () => {
    const seg = ui.segments[ui.sel];
    ui.prompts[seg.index] = prompt.value;
    writePrompts(node, ui.prompts);
    layoutSegments(ui);
  };
  return ui;
}

app.registerExtension({
  name: "WhatDreamsCost.LTXExtendPromptStudio",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LTXExtendPromptStudio") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      // hide the raw prompts_json widget — managed by the studio UI
      const pj = getWidget(this, "prompts_json");
      if (pj) { pj.hidden = true; pj.computeSize = () => [0, -4]; }

      const ui = buildUI(this);
      this._ltxstUI = ui;
      if (typeof this.addDOMWidget === "function") {
        const w = this.addDOMWidget("Studio", "div", ui.wrap, { serialize: false, hideOnZoom: false });
        w.computeSize = () => [0, 200];
      }
      requestAnimationFrame(() => {
        if (this.size[0] < 380) this.size[0] = 380;
        if (this.size[1] < 420) this.size[1] = 420;
        layoutSegments(ui);
        selectSegment(ui, 0);
        drawTimeline(ui);
      });
      // rebuild segments when `total` changes
      const totalW = getWidget(this, "total");
      if (totalW) {
        const cb = totalW.callback;
        totalW.callback = function () {
          const rr = cb ? cb.apply(this, arguments) : undefined;
          if (!ui.hasBackendSegments) ui.segments = fallbackSegments(ui.node);
          selectSegment(ui, ui.sel);
          return rr;
        };
      }

      // PERSISTENCE. onNodeCreated runs BEFORE ComfyUI restores saved widget values, so the
      // prompts read at build time are empty on reload. Two safeguards:
      //  1) prompts_json is a normal serialized widget -> its value goes into the workflow AND to
      //     the backend. onConfigure (fires AFTER widgets_values are applied) re-pulls it into the UI.
      //  2) Belt-and-suspenders: also stash prompts in the node's own serialized data, in case a
      //     frontend drops hidden widgets from widgets_values — and re-hydrate the widget on load so
      //     the backend still receives them.
      const onSerialize = this.onSerialize;
      this.onSerialize = function (o) {
        if (onSerialize) onSerialize.apply(this, arguments);
        try { o.ltxst_prompts = ui.prompts; } catch (e) {}
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function (info) {
        const rr = onConfigure ? onConfigure.apply(this, arguments) : undefined;
        let p = readPrompts(this);
        if ((!p || !Object.keys(p).length) && info && info.ltxst_prompts && Object.keys(info.ltxst_prompts).length) {
          p = info.ltxst_prompts;
          writePrompts(this, p);  // re-hydrate prompts_json so the backend gets them on run
        }
        ui.prompts = p || {};
        if (!ui.hasBackendSegments) ui.segments = fallbackSegments(ui.node);
        selectSegment(ui, ui.sel || 0);
        layoutSegments(ui);
        return rr;
      };
      return r;
    };
  },
});

api.addEventListener("ltx_studio_layout", (e) => {
  const d = e.detail || {};
  const g = app.graph;
  const node = g && g.getNodeById ? (g.getNodeById(Number(d.node_id)) || g.getNodeById(d.node_id)) : null;
  const ui = node && node._ltxstUI;
  if (!ui) return;
  if (Array.isArray(d.segments) && d.segments.length) {
    ui.segments = d.segments;
    ui.hasBackendSegments = true;
  }
  if (d.audio_url) {
    ui.audio = new Audio(d.audio_url);
    decodePeaks(d.audio_url)
      .then(({ peaks, duration }) => { ui.peaks = peaks; ui.duration = duration; drawTimeline(ui); layoutSegments(ui); })
      .catch(() => {});
  }
  selectSegment(ui, Math.min(ui.sel, ui.segments.length - 1));
});
