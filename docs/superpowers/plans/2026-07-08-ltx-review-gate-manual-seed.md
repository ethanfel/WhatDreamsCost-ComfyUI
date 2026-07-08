# LTX Review Gate Manual Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `LTX Review Gate` play preview audio/video on hover and control a normal seed node outside the extend loop.

**Architecture:** Keep the backend seed node stateless: it only exposes a normal `seed` widget and returns its value. Put graph-aware behavior in `js/ltx_review.js`, where the Review Gate UI already lives, so the gate can increment the seed widget in manual/passthrough workflows without changing existing loop reroll behavior.

**Tech Stack:** Python ComfyUI node classes, ComfyUI frontend extension JavaScript, pytest, `py_compile`, Node syntax checking.

---

## File Structure

- Modify: `ltx_director.py`
  - Add `LTXReviewSeed`, a small stateless node with `seed` and `gate_id` widgets.
- Modify: `__init__.py`
  - Import and register `LTXReviewSeed` with display name `LTX Review Seed`.
- Modify: `js/ltx_review.js`
  - Change preview media from autoplay-on-arrival to hover-to-play.
  - Keep `Reroll seed` enabled in passthrough mode.
  - Increment the matching `LTX Review Seed.seed` widget by `+1` before backend reroll decisions.
- Create: `tests/test_ltx_review_seed.py`
  - Reuse lightweight ComfyUI stubs to test seed output and unsigned 64-bit normalization.

---

### Task 1: Add Failing Review Seed Tests

**Files:**
- Create: `tests/test_ltx_review_seed.py`

- [ ] **Step 1: Create tests for the new seed node**

Create `tests/test_ltx_review_seed.py` with:

```python
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class _Routes:
    def get(self, _path):
        return lambda fn: fn

    def post(self, _path):
        return lambda fn: fn


class _PromptServer:
    instance = types.SimpleNamespace(routes=_Routes(), send_sync=lambda *args, **kwargs: None)


class _IOType:
    @staticmethod
    def Input(*args, **kwargs):
        return ("INPUT", args, kwargs)

    @staticmethod
    def Output(*args, **kwargs):
        return ("OUTPUT", args, kwargs)


class _NodeOutput(tuple):
    def __new__(cls, *values):
        return tuple.__new__(cls, values)


class _IO:
    ComfyNode = object
    NodeOutput = _NodeOutput
    Model = _IOType
    Clip = _IOType
    Latent = _IOType
    Image = _IOType
    Int = _IOType
    Float = _IOType
    String = _IOType
    Audio = _IOType
    Vae = _IOType
    Combo = _IOType
    Boolean = _IOType

    @staticmethod
    def Custom(_name):
        return _IOType

    @staticmethod
    def Schema(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def _load_ltx_director(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(ROOT),
        get_output_directory=lambda: str(ROOT),
        get_temp_directory=lambda: str(ROOT),
        get_filename_list=lambda _name: [],
    ))

    comfy_mod = types.ModuleType("comfy")
    comfy_mod.model_management = types.SimpleNamespace(
        intermediate_device=lambda: torch.device("cpu"),
        throw_exception_if_processing_interrupted=lambda: None,
    )
    comfy_mod.utils = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_management", comfy_mod.model_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_mod.utils)

    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=_PromptServer))
    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(web=types.SimpleNamespace(json_response=lambda data, status=200: data)))
    monkeypatch.setitem(sys.modules, "aiohttp.web", types.SimpleNamespace(json_response=lambda data, status=200: data))

    api_latest = types.SimpleNamespace(io=_IO)
    monkeypatch.setitem(sys.modules, "comfy_api", types.SimpleNamespace(latest=api_latest))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", api_latest)

    pkg = types.ModuleType("wdc")
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "wdc", pkg)
    monkeypatch.setitem(sys.modules, "wdc.prompt_relay", types.SimpleNamespace(
        get_raw_tokenizer=lambda clip: None,
        map_token_indices=lambda tokenizer, global_prompt, locals_list: (global_prompt, []),
        build_segments=lambda *args, **kwargs: None,
        create_mask_fn=lambda *args, **kwargs: None,
        distribute_segment_lengths=lambda count, latent_frames, parsed_lengths: [latent_frames],
    ))
    monkeypatch.setitem(sys.modules, "wdc.patches", types.SimpleNamespace(
        detect_model_type=lambda model: ("ltx", (1, 1, 1), 8),
        apply_patches=lambda model, arch, mask_fn: None,
    ))

    module_name = "wdc.ltx_director"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "ltx_director.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_review_seed_outputs_seed(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)

    assert ltx_director.LTXReviewSeed().seed(123, gate_id="") == (123,)


def test_review_seed_wraps_unsigned_64(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)

    assert ltx_director.LTXReviewSeed().seed(0xffffffffffffffff + 2, gate_id="42") == (1,)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_ltx_review_seed.py -q
```

Expected: FAIL because `ltx_director.LTXReviewSeed` does not exist.

---

### Task 2: Implement and Register `LTX Review Seed`

**Files:**
- Modify: `ltx_director.py`
- Modify: `__init__.py`
- Test: `tests/test_ltx_review_seed.py`

- [ ] **Step 1: Add the node class**

In `ltx_director.py`, insert this class immediately before `LTXReviewGate`:

```python
class LTXReviewSeed:
    """Small seed source controlled by the LTX Review Gate frontend.

    The node is stateless on the backend: the frontend increments the visible `seed`
    widget and this node returns that widget value on the next queue/run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "tooltip": "Seed value. LTX Review Gate can increment this by +1 from its Reroll seed button.",
                }),
            },
            "optional": {
                "gate_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional Review Gate node id to bind to. Leave blank when there is only one LTX Review Seed in the graph.",
                }),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "seed"
    CATEGORY = "WhatDreamsCost"

    def seed(self, seed=0, gate_id=""):
        return (int(seed) & 0xffffffffffffffff,)
```

- [ ] **Step 2: Export the node in `__init__.py`**

Change the import:

```python
from .ltx_director import LTXDirector, LTXKeyframeOut, LTXAutoExtend, LTXExtendInit, LTXExtendStep, LTXExtendCollect, LTXReviewGate, LTXReviewSeed
```

Add to `NODE_CLASS_MAPPINGS`:

```python
    "LTXReviewSeed": LTXReviewSeed,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS`:

```python
    "LTXReviewSeed": "LTX Review Seed",
```

- [ ] **Step 3: Run the seed node tests**

Run:

```bash
pytest tests/test_ltx_review_seed.py -q
```

Expected: PASS.

---

### Task 3: Change Review Gate Preview To Hover Playback

**Files:**
- Modify: `js/ltx_review.js`

- [ ] **Step 1: Update media helpers**

Replace the eager playback logic in `showMedia(ui, d)` with helper functions that only play on hover:

```javascript
function pauseMedia(ui) {
  if (ui.video) { try { ui.video.pause(); } catch (e) {} }
  if (ui.audio) { try { ui.audio.pause(); } catch (e) {} }
}

function playMedia(ui) {
  if (ui.video && ui.video.style.display !== "none" && ui.video.src) {
    ui.video.play().catch(() => {});
  }
  if (ui.audio && ui.audio.src) {
    try {
      if (ui.video && ui.video.style.display !== "none") {
        ui.audio.currentTime = ui.video.currentTime || 0;
      }
    } catch (e) {}
    ui.audio.play().catch(() => {});
  }
}
```

Inside `showMedia(ui, d)`, set video/audio sources and remove `ui.video.onloadeddata = play; play();`. The preview should be ready but paused until hover.

- [ ] **Step 2: Wire hover events**

In `buildUI(node)`, after creating `ui`, add:

```javascript
  view.addEventListener("mouseenter", () => playMedia(ui));
  view.addEventListener("mouseleave", () => pauseMedia(ui));
```

- [ ] **Step 3: Syntax check the frontend**

Run:

```bash
node --check js/ltx_review.js
```

Expected: no syntax errors.

---

### Task 4: Add Gate-Controlled Seed Increment

**Files:**
- Modify: `js/ltx_review.js`

- [ ] **Step 1: Add seed-node discovery helpers**

Add helper functions near `uiForNodeId`:

```javascript
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
```

- [ ] **Step 2: Register seed node frontend marker**

Extend `beforeRegisterNodeDef` so `LTXReviewSeed` nodes are marked:

```javascript
    if (nodeData.name === "LTXReviewSeed") {
      const onCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const r = onCreated ? onCreated.apply(this, arguments) : undefined;
        this._ltxReviewSeedNode = true;
        return r;
      };
      return;
    }
```

Keep the existing `LTXReviewGate` UI branch after this marker branch.

- [ ] **Step 3: Change reroll click behavior**

In `buildUI(node)`, replace:

```javascript
  reroll.onclick = () => decide(String(node.id), "reroll", ui);
```

with:

```javascript
  reroll.onclick = () => {
    incrementControlledSeed(node, ui);
    if (!ui.passthrough) decide(String(node.id), "reroll", ui);
  };
```

- [ ] **Step 4: Keep reroll enabled in passthrough mode**

Replace `setButtons(ui, enabled)` with:

```javascript
function setButtons(ui, enabled) {
  for (const b of ui.buttons) b.disabled = !enabled;
}

function setPassthroughButtons(ui) {
  ui.passButton.disabled = true;
  ui.rerollButton.disabled = false;
  ui.reloadButton.disabled = true;
}
```

Change the `ui` object in `buildUI(node)` to include named buttons:

```javascript
  const ui = {
    wrap, view, img, video, audio, status,
    buttons: [pass, reroll, reload],
    passButton: pass,
    rerollButton: reroll,
    reloadButton: reload,
    frames: [], idx: 0, timer: null, passthrough: false,
  };
```

In the `ltx_review_show` listener, set `ui.passthrough = !!d.passthrough`; use `setPassthroughButtons(ui)` when passthrough is true and `setButtons(ui, true)` otherwise.

- [ ] **Step 5: Syntax check the frontend**

Run:

```bash
node --check js/ltx_review.js
```

Expected: no syntax errors.

---

### Task 5: Final Verification And Commit

**Files:**
- Verify: `ltx_director.py`
- Verify: `__init__.py`
- Verify: `js/ltx_review.js`
- Verify: `tests/test_ltx_review_seed.py`

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
pytest tests/test_ltx_review_seed.py tests/test_ltx_extend_loop_state.py tests/test_ltx_autoextend_anchor.py -q
```

Expected: PASS.

- [ ] **Step 2: Compile Python files**

Run:

```bash
python -m py_compile ltx_director.py __init__.py tests/test_ltx_review_seed.py
```

Expected: no output.

- [ ] **Step 3: Syntax check frontend**

Run:

```bash
node --check js/ltx_review.js
```

Expected: no output.

- [ ] **Step 4: Review staged diff**

Run:

```bash
git diff -- ltx_director.py __init__.py js/ltx_review.js tests/test_ltx_review_seed.py
```

Expected: only the Review Gate seed node, hover playback, passthrough reroll enablement, and tests are present.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add ltx_director.py __init__.py js/ltx_review.js tests/test_ltx_review_seed.py docs/superpowers/plans/2026-07-08-ltx-review-gate-manual-seed.md
git commit -m "Add review gate manual seed control"
```

Expected: commit succeeds.

---

## Self-Review

- Spec coverage: hover media playback is Task 3; `LTX Review Seed` is Task 2; gate-to-seed matching, passthrough reroll, and `+1` increment are Task 4; tests and verification are Tasks 1 and 5.
- Placeholder scan: no placeholders are present.
- Type consistency: Python node uses `seed(seed=0, gate_id="") -> ("INT",)`. Frontend matches widget names `seed` and `gate_id`, and class name `LTXReviewSeed`.

