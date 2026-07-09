# LTX Guide Data Image Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `LTX Guide Data Image Compression` node that changes only image keyframes inside `guide_data["images"]` for pass-specific compression experiments.

**Architecture:** Implement the node in `ltx_director.py` next to the existing Review/Guide helper nodes because it reuses `_compress_image()` and the `GuideData` type. The node copies the incoming guide-data dictionary and list fields, recompresses only `images`, and preserves `guide_latents` and all metadata unchanged.

**Tech Stack:** Python ComfyUI custom node mappings, PyTorch tensors, pytest, `py_compile`.

---

## File Structure

- Modify: `ltx_director.py`
  - Add `LTXGuideDataImageCompression`.
- Modify: `__init__.py`
  - Import and register `LTXGuideDataImageCompression`.
- Create: `tests/test_ltx_guide_data_image_compression.py`
  - Test CRF `0`, CRF `>0`, metadata preservation, list copying, and latent identity preservation.
- Modify: `docs/superpowers/plans/2026-07-09-ltx-guide-data-image-compression.md`
  - Track the implementation plan.

---

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_ltx_guide_data_image_compression.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_ltx_guide_data_image_compression.py` with:

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


def _guide_data():
    image_a = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    image_b = torch.ones((1, 4, 4, 3), dtype=torch.float32)
    original = torch.full((1, 8, 8, 3), 0.5, dtype=torch.float32)
    latent = torch.ones((1, 128, 1, 2, 2), dtype=torch.float32)
    return {
        "images": [image_a, image_b],
        "original_images": [original],
        "insert_frames": [0, 8],
        "strengths": [1.0, 0.5],
        "segment_numbers": [0, 1],
        "guide_latents": [latent, None],
        "frame_rate": 24.0,
        "timeline_data": "{}",
    }, image_a, image_b, original, latent


def test_crf_zero_preserves_image_values_and_latents(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, image_a, image_b, original, latent = _guide_data()

    (out,) = ltx_director.LTXGuideDataImageCompression().execute(gd, img_compression=0)

    assert out is not gd
    assert out["images"] is not gd["images"]
    assert torch.equal(out["images"][0], image_a)
    assert torch.equal(out["images"][1], image_b)
    assert out["original_images"][0] is original
    assert out["guide_latents"][0] is latent
    assert out["insert_frames"] == [0, 8]
    assert out["strengths"] == [1.0, 0.5]
    assert out["segment_numbers"] == [0, 1]


def test_positive_crf_compresses_only_images(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, image_a, image_b, original, latent = _guide_data()
    calls = []

    def fake_compress(tensor, crf):
        calls.append((tensor, crf))
        return tensor + 0.25

    monkeypatch.setattr(ltx_director, "_compress_image", fake_compress)

    (out,) = ltx_director.LTXGuideDataImageCompression().execute(gd, img_compression=36)

    assert calls == [(image_a, 36), (image_b, 36)]
    assert torch.equal(out["images"][0], image_a + 0.25)
    assert torch.equal(out["images"][1], image_b + 0.25)
    assert out["original_images"][0] is original
    assert out["guide_latents"][0] is latent
    assert gd["images"][0] is image_a
    assert gd["images"][1] is image_b
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest tests/test_ltx_guide_data_image_compression.py -q
```

Expected: FAIL with `AttributeError: module 'wdc.ltx_director' has no attribute 'LTXGuideDataImageCompression'`.

---

### Task 2: Implement And Register The Node

**Files:**
- Modify: `ltx_director.py`
- Modify: `__init__.py`
- Test: `tests/test_ltx_guide_data_image_compression.py`

- [ ] **Step 1: Add the backend node**

In `ltx_director.py`, add this class immediately before `LTXReviewSeed`:

```python
class LTXGuideDataImageCompression:
    """Pass-specific image compression for Director guide_data keyframe images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guide_data": (GuideData,),
                "img_compression": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "H.264 CRF compression to apply to guide_data image keyframes only. 0 = no compression.",
                }),
            },
        }

    RETURN_TYPES = (GuideData,)
    RETURN_NAMES = ("guide_data",)
    FUNCTION = "execute"
    CATEGORY = "WhatDreamsCost"

    def execute(self, guide_data, img_compression=0):
        src = guide_data or {}
        out = dict(src)
        for key, value in src.items():
            if isinstance(value, list):
                out[key] = list(value)

        images = list(src.get("images", []) or [])
        crf = max(0, min(100, int(img_compression)))
        if crf > 0:
            images = [_compress_image(img, crf) for img in images]
        out["images"] = images
        return (out,)
```

- [ ] **Step 2: Register the node**

In `__init__.py`, update the import:

```python
from .ltx_director import LTXDirector, LTXKeyframeOut, LTXAutoExtend, LTXExtendInit, LTXExtendStep, LTXExtendCollect, LTXReviewGate, LTXReviewSeed, LTXGuideDataImageCompression
```

Add to `NODE_CLASS_MAPPINGS`:

```python
    "LTXGuideDataImageCompression": LTXGuideDataImageCompression,
```

Add to `NODE_DISPLAY_NAME_MAPPINGS`:

```python
    "LTXGuideDataImageCompression": "LTX Guide Data Image Compression",
```

- [ ] **Step 3: Run focused tests and verify GREEN**

Run:

```bash
pytest tests/test_ltx_guide_data_image_compression.py -q
```

Expected: PASS.

---

### Task 3: Verify Existing Extension Tests

**Files:**
- Verify: `ltx_director.py`
- Verify: `__init__.py`
- Verify: `tests/test_ltx_guide_data_image_compression.py`
- Verify: existing focused tests

- [ ] **Step 1: Run focused Python tests**

Run:

```bash
pytest tests/test_ltx_guide_data_image_compression.py tests/test_ltx_review_seed.py tests/test_ltx_extend_loop_state.py tests/test_ltx_autoextend_anchor.py -q
```

Expected: PASS.

- [ ] **Step 2: Compile changed Python files**

Run:

```bash
python -m py_compile ltx_director.py __init__.py tests/test_ltx_guide_data_image_compression.py
```

Expected: no output.

- [ ] **Step 3: Review intended diff**

Run:

```bash
git diff -- ltx_director.py __init__.py tests/test_ltx_guide_data_image_compression.py docs/superpowers/plans/2026-07-09-ltx-guide-data-image-compression.md
```

Expected: only the compression node, registration, tests, and plan are present.

---

### Task 4: Commit And Push

**Files:**
- Commit: `ltx_director.py`
- Commit: `__init__.py`
- Commit: `tests/test_ltx_guide_data_image_compression.py`
- Commit: `docs/superpowers/plans/2026-07-09-ltx-guide-data-image-compression.md`

- [ ] **Step 1: Stage intended files only**

Run:

```bash
git add ltx_director.py __init__.py tests/test_ltx_guide_data_image_compression.py docs/superpowers/plans/2026-07-09-ltx-guide-data-image-compression.md
```

- [ ] **Step 2: Commit**

Run:

```bash
git commit -m "Add guide data image compression node"
```

Expected: commit succeeds.

- [ ] **Step 3: Push branch**

Run:

```bash
git push
```

Expected: `custom -> custom` push succeeds.

---

## Self-Review

- Spec coverage: Task 2 implements a node that mutates only `guide_data["images"]`; tests verify CRF `0`, CRF `>0`, metadata preservation, and guide-latent identity preservation.
- Placeholder scan: no postponed implementation language is present.
- Type consistency: The node class is `LTXGuideDataImageCompression`, the ComfyUI display name is `LTX Guide Data Image Compression`, and the method is `execute(guide_data, img_compression=0)`.
