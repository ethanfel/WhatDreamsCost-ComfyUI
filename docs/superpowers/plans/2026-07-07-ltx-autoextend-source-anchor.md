# LTX Autoextend Source Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional weak original/source keyframe anchor to LTX autoextend passes while preserving the current tail-latent continuation behavior.

**Architecture:** Keep `_build_extend_pass()` as the single construction point for extension guide data. Add small pure helpers in `ltx_director.py` to resolve anchor mode, cadence, image validity, and latent slicing. `LTXAutoExtend` exposes manual-chain anchor inputs; `LTXExtendInit` stores loop-chain anchor state; `LTXExtendStep` passes that state into `_build_extend_pass()`. Native `LTXExtendLoopOpen` / `LTXExtendLoopClose` remain unchanged except for tests proving state preservation.

**Tech Stack:** Python 3, PyTorch tensors, ComfyUI node schemas, pytest with lightweight ComfyUI module stubs.

---

## File Structure

- Modify: `ltx_director.py`
  - Add anchor helper functions near `_build_extend_pass()`.
  - Extend `_build_extend_pass()` with optional anchor arguments.
  - Add optional anchor inputs to `LTXAutoExtend`.
  - Add optional anchor fields to `LTXExtendInit`.
  - Pass anchor state from `LTXExtendStep` to `_build_extend_pass()`.
- Test: `tests/test_ltx_autoextend_anchor.py`
  - Stub ComfyUI-only imports.
  - Test anchor helper behavior and `_build_extend_pass()` guide-data output.
  - Test `LTXExtendInit` stores anchor state and `LTXExtendStep` honors cadence.
- Test: `tests/test_ltx_extend_loop_state.py`
  - Test native loop state folding preserves anchor fields.
- Modify: `README.md`
  - Add a short usage note for the optional source anchor.

---

### Task 1: Add Failing Anchor Helper Tests

**Files:**
- Create: `tests/test_ltx_autoextend_anchor.py`

- [ ] **Step 1: Create the pytest harness and failing helper tests**

Create `tests/test_ltx_autoextend_anchor.py` with this content:

```python
import importlib.util
import sys
import types
from pathlib import Path

import pytest
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


def _install_stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(ROOT),
        get_output_directory=lambda: str(ROOT),
        get_filename_list=lambda _name: [],
    ))

    comfy_mod = types.ModuleType("comfy")
    comfy_mod.model_management = types.SimpleNamespace(intermediate_device=lambda: torch.device("cpu"))
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


@pytest.fixture()
def ltx_director(monkeypatch):
    _install_stubs(monkeypatch)
    module_name = "wdc.ltx_director"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "ltx_director.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_encode_relay", lambda model, clip, latent, prompt, local_prompts, segment_lengths, epsilon: (model, {"prompt": prompt}))
    return module


def _latent(frames=6, channels=128, height=2, width=3, start=0.0):
    samples = torch.arange(
        start,
        start + channels * frames * height * width,
        dtype=torch.float32,
    ).reshape(1, channels, frames, height, width)
    return {"samples": samples}


def test_anchor_mode_defaults_to_off_for_unknown_values(ltx_director):
    assert ltx_director._normalize_anchor_mode(None) == "off"
    assert ltx_director._normalize_anchor_mode("bad") == "off"
    assert ltx_director._normalize_anchor_mode(" AUTO ") == "auto"


def test_anchor_cadence_uses_one_based_loop_steps(ltx_director):
    assert ltx_director._should_emit_anchor("auto", 0.25, step_index=1, anchor_every_n_steps=2) is True
    assert ltx_director._should_emit_anchor("auto", 0.25, step_index=2, anchor_every_n_steps=2) is False
    assert ltx_director._should_emit_anchor("auto", 0.25, step_index=3, anchor_every_n_steps=2) is True
    assert ltx_director._should_emit_anchor("off", 0.25, step_index=1, anchor_every_n_steps=1) is False
    assert ltx_director._should_emit_anchor("auto", 0.0, step_index=1, anchor_every_n_steps=1) is False


def test_build_extend_pass_without_anchor_keeps_single_tail_guide(ltx_director):
    result = ltx_director._build_extend_pass(
        model=object(),
        clip=object(),
        latent=_latent(frames=6),
        prompt="continue",
        extension_seconds=1.0,
        guide_overlap_seconds=0.5,
        frame_rate=24.0,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is False
    assert len(guide_data["images"]) == 1
    assert len(guide_data["guide_latents"]) == 1
    assert guide_data["strengths"] == [1.0]


def test_build_extend_pass_with_image_anchor_adds_weak_second_guide(ltx_director):
    anchor_image = torch.ones((1, 64, 96, 3), dtype=torch.float32)

    result = ltx_director._build_extend_pass(
        model=object(),
        clip=object(),
        latent=_latent(frames=6),
        prompt="continue",
        extension_seconds=1.0,
        guide_overlap_seconds=0.5,
        frame_rate=24.0,
        anchor_image=anchor_image,
        anchor_mode="image",
        anchor_strength=0.25,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is True
    assert len(guide_data["images"]) == 2
    assert guide_data["images"][1] is anchor_image
    assert guide_data["original_images"][1] is anchor_image
    assert guide_data["insert_frames"] == [0, 0]
    assert guide_data["strengths"] == [1.0, 0.25]
    assert guide_data["segment_numbers"] == [0, -1]
    assert guide_data["guide_latents"][1] is None


def test_build_extend_pass_with_latent_anchor_crops_to_one_frame(ltx_director):
    anchor_latent = _latent(frames=4, start=1000.0)

    result = ltx_director._build_extend_pass(
        model=object(),
        clip=object(),
        latent=_latent(frames=6),
        prompt="continue",
        extension_seconds=1.0,
        guide_overlap_seconds=0.5,
        frame_rate=24.0,
        anchor_latent=anchor_latent,
        anchor_mode="latent",
        anchor_strength=0.2,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is True
    assert len(guide_data["guide_latents"]) == 2
    assert guide_data["guide_latents"][1].shape == (1, 128, 1, 2, 3)
    assert torch.equal(guide_data["guide_latents"][1], anchor_latent["samples"][:, :, :1].clone())


def test_build_extend_pass_auto_falls_back_from_bad_latent_to_image(ltx_director):
    bad_latent = _latent(frames=4, channels=64)
    anchor_image = torch.ones((1, 64, 96, 3), dtype=torch.float32)

    result = ltx_director._build_extend_pass(
        model=object(),
        clip=object(),
        latent=_latent(frames=6),
        prompt="continue",
        extension_seconds=1.0,
        guide_overlap_seconds=0.5,
        frame_rate=24.0,
        anchor_image=anchor_image,
        anchor_latent=bad_latent,
        anchor_mode="auto",
        anchor_strength=0.25,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is True
    assert guide_data["images"][1] is anchor_image
    assert guide_data["guide_latents"][1] is None
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
python -m pytest tests/test_ltx_autoextend_anchor.py -q
```

Expected: FAIL with missing helper attributes such as `_normalize_anchor_mode` or missing `source_anchor_added`.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_ltx_autoextend_anchor.py
git commit -m "test: cover LTX source anchor guide construction"
```

---

### Task 2: Implement Anchor Helpers and Guide Construction

**Files:**
- Modify: `ltx_director.py:1829-1923`
- Test: `tests/test_ltx_autoextend_anchor.py`

- [ ] **Step 1: Add helper functions above `_build_extend_pass()`**

In `ltx_director.py`, insert this code immediately before `_build_extend_pass()`:

```python
_ANCHOR_MODES = {"off", "auto", "image", "latent"}


def _normalize_anchor_mode(anchor_mode):
    mode = str(anchor_mode or "off").strip().lower()
    return mode if mode in _ANCHOR_MODES else "off"


def _positive_int(value, default=1):
    try:
        return max(1, int(value))
    except Exception:
        return int(default)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _should_emit_anchor(anchor_mode, anchor_strength, step_index=1, anchor_every_n_steps=1):
    mode = _normalize_anchor_mode(anchor_mode)
    if mode == "off":
        return False
    if _safe_float(anchor_strength, 0.0) <= 0.0:
        return False
    every = _positive_int(anchor_every_n_steps, 1)
    idx = _positive_int(step_index, 1)
    return ((idx - 1) % every) == 0


def _valid_anchor_image(anchor_image):
    return (
        isinstance(anchor_image, torch.Tensor)
        and anchor_image.ndim == 4
        and int(anchor_image.shape[-1]) >= 3
        and int(anchor_image.shape[0]) >= 1
    )


def _extract_anchor_latent(anchor_latent, target_channels):
    if not isinstance(anchor_latent, dict):
        return None
    samples = anchor_latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        return None
    if int(samples.shape[1]) != int(target_channels):
        return None
    if int(samples.shape[2]) < 1:
        return None
    return samples[0:1, :, :1, :, :].clone()


def _append_source_anchor_guide(
    guide_data,
    dummy_img,
    target_channels,
    anchor_image=None,
    anchor_latent=None,
    anchor_mode="off",
    anchor_strength=0.25,
    anchor_every_n_steps=1,
    step_index=1,
    log_tag="LTXAutoExtend",
):
    if not _should_emit_anchor(anchor_mode, anchor_strength, step_index, anchor_every_n_steps):
        return False

    mode = _normalize_anchor_mode(anchor_mode)
    image_ok = _valid_anchor_image(anchor_image)
    latent_guide = None

    if mode in ("auto", "latent"):
        latent_guide = _extract_anchor_latent(anchor_latent, target_channels)
        if latent_guide is None and anchor_latent is not None:
            log.warning("[%s] source anchor latent is incompatible; falling back to image anchor.", log_tag)

    if latent_guide is None and mode == "latent" and not image_ok:
        return False
    if latent_guide is None and mode in ("auto", "image") and not image_ok:
        return False

    active_image = anchor_image if image_ok and latent_guide is None else dummy_img
    original_image = anchor_image if image_ok else active_image

    guide_data["images"].append(active_image)
    guide_data["original_images"].append(original_image)
    guide_data["insert_frames"].append(0)
    guide_data["strengths"].append(_safe_float(anchor_strength, 0.25))
    guide_data["segment_numbers"].append(-1)
    guide_data["guide_latents"].append(latent_guide)

    log.info(
        "[%s] source anchor active mode=%s latent=%s strength=%.3f step=%d every=%d",
        log_tag,
        mode,
        latent_guide is not None,
        _safe_float(anchor_strength, 0.25),
        _positive_int(step_index, 1),
        _positive_int(anchor_every_n_steps, 1),
    )
    return True
```

- [ ] **Step 2: Extend `_build_extend_pass()` signature**

Change the signature to:

```python
def _build_extend_pass(model, clip, latent, prompt, extension_seconds, guide_overlap_seconds,
                       frame_rate, audio=None, audio_vae=None, guide_strength=1.0, epsilon=1e-3,
                       audio_base_px=0, log_tag="LTXAutoExtend", anchor_image=None,
                       anchor_latent=None, anchor_mode="off", anchor_strength=0.25,
                       anchor_every_n_steps=1, step_index=1):
```

- [ ] **Step 3: Append the optional source anchor after the tail guide**

Immediately after `guide_data` is created, insert:

```python
    source_anchor_added = _append_source_anchor_guide(
        guide_data,
        dummy_img,
        C,
        anchor_image=anchor_image,
        anchor_latent=anchor_latent,
        anchor_mode=anchor_mode,
        anchor_strength=anchor_strength,
        anchor_every_n_steps=anchor_every_n_steps,
        step_index=step_index,
        log_tag=log_tag,
    )
```

Then add this key to the returned dict:

```python
        "source_anchor_added": source_anchor_added,
```

The final return metadata block should include:

```python
        "rel_off_px": tail_start_lat * tsf, "ltxv_length": ltxv_length, "latent_t": latent_t,
        "n_overlap_lat": n_overlap_lat, "window_px": window_px,
        "source_anchor_added": source_anchor_added,
```

- [ ] **Step 4: Run helper and guide construction tests**

Run:

```bash
python -m pytest tests/test_ltx_autoextend_anchor.py -q
```

Expected: PASS for helper and `_build_extend_pass()` tests that do not depend on node input wiring.

- [ ] **Step 5: Commit helper implementation**

Run:

```bash
git add ltx_director.py tests/test_ltx_autoextend_anchor.py
git commit -m "feat: add LTX source anchor guide construction"
```

---

### Task 3: Wire Manual and Loop Node Inputs

**Files:**
- Modify: `ltx_director.py:1940-2152`
- Modify: `tests/test_ltx_autoextend_anchor.py`

- [ ] **Step 1: Add failing tests for loop state and cadence**

Append these tests to `tests/test_ltx_autoextend_anchor.py`:

```python
def test_extend_init_stores_anchor_state(ltx_director):
    anchor_image = torch.ones((1, 64, 96, 3), dtype=torch.float32)
    anchor_latent = _latent(frames=4, start=1000.0)

    (state,) = ltx_director.LTXExtendInit().init(
        seed_latent=_latent(frames=6),
        model="model",
        clip="clip",
        base_seed=100,
        anchor_image=anchor_image,
        anchor_latent=anchor_latent,
        anchor_mode="auto",
        anchor_strength=0.3,
        anchor_every_n_steps=2,
    )

    assert state["anchor_image"] is anchor_image
    assert state["anchor_latent"] is anchor_latent
    assert state["anchor_mode"] == "auto"
    assert state["anchor_strength"] == 0.3
    assert state["anchor_every_n_steps"] == 2


def test_extend_step_honors_anchor_every_n_steps(ltx_director):
    state = {
        "model": object(),
        "clip": object(),
        "latent": _latent(frames=6),
        "base_seed": 100,
        "prompts": ["one", "two"],
        "global_prompt": "",
        "extension_seconds": 1.0,
        "guide_overlap_seconds": 0.5,
        "frame_rate": 24.0,
        "guide_strength": 1.0,
        "epsilon": 1e-3,
        "abs_pos_px": 0,
        "master_audio": None,
        "audio_vae": None,
        "anchor_image": torch.ones((1, 64, 96, 3), dtype=torch.float32),
        "anchor_latent": None,
        "anchor_mode": "image",
        "anchor_strength": 0.25,
        "anchor_every_n_steps": 2,
    }

    step = ltx_director.LTXExtendStep()
    first = step.step(state, index=1)
    second = step.step(state, index=2)

    first_guide_data = first[4]
    second_guide_data = second[4]
    assert len(first_guide_data["images"]) == 2
    assert len(second_guide_data["images"]) == 1
```

- [ ] **Step 2: Run tests and verify wiring failures**

Run:

```bash
python -m pytest tests/test_ltx_autoextend_anchor.py -q
```

Expected: FAIL because `LTXExtendInit.init()` does not accept anchor arguments and `LTXExtendStep.step()` does not pass anchor state yet.

- [ ] **Step 3: Add optional anchor inputs to `LTXAutoExtend.define_schema()`**

In `LTXAutoExtend.define_schema()`, append these inputs after the existing `epsilon` input:

```python
                io.Image.Input(
                    "anchor_image", optional=True,
                    tooltip="Optional pristine/source keyframe used as a weak identity anchor for long extensions.",
                ),
                io.Latent.Input(
                    "anchor_latent", optional=True,
                    tooltip="Optional source latent anchor. When compatible, this avoids VAE re-encoding and uses the first latent frame.",
                ),
                io.Combo.Input(
                    "anchor_mode", options=["off", "auto", "image", "latent"], default="off", optional=True,
                    tooltip="'off' preserves old behavior. 'auto' uses anchor_latent when valid, otherwise anchor_image.",
                ),
                io.Float.Input(
                    "anchor_strength", default=0.25, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Weak source-anchor guide strength. Recommended range: 0.15-0.35.",
                ),
```

- [ ] **Step 4: Pass manual anchor inputs into `_build_extend_pass()`**

Change `LTXAutoExtend.execute()` to this signature:

```python
    def execute(cls, model, clip, latent, prompt="", seed=0, extension_seconds=12.0,
                guide_overlap_seconds=3.0, frame_rate=24.0, audio=None, audio_vae=None,
                guide_strength=1.0, epsilon=1e-3, anchor_image=None, anchor_latent=None,
                anchor_mode="off", anchor_strength=0.25) -> io.NodeOutput:
```

Update the `_build_extend_pass()` call inside `LTXAutoExtend.execute()`:

```python
        r = _build_extend_pass(
            model, clip, latent, prompt, extension_seconds, guide_overlap_seconds, frame_rate,
            audio=audio, audio_vae=audio_vae, guide_strength=guide_strength, epsilon=epsilon,
            audio_base_px=0, log_tag="LTXAutoExtend", anchor_image=anchor_image,
            anchor_latent=anchor_latent, anchor_mode=anchor_mode, anchor_strength=anchor_strength,
        )
```

- [ ] **Step 5: Add anchor state inputs to `LTXExtendInit.INPUT_TYPES()`**

Append these optional entries after `resume_from_seconds`:

```python
                "anchor_image": ("IMAGE", {"tooltip": "Optional pristine/source keyframe to carry as a weak identity anchor."}),
                "anchor_latent": ("LATENT", {"tooltip": "Optional source latent anchor. Uses the first latent frame when compatible."}),
                "anchor_mode": (["off", "auto", "image", "latent"], {"default": "off", "tooltip": "'off' preserves old behavior. 'auto' prefers latent then image."}),
                "anchor_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weak source-anchor guide strength. Recommended range: 0.15-0.35."}),
                "anchor_every_n_steps": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "Emit the source anchor every N extension steps. 1 = every step."}),
```

- [ ] **Step 6: Store anchor fields in `LTXExtendInit.init()`**

Change the signature to:

```python
    def init(self, seed_latent, model, clip, base_seed, prompts=None, global_prompt="",
             audio=None, audio_vae=None, extension_seconds=12.0, guide_overlap_seconds=3.0,
             frame_rate=24.0, guide_strength=1.0, epsilon=1e-3, resume_from_seconds=-1.0,
             anchor_image=None, anchor_latent=None, anchor_mode="off", anchor_strength=0.25,
             anchor_every_n_steps=1):
```

Add these keys to the `state` dict:

```python
            "anchor_image": anchor_image, "anchor_latent": anchor_latent,
            "anchor_mode": _normalize_anchor_mode(anchor_mode),
            "anchor_strength": _safe_float(anchor_strength, 0.25),
            "anchor_every_n_steps": _positive_int(anchor_every_n_steps, 1),
```

- [ ] **Step 7: Pass loop anchor state from `LTXExtendStep.step()`**

Update the `_build_extend_pass()` call inside `LTXExtendStep.step()`:

```python
        r = _build_extend_pass(
            st.get("model"), st.get("clip"), st.get("latent"), prompt,
            st.get("extension_seconds", 12.0), st.get("guide_overlap_seconds", 3.0),
            st.get("frame_rate", 24.0), audio=st.get("master_audio"), audio_vae=st.get("audio_vae"),
            guide_strength=st.get("guide_strength", 1.0), epsilon=st.get("epsilon", 1e-3),
            audio_base_px=int(st.get("abs_pos_px", 0)), log_tag="LTXExtendStep",
            anchor_image=st.get("anchor_image"), anchor_latent=st.get("anchor_latent"),
            anchor_mode=st.get("anchor_mode", "off"), anchor_strength=st.get("anchor_strength", 0.25),
            anchor_every_n_steps=st.get("anchor_every_n_steps", 1), step_index=idx,
        )
```

- [ ] **Step 8: Run all anchor tests**

Run:

```bash
python -m pytest tests/test_ltx_autoextend_anchor.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit node wiring**

Run:

```bash
git add ltx_director.py tests/test_ltx_autoextend_anchor.py
git commit -m "feat: wire LTX source anchor inputs"
```

---

### Task 4: Test Native Loop State Preservation

**Files:**
- Create: `tests/test_ltx_extend_loop_state.py`
- Modify: `ltx_extend_loop.py` only if the test exposes state loss.

- [ ] **Step 1: Add native loop state preservation tests**

Create `tests/test_ltx_extend_loop_state.py` with this content:

```python
import torch

import ltx_extend_loop


def _latent(frames=6):
    return {"samples": torch.zeros((1, 128, frames, 2, 3), dtype=torch.float32)}


def test_fold_latent_preserves_anchor_fields():
    anchor_image = torch.ones((1, 64, 96, 3), dtype=torch.float32)
    anchor_latent = _latent(frames=4)
    state = {
        "latent": _latent(frames=6),
        "frame_rate": 24.0,
        "guide_overlap_seconds": 0.5,
        "abs_pos_px": 0,
        "anchor_image": anchor_image,
        "anchor_latent": anchor_latent,
        "anchor_mode": "auto",
        "anchor_strength": 0.25,
        "anchor_every_n_steps": 2,
    }
    passed_latent = _latent(frames=5)

    folded = ltx_extend_loop._fold_latent(state, passed_latent)

    assert folded["latent"] is passed_latent
    assert folded["anchor_image"] is anchor_image
    assert folded["anchor_latent"] is anchor_latent
    assert folded["anchor_mode"] == "auto"
    assert folded["anchor_strength"] == 0.25
    assert folded["anchor_every_n_steps"] == 2
    assert folded["abs_pos_px"] == 32
```

- [ ] **Step 2: Run the native loop test**

Run:

```bash
python -m pytest tests/test_ltx_extend_loop_state.py -q
```

Expected: PASS. `_fold_latent()` copies the state dict and should preserve unknown anchor keys without code changes.

- [ ] **Step 3: Commit native loop test**

Run:

```bash
git add tests/test_ltx_extend_loop_state.py
git commit -m "test: cover LTX native loop anchor state"
```

---

### Task 5: Document Source Anchor Usage

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README guidance**

In `README.md`, add this section near the LTX Director / extension documentation:

```markdown
### LTX Auto Extend Source Anchor

For long LTX 2.3 extension runs, keep the normal tail-latent continuation guide enabled and optionally add a weak source anchor:

- Connect `LTX Keyframe Out.original` to `anchor_image` when you want the pristine source frame to stabilize identity.
- Use `anchor_mode: auto` for new long runs. Existing workflows default to `off`.
- Keep `anchor_strength` low. Recommended range: `0.15` to `0.35`; the default is `0.25`.
- In loop workflows, `anchor_every_n_steps` controls how often the source anchor is emitted. Use `1` for maximum identity stability, or `2+` when the anchor pulls too hard.

The previous-pass tail latent remains the primary continuity guide. The source anchor is only a weak identity and texture reference.
```

- [ ] **Step 2: Review the README placement**

Run:

```bash
rg -n "LTX Auto Extend Source Anchor|anchor_strength|anchor_every_n_steps" README.md
```

Expected: The new section and key settings are listed once.

- [ ] **Step 3: Commit documentation**

Run:

```bash
git add README.md
git commit -m "docs: explain LTX source anchor usage"
```

---

### Task 6: Final Verification

**Files:**
- Verify: `ltx_director.py`
- Verify: `ltx_extend_loop.py`
- Verify: `tests/test_ltx_autoextend_anchor.py`
- Verify: `tests/test_ltx_extend_loop_state.py`
- Verify: `README.md`

- [ ] **Step 1: Run all tests**

Run:

```bash
python -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 2: Compile touched Python files**

Run:

```bash
python -m py_compile ltx_director.py ltx_extend_loop.py tests/test_ltx_autoextend_anchor.py tests/test_ltx_extend_loop_state.py
```

Expected: No output and exit code 0.

- [ ] **Step 3: Check working tree**

Run:

```bash
git status --short
```

Expected: No modified tracked files from this feature. Existing unrelated untracked patch files may still appear.

- [ ] **Step 4: Inspect recent commits**

Run:

```bash
git log --oneline -6
```

Expected: The feature commits appear above the plan/spec commits.

---

## Self-Review Notes

Spec coverage:

- Tail latent remains first and unchanged: Task 2.
- Weak source anchor guide: Task 2.
- Manual chain inputs: Task 3.
- Loop state inputs and cadence: Task 3.
- Native loop preservation: Task 4.
- Backward-compatible default `off`: Task 3.
- Testing and documentation: Tasks 1, 4, 5, 6.

Type consistency:

- `anchor_image` uses `IMAGE` / tensor shape `[B,H,W,C]`.
- `anchor_latent` uses `LATENT` / dict with `samples`.
- `anchor_mode` values are `off`, `auto`, `image`, `latent`.
- `anchor_strength` is a float.
- `anchor_every_n_steps` and `step_index` are positive integers.
