# LTX Guide Data Chainable Degrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three chainable `GUIDE_DATA` transform nodes: image resample degradation, deterministic image noise, and keyframe strength override.

**Architecture:** Keep each node small and single-purpose in `ltx_director.py`, next to `LTXGuideDataImageCompression`. Add shared helper functions for copying guide data, transforming image lists, parsing strength lists, and resizing image tensors. Register each legacy node with string socket types such as `"GUIDE_DATA"` so `/object_info` stays JSON serializable.

**Tech Stack:** Python ComfyUI custom node mappings, PyTorch tensor image operations, pytest, `py_compile`.

---

## File Structure

- Modify: `ltx_director.py`
  - Add guide-data copy helpers.
  - Add `LTXGuideDataImageResampleDegrade`.
  - Add `LTXGuideDataImageNoise`.
  - Add `LTXGuideDataStrengthOverride`.
- Modify: `__init__.py`
  - Import and register the three new nodes.
- Modify: `tests/test_ltx_guide_data_image_compression.py`
  - Extend existing guide-data tests with the new node behavior and metadata checks.
- Add: `docs/superpowers/plans/2026-07-09-ltx-guide-data-chainable-degrade.md`

---

### Task 1: Add Failing Tests

**Files:**
- Modify: `tests/test_ltx_guide_data_image_compression.py`

- [ ] **Step 1: Add tests for the three new nodes**

Append these tests to `tests/test_ltx_guide_data_image_compression.py`:

```python
def _pattern_image():
    vals = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return vals.reshape(1, 4, 4, 1).repeat(1, 1, 1, 3)


def test_resample_degrade_changes_images_only(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, _image_a, _image_b, original, latent = _guide_data()
    pattern = _pattern_image()
    gd["images"] = [pattern]

    (out,) = ltx_director.LTXGuideDataImageResampleDegrade().execute(gd, scale=0.5, method="bilinear")

    assert out is not gd
    assert out["images"] is not gd["images"]
    assert out["images"][0].shape == pattern.shape
    assert not torch.equal(out["images"][0], pattern)
    assert torch.equal(gd["images"][0], pattern)
    assert out["original_images"][0] is original
    assert out["guide_latents"][0] is latent


def test_noise_degrade_is_seeded_and_changes_images_only(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, _image_a, _image_b, original, latent = _guide_data()
    base = torch.full((1, 4, 4, 3), 0.5, dtype=torch.float32)
    gd["images"] = [base]

    node = ltx_director.LTXGuideDataImageNoise()
    (out_a,) = node.execute(gd, amount=0.05, seed=123)
    (out_b,) = node.execute(gd, amount=0.05, seed=123)
    (out_c,) = node.execute(gd, amount=0.05, seed=124)

    assert torch.equal(out_a["images"][0], out_b["images"][0])
    assert not torch.equal(out_a["images"][0], out_c["images"][0])
    assert not torch.equal(out_a["images"][0], base)
    assert out_a["original_images"][0] is original
    assert out_a["guide_latents"][0] is latent


def test_strength_override_replace_by_keyframe_order(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, *_ = _guide_data()

    (out,) = ltx_director.LTXGuideDataStrengthOverride().execute(
        gd, strengths="0.75, 1.0, 0.25", mode="replace",
    )

    assert out["strengths"] == [0.75, 1.0]
    assert out["images"] == gd["images"]
    assert out["guide_latents"] == gd["guide_latents"]


def test_strength_override_multiply_by_keyframe_order(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
    gd, *_ = _guide_data()

    (out,) = ltx_director.LTXGuideDataStrengthOverride().execute(
        gd, strengths="0.5", mode="multiply",
    )

    assert out["strengths"] == [0.5, 0.5]
    assert gd["strengths"] == [1.0, 0.5]
```

- [ ] **Step 2: Extend metadata serialization coverage**

Change `test_node_metadata_is_json_serializable` so it loops over:

```python
for name in [
    "LTXGuideDataImageCompression",
    "LTXGuideDataImageResampleDegrade",
    "LTXGuideDataImageNoise",
    "LTXGuideDataStrengthOverride",
]:
    node = getattr(ltx_director, name)
    json.dumps({
        "input": node.INPUT_TYPES(),
        "return": node.RETURN_TYPES,
        "return_names": node.RETURN_NAMES,
        "function": node.FUNCTION,
        "category": node.CATEGORY,
    })
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
pytest tests/test_ltx_guide_data_image_compression.py -q
```

Expected: FAIL with missing attributes for the new classes.

---

### Task 2: Implement Chainable Nodes

**Files:**
- Modify: `ltx_director.py`
- Modify: `__init__.py`

- [ ] **Step 1: Add helpers in `ltx_director.py` before `LTXGuideDataImageCompression`**

```python
def _copy_guide_data(guide_data):
    src = guide_data or {}
    out = dict(src)
    for key, value in src.items():
        if isinstance(value, list):
            out[key] = list(value)
    return out


def _guide_images(src):
    return list((src or {}).get("images", []) or [])
```

- [ ] **Step 2: Update `LTXGuideDataImageCompression.execute()`**

Use the helpers:

```python
src = guide_data or {}
out = _copy_guide_data(src)
images = _guide_images(src)
```

- [ ] **Step 3: Add `LTXGuideDataImageResampleDegrade`**

Implement a legacy node with `"GUIDE_DATA"` input/output, `scale` float, `method` combo `["nearest", "bilinear", "bicubic", "area"]`, and image-only downscale/upscale using `torch.nn.functional.interpolate`.

- [ ] **Step 4: Add `LTXGuideDataImageNoise`**

Implement a legacy node with `"GUIDE_DATA"` input/output, `amount` float, `seed` int, deterministic CPU generator, and image-only `clamp(0, 1)` output.

- [ ] **Step 5: Add `LTXGuideDataStrengthOverride`**

Implement a legacy node with `"GUIDE_DATA"` input/output, `strengths` string, `mode` combo `["replace", "multiply"]`, and keyframe-order mapping to `guide_data["strengths"]`.

- [ ] **Step 6: Register nodes in `__init__.py`**

Import and add mappings/display names:

```python
LTXGuideDataImageResampleDegrade
LTXGuideDataImageNoise
LTXGuideDataStrengthOverride
```

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```bash
pytest tests/test_ltx_guide_data_image_compression.py -q
```

Expected: PASS.

---

### Task 3: Verify And Commit

**Files:**
- Verify: `ltx_director.py`
- Verify: `__init__.py`
- Verify: `tests/test_ltx_guide_data_image_compression.py`
- Verify: `docs/superpowers/plans/2026-07-09-ltx-guide-data-chainable-degrade.md`

- [ ] **Step 1: Run focused test suite**

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

- [ ] **Step 3: Stage, commit, push**

Run:

```bash
git add ltx_director.py __init__.py tests/test_ltx_guide_data_image_compression.py docs/superpowers/plans/2026-07-09-ltx-guide-data-chainable-degrade.md
git commit -m "Add chainable guide data degrade nodes"
git push
```

Expected: push succeeds to `origin/custom`.

---

## Self-Review

- Spec coverage: image resample, image noise, and strength override each have a task and tests.
- Placeholder scan: no postponed implementation language is present.
- Type consistency: all legacy node sockets use JSON-serializable string type names.

