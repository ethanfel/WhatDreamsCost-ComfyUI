import importlib.util
import json
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


def _pattern_image():
    vals = torch.tensor(
        [[0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return vals.reshape(1, 4, 4, 1).repeat(1, 1, 1, 3)


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


def test_node_metadata_is_json_serializable(monkeypatch):
    ltx_director = _load_ltx_director(monkeypatch)
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
