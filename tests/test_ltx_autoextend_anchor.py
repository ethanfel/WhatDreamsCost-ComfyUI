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
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "ltx_director.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
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


@pytest.mark.parametrize("anchor_strength", [float("nan"), float("inf"), float("-inf")])
def test_build_extend_pass_skips_non_finite_anchor_strengths(ltx_director, anchor_strength):
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
        anchor_strength=anchor_strength,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is False
    assert len(guide_data["images"]) == 1
    assert guide_data["strengths"] == [1.0]


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
    assert torch.equal(guide_data["images"][1], anchor_image)
    assert torch.equal(guide_data["original_images"][1], anchor_image)
    assert guide_data["insert_frames"] == [0, 0]
    assert guide_data["strengths"] == [1.0, 0.25]
    assert guide_data["segment_numbers"] == [0, -1]
    assert guide_data["guide_latents"][1] is None


@pytest.mark.parametrize(
    "anchor_image",
    [
        torch.ones((1, 0, 96, 3), dtype=torch.float32),
        torch.ones((1, 64, 0, 3), dtype=torch.float32),
    ],
)
def test_build_extend_pass_skips_empty_anchor_images(ltx_director, anchor_image):
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
    assert result["source_anchor_added"] is False
    assert len(guide_data["images"]) == 1
    assert len(guide_data["guide_latents"]) == 1


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


@pytest.mark.parametrize(
    "samples",
    [
        torch.empty((0, 128, 4, 2, 3), dtype=torch.float32),
        torch.empty((1, 128, 4, 0, 3), dtype=torch.float32),
        torch.empty((1, 128, 4, 2, 0), dtype=torch.float32),
    ],
)
def test_build_extend_pass_skips_empty_anchor_latents(ltx_director, samples):
    result = ltx_director._build_extend_pass(
        model=object(),
        clip=object(),
        latent=_latent(frames=6),
        prompt="continue",
        extension_seconds=1.0,
        guide_overlap_seconds=0.5,
        frame_rate=24.0,
        anchor_latent={"samples": samples},
        anchor_mode="latent",
        anchor_strength=0.25,
    )

    guide_data = result["guide_data"]
    assert result["source_anchor_added"] is False
    assert len(guide_data["images"]) == 1
    assert len(guide_data["guide_latents"]) == 1


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
    assert torch.equal(guide_data["images"][1], anchor_image)
    assert guide_data["guide_latents"][1] is None
