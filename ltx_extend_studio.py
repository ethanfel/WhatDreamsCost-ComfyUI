"""LTX Extend Prompt Studio — audition the master audio per extension step and write its prompt.

Does LTX Extend Init's job (packs the loop 'state') PLUS an interactive, audio-segmented prompt editor.
It slices the master audio into the exact windows the extend loop will generate (same pixel-frame math),
lets you play each section and write its prompt on a Director-style timeline, and outputs the full state
(with the ordered prompts baked in) straight to LTX Extend Loop Open — no separate Init needed.

Frontend (js/ltx_extend_studio.js) authors the prompts into the hidden 'prompts_json' widget and reads
the segment layout this node broadcasts on execute (it also serves the master audio for playback).
"""

import json
import logging
import math
import os

try:
    import torch
except Exception:
    torch = None

try:
    import folder_paths
    from server import PromptServer
except Exception:
    folder_paths = None
    PromptServer = None

log = logging.getLogger(__name__)

_TSF = 8  # LTX temporal downscale (8n+1)


class _StudioAny(str):
    def __ne__(self, _other):
        return False


_STUDIO_ANY = _StudioAny("*")


def _extend_segments(seed_lat_frames, extension_s, overlap_s, fps, total):
    """Per-step windows in pixel frames, matching LTX Extend Loop / Step exactly.

    step start(i) = seed_offset + (i-1)*steady_advance  (1-based i)
      seed_offset   = (seed_lat_frames - n_overlap) * 8         (window 1 begins here)
      steady_advance = (T_window - n_overlap) * 8               (new-content length per step)
    Each window is ltxv_length px; the first `overlap_decoded` px of a window is the continuation
    from the previous step (shaded in the UI)."""
    tsf = _TSF
    fps = float(fps) if fps else 24.0
    overlap_px = max(1, int(round(float(overlap_s) * fps)))
    ext_px = max(1, int(round(float(extension_s) * fps)))
    window_px = overlap_px + ext_px
    ltxv = int(math.ceil((window_px - 1) / 8.0) * 8) + 1
    t_win = ((ltxv - 1) // 8) + 1
    n_ov_seed = max(1, min(int(seed_lat_frames), (overlap_px + tsf - 1) // tsf))
    n_ov_win = max(1, min(t_win, (overlap_px + tsf - 1) // tsf))
    seed_off = max(0, (int(seed_lat_frames) - n_ov_seed) * tsf)
    steady = (t_win - n_ov_win) * tsf
    overlap_dec = (n_ov_win - 1) * tsf + 1
    segs = []
    for i in range(int(total)):
        start = seed_off + i * steady
        segs.append({
            "index": i + 1,
            "start_px": start,
            "len_px": ltxv,
            "overlap_px": overlap_dec,          # leading continuation region (shade it)
            "start_s": round(start / fps, 3),
            "len_s": round(ltxv / fps, 3),
        })
    return segs


def _parse_prompts_json(prompts_json, total):
    """The JS stores an object {"1": "...", "2": "..."} (or a list). Return an ordered list length=total."""
    out = ["" for _ in range(int(total))]
    if not prompts_json:
        return out
    try:
        data = json.loads(prompts_json) if isinstance(prompts_json, str) else prompts_json
    except Exception:
        return out
    if isinstance(data, list):
        for i, v in enumerate(data):
            if i < len(out):
                out[i] = str(v or "")
    elif isinstance(data, dict):
        for k, v in data.items():
            try:
                i = int(k) - 1  # keys are 1-based step numbers
            except (ValueError, TypeError):
                continue
            if 0 <= i < len(out):
                out[i] = str(v or "")
    return out


def _serve_master_audio(audio, node_id, segments):
    """Save the master audio to the temp dir and broadcast its /view URL + the segment layout to the
    frontend so the Studio can render the waveform and play each section. Best-effort."""
    if PromptServer is None or folder_paths is None or torch is None:
        return
    url = None
    try:
        if isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor):
            import torchaudio  # optional; only for playback
            wf = audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]
            sr = int(audio.get("sample_rate", 44100))
            tmp = folder_paths.get_temp_directory()
            os.makedirs(tmp, exist_ok=True)
            fname = f"ltx_studio_{node_id}.wav"
            torchaudio.save(os.path.join(tmp, fname), wf.cpu().float().clamp(-1, 1), sr)
            url = f"/view?filename={fname}&type=temp&rand={node_id}"
    except Exception as e:
        log.info("[LTXExtendPromptStudio] audio preview unavailable (%s) — segments/prompts still work.", e)
    try:
        PromptServer.instance.send_sync("ltx_studio_layout",
                                        {"node_id": str(node_id), "audio_url": url, "segments": segments})
    except Exception:
        pass


class LTXExtendPromptStudio:
    """Init + audio-segmented prompt studio: outputs the extend loop 'state' with per-step prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_latent": ("LATENT", {"tooltip": "The first clip's latent (seed to extend). Its length sets the first window."}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "master_audio": ("AUDIO", {"tooltip": "Full master audio, aligned to the seed's start. Segmented into per-step windows."}),
                "total": ("INT", {"default": 14, "min": 1, "max": 100000, "step": 1, "tooltip": "Number of extension steps (segments)."}),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "Audio VAE -> per-pass preserve-masked audio_latent for lipsync."}),
                "global_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Fallback prompt for any step left blank in the studio."}),
                "extension_seconds": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "guide_overlap_seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "prompts_json": ("STRING", {"default": "", "multiline": False, "tooltip": "Authored per-step prompts (managed by the Studio UI)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LTX_EXTEND_STATE", "STRING")
    RETURN_NAMES = ("state", "prompts")
    FUNCTION = "build"
    CATEGORY = "WhatDreamsCost"

    def build(self, seed_latent, model, clip, master_audio, total, base_seed,
              audio_vae=None, global_prompt="", extension_seconds=12.0, guide_overlap_seconds=3.0,
              frame_rate=24.0, guide_strength=1.0, epsilon=1e-3, prompts_json="", unique_id=None):
        fps = float(frame_rate) if frame_rate else 24.0
        seed_frames = int(seed_latent["samples"].shape[2]) if isinstance(seed_latent, dict) else 1
        segments = _extend_segments(seed_frames, extension_seconds, guide_overlap_seconds, fps, total)
        prompts = _parse_prompts_json(prompts_json, total)

        state = {
            "model": model, "clip": clip, "audio_vae": audio_vae, "master_audio": master_audio,
            "prompts": prompts, "global_prompt": global_prompt or "", "base_seed": int(base_seed),
            "latent": seed_latent, "abs_pos_px": 0,
            "extension_seconds": float(extension_seconds), "guide_overlap_seconds": float(guide_overlap_seconds),
            "frame_rate": fps, "guide_strength": float(guide_strength), "epsilon": float(epsilon),
        }

        _serve_master_audio(master_audio, unique_id, segments)
        nfilled = sum(1 for p in prompts if str(p).strip())
        log.info("[LTXExtendPromptStudio] %d segments (seed %d lat frames), %d/%d prompts filled, base_seed=%d",
                 len(segments), seed_frames, nfilled, int(total), int(base_seed))
        return (state, "\n".join(prompts))


STUDIO_NODE_CLASS_MAPPINGS = {"LTXExtendPromptStudio": LTXExtendPromptStudio}
STUDIO_NODE_DISPLAY_NAME_MAPPINGS = {"LTXExtendPromptStudio": "LTX Extend Prompt Studio"}
