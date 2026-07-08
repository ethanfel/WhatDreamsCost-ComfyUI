import logging
import asyncio
import json
import base64
import io as _io
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
import av
from PIL import Image

import os
import platform
import folder_paths
import comfy.model_management
from server import PromptServer
from aiohttp import web

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)

from .patches import detect_model_type, apply_patches

log = logging.getLogger(__name__)

# Setup global event loop exception handler to silence ConnectionResetError (WinError 10054/10053) on Windows
try:
    loop = None
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except Exception:
            pass

    if loop is not None:
        old_handler = loop.get_exception_handler()
        
        def silence_connection_reset_handler(loop, context):
            exception = context.get('exception')
            if (isinstance(exception, (ConnectionResetError, ConnectionAbortedError)) or 
                (isinstance(exception, OSError) and getattr(exception, 'winerror', None) in (10054, 10053))):
                # Suppress WinError 10054 and WinError 10053 tracebacks in logging
                return
            if old_handler:
                old_handler(loop, context)
            else:
                loop.default_exception_handler(context)
                
        loop.set_exception_handler(silence_connection_reset_handler)
except Exception:
    pass

# Custom socket type shared with LTXSequencer
GuideData = io.Custom("GUIDE_DATA")
MotionGuideData = io.Custom("MOTION_GUIDE_DATA")

# --- File Check Endpoint for Deduplication ---
@PromptServer.instance.routes.get("/ltx_director_check_file")
async def ltx_director_check_file(request):
    filename = request.query.get("filename", "")
    file_size = request.query.get("size", "")
    if not filename:
        return web.json_response({"exists": False})

    upload_dir = folder_paths.get_input_directory()
    temp_dir = os.path.join(upload_dir, "whatdreamscost")
    
    # 1. Check if the exact filename exists in whatdreamscost or root input dir
    possible_paths = [
        os.path.join(temp_dir, filename),
        os.path.join(upload_dir, filename)
    ]
    
    found_path = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            if file_size:
                try:
                    if os.path.getsize(p) == int(file_size):
                        found_path = p
                        break
                except ValueError:
                    found_path = p
                    break
            else:
                found_path = p
                break
                
    if found_path:
        rel_name = os.path.relpath(found_path, upload_dir).replace('\\', '/')
        return web.json_response({"exists": True, "name": rel_name})

    # 2. Suffix search if exact match not found
    base_name = os.path.basename(filename)
    suffix = f"_{base_name}"
    try:
        for search_dir in [temp_dir, upload_dir]:
            if os.path.exists(search_dir):
                for f_name in os.listdir(search_dir):
                    if f_name.endswith(suffix) or f_name == base_name:
                        pot_path = os.path.join(search_dir, f_name)
                        if os.path.isfile(pot_path):
                            if file_size:
                                try:
                                    if os.path.getsize(pot_path) == int(file_size):
                                        rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                        return web.json_response({"exists": True, "name": rel_name})
                                except ValueError:
                                    pass
                            else:
                                rel_name = os.path.relpath(pot_path, upload_dir).replace('\\', '/')
                                return web.json_response({"exists": True, "name": rel_name})
    except Exception as e:
        log.warning(f"[LTXDirector] Error listing input directory: {e}")

    return web.json_response({"exists": False})


def read_wav_peaks(wav_path):
    import wave
    peaks = []
    with wave.open(wav_path, 'rb') as w:
        n_frames = w.getnframes()
        if n_frames > 0:
            frames_bytes = w.readframes(n_frames)
            samples = np.frombuffer(frames_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
        else:
            peaks = [0.0] * 200
    return peaks


def extract_audio_from_video(video_path):
    import wave
    try:
        base, _ = os.path.splitext(video_path)
        output_wav = base + "_extracted_audio.wav"
        
        # Check if already exists, is not empty, and has the correct 44100Hz sample rate
        if os.path.exists(output_wav) and os.path.getsize(output_wav) > 44:
            try:
                with wave.open(output_wav, 'rb') as w_check:
                    if w_check.getframerate() == 44100:
                        peaks = read_wav_peaks(output_wav)
                        input_dir = folder_paths.get_input_directory()
                        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
                        return rel_output, peaks
            except Exception:
                pass

        # Decode the video using PyAV
        with av.open(video_path) as container:
            if not container.streams.audio:
                return None, None
            stream = container.streams.audio[0]
            
            # Setup resampler to 44100Hz, Mono, signed 16-bit integer (s16)
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=44100,
            )
            
            audio_bytes = bytearray()
            
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
                    
            # Flush resampler
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
                
            if not audio_bytes:
                return None, None
                
            # Write WAV file
            with wave.open(output_wav, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2) # 16-bit
                w.setframerate(44100)
                w.writeframes(audio_bytes)
                
        # Calculate peaks
        peaks = []
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        num_peaks = 200
        step = max(1, len(samples) // num_peaks)
        for i in range(num_peaks):
            chunk = samples[i * step : (i + 1) * step]
            if len(chunk) > 0:
                max_val = np.max(np.abs(chunk)) / 32767.0
                peaks.append(float(max_val))
            else:
                peaks.append(0.0)
                
        input_dir = folder_paths.get_input_directory()
        rel_output = os.path.relpath(output_wav, input_dir).replace('\\', '/')
        return rel_output, peaks
    except Exception as e:
        print(f"[LTXDirector] Server audio extraction failed: {e}")
        return None, None


def get_audio_peaks(audio_path):
    import wave
    # If it is already a WAV file, read peaks directly
    _, ext = os.path.splitext(audio_path)
    if ext.lower() == ".wav":
        try:
            return read_wav_peaks(audio_path)
        except Exception:
            pass # fallback to PyAV
            
    # Use PyAV to decode and resample the audio file
    try:
        with av.open(audio_path) as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=8000,
            )
            audio_bytes = bytearray()
            for frame in container.decode(stream):
                for resampled_frame in resampler.resample(frame):
                    arr = resampled_frame.to_ndarray()
                    audio_bytes.extend(arr.tobytes())
            for resampled_frame in resampler.resample(None):
                arr = resampled_frame.to_ndarray()
                audio_bytes.extend(arr.tobytes())
                
            if not audio_bytes:
                return None
                
            peaks = []
            samples = np.frombuffer(audio_bytes, dtype=np.int16)
            num_peaks = 200
            step = max(1, len(samples) // num_peaks)
            for i in range(num_peaks):
                chunk = samples[i * step : (i + 1) * step]
                if len(chunk) > 0:
                    max_val = np.max(np.abs(chunk)) / 32767.0
                    peaks.append(float(max_val))
                else:
                    peaks.append(0.0)
            return peaks
    except Exception as e:
        print(f"[LTXDirector] Failed to get audio peaks via PyAV: {e}")
        return None


@PromptServer.instance.routes.get("/ltx_director_get_audio")
async def ltx_director_get_audio(request):
    filename = request.query.get("filename")
    if not filename:
        return web.json_response({"error": "Missing filename"}, status=400)

    upload_dir = folder_paths.get_input_directory()
    
    clean_filename = filename.replace('\\', '/')
    file_path = os.path.join(upload_dir, clean_filename)
    if not os.path.exists(file_path):
        basename = os.path.basename(clean_filename)
        temp_path = os.path.join(upload_dir, "whatdreamscost", basename)
        if os.path.exists(temp_path):
            file_path = temp_path
        else:
            file_path = os.path.join(upload_dir, basename)
        
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return web.json_response({"error": "File not found"}, status=404)

    _, ext = os.path.splitext(file_path)
    is_audio = ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]
    
    if is_audio:
        peaks = None
        try:
            peaks = get_audio_peaks(file_path)
        except Exception as e:
            print(f"[LTXDirector] Failed to get audio peaks for audio file: {e}")
            
        rel_path = os.path.relpath(file_path, upload_dir).replace('\\', '/')
        return web.json_response({
            "audio_file": rel_path,
            "peaks": peaks
        })

    audio_file, peaks = None, None
    try:
        loop = asyncio.get_event_loop()
        audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
    except Exception as e:
        print(f"[LTXDirector] Error extracting audio: {e}")

    return web.json_response({
        "audio_file": audio_file,
        "peaks": peaks
    })


@PromptServer.instance.routes.get("/ltx_director_open_folder")
async def ltx_director_open_folder(request):
    upload_dir = os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)
    try:
        if hasattr(os, "startfile"):
            os.startfile(upload_dir)
        else:
            import webbrowser
            webbrowser.open(os.path.abspath(upload_dir))
        return web.json_response({"success": True})
    except Exception as e:
        print(f"[LTXDirector] Failed to open workspace folder: {e}")
        return web.json_response({"success": False, "error": str(e)}, status=500)


@PromptServer.instance.routes.post("/ltx_director_extract_marked_audio")
async def ltx_director_extract_marked_audio(request):
    """Extract the timeline audio inside the marked zone [start_frame, end_frame) and
    save it as a WAV in the ComfyUI output directory. Reuses _build_combined_audio."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"success": False, "error": "invalid JSON"}, status=400)

    timeline_data = data.get("timeline_data", "") or ""
    try:
        start_frame = int(data.get("start_frame", 0))
        end_frame = int(data.get("end_frame", 0))
        frame_rate = float(data.get("frame_rate", 24) or 24)
    except (TypeError, ValueError):
        return web.json_response({"success": False, "error": "bad numeric params"}, status=400)
    override_audio = bool(data.get("override_audio", False))
    name_hint = str(data.get("name", "") or "marked_audio")

    duration = end_frame - start_frame
    if duration <= 0:
        return web.json_response({"success": False, "error": "empty marked zone (end must be after start)"}, status=400)
    if not timeline_data:
        return web.json_response({"success": False, "error": "no timeline data"}, status=400)

    def _work():
        import wave
        import re
        audio = _build_combined_audio(timeline_data, start_frame, duration, frame_rate, override_audio)
        wf = audio.get("waveform")
        sr = int(audio.get("sample_rate", 44100))
        if wf is None or wf.numel() == 0:
            return None, None, True
        arr = wf[0].clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).cpu().numpy()  # [channels, samples]
        if arr.ndim == 1:
            arr = np.stack([arr, arr], axis=0)
        channels, n = arr.shape[0], arr.shape[1]
        is_silent = bool(n == 0 or int(np.max(np.abs(arr))) == 0)
        interleaved = np.empty((n * channels,), dtype=np.int16)
        for c in range(channels):
            interleaved[c::channels] = arr[c]

        out_dir = folder_paths.get_output_directory()
        os.makedirs(out_dir, exist_ok=True)
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name_hint).strip('_') or "marked_audio"
        base = f"{safe}_{start_frame}-{end_frame}"
        fname = base + ".wav"
        path = os.path.join(out_dir, fname)
        i = 1
        while os.path.exists(path):
            fname = f"{base}_{i}.wav"
            path = os.path.join(out_dir, fname)
            i += 1
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(interleaved.tobytes())
        return fname, path, is_silent

    try:
        loop = asyncio.get_event_loop()
        fname, path, is_silent = await loop.run_in_executor(None, _work)
    except Exception as e:
        log.error("[LTXDirector] extract marked audio failed: %s", e)
        return web.json_response({"success": False, "error": str(e)}, status=500)

    if fname is None:
        return web.json_response({"success": False, "error": "no audio produced"}, status=200)

    return web.json_response({
        "success": True,
        "filename": fname,
        "path": path,
        "url": f"/view?filename={fname}&type=output",
        "silent": is_silent,
    })


def _read_and_write_file_chunk(file, file_path, mode):
    chunk_bytes = file.file.read()
    with open(file_path, mode) as f:
        f.write(chunk_bytes)


# --- LTX Director Chunked Video Upload Endpoint ---
# Bypasses the 413 Payload Too Large error for large video files.
# This endpoint is self-contained and independent of any other node.
@PromptServer.instance.routes.post("/ltx_director_upload_chunk")
async def ltx_director_upload_chunk(request):
    post = await request.post()
    file = post.get("file")
    filename = post.get("filename")
    chunk_index = int(post.get("chunk_index"))
    total_chunks = int(post.get("total_chunks"))

    upload_dir = os.path.join(folder_paths.get_input_directory(), "whatdreamscost")
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename to prevent path traversal attacks (e.g. ../../etc/passwd)
    filename = os.path.basename(filename)
    file_path = os.path.join(upload_dir, filename)

    # Belt-and-suspenders: confirm the resolved path is still inside the upload directory
    if not os.path.realpath(file_path).startswith(os.path.realpath(upload_dir)):
        return web.json_response({"error": "Invalid filename"}, status=400)

    # Append chunk to file (write fresh on first chunk, append on subsequent)
    mode = "ab" if chunk_index > 0 else "wb"
    
    # Offload the blocking read/write disk I/O to a thread executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _read_and_write_file_chunk, file, file_path, mode)

    if chunk_index == total_chunks - 1:
        audio_file, peaks = None, None
        try:
            audio_file, peaks = await loop.run_in_executor(None, extract_audio_from_video, file_path)
        except Exception as e:
            print(f"[LTXDirector] Error in final chunk audio extraction: {e}")
            
        return web.json_response({
            "name": f"whatdreamscost/{filename}",
            "audio_file": audio_file,
            "peaks": peaks
        })
    return web.json_response({"status": "ok"})



def _load_image_tensor(seg: dict) -> torch.Tensor:
    """Decode an image from the ComfyUI input folder (if imageFile provided) or fallback to base64
    to a ComfyUI-style image tensor of shape [1, H, W, 3], float32 in [0, 1]."""
    if seg.get("imageFile"):
        file_path = os.path.join(folder_paths.get_input_directory(), seg["imageFile"])
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if not b64_str or b64_str.startswith("/view?"):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    
    try:
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

def _load_video_tensor(seg: dict, frame_rate: float) -> torch.Tensor:
    """Extracts a sequence of frames from a video file based on the segment's trim parameters,
    and returns them as an [N, H, W, 3] float32 tensor."""
    file_path = os.path.join(folder_paths.get_input_directory(), seg.get("imageFile", ""))
    
    if not os.path.exists(file_path):
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

    trim_start_frames = float(seg.get("trimStart", 0))
    length_frames = float(seg.get("length", 1))
    start_sec = trim_start_frames / frame_rate
    
    frames = []
    try:
        with av.open(file_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            
            # Seek slightly before target to hit a keyframe
            if stream.time_base:
                seek_pts = int((max(0, start_sec - 0.5)) / float(stream.time_base))
            else:
                seek_pts = int((max(0, start_sec - 0.5)) * av.time_base)
            
            container.seek(seek_pts, stream=stream, backward=True)
            
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None and frame.pts is not None and stream.time_base:
                    frame_time = float(frame.pts * stream.time_base)
                    
                if frame_time is None:
                    frame_time = 0.0
                    
                if frame_time < start_sec - 0.01:
                    continue
                    
                frames.append(frame.to_ndarray(format='rgb24'))
                
                if len(frames) >= int(length_frames):
                    break
    except Exception as e:
        log.warning(f"[PromptRelay] Video extract error: {e}")
        
    if not frames:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
    frames_np = np.array(frames, dtype=np.float32) / 255.0
    return torch.from_numpy(frames_np)

def _read_latent_tensor(path):
    """Read a latent tensor [B,C,F,H,W] from a .latent file, tolerant of format:
    ComfyUI SaveLatent (safetensors with 'latent_tensor') OR a torch-pickled save
    (raw tensor, {'samples':...}, {'latent_tensor':...}, or {'latent':{'samples':...}})."""
    # 1) ComfyUI SaveLatent format (safetensors)
    try:
        import safetensors.torch
        data = safetensors.torch.load_file(path, device="cpu")
        t = data.get("latent_tensor")
        if t is not None:
            t = t.to(torch.float32)
            if "latent_format_version_0" not in data:
                t = t * (1.0 / 0.18215)  # legacy scaled tensor
            return t
        for v in data.values():  # unexpected keys: first 3D+ tensor
            if isinstance(v, torch.Tensor) and v.dim() >= 3:
                return v.to(torch.float32)
        return None
    except Exception:
        pass  # not safetensors -> try torch pickle
    # 2) torch-pickled fallback
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.to(torch.float32)
    if isinstance(obj, dict):
        for k in ("latent_tensor", "samples"):
            v = obj.get(k)
            if isinstance(v, torch.Tensor):
                return v.to(torch.float32)
        lat = obj.get("latent")
        if isinstance(lat, dict) and isinstance(lat.get("samples"), torch.Tensor):
            return lat["samples"].to(torch.float32)
    return None


def _load_sibling_latent(seg, extra_dirs=None):
    """If a '<clip>.latent' file exists (next to the clip, in input/output, or in a
    user-provided folder/path in `extra_dirs`), load and return its raw latent tensor
    [B, C, F, H, W]; otherwise None. Lets a clip be inserted directly (max quality)
    instead of being decoded->re-encoded through the VAE."""
    import re
    media = seg.get("imageFile") or ""      # uploaded name, e.g. "1720000000000_clip.mp4"
    orig = seg.get("fileName") or ""        # original name the user dropped in, e.g. "clip.mp4"
    if not media and not orig:
        return None

    def _latent_names(name):
        out = set()
        base = os.path.basename(name or "")
        if not base:
            return out
        root, _ = os.path.splitext(base)
        out.add(root + ".latent")
        stripped = re.sub(r"^\d+_", "", root)  # drop the upload timestamp prefix
        if stripped and stripped != root:
            out.add(stripped + ".latent")
        return out

    latent_names = _latent_names(media) | _latent_names(orig)
    if not latent_names:
        return None

    input_dir = folder_paths.get_input_directory()
    try:
        output_dir = folder_paths.get_output_directory()
    except Exception:
        output_dir = None

    # User-provided path(s) take priority: an explicit '.latent' file (used as-is), or a
    # folder to search. Relative paths resolve under output/ then input/.
    search_dirs = []
    candidates = []
    for d in (extra_dirs or []):
        d = (d or "").strip()
        if not d:
            continue
        if os.path.isfile(d) and d.lower().endswith(".latent"):
            candidates.append(d)  # exact file
        elif os.path.isabs(d):
            search_dirs.append(d)  # absolute folder
        else:
            for base in ([output_dir, input_dir] if output_dir else [input_dir]):
                search_dirs.append(os.path.join(base, d))  # relative -> output/ then input/

    search_dirs += [input_dir, os.path.join(input_dir, "whatdreamscost")]
    if output_dir:
        search_dirs += [output_dir, os.path.join(output_dir, "whatdreamscost")]

    if media:  # also honor the imageFile's own relative subfolder in input/
        candidates.append(os.path.join(input_dir, os.path.splitext(media)[0] + ".latent"))
    for dd in search_dirs:
        for nm in latent_names:
            candidates.append(os.path.join(dd, nm))

    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        log.info("[LTXDirector] No sibling latent for %s. Checked these paths:\n  %s\n"
                 "  (tip: set the node's latent_dir input to the folder, e.g. output/LTX_video)",
                 (orig or media), "\n  ".join(candidates))
        return None
    try:
        t = _read_latent_tensor(path)
        if t is None:
            log.warning("[LTXDirector] %s: no latent tensor found inside (expected a ComfyUI "
                        "SaveLatent .latent or a torch-saved LATENT).", path)
            return None
        log.info("[LTXDirector] Loaded sibling latent %s shape=%s", path, tuple(t.shape))
        return t
    except Exception as e:
        log.warning("[LTXDirector] Failed to load sibling latent %s: %s", path, e)
        return None


def _resolve_resize_algo(name):
    """Map a resize-algo dropdown option to (torch interpolate mode, antialias).

    Antialias is only offered for bilinear/bicubic (the only modes torch supports it on);
    it reduces aliasing when DOWNSCALING and is a no-op when upscaling."""
    table = {
        "bilinear": ("bilinear", False),
        "bilinear (antialias)": ("bilinear", True),
        "bicubic": ("bicubic", False),
        "bicubic (antialias)": ("bicubic", True),
        "area": ("area", False),
        "nearest": ("nearest", False),
    }
    return table.get(name, ("bilinear", False))


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int, resize_algo: str = "bilinear") -> torch.Tensor:
    """Resize an [N, H, W, 3] float32 tensor to target dimensions using the given method,
    then snap the final dimensions to be divisible by `divisible_by`. `resize_algo` picks the
    interpolation (bilinear/bicubic/area/nearest, optionally antialiased)."""

    def snap(val, div):
        return max(div, (val // div) * div)

    mode, antialias = _resolve_resize_algo(resize_algo)

    def interp(t, size):
        # area/nearest don't accept align_corners or antialias in torch.
        if mode in ("area", "nearest"):
            return F.interpolate(t, size=size, mode=mode)
        return F.interpolate(t, size=size, mode=mode, align_corners=False, antialias=antialias)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)

    N, H, W, C = tensor.shape
    if H == th and W == tw:
        return tensor

    t_nchw = tensor.permute(0, 3, 1, 2)

    if method == "stretch to fit":
        resized = interp(t_nchw, (th, tw))

    elif method == "maintain aspect ratio":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        resized = interp(t_nchw, (new_h, new_w))

    elif method == "pad" or method == "pad green":
        ratio = min(tw / W, th / H)
        new_w = snap(int(W * ratio), divisible_by)
        new_h = snap(int(H * ratio), divisible_by)
        inner = interp(t_nchw, (new_h, new_w))

        pad_l = (tw - new_w) // 2
        pad_t = (th - new_h) // 2

        if method == "pad green":
            resized = torch.zeros((N, C, th, tw), dtype=t_nchw.dtype, device=t_nchw.device)
            # #66FF00 is roughly R: 102/255, G: 255/255, B: 0
            resized[:, 0, :, :] = 102 / 255.0
            resized[:, 1, :, :] = 1.0
            resized[:, 2, :, :] = 0.0
            resized[:, :, pad_t:pad_t+new_h, pad_l:pad_l+new_w] = inner
        else:
            resized = F.pad(inner, (pad_l, tw - new_w - pad_l, pad_t, th - new_h - pad_t), mode="constant", value=0)

    elif method == "crop":
        ratio = max(tw / W, th / H)
        new_w = int(W * ratio)
        new_h = int(H * ratio)
        inner = interp(t_nchw, (new_h, new_w))

        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        resized = inner[:, :, top:top+th, left:left+tw]

    else:
        resized = interp(t_nchw, (th, tw))

    # bicubic can overshoot [0,1] (ringing); keep guide pixels valid. No-op for bilinear.
    return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _compress_image(tensor: torch.Tensor, crf: int) -> torch.Tensor:
    """Apply H.264 compression artefacts to an [N, H, W, 3] float32 tensor (ComfyUI image format).
    crf=0 means no compression. Uses PyAV to encode/decode frames in-memory."""
    if crf == 0:
        return tensor
        
    N, H, W, C = tensor.shape
    
    # Dimensions must be even for H.264
    h = (H // 2) * 2
    w = (W // 2) * 2
    
    # uint8 [N, H, W, 3]
    tensor_bytes = (tensor[:, :h, :w, :] * 255.0).byte().cpu().numpy()
    
    try:
        buf = _io.BytesIO()
        container = av.open(buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=24)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "ultrafast"}
        
        for i in range(N):
            frame = av.VideoFrame.from_ndarray(tensor_bytes[i], format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
                
        for pkt in stream.encode(None):
            container.mux(pkt)
            
        container.close()
        
        buf.seek(0)
        container_r = av.open(buf, mode="r")
        decoded = [frame_r.to_ndarray(format="rgb24") for frame_r in container_r.decode(video=0)]
        container_r.close()
        
        if not decoded:
            return tensor
            
        decoded_np = np.stack(decoded).astype(np.float32) / 255.0
        
        # Re-embed into original tensor shape (may have been cropped by even-rounding)
        out = tensor.clone()
        dec_N = min(N, len(decoded))
        out[:dec_N, :h, :w] = torch.from_numpy(decoded_np[:dec_N]).to(tensor.device, tensor.dtype)
        
        return out
        
    except Exception as e:
        log.warning("[PromptRelay] img_compression encode/decode failed: %s", e)
        return tensor


def _build_combined_audio(timeline_data_str: str, start_frame: int, duration_frames: int, frame_rate: float, override_audio: bool = False) -> dict:
    """Parses timeline JSON, loads/trims audio directly from memory using PyAV, 
    and aligns to a global timeline yielding ComfyUI's format.
    Output length explicitly mimics the timeline's duration_frames length."""
    target_sr = 44100
    total_samples = max(1, int(math.ceil(duration_frames / frame_rate * target_sr)))
    empty_audio = {"waveform": torch.zeros((1, 2, total_samples), dtype=torch.float32), "sample_rate": target_sr}

    if not timeline_data_str:
        return empty_audio

    try:
        data = json.loads(timeline_data_str)
        is_retake = data.get("retakeMode", False)
        if is_retake and data.get("retakeVideo"):
            retake_vid = data.get("retakeVideo")
            audio_segs = [{
                "videoFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "audioFile": retake_vid.get("imageFile") or retake_vid.get("fileName"),
                "start": 0,
                "length": retake_vid.get("videoDurationFrames", duration_frames),
                "trimStart": 0
            }]
            override_audio = True
        elif override_audio:
            audio_segs = data.get("motionSegments", [])
        else:
            audio_segs = data.get("audioSegments", [])
    except Exception:
        return empty_audio

    if not audio_segs:
        return empty_audio

    out_waveform = torch.zeros((2, total_samples), dtype=torch.float32)

    for seg in audio_segs:
        buffer = None
        file_key = "videoFile" if override_audio else "audioFile"
        if seg.get(file_key):
            file_path = os.path.join(folder_paths.get_input_directory(), seg[file_key])
            if not os.path.exists(file_path):
                # Try fallback under whatdreamscost subfolder
                basename = os.path.basename(seg[file_key])
                fallback_path = os.path.join(folder_paths.get_input_directory(), "whatdreamscost", basename)
                if os.path.exists(fallback_path):
                    file_path = fallback_path

            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    buffer = _io.BytesIO(f.read())
        
        if not override_audio and not buffer and seg.get("audioB64"):
            b64 = seg.get("audioB64")
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            try:
                audio_bytes = base64.b64decode(b64)
                buffer = _io.BytesIO(audio_bytes)
            except:
                pass
                
        if not buffer:
            continue

        try:
            clip_frames = []
            
            # Use PyAV to decode directly from memory buffer
            with av.open(buffer) as container:
                if not container.streams.audio:
                    continue
                stream = container.streams.audio[0]
                
                # Setup resampler to ensure output is 44.1kHz, Stereo, Float32 Planar
                resampler = av.AudioResampler(
                    format='fltp',
                    layout='stereo',
                    rate=target_sr,
                )
                
                for frame in container.decode(stream):
                    for resampled_frame in resampler.resample(frame):
                        # to_ndarray() on fltp gives shape (channels, samples)
                        arr = resampled_frame.to_ndarray()
                        clip_frames.append(torch.from_numpy(arr))
                
                # Flush the resampler to get any remaining samples
                for resampled_frame in resampler.resample(None):
                    arr = resampled_frame.to_ndarray()
                    clip_frames.append(torch.from_numpy(arr))

            if not clip_frames:
                continue

            # Concatenate all frame blocks along the samples dimension (dim 1)
            waveform = torch.cat(clip_frames, dim=1) # Shape: [2, total_clip_samples]

            # Calculate interactive trim boundaries
            trim_start_frames = float(seg.get("trimStart", 0))
            length_frames = float(seg.get("length", 1))
            start_frames = float(seg.get("start", 0))
            
            if start_frames + length_frames <= start_frame:
                continue
                
            offset = max(0, start_frame - start_frames)
            trim_start_frames += offset
            length_frames = max(1, length_frames - offset)
            start_frames = max(0, start_frames - start_frame)

            start_sample_src = int(trim_start_frames / frame_rate * target_sr)
            length_samples = int(length_frames / frame_rate * target_sr)
            end_sample_src = start_sample_src + length_samples

            if start_sample_src < 0: start_sample_src = 0
            if end_sample_src > waveform.shape[1]:
                end_sample_src = waveform.shape[1]

            actual_length = end_sample_src - start_sample_src
            if actual_length <= 0: continue

            # Extract the correct segment of the audio
            clip_waveform = waveform[:, start_sample_src:end_sample_src]

            # Position onto the timeline
            start_sample_dst = int(start_frames / frame_rate * target_sr)
            
            if start_sample_dst >= out_waveform.shape[1]:
                continue
                
            end_sample_dst = start_sample_dst + actual_length

            # Clip any trailing overflow so we don't index past the timeline bounds
            if end_sample_dst > out_waveform.shape[1]:
                actual_length = out_waveform.shape[1] - start_sample_dst
                clip_waveform = clip_waveform[:, :actual_length]
                end_sample_dst = start_sample_dst + actual_length
                
            if actual_length <= 0:
                continue

            # Additive composite (allows clips overlapping to sum together naturally)
            out_waveform[:, start_sample_dst:end_sample_dst] += clip_waveform

        except Exception as e:
            log.warning("[PromptRelay] Audio process error for segment %s: %s", seg.get("fileName"), e)
            continue

    return {"waveform": out_waveform.unsqueeze(0), "sample_rate": target_sr}


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    """Convert pixel-space segment lengths to integer latent-space lengths using the
    largest-remainder method. Targets the full `latent_frames` when the pixel sum looks
    like full coverage (within one stride of latent_frames * stride). Otherwise targets
    round(total_pixel / temporal_stride) so partial-coverage timelines stay partial.
    """
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    # Within one frame of full → user clearly intended full coverage; pin to latent_frames.
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    # Ensure every segment has ≥ 1 latent frame (steal from the largest if needed).
    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result


def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(
                f"PromptRelay: '{name}' arrived as None. "
                "Likely causes: a stale workflow JSON saved with null, the timeline "
                "editor's web extension failing to load, or an upstream node returning None. "
                "Set the field to an empty string or fix the upstream connection."
            )

    # Split prompts but do NOT filter out empty ones yet, so we can detect them
    locals_list = [p.strip() for p in local_prompts.split("|")]
    
    # If there are no visual segments on the timeline (e.g., only using IC-LoRA motion track),
    # bypass the local prompt chunking entirely and just use the global prompt.
    if not locals_list or (len(locals_list) == 1 and not locals_list[0]):
        log.info("[PromptRelay] No local segments found. Using global prompt exclusively.")
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(global_prompt))
        return model.clone(), conditioning

    # Check if any specific segment is empty and apply fallbacks
    for i, p in enumerate(locals_list):
        if not p:
            fallback = global_prompt.strip() if global_prompt else "video"
            if not fallback:
                fallback = "video"
            locals_list[i] = fallback

    arch, patch_size, temporal_stride = detect_model_type(model)

    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(float(x.strip())) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)

    log.info("[PromptRelay] Global: tokens [0:%d] (%d tokens)", token_ranges[0][0], token_ranges[0][0])
    for i, (s, e) in enumerate(token_ranges):
        log.info("[PromptRelay] Segment %d: tokens [%d:%d] (%d tokens)", i, s, e, e - s)

    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)

    log.info(
        "[PromptRelay] Latent: %d frames, %d tokens/frame, segments: %s",
        latent_frames, tokens_per_frame, effective_lengths,
    )

    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, None)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning


def _parse_beats_json(beats_json, default_fps=24):
    """Parse a 'beats' / shot-list JSON into prompt-relay inputs.

    Accepts {fps, global_prompt, negative_prompt, beats:[{prompt|camera+delta,
    frames|duration_s}, ...]} (beats may also be named 'segments' or 'shots').
    Returns a dict with local_prompts (" | "-joined), segment_lengths
    (","-joined), global_prompt and fps, or None if nothing usable was found.
    """
    try:
        root = json.loads(beats_json)
    except Exception as e:
        log.warning("[LTXDirector] beats_json parse error: %s", e)
        return None
    if not isinstance(root, dict):
        return None

    beats = root.get("beats")
    if not isinstance(beats, list):
        beats = root.get("segments")
    if not isinstance(beats, list):
        beats = root.get("shots")
    if not isinstance(beats, list) or not beats:
        return None

    try:
        fps = int(round(float(root.get("fps", default_fps))))
    except Exception:
        fps = default_fps
    if fps <= 0:
        fps = default_fps

    prompts, lengths = [], []
    for b in beats:
        if not isinstance(b, dict):
            continue
        length = 0
        try:
            if b.get("frames") is not None and float(b["frames"]) > 0:
                length = int(round(float(b["frames"])))
            elif b.get("duration_s") is not None and float(b["duration_s"]) > 0:
                length = int(round(float(b["duration_s"]) * fps))
        except Exception:
            length = 0
        if length <= 0:
            length = fps

        prompt = ""
        if isinstance(b.get("prompt"), str) and b["prompt"].strip():
            prompt = b["prompt"].strip()
        else:
            parts = [b.get("camera"), b.get("delta")]
            prompt = ", ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())

        prompts.append(prompt)
        lengths.append(max(1, length))

    if not prompts:
        return None

    gp = root.get("global_prompt")
    gp = gp.strip() if isinstance(gp, str) else ""
    return {
        "global_prompt": gp,
        "local_prompts": " | ".join(prompts),
        "segment_lengths": ",".join(str(x) for x in lengths),
        "fps": fps,
        "total_frames": sum(lengths),
    }


class LTXDirector(io.ComfyNode):
    """WYSIWYG timeline variant — segments and lengths come from a visual editor in the node UI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXDirector",
            display_name="LTX Director",
            category="WhatDreamsCost",
            description=(
                "Same as Prompt Relay Encode, but local prompts and segment lengths are edited "
                "visually as draggable blocks on a timeline. The duration_frames input only sets the "
                "timeline scale (pixel space) — actual frame count is still read from the latent."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Optional. Connect an Audio VAE to generate audio latents."),
                io.Latent.Input("optional_latent", optional=True, tooltip="Optional. Connect a latent to override the auto-generated one."),
                io.String.Input(
                    "global_prompt", multiline=True, default="", force_input=True, optional=True,
                    tooltip="Conditions the entire video. Anchors persistent characters, objects, and scene context.",
                ),
                io.Float.Input(
                    "start_second", default=0.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="Start time in seconds of the timeline generation.",
                ),
                io.Float.Input(
                    "end_second", default=5.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="End time in seconds of the timeline generation.",
                ),
                io.Float.Input(
                    "duration_seconds", default=5.0, min=0.1, max=1000.0, step=0.01,
                    tooltip="Total timeline duration in seconds (computed/synced from frames).",
                ),
                io.Int.Input(
                    "start_frame", default=0, min=0, max=10000, step=1,
                    tooltip="Start frame of the timeline generation.",
                ),
                io.Int.Input(
                    "end_frame", default=120, min=1, max=10000, step=1,
                    tooltip="End frame of the timeline generation.",
                ),
                io.Int.Input(
                    "duration_frames", default=120, min=1, max=10000, step=1,
                    tooltip="Total timeline length in pixel-space frames. Used by the editor for visual scale only.",
                ),
                io.String.Input(
                    "timeline_data", default="",
                    tooltip="JSON state of the timeline editor (auto-managed; do not edit by hand).",
                ),
                io.Boolean.Input(
                    "use_custom_audio", default=False, optional=True,
                    tooltip="Toggle between using timeline audio (ON) and generating audio from scratch (OFF).",
                ),
                io.Boolean.Input(
                    "use_custom_motion", default=True, optional=True,
                    tooltip="Toggle between using timeline motion guidance (ON) and ignoring motion video segments (OFF).",
                ),
                io.Boolean.Input(
                    "inpaint_audio", default=True, optional=True,
                    tooltip="Toggle whether empty gaps in the audio track are inpainted with generated audio.",
                ),
                io.String.Input(
                    "local_prompts", multiline=True, default="",
                    tooltip="Auto-populated from the timeline editor.",
                ),
                io.String.Input(
                    "segment_lengths", default="",
                    tooltip="Auto-populated from the timeline editor (pixel-space frame counts).",
                ),
                io.Float.Input(
                    "epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001,
                    tooltip="Penalty decay parameter. Values below ~0.1 all produce sharp boundaries (paper default 0.001). For softer transitions, try 0.5 or higher.",
                ),
                io.Float.Input(
                    "frame_rate", default=24, min=1, max=240, step=1, optional=True,
                    tooltip="Frames per second — only affects how time is displayed in the timeline editor when time_units is set to 'seconds'.",
                ),
                io.Combo.Input(
                    "display_mode", options=["frames", "seconds"], default="seconds", optional=True,
                    tooltip="Display the ruler, segment ranges, length input, and total in frames or seconds. Internal storage is always pixel-space frames.",
                ),
                io.String.Input(
                    "guide_strength", default="",
                    tooltip="Auto-populated from the timeline editor (comma-separated guide strengths for image segments).",
                ),
                io.Int.Input(
                    "custom_width", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="Target output width for all image segments. Set to 0 to use the original image width.",
                ),
                io.Int.Input(
                    "custom_height", default=0, min=0, max=8192, step=1, optional=True,
                    tooltip="Target output height for all image segments. Set to 0 to use the original image height.",
                ),
                io.Combo.Input(
                    "resize_method",
                    options=["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"],
                    default="maintain aspect ratio",
                    optional=True,
                    tooltip="How to resize image segments to fit the target dimensions.",
                ),
                io.Int.Input(
                    "divisible_by", default=32, min=1, max=256, step=1, optional=True,
                    tooltip="Snap the final output image dimensions to be divisible by this number (e.g. 32 for LTX).",
                ),
                io.Int.Input(
                    "img_compression", default=18, min=0, max=100, step=1, optional=True,
                    tooltip="H.264 CRF compression to apply to each guide image. 0 = no compression, higher = more artefacts.",
                ),
                io.Boolean.Input(
                    "override_audio", default=False, optional=True,
                    tooltip="Use the audio from the IC-LoRA video instead of using the audio track.",
                ),
                io.Combo.Input(
                    "resize_algo",
                    options=["bilinear", "bilinear (antialias)", "bicubic", "bicubic (antialias)", "area", "nearest"],
                    default="bilinear",
                    optional=True,
                    tooltip=(
                        "Interpolation used when resizing guide images. The '(antialias)' variants "
                        "reduce aliasing/shimmer when DOWNSCALING (only bilinear/bicubic support it; "
                        "no effect when upscaling). 'area' is a solid antialiased downscale; 'nearest' "
                        "is hard-edged. (Kept last in the input list so it never shifts other widgets.)"
                    ),
                ),
                io.String.Input(
                    "latent_dir", default="", optional=True,
                    tooltip=(
                        "Optional. Folder (or an exact .latent file) where per-clip '<name>.latent' files "
                        "live — used to insert a clip's saved latent directly (max quality, no VAE re-encode). "
                        "Relative paths resolve under output/ then input/ (e.g. 'LTX_video'). Name each .latent "
                        "after the clip's ORIGINAL filename. Leave empty to just look next to the clip / in output/."
                    ),
                ),
                io.String.Input(
                    "beats_json", multiline=True, default="", optional=True, force_input=True,
                    tooltip=(
                        "Optional. A 'beats'/shot-list JSON (fps, global_prompt, beats:[{prompt, "
                        "frames|duration_s}, ...]). When connected, it drives a pure text prompt-relay "
                        "for this run, overriding the visual timeline. negative_prompt is ignored here "
                        "(set it downstream at the guide/sampler)."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent", tooltip="Auto-generated LTXV empty latent (only populated when no latent is connected)."),
                io.Latent.Output(display_name="audio_latent", tooltip="Auto-generated audio latent (uses custom audio if enabled)."),
                GuideData.Output(display_name="guide_data"),
                MotionGuideData.Output(display_name="motion_guide_data"),
                io.Float.Output(display_name="frame_rate", tooltip="The frame rate used for the timeline."),
                io.Audio.Output(display_name="combined_audio", tooltip="Combined timeline audio layout."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, start_second, end_second, duration_seconds, start_frame, end_frame, duration_frames,
                timeline_data, local_prompts, segment_lengths, global_prompt="", guide_strength="", epsilon=1e-3,
                frame_rate=24, display_mode="seconds",
                custom_width=768, custom_height=512, resize_method="maintain aspect ratio",
                resize_algo="bilinear", divisible_by=32, img_compression=0, audio_vae=None, optional_latent=None,
                use_custom_audio=False, inpaint_audio=True, use_custom_motion=True, override_audio=False,
                beats_json="", latent_dir="") -> io.NodeOutput:

        # Parse timeline data
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
        except Exception as e:
            log.error(f"[LTXDirector] execute timeline_data parse error: {e}")
            tdata = {}

        is_retake_mode = tdata.get("retakeMode", False)
        is_retake_active = is_retake_mode and tdata.get("retakeVideo") is not None

        # Extract global_prompt from timeline_data if not connected/empty
        if not global_prompt:
            if is_retake_mode:
                global_prompt = tdata.get("retake_global_prompt", "")
            else:
                global_prompt = tdata.get("global_prompt", "")

        log.info(f"[LTXDirector] execute RECEIVED global_prompt: {repr(global_prompt)}")

        # --- Optional 'beats' JSON override (node input) ---
        # When connected, the beats drive a pure text prompt-relay for this run:
        # override prompts/lengths/fps and ignore the visual-timeline guides.
        if beats_json and beats_json.strip():
            beats = _parse_beats_json(beats_json, default_fps=int(frame_rate) if frame_rate else 24)
            if beats:
                if not global_prompt:
                    global_prompt = beats["global_prompt"]
                local_prompts = beats["local_prompts"]
                segment_lengths = beats["segment_lengths"]
                frame_rate = beats["fps"]
                duration_frames = max(1, beats["total_frames"])
                # Text-only: drop visual-timeline guides/motion for this run.
                timeline_data = "{}"
                tdata = {}
                log.info(
                    "[LTXDirector] beats_json override: %d segments, %d frames @ %d fps",
                    len(beats["segment_lengths"].split(",")), duration_frames, frame_rate,
                )

        # --- Build guide_data from image segments FIRST (to derive output dimensions) ---
        # "segment_numbers" carries the 0-based timeline position of each guide (see below).
        # "original_images" holds each guide's pre-resize image (parallel to "images").
        guide_data = {"images": [], "original_images": [], "insert_frames": [], "strengths": [], "segment_numbers": [], "guide_latents": [], "frame_rate": frame_rate}
        derived_w, derived_h = custom_width, custom_height
        try:
            img_segs = [
                s for s in tdata.get("segments", [])
                if s.get("type", "image") in ("image", "video")
                and (s.get("imageFile") or s.get("imageB64"))
                and int(s.get("start", 0)) < start_frame + duration_frames
                and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
            ]
            img_segs.sort(key=lambda s: s["start"])

            # Map every main-track segment overlapping the generated window to its 0-based
            # timeline position (left-to-right by start frame). Each guide is then tagged
            # with the slot of the block it belongs to — not merely its rank among the
            # image-bearing segments — so text-only blocks still occupy a position.
            window_segs = sorted(
                (
                    s for s in tdata.get("segments", [])
                    if int(s.get("start", 0)) < start_frame + duration_frames
                    and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
                ),
                key=lambda s: int(s.get("start", 0)),
            )
            seg_position = {}
            for pos, s in enumerate(window_segs):
                sid = s.get("id")
                if sid is not None and sid not in seg_position:
                    seg_position[sid] = pos

            strengths = []
            if guide_strength.strip():
                strengths = [float(x.strip()) for x in guide_strength.split(",") if x.strip()]

            for idx, seg in enumerate(img_segs):
                seg_start = int(seg.get("start", 0))
                offset = max(0, start_frame - seg_start)

                if seg.get("type") == "video":
                    if offset > 0:
                        seg["trimStart"] = float(seg.get("trimStart", 0)) + offset
                        seg["length"] = max(1, int(seg.get("length", 1)) - offset)
                    tensor = _load_video_tensor(seg, float(frame_rate))
                else:
                    tensor = _load_image_tensor(seg)

                # Keep the original (pre-resize) image; resize returns a new tensor so this ref stays intact.
                original_tensor = tensor

                # Apply resize
                src_h, src_w = tensor.shape[1], tensor.shape[2]

                def snap(val, div):
                    return max(div, (val // div) * div)

                if custom_width > 0 and custom_height > 0:
                    # Both dimensions set — apply selected resize_method (pad, crop, stretch, maintain AR)
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by, resize_algo)
                elif custom_width > 0:
                    # Width only — scale height from AR, snap both, then resize to exact dimensions
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by, resize_algo)
                elif custom_height > 0:
                    # Height only — scale width from AR, snap both, then resize to exact dimensions
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by, resize_algo)
                else:
                    # Both zero — keep original dimensions, just snap to divisible_by
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by, resize_algo)


                # Apply compression
                if img_compression > 0:
                    tensor = _compress_image(tensor, img_compression)

                # Record dimensions of the first processed image for latent generation
                if idx == 0:
                    derived_h = tensor.shape[1]
                    derived_w = tensor.shape[2]

                if seg.get("isEndFrame"):
                    insert_frame = max(0, seg_start + int(seg.get("length", 1)) - 1 - start_frame)
                else:
                    insert_frame = max(0, seg_start - start_frame)
                strength = strengths[idx] if idx < len(strengths) else 1.0

                # If a '<clip>.latent' sibling exists, feed it as this guide's latent instead of
                # VAE-re-encoding the clip (max quality). It still flows through the normal
                # keyframe-guide path (append_keyframe) in the Guide, which is what conditions the
                # model to continue — that's why the decoded video already extends correctly.
                sibling_latent = _load_sibling_latent(seg, [latent_dir])
                if sibling_latent is not None:
                    # The sibling .latent is the FULL clip's latent; slice it to the timeline trim
                    # (trimStart/length are already window-adjusted above) so it matches what the
                    # decoded-video path sends — otherwise the guide is the START of the clip.
                    if seg.get("type") == "video":
                        _tsf = 8  # LTX temporal downscale (8n+1) — same factor used elsewhere here
                        _n = int(sibling_latent.shape[2])
                        _ts = min(max(0, int(float(seg.get("trimStart", 0))) // _tsf), max(0, _n - 1))
                        _ln = max(1, (int(seg.get("length", 1)) + _tsf - 1) // _tsf)
                        _end = min(_n, _ts + _ln)
                        if _end > _ts and (_ts > 0 or _end < _n):
                            sibling_latent = sibling_latent[:, :, _ts:_end]
                            log.info("[LTXDirector] Trimmed sibling latent to frames %d:%d (timeline trim %s+%spx)",
                                     _ts, _end, seg.get("trimStart", 0), seg.get("length", 1))
                    log.info("[LTXDirector] Using sibling latent for %s (%d latent frames)",
                             seg.get("imageFile"), int(sibling_latent.shape[2]))

                guide_data["images"].append(tensor)
                guide_data["original_images"].append(original_tensor)
                guide_data["insert_frames"].append(insert_frame)
                guide_data["strengths"].append(float(strength))
                guide_data["segment_numbers"].append(int(seg_position.get(seg.get("id"), idx)))
                guide_data["guide_latents"].append(sibling_latent)
            
            # If no images were loaded from the timeline, create a dummy image at strength 0
            # to prevent artifacts in text-to-video mode.
            if not guide_data["images"] and optional_latent is None:
                src_w = derived_w if derived_w > 0 else 768
                src_h = derived_h if derived_h > 0 else 512
                
                # If there's an IC-LoRA video or retake base video on the timeline, extract its dimensions for accurate aspect ratio scaling
                tdata_motion = json.loads(timeline_data) if timeline_data else {}
                found_dims = False
                
                # Check for retake base video first
                is_retake = tdata_motion.get("retakeMode", False)
                retake_vid = tdata_motion.get("retakeVideo") or {}
                retake_file = retake_vid.get("imageFile", "") if isinstance(retake_vid, dict) else ""
                if is_retake and retake_file:
                    r_path = os.path.join(folder_paths.get_input_directory(), retake_file)
                    if not os.path.exists(r_path):
                        basename = os.path.basename(retake_file)
                        fallback_path = os.path.join(folder_paths.get_input_directory(), "whatdreamscost", basename)
                        if os.path.exists(fallback_path):
                            r_path = fallback_path
                    if os.path.exists(r_path):
                        try:
                            with av.open(r_path) as container:
                                stream = container.streams.video[0]
                                src_w = stream.width or stream.codec_context.width
                                src_h = stream.height or stream.codec_context.height
                                found_dims = True
                        except:
                            pass
                
                # Fallback to normal motion segments
                if not found_dims:
                    for mseg in tdata_motion.get("motionSegments", []):
                        v_file = mseg.get("videoFile")
                        if v_file:
                            v_path = os.path.join(folder_paths.get_input_directory(), v_file)
                            if not os.path.exists(v_path):
                                basename = os.path.basename(v_file)
                                fallback_path = os.path.join(folder_paths.get_input_directory(), "whatdreamscost", basename)
                                if os.path.exists(fallback_path):
                                    v_path = fallback_path
                            if os.path.exists(v_path):
                                try:
                                    with av.open(v_path) as container:
                                        stream = container.streams.video[0]
                                        src_w = stream.width or stream.codec_context.width
                                        src_h = stream.height or stream.codec_context.height
                                        found_dims = True
                                        break
                                except:
                                    pass

                # Create a dummy tensor of the exact source dimensions
                tensor = torch.zeros((1, src_h, src_w, 3), dtype=torch.float32)

                def snap(val, div):
                    return max(div, (val // div) * div)

                # Route the dummy tensor through the exact same resizing pipeline
                if custom_width > 0 and custom_height > 0:
                    tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by, resize_algo)
                elif custom_width > 0:
                    tgt_w = snap(custom_width, divisible_by)
                    tgt_h = snap(int(src_h * tgt_w / src_w), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by, resize_algo)
                elif custom_height > 0:
                    tgt_h = snap(custom_height, divisible_by)
                    tgt_w = snap(int(src_w * tgt_h / src_h), divisible_by)
                    tensor = _resize_image(tensor, tgt_w, tgt_h, "stretch to fit", divisible_by, resize_algo)
                else:
                    tensor = _resize_image(tensor, src_w, src_h, "maintain aspect ratio", divisible_by, resize_algo)
                
                guide_data["images"].append(tensor)
                guide_data["original_images"].append(tensor)
                guide_data["insert_frames"].append(0)
                guide_data["strengths"].append(0.0)
                guide_data["segment_numbers"].append(0)
                guide_data["guide_latents"].append(None)

                derived_w = tensor.shape[2]
                derived_h = tensor.shape[1]

        except Exception as e:
            log.warning("[PromptRelay] Could not build guide_data: %s", e)

        # --- Auto-generate LTXV latent if none was provided ---
        # Apply the community 8n+1 rule directly to the timeline's duration_frames:
        # int(ceil(((duration_frames) - 1) / 8) * 8) + 1
        # This ensures we get AT LEAST the requested frames, snapped to LTXV's requirements.
        ltxv_length = int(math.ceil((duration_frames - 1) / 8.0) * 8) + 1
        
        if optional_latent is None:
            latent_w = max(32, (derived_w // 32) * 32)
            latent_h = max(32, (derived_h // 32) * 32)
            # LTXV temporal: ((length - 1) // 8) + 1 latent frames; invert to get pixel frames -> length
            latent_t = ((ltxv_length - 1) // 8) + 1
            samples = torch.zeros(
                [1, 128, latent_t, latent_h // 32, latent_w // 32],
                device=comfy.model_management.intermediate_device(),
            )
            latent = {"samples": samples}
            log.info(
                "[PromptRelay] Auto-generated LTXV latent: %dx%d, %d pixel frames (%d latent frames)",
                latent_w, latent_h, ltxv_length, latent_t,
            )
        else:
            latent = optional_latent

        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon,
        )

        # --- Build Audio Output ---
        audio_out = _build_combined_audio(timeline_data, start_frame, ltxv_length, float(frame_rate), override_audio=override_audio)

        # --- Audio Latent Generation ---
        audio_latent = {}
        
        if audio_vae is not None:
            # Helper to generate empty latent
            def get_empty_latent():
                # Support both raw AudioVAE objects and ComfyUI VAE wrappers.
                inner = getattr(audio_vae, "first_stage_model", audio_vae)
                z_channels = audio_vae.latent_channels
                audio_freq = inner.latent_frequency_bins
                num_audio_latents = inner.num_of_latents_from_frames(ltxv_length, float(frame_rate))
                audio_latents = torch.zeros(
                    (1, z_channels, num_audio_latents, audio_freq),
                    device=comfy.model_management.intermediate_device(),
                )
                return {"samples": audio_latents, "type": "audio"}

            if use_custom_audio or override_audio or is_retake_active:
                try:
                    if audio_out is not None:
                        # 1. Encode audio waveform into latent space
                        waveform = audio_out["waveform"]
                        if waveform.ndim == 2:
                            waveform = waveform.unsqueeze(0)
                        if waveform.ndim != 3:
                            raise ValueError(
                                f"Expected custom audio waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}"
                            )

                        # Wrapped ComfyUI VAE expects (batch, samples, channels);
                        # raw AudioVAE expects a dict with waveform in (batch, channels, samples).
                        if hasattr(audio_vae, "first_stage_model"):
                            latent_samples = audio_vae.encode(waveform.movedim(1, -1))
                        else:
                            latent_samples = audio_vae.encode({
                                "waveform": waveform,
                                "sample_rate": audio_out["sample_rate"],
                            })
                        
                        if latent_samples.numel() == 0:
                            raise ValueError("Encoded audio latent is empty (0 elements).")
                        
                        # 2. Create a 3D gap mask [B, F, H] to avoid accidental broadcasting to the 5D video latent 
                        # which also has 128 channels. A 4D audio mask [1, 128, F, H] confuses ComfyUI's KSampler 
                        # into masking the video latent as well, causing black frames.
                        B, C, F_len, H_len = latent_samples.shape
                        
                        if is_retake_active:
                            gap_mask = torch.zeros((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)
                            
                            retake_start = float(tdata.get("retakeStart", 0))
                            retake_len = float(tdata.get("retakeLength", 0))
                            
                            overlap_start = max(start_frame, retake_start)
                            overlap_end = min(start_frame + ltxv_length, retake_start + retake_len)
                            
                            if overlap_end > overlap_start:
                                rel_start = overlap_start - start_frame
                                rel_len = overlap_end - overlap_start
                                
                                start_sec = rel_start / float(frame_rate)
                                len_sec = rel_len / float(frame_rate)
                                total_sec = ltxv_length / float(frame_rate)
                                
                                start_idx = int((start_sec / total_sec) * F_len)
                                end_idx = int(((start_sec + len_sec) / total_sec) * F_len)
                                
                                start_idx = max(0, min(F_len, start_idx))
                                end_idx = max(0, min(F_len, end_idx))
                                
                                gap_mask[:, start_idx:end_idx, :] = 1.0
                        else:
                            gap_mask = torch.ones((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)
                            
                            audio_segs_key = "motionSegments" if override_audio else "audioSegments"
                            file_key = "videoFile" if override_audio else "audioFile"
                            for seg in tdata.get(audio_segs_key, []):
                                if not seg.get(file_key):
                                    continue
                                
                                seg_start = float(seg.get("start", 0))
                                seg_len = float(seg.get("length", 1))
                                
                                if seg_start + seg_len <= start_frame or seg_start >= start_frame + ltxv_length:
                                    continue
                                    
                                offset = max(0, start_frame - seg_start)
                                seg_len = max(1.0, seg_len - offset)
                                seg_start = max(0, seg_start - start_frame)

                                start_sec = seg_start / float(frame_rate)
                                len_sec = seg_len / float(frame_rate)
                                total_sec = ltxv_length / float(frame_rate)

                                start_idx = int((start_sec / total_sec) * F_len)
                                end_idx = int(((start_sec + len_sec) / total_sec) * F_len)
                                gap_mask[:, start_idx:end_idx, :] = 0.0
                                
                        if inpaint_audio:
                            # Generate new audio in the gaps, preserve custom audio segments
                            mask = gap_mask
                        else:
                            # Preserve the entire audio latent (no generation). 
                            # We use a 3D zeros mask to prevent video blackouts.
                            mask = torch.zeros((B, F_len, H_len), dtype=torch.float32, device=latent_samples.device)
                        
                        audio_latent = {
                            "samples": latent_samples,
                            "type": "audio",
                            "noise_mask": mask
                        }
                        log.info("[PromptRelay] Generated custom audio latent with dynamic noise mask.")
                    else:
                        raise ValueError("No audio waveform to encode.")
                except Exception as e:
                    log.error("[PromptRelay] Failed to generate custom audio latent: %s", e)
                    raise e
            else:
                # Generate empty latent
                try:
                    audio_latent = get_empty_latent()
                    log.info("[PromptRelay] Auto-generated empty audio latent.")
                except Exception as e:
                    log.error("[PromptRelay] Could not generate empty audio latent: %s", e)
                    raise e

        # --- Motion guide output from timeline video segments ---
        motion_guide_data = {"segments": [], "frame_rate": float(frame_rate), "duration_frames": int(duration_frames), "resize_method": resize_method}
        try:
            tdata = json.loads(timeline_data) if timeline_data else {}
            if use_custom_motion:
                motion_segments = tdata.get("motionSegments", [])
            else:
                motion_segments = []

            # 0-based timeline position for each motion segment (left-to-right by start
            # frame), numbered independently of the image guide track.
            window_motion = sorted(
                (
                    s for s in motion_segments
                    if s.get("videoFile")
                    and int(s.get("start", 0)) < start_frame + duration_frames
                    and int(s.get("start", 0)) + int(s.get("length", 1)) > start_frame
                ),
                key=lambda s: int(s.get("start", 0)),
            )
            motion_position = {}
            for pos, s in enumerate(window_motion):
                sid = s.get("id")
                if sid is not None and sid not in motion_position:
                    motion_position[sid] = pos

            for seg in motion_segments:
                seg_start = int(seg.get("start", 0))
                length = int(seg.get("length", 1))
                if seg_start >= start_frame + duration_frames or seg_start + length <= start_frame:
                    continue
                if not seg.get("videoFile"):
                    continue
                    
                offset = max(0, start_frame - seg_start)
                new_start = max(0, seg_start - start_frame)
                
                # Trim length so it doesn't extend beyond duration_frames
                clipped_len = min(length - offset, duration_frames - new_start)
                if clipped_len <= 0:
                    continue
                    
                clean = dict(seg)
                clean["start"] = new_start
                clean["length"] = clipped_len
                clean["trimStart"] = float(seg.get("trimStart", 0)) + offset
                clean["segment_number"] = int(motion_position.get(seg.get("id"), len(motion_guide_data["segments"])))
                motion_guide_data["segments"].append(clean)
        except Exception as e:
            log.warning("[LTXDirector] Could not build motion_guide_data: %s", e)

        # Inject raw timeline details for downstream masking in Retake Mode
        guide_data["timeline_data"] = timeline_data
        guide_data["start_frame"] = start_frame
        guide_data["duration_frames"] = duration_frames
        guide_data["resize_method"] = resize_method
        # Surface the assembled per-segment prompts (" | "-joined, timeline order) + global, so the
        # LTX Timeline -> Extend Prompts adapter can pull the ordered prompt list without re-parsing.
        guide_data["local_prompts"] = local_prompts
        guide_data["global_prompt"] = global_prompt

        return io.NodeOutput(patched, conditioning, latent, audio_latent, guide_data, motion_guide_data, float(frame_rate), audio_out)


class LTXKeyframeOut(io.ComfyNode):
    """Extract a single keyframe's original (pre-resize) and resized image from an
    LTX Director guide_data output, selected by its timeline segment number or by order."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXKeyframeOut",
            display_name="LTX Keyframe Out",
            category="WhatDreamsCost",
            description=(
                "Pull one keyframe's original (pre-resize) and resized image out of LTX "
                "Director's guide_data. Select by the keyframe's timeline segment number, "
                "or by its order among the keyframes."
            ),
            inputs=[
                GuideData.Input("guide_data", tooltip="guide_data output from LTX Director."),
                io.Int.Input(
                    "keyframe", default=0, min=0, max=9999, step=1,
                    tooltip="Which keyframe to extract (interpreted per 'select_by').",
                ),
                io.Combo.Input(
                    "select_by", options=["segment number", "keyframe order"],
                    default="segment number", optional=True,
                    tooltip=(
                        "'segment number' = the guide's timeline position (matches segment_number); "
                        "'keyframe order' = its 0-based index among the keyframes. Falls back to order "
                        "if the requested segment number has no keyframe."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="original"),
                io.Image.Output(display_name="resized"),
                io.Int.Output(display_name="segment_number"),
                io.Int.Output(display_name="insert_frame"),
                io.Int.Output(display_name="keyframe_count"),
            ],
        )

    @classmethod
    def execute(cls, guide_data, keyframe=0, select_by="segment number") -> io.NodeOutput:
        gd = guide_data or {}
        images = gd.get("images", []) or []
        originals = gd.get("original_images", []) or []
        seg_nums = gd.get("segment_numbers", []) or []
        insert_frames = gd.get("insert_frames", []) or []
        count = len(images)

        if count == 0:
            empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return io.NodeOutput(empty, empty, -1, 0, 0)

        # Resolve which keyframe to return.
        i = None
        if select_by == "segment number":
            try:
                i = seg_nums.index(int(keyframe))
            except (ValueError, TypeError):
                i = None
        if i is None:
            i = max(0, min(int(keyframe), count - 1))

        resized = images[i]
        original = originals[i] if i < len(originals) else resized
        seg_no = int(seg_nums[i]) if i < len(seg_nums) else i
        ins = int(insert_frames[i]) if i < len(insert_frames) else 0
        return io.NodeOutput(original, resized, seg_no, ins, count)


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
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _anchor_strength_for_state(anchor_strength, default=0.25):
    if anchor_strength is None:
        return float(default)
    try:
        result = float(anchor_strength)
    except Exception:
        return 0.0
    return result if math.isfinite(result) else 0.0


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
        and int(anchor_image.shape[1]) > 0
        and int(anchor_image.shape[2]) > 0
    )


def _extract_anchor_latent(anchor_latent, target_channels):
    if not isinstance(anchor_latent, dict):
        return None
    samples = anchor_latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        return None
    if int(samples.shape[0]) < 1:
        return None
    if int(samples.shape[1]) != int(target_channels):
        return None
    if int(samples.shape[2]) < 1:
        return None
    if int(samples.shape[3]) < 1 or int(samples.shape[4]) < 1:
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
            if mode == "auto":
                log.warning("[%s] source anchor latent is incompatible; falling back to image anchor.", log_tag)
            else:
                log.warning("[%s] source anchor latent is incompatible; skipping latent anchor.", log_tag)

    if latent_guide is None and mode == "latent":
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


def _build_extend_pass(model, clip, latent, prompt, extension_seconds, guide_overlap_seconds,
                       frame_rate, audio=None, audio_vae=None, vae=None, guide_strength=1.0, epsilon=1e-3,
                       audio_base_px=0, log_tag="LTXAutoExtend", anchor_image=None,
                       anchor_latent=None, anchor_mode="off", anchor_strength=0.25,
                       anchor_every_n_steps=1, step_index=1):
    """Core of ONE extension pass, shared by LTX Auto Extend (manual chain) and LTX Extend Step
    (loop). Slices the incoming latent's tail as the continuation guide, builds the empty extended
    latent + Director-shaped guide_data, encodes the prompt, and cuts this pass's audio window
    (starting audio_base_px + tail_start*8 pixel frames into the given audio). Returns a dict.
    audio_base_px = 0 for the manual node (fed audio starts at the incoming latent's frame 0);
    the loop passes the running master-timeline position so it cuts the master in place."""
    fps = float(frame_rate) if frame_rate else 24.0
    tsf = 8  # LTX temporal downscale (8n+1)
    target_sr = 44100

    in_samples = latent["samples"]
    C = int(in_samples.shape[1]); T_in = int(in_samples.shape[2])
    Hl = int(in_samples.shape[3]); Wl = int(in_samples.shape[4])
    overlap_px = max(1, int(round(float(guide_overlap_seconds) * fps)))
    ext_px = max(1, int(round(float(extension_seconds) * fps)))
    n_overlap_lat = max(1, min(T_in, (overlap_px + tsf - 1) // tsf))
    tail_latent = in_samples[0:1, :, T_in - n_overlap_lat:, :, :].clone()

    window_px = overlap_px + ext_px
    ltxv_length = int(math.ceil((window_px - 1) / 8.0) * 8) + 1
    latent_t = ((ltxv_length - 1) // 8) + 1
    empty = torch.zeros([1, C, latent_t, Hl, Wl], device=comfy.model_management.intermediate_device())
    video_latent = {"samples": empty}

    dummy_img = torch.zeros((1, Hl * 32, Wl * 32, 3), dtype=torch.float32)
    # Guide source: raw tail latent by default (sharpest when the working res matches). If a video VAE
    # is provided (HYBRID), decode to images and feed the image-guide path instead — the Guide resizes
    # PIXELS then re-encodes at the working res, staying sharp under scale_by (vs. bilinear-resizing the
    # latent, which decodes soft). DECODE-THEN-CUT: decode the WHOLE incoming latent so the tail has full
    # temporal context (no causal-blur front frame), THEN slice the sharp tail from the decoded images.
    guide_img = dummy_img
    guide_lat = tail_latent
    if vae is not None:
        try:
            dec = vae.decode(in_samples[0:1])
            if hasattr(dec, "ndim") and dec.ndim == 5:  # [B,T,H,W,C] -> [B*T,H,W,C]
                _b, _t, _h, _w, _c = dec.shape
                dec = dec.reshape(_b * _t, _h, _w, _c)
            if hasattr(dec, "ndim") and dec.ndim == 4 and int(dec.shape[-1]) >= 3:
                tail_px = (n_overlap_lat - 1) * tsf + 1            # overlap length in pixel frames (8n+1)
                n_full = int(dec.shape[0])
                tail = dec[-tail_px:] if n_full > tail_px else dec  # sharp tail from the full-context decode
                guide_img = tail[..., :3].to(torch.float32)
                guide_lat = None  # -> Guide uses the image path (sharp resize + re-encode)
                log.info("[%s] hybrid guide: decoded FULL incoming latent (%d px) -> sharp tail %d px (image path)",
                         log_tag, n_full, int(guide_img.shape[0]))
            else:
                log.warning("[%s] guide decode gave shape %s — keeping raw latent guide.",
                            log_tag, tuple(getattr(dec, "shape", ())))
        except Exception as e:
            log.error("[%s] guide decode failed (%s) — keeping raw latent guide.", log_tag, e)
    guide_data = {
        "images": [guide_img], "original_images": [guide_img], "insert_frames": [0],
        "strengths": [float(guide_strength)], "segment_numbers": [0], "guide_latents": [guide_lat],
        "frame_rate": fps, "timeline_data": "", "start_frame": 0,
        "duration_frames": int(window_px), "resize_method": "maintain aspect ratio",
    }
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
    motion_guide_data = {
        "segments": [], "frame_rate": fps,
        "duration_frames": int(window_px), "resize_method": "maintain aspect ratio",
    }

    patched, conditioning = _encode_relay(model, clip, video_latent, prompt or "", "", "", float(epsilon))

    tail_start_lat = T_in - n_overlap_lat
    off_px = max(0, int(audio_base_px) + tail_start_lat * tsf)

    def _empty_audio(nsamp):
        return {"waveform": torch.zeros((1, 2, max(1, nsamp)), dtype=torch.float32), "sample_rate": target_sr}

    if isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor):
        wf = audio["waveform"]
        if wf.ndim == 2:
            wf = wf.unsqueeze(0)
        sr = int(audio.get("sample_rate", target_sr))
        off_s = max(0, int(round(off_px / fps * sr)))
        win_s = max(1, int(round(ltxv_length / fps * sr)))   # window matches the ACTUAL video length
        win = wf[..., off_s:off_s + win_s]
        if win.shape[-1] < win_s:  # pad if the audio runs short at the tail
            pad = torch.zeros((*win.shape[:-1], win_s - win.shape[-1]), dtype=win.dtype, device=win.device)
            win = torch.cat([win, pad], dim=-1)
        combined_audio = {"waveform": win.contiguous(), "sample_rate": sr}
        rem = wf[..., off_s:]
        if rem.shape[-1] <= 0:
            rem = torch.zeros((*wf.shape[:-1], 1), dtype=wf.dtype, device=wf.device)
        remaining_audio = {"waveform": rem.contiguous(), "sample_rate": sr}
        log.info("[%s] audio: base %d + tail %d = off %d px (%.2fs), window %d px (%.2fs), sr=%d",
                 log_tag, int(audio_base_px), tail_start_lat * tsf, off_px, off_px / fps,
                 ltxv_length, ltxv_length / fps, sr)
    else:
        combined_audio = _empty_audio(int(round(ltxv_length / fps * target_sr)))
        remaining_audio = _empty_audio(1)

    audio_latent = {}
    if audio_vae is not None:
        try:
            waveform = combined_audio["waveform"]
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            if hasattr(audio_vae, "first_stage_model"):
                latent_samples = audio_vae.encode(waveform.movedim(1, -1))
            else:
                latent_samples = audio_vae.encode({"waveform": waveform, "sample_rate": combined_audio["sample_rate"]})
            Bc, Cc, F_len, H_len = latent_samples.shape
            mask = torch.zeros((Bc, F_len, H_len), dtype=torch.float32, device=latent_samples.device)
            audio_latent = {"samples": latent_samples, "type": "audio", "noise_mask": mask}
            log.info("[%s] Built preserve-masked audio_latent %s.", log_tag, tuple(latent_samples.shape))
        except Exception as e:
            log.error("[%s] audio_latent build failed (%s) — leaving it empty.", log_tag, e)
            audio_latent = {}

    return {
        "model": patched, "positive": conditioning, "video_latent": video_latent,
        "audio_latent": audio_latent, "guide_data": guide_data, "motion_guide_data": motion_guide_data,
        "frame_rate": fps, "combined_audio": combined_audio, "remaining_audio": remaining_audio,
        "rel_off_px": tail_start_lat * tsf, "ltxv_length": ltxv_length, "latent_t": latent_t,
        "n_overlap_lat": n_overlap_lat, "window_px": window_px,
        "source_anchor_added": source_anchor_added,
    }


class LTXAutoExtend(io.ComfyNode):
    """Headless one-pass video extender for the LTX Director chain.

    Sits between one generation's final (second-sampler) latent and the next pass's
    sampler: it takes the TAIL of the incoming latent as a continuation guide — fed
    straight through the Guide's append_keyframe path (no VAE re-encode, max quality) —
    and emits outputs shaped exactly like LTX Director, so your existing Guide + samplers
    keep working unchanged. Chain N of these (each with its own prompt/seed) to build a
    long clip. Audio self-threads: feed the master audio into pass 1; each node uses its
    (overlap + extension) window and passes the remainder out to the next node.

    The overlap region is regenerated each pass (it's the frozen continuation guide); crop
    it off the front when you stitch the pass outputs — same manual step as before."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXAutoExtend",
            display_name="LTX Auto Extend",
            category="WhatDreamsCost",
            description=(
                "Automate the manual reinsert->trim->prompt loop for iterative extension. Takes "
                "the previous pass's latent, reuses its last 'guide_overlap_seconds' as the "
                "continuation guide (no re-encode), and outputs Director-shaped model/positive/"
                "latent/guide_data/audio for the next Guide + sampler. Audio self-threads pass to pass."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input(
                    "latent",
                    tooltip="The previous pass's final (second-sampler) video latent to continue from.",
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip="Prompt for THIS extension beat.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xffffffffffffffff, step=1,
                    tooltip="Seed for this extension. Passed through on the 'seed' output — wire it into your sampler's noise seed.",
                ),
                io.Float.Input(
                    "extension_seconds", default=12.0, min=0.1, max=600.0, step=0.1,
                    tooltip="How many NEW seconds to generate this pass.",
                ),
                io.Float.Input(
                    "guide_overlap_seconds", default=3.0, min=0.1, max=60.0, step=0.1,
                    tooltip="How much of the incoming latent's TAIL to reuse as the continuation guide (the overlap you crop when stitching).",
                ),
                io.Float.Input(
                    "frame_rate", default=24.0, min=1.0, max=120.0, step=0.001,
                    tooltip="Timeline fps — match your generation.",
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Master audio (pass 1, aligned so it starts at this pass's overlap point) or the previous node's remaining_audio. This pass uses the front (overlap+extension) window.",
                ),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip="Optional Audio VAE. When connected, builds a preserve-masked audio_latent for the window (fixed audio drives the video). Leave unconnected to just pass the audio window through.",
                ),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="Optional VIDEO VAE (hybrid guide). When connected, the guide tail is decoded to images and fed through the image path (sharp resize + re-encode at the working res) instead of the raw latent — fixes the blurry guide under scale_by. Leave unconnected to use the raw latent (sharpest at native res).",
                ),
                io.Float.Input(
                    "guide_strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True,
                    tooltip="Keyframe strength for the continuation guide.",
                ),
                io.Float.Input(
                    "epsilon", default=1e-3, min=0.0, max=1.0, step=1e-4, optional=True,
                ),
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
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
                GuideData.Output(display_name="guide_data"),
                MotionGuideData.Output(display_name="motion_guide_data"),
                io.Float.Output(display_name="frame_rate"),
                io.Audio.Output(display_name="combined_audio", tooltip="This pass's audio window (overlap+extension)."),
                io.Audio.Output(display_name="remaining_audio", tooltip="Audio advanced by extension_seconds — wire to the next LTX Auto Extend."),
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, prompt="", seed=0, extension_seconds=12.0,
                guide_overlap_seconds=3.0, frame_rate=24.0, audio=None, audio_vae=None, vae=None,
                guide_strength=1.0, epsilon=1e-3, anchor_image=None, anchor_latent=None,
                anchor_mode="off", anchor_strength=0.25) -> io.NodeOutput:
        # Manual chain: the fed audio starts at the incoming latent's frame 0, so audio_base_px = 0.
        r = _build_extend_pass(
            model, clip, latent, prompt, extension_seconds, guide_overlap_seconds, frame_rate,
            audio=audio, audio_vae=audio_vae, vae=vae, guide_strength=guide_strength, epsilon=epsilon,
            audio_base_px=0, log_tag="LTXAutoExtend", anchor_image=anchor_image,
            anchor_latent=anchor_latent, anchor_mode=anchor_mode, anchor_strength=anchor_strength,
        )
        log.info("[LTXAutoExtend] guide tail=%d latent frames -> window %d px (%d latent), +%.2fs new, seed=%d",
                 r["n_overlap_lat"], r["window_px"], r["latent_t"], float(extension_seconds), int(seed))
        return io.NodeOutput(
            r["model"], r["positive"], r["video_latent"], r["audio_latent"], r["guide_data"],
            r["motion_guide_data"], r["frame_rate"], r["combined_audio"], r["remaining_audio"], int(seed),
        )


class _ExtendAny(str):
    """Permissive type so a connected list (prompts) or the loop's ANY-typed carried value wires in."""
    def __ne__(self, _other):
        return False


_EXTEND_ANY = _ExtendAny("*")


def _pick_prompt(prompts, global_prompt, index):
    """1-based loop index -> the index-th prompt from a connected list (or newline text). Falls back
    to global_prompt (or 'video') when the list is missing/short/blank. Past the end reuses the last."""
    gp = (global_prompt or "").strip()
    items = None
    if isinstance(prompts, (list, tuple)):
        items = [str(p) for p in prompts]
    elif isinstance(prompts, str) and prompts.strip():
        items = list(prompts.splitlines())
    if items:
        i = int(index) - 1  # loop index is 1-based; list is 0-based
        if 0 <= i < len(items):
            if str(items[i]).strip():
                return str(items[i])
            # in range but blank -> fall through to the global fallback
        elif i >= len(items) and str(items[-1]).strip():
            return str(items[-1])  # past the end -> reuse the last prompt
    return gp or "video"


class LTXExtendInit:
    """Pack the seed clip's latent + everything the extend loop needs into ONE 'state' signal.

    Wire the output into SxCP For Loop Start's initial_value. The loop's 1-based index drives which
    prompt (from a connected list) and seed (base_seed + index) each pass uses; audio auto-cuts from
    the master. Resume a dead run by feeding the latent you stopped at + setting resume_from_seconds
    to that latent's master-timeline position (printed in the previous run's LTX Extend Step log)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_latent": ("LATENT", {"tooltip": "The first clip's latent (seed to extend). On resume, feed the latent you stopped at."}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "base_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Per-step seed = base_seed + loop index (deterministic, resumable)."}),
            },
            "optional": {
                "prompts": (_EXTEND_ANY, {"tooltip": "Connected list of per-step prompts (index N -> Nth). Newline text also works. Blank/short entries fall back to global_prompt."}),
                "global_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Fallback prompt for steps with no list entry."}),
                "audio": ("AUDIO", {"tooltip": "Full master audio, aligned to the seed's start. The loop auto-cuts each pass's window; no manual pre-cut."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE -> per-pass preserve-masked audio_latent for lipsync."}),
                "vae": ("VAE", {"tooltip": "Optional VIDEO VAE (hybrid guide). Connected -> the guide tail is decoded to images and fed through the image path (sharp resize under scale_by) instead of the raw latent. Leave empty to use the raw latent."}),
                "extension_seconds": ("FLOAT", {"default": 12.0, "min": 0.1, "max": 600.0, "step": 0.1}),
                "guide_overlap_seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "guide_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "epsilon": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 1.0, "step": 1e-4}),
                "resume_from_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 100000.0, "step": 0.01, "tooltip": "Fresh run: leave -1 (seed starts at master 0). Resume: master-timeline position of the latent you're feeding (from the previous Step log)."}),
                "anchor_image": ("IMAGE", {"tooltip": "Optional pristine/source keyframe to carry as a weak identity anchor."}),
                "anchor_latent": ("LATENT", {"tooltip": "Optional source latent anchor. Uses the first latent frame when compatible."}),
                "anchor_mode": (["off", "auto", "image", "latent"], {"default": "off", "tooltip": "'off' preserves old behavior. 'auto' prefers latent then image."}),
                "anchor_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weak source-anchor guide strength. Recommended range: 0.15-0.35."}),
                "anchor_every_n_steps": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "Emit the source anchor every N extension steps. 1 = every step."}),
                "combine_global": ("BOOLEAN", {"default": False, "tooltip": "ON: global_prompt is prepended to EVERY step ('<global>, <step>') as a shared style; a blank step becomes just the global. OFF: global is only a fallback for blank steps. (Kept last so it never shifts other widgets.)"}),
            },
        }

    RETURN_TYPES = ("LTX_EXTEND_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "init"
    CATEGORY = "WhatDreamsCost"

    def init(self, seed_latent, model, clip, base_seed, prompts=None, global_prompt="",
             audio=None, audio_vae=None, vae=None, extension_seconds=12.0, guide_overlap_seconds=3.0,
             frame_rate=24.0, guide_strength=1.0, epsilon=1e-3, resume_from_seconds=-1.0,
             anchor_image=None, anchor_latent=None, anchor_mode="off", anchor_strength=0.25,
             anchor_every_n_steps=1, combine_global=False):
        fps = float(frame_rate) if frame_rate else 24.0
        abs_pos_px = 0 if (resume_from_seconds is None or float(resume_from_seconds) < 0) else int(round(float(resume_from_seconds) * fps))
        state = {
            "model": model, "clip": clip, "audio_vae": audio_vae, "master_audio": audio,
            "prompts": prompts, "global_prompt": global_prompt or "", "base_seed": int(base_seed),
            "combine_global": bool(combine_global), "vae": vae,
            "latent": seed_latent, "abs_pos_px": abs_pos_px,
            "extension_seconds": float(extension_seconds), "guide_overlap_seconds": float(guide_overlap_seconds),
            "frame_rate": fps, "guide_strength": float(guide_strength), "epsilon": float(epsilon),
            "anchor_image": anchor_image, "anchor_latent": anchor_latent,
            "anchor_mode": _normalize_anchor_mode(anchor_mode),
            "anchor_strength": _anchor_strength_for_state(anchor_strength, 0.25),
            "anchor_every_n_steps": _positive_int(anchor_every_n_steps, 1),
        }
        nprompts = len(prompts) if isinstance(prompts, (list, tuple)) else ("text" if prompts else "none")
        log.info("[LTXExtendInit] packed state: seed latent T=%d, abs_pos=%d px (%.2fs), base_seed=%d, prompts=%s",
                 int(seed_latent["samples"].shape[2]), abs_pos_px, abs_pos_px / fps, int(base_seed), nprompts)
        return (state,)


class LTXExtendStep:
    """One loop pass. Reads the state + the loop index -> picks prompts[index], seed = base_seed +
    index, cuts this pass's audio window from the master, and emits Director-shaped outputs for your
    Guide + two samplers. Pass 'state' through to LTX Extend Collect after the second sampler, and
    wire 'seed' into the sampler's noise seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_EXTEND_ANY,),
                "index": ("INT", {"default": 1, "min": 0, "max": 0xffffffff, "forceInput": True, "tooltip": "Loop index from SxCP For Loop Start (1-based)."}),
            },
            "optional": {
                "seed_offset": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "forceInput": True, "tooltip": "Seed rotation from a review retry loop (LTX Review Gate 'attempt'). seed = base_seed + index + seed_offset."}),
                "prompts": (_EXTEND_ANY, {"tooltip": "Optional LIVE prompts list override (re-read every run). Wire your prompt source here too so 'Reload' re-pulls a changed prompt; else the Init-time list is used."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "LATENT", "LATENT", "GUIDE_DATA", "MOTION_GUIDE_DATA", "FLOAT", "AUDIO", "LTX_EXTEND_STATE", "INT", "STRING")
    RETURN_NAMES = ("model", "positive", "video_latent", "audio_latent", "guide_data", "motion_guide_data", "frame_rate", "combined_audio", "state", "seed", "prompt")
    FUNCTION = "step"
    CATEGORY = "WhatDreamsCost"

    def step(self, state, index, seed_offset=0, prompts=None):
        st = state or {}
        idx = int(index)
        prompt_src = prompts if prompts is not None else st.get("prompts")
        gp = str(st.get("global_prompt", "")).strip()
        raw = _pick_prompt(prompt_src, gp, idx)
        # combine_global ON -> global is a shared prefix on every step ("<global>, <step>"); a blank
        # step becomes just the global. OFF -> global stays a fallback only (raw unchanged).
        if bool(st.get("combine_global", False)) and gp and raw and raw != gp:
            prompt = gp + ", " + raw
        else:
            prompt = raw
        seed = (int(st.get("base_seed", 0)) + idx + int(seed_offset)) & 0xffffffffffffffff
        fps = float(st.get("frame_rate", 24.0)) or 24.0
        r = _build_extend_pass(
            st.get("model"), st.get("clip"), st.get("latent"), prompt,
            st.get("extension_seconds", 12.0), st.get("guide_overlap_seconds", 3.0),
            st.get("frame_rate", 24.0), audio=st.get("master_audio"), audio_vae=st.get("audio_vae"),
            vae=st.get("vae"),
            guide_strength=st.get("guide_strength", 1.0), epsilon=st.get("epsilon", 1e-3),
            audio_base_px=int(st.get("abs_pos_px", 0)), log_tag="LTXExtendStep",
            anchor_image=st.get("anchor_image"), anchor_latent=st.get("anchor_latent"),
            anchor_mode=st.get("anchor_mode", "off"), anchor_strength=st.get("anchor_strength", 0.25),
            anchor_every_n_steps=st.get("anchor_every_n_steps", 1), step_index=idx,
        )
        preview = (prompt[:60] + "...") if len(prompt) > 60 else prompt
        log.info("[LTXExtendStep] index=%d seed=%d prompt=%r -> window %d px (%d latent) at master %.2fs",
                 idx, seed, preview, r["window_px"], r["latent_t"], int(st.get("abs_pos_px", 0)) / fps)
        return (r["model"], r["positive"], r["video_latent"], r["audio_latent"], r["guide_data"],
                r["motion_guide_data"], r["frame_rate"], r["combined_audio"], st, int(seed), prompt)


class LTXExtendCollect:
    """Fold a finished pass's latent back into the loop state. Wire the second sampler's LATENT here
    plus the 'state' from LTX Extend Step; send the output to SxCP For Loop End's initial_value.
    Advances the master audio position by this pass's new-content length so the next pass auto-aligns."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (_EXTEND_ANY,),
                "latent": ("LATENT", {"tooltip": "This pass's final (second-sampler) latent."}),
            },
        }

    RETURN_TYPES = ("LTX_EXTEND_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "collect"
    CATEGORY = "WhatDreamsCost"

    def collect(self, state, latent):
        st = dict(state or {})
        tsf = 8
        fps = float(st.get("frame_rate", 24.0)) or 24.0
        overlap_px = max(1, int(round(float(st.get("guide_overlap_seconds", 3.0)) * fps)))
        old = st.get("latent")
        rel_off = 0
        if isinstance(old, dict) and isinstance(old.get("samples"), torch.Tensor):
            T_old = int(old["samples"].shape[2])
            n_ov = max(1, min(T_old, (overlap_px + tsf - 1) // tsf))
            rel_off = (T_old - n_ov) * tsf   # same advance the Step used to cut this pass's audio
        new_abs = int(st.get("abs_pos_px", 0)) + rel_off
        st["latent"] = latent
        st["abs_pos_px"] = new_abs
        next_t = int(latent["samples"].shape[2]) if isinstance(latent, dict) and isinstance(latent.get("samples"), torch.Tensor) else -1
        log.info("[LTXExtendCollect] advanced audio %d px -> abs_pos %d px (%.2fs); next latent T=%d",
                 rel_off, new_abs, new_abs / fps, next_t)
        return (st,)


# --- LTX Review Gate: human-in-the-loop pass / reroll / reload for the extend retry loop ---
_review_events = {}    # node_id -> threading.Event
_review_actions = {}   # node_id -> "pass" | "reroll" | "reload"


@PromptServer.instance.routes.post("/ltx_review_decide")
async def ltx_review_decide(request):
    """Frontend button -> unblock the waiting LTX Review Gate with a decision."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON"}, status=400)
    key = str(data.get("node_id", ""))
    action = str(data.get("action", "")).lower()
    if action not in ("pass", "reroll", "reload"):
        return web.json_response({"ok": False, "error": "bad action"}, status=400)
    _review_actions[key] = action
    ev = _review_events.get(key)
    if ev is not None:
        ev.set()
    return web.json_response({"ok": True})


def _review_preview_frames(images, max_frames=16, quality=70):
    """Sample up to max_frames evenly from an IMAGE tensor (B,H,W,3 in 0..1) -> base64 JPEG data URIs."""
    out = []
    try:
        n = int(images.shape[0])
        if n <= 0:
            return out
        count = min(max_frames, n)
        idxs = [0] if count == 1 else [int(round(i * (n - 1) / (count - 1))) for i in range(count)]
        for i in idxs:
            arr = (images[i].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            im = Image.fromarray(arr)
            buf = _io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    except Exception as e:
        log.error("[LTXReviewGate] preview encode failed: %s", e)
    return out


def _review_temp_name(key, attempt, ext):
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(key))
    return f"ltx_review_{safe}_{int(attempt)}.{ext}"


def _review_encode_one(images, audio, fps, key, attempt, with_audio):
    """One mp4 encode attempt: frames at fps, optionally MUXING the audio window into the same file so
    video+audio are a single synced stream (scrubs together, no drift). Returns the /view URL."""
    n = int(images.shape[0]); H = int(images.shape[1]); W = int(images.shape[2])
    if n <= 0:
        raise ValueError("no frames")
    W2 = W - (W % 2); H2 = H - (H % 2)  # yuv420p needs even dims
    fps_i = max(1, int(round(float(fps) or 24)))
    tmp = folder_paths.get_temp_directory(); os.makedirs(tmp, exist_ok=True)
    fname = _review_temp_name(key, attempt, "mp4")
    container = av.open(os.path.join(tmp, fname), mode="w")
    try:
        vs = container.add_stream("libx264", rate=fps_i)
        vs.width = W2; vs.height = H2; vs.pix_fmt = "yuv420p"
        astream = None; samples = None; sr = 44100; layout = "stereo"
        if with_audio and isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor):
            wf = audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]
            wf = wf.cpu().float().clamp(-1, 1).contiguous()
            ch = int(wf.shape[0]); layout = "stereo" if ch >= 2 else "mono"
            sr = int(audio.get("sample_rate", 44100))
            samples = np.ascontiguousarray(wf.numpy())  # [ch, N] float32 (planar = fltp)
            astream = container.add_stream("aac", rate=sr)
            astream.layout = layout
        for i in range(n):  # video frames
            arr = (images[i, :H2, :W2, :3].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for pkt in vs.encode(vf):
                container.mux(pkt)
        for pkt in vs.encode():
            container.mux(pkt)
        if astream is not None:  # audio frames, chunked to the aac frame size
            nsamp = samples.shape[1]; pts = 0
            for s in range(0, nsamp, 1024):
                seg = np.ascontiguousarray(samples[:, s:s + 1024])
                af = av.AudioFrame.from_ndarray(seg, format="fltp", layout=layout)
                af.sample_rate = sr; af.pts = pts; pts += seg.shape[1]
                for pkt in astream.encode(af):
                    container.mux(pkt)
            for pkt in astream.encode():
                container.mux(pkt)
    finally:
        container.close()
    return f"/view?filename={fname}&type=temp&t={int(attempt)}"


def _review_encode_video(images, audio, fps, key, attempt):
    """Encode a temp mp4 at fps, muxing the audio in so it's ONE synced file. Returns (url, has_audio):
    tries video+audio first, falls back to video-only, then None (JS uses the frame slideshow)."""
    if PromptServer is None or folder_paths is None or not hasattr(images, "shape"):
        return None, False
    has_audio = isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor)
    if has_audio:
        try:
            return _review_encode_one(images, audio, fps, key, attempt, True), True
        except Exception as e:
            log.info("[LTXReviewGate] muxed encode failed (%s) — trying video-only.", e)
    try:
        return _review_encode_one(images, None, fps, key, attempt, False), False
    except Exception as e:
        log.info("[LTXReviewGate] video encode failed (%s) — using frame slideshow.", e)
        return None, False


def _review_serve_audio(audio, key, attempt):
    """Save this pass's audio window to a temp WAV -> /view URL, played alongside the preview video."""
    if PromptServer is None or folder_paths is None:
        return None
    if not (isinstance(audio, dict) and isinstance(audio.get("waveform"), torch.Tensor)):
        return None
    try:
        import torchaudio
        wf = audio["waveform"]
        if wf.ndim == 3:
            wf = wf[0]
        sr = int(audio.get("sample_rate", 44100))
        tmp = folder_paths.get_temp_directory(); os.makedirs(tmp, exist_ok=True)
        fname = _review_temp_name(key, attempt, "wav")
        torchaudio.save(os.path.join(tmp, fname), wf.cpu().float().clamp(-1, 1), sr)
        return f"/view?filename={fname}&type=temp&t={int(attempt)}"
    except Exception as e:
        log.info("[LTXReviewGate] audio preview unavailable (%s)", e)
        return None


class LTXReviewSeed:
    """Small seed source controlled by the LTX Review Gate frontend."""

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


class LTXReviewGate:
    """Human-in-the-loop review for the extend retry loop. BLOCKS, plays this attempt's decoded frames
    in the node, and waits for Pass / Reroll / Reload.

    Use as the CONDITION of an SxCP While Loop wrapped around [LTX Extend Step -> Guide -> samplers ->
    VAE decode]. 'continue' drives the While Loop End condition (True = redo this step), 'attempt'
    carries the seed-rotation counter (wire it back to the loop and into Step.seed_offset), and 'latent'
    carries the approved result out. Reroll -> attempt+1 (new seed); Reload -> re-run so Step re-pulls
    its live prompts input; Pass -> exit the retry loop with the shown latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The decoded frames of THIS attempt (VAE decode of the pass's latent)."}),
                "latent": ("LATENT", {"tooltip": "This attempt's latent (passed through on Pass)."}),
            },
            "optional": {
                "attempt": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "forceInput": True, "tooltip": "Optional: the current retry counter (LTX Extend Loop Open 'attempt'), shown in the preview label."}),
                "auto_pass_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 1.0, "tooltip": "0 = wait forever for a button. >0 = auto-Pass after N seconds (unattended runs)."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.001, "tooltip": "Playback fps for the preview — wire your generation frame_rate so it plays at the right speed."}),
                "audio": ("AUDIO", {"tooltip": "Optional: this pass's audio window (Step combined_audio). Played in sync with the preview."}),
                "passthrough": ("BOOLEAN", {"default": False, "tooltip": "ON: use the gate as a PREVIEW only — show the video but never block; always 'pass' so the loop runs unattended. OFF: block and wait for Pass / Reroll / Reload."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "dynprompt": "DYNPROMPT"},
        }

    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("decision", "latent")
    FUNCTION = "review"
    CATEGORY = "WhatDreamsCost"

    def review(self, images, latent, attempt=0, auto_pass_seconds=0.0, fps=24.0, audio=None,
               passthrough=False, unique_id=None, dynprompt=None):
        # Key by the DISPLAY node id, not unique_id: inside the loop's GraphBuilder expansion the node is
        # cloned with a new id each iteration, but its display id stays the frontend node.id the buttons use.
        key = str(unique_id)
        if dynprompt is not None:
            try:
                key = str(dynprompt.get_display_node_id(str(unique_id)))
            except Exception:
                key = str(unique_id)

        # Always push the preview to the node. The mp4 muxes the audio in (one synced <video>); a
        # separate audio_url is only sent when the audio ISN'T in the video (video-only or slideshow).
        video_url, video_has_audio = _review_encode_video(images, audio, fps, key, attempt)
        audio_url = None if (video_url and video_has_audio) else _review_serve_audio(audio, key, attempt)
        frames = [] if video_url else _review_preview_frames(images)  # frames only as a fallback
        try:
            PromptServer.instance.send_sync("ltx_review_show",
                                            {"node_id": key, "frames": frames, "attempt": int(attempt),
                                             "fps": float(fps) if fps else 24.0,
                                             "video_url": video_url, "audio_url": audio_url,
                                             "passthrough": bool(passthrough)})
        except Exception as e:
            log.error("[LTXReviewGate] preview push failed: %s", e)

        # Passthrough: use the gate purely as a preview — never block, always pass.
        if passthrough:
            log.info("[LTXReviewGate] passthrough -> preview only, auto-pass (attempt %s)", int(attempt))
            return ("pass", latent)

        # Otherwise block the worker thread until a button arrives (or auto-pass / user interrupt).
        ev = threading.Event()
        _review_events[key] = ev
        _review_actions.pop(key, None)
        waited = 0.0
        while not ev.wait(timeout=0.5):
            comfy.model_management.throw_exception_if_processing_interrupted()
            waited += 0.5
            if auto_pass_seconds and waited >= float(auto_pass_seconds):
                _review_actions[key] = "pass"
                break

        action = _review_actions.pop(key, "pass")
        _review_events.pop(key, None)
        try:
            PromptServer.instance.send_sync("ltx_review_done", {"node_id": key, "action": action})
        except Exception:
            pass

        # decision string is consumed by LTX Extend Loop Close: reroll -> attempt+1 + redo step;
        # reload -> redo step (Step re-pulls its live prompts); pass -> fold latent + advance.
        log.info("[LTXReviewGate] decision=%s (attempt %s)", action, int(attempt))
        return (action, latent)


NODE_CLASS_MAPPINGS = {
    "LTXDirector": LTXDirector,
    "LTXKeyframeOut": LTXKeyframeOut,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
}
