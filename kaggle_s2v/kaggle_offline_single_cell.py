"""
Kaggle: single-cell OFFLINE Speech-to-Video (image + audio -> video) for LTX-2.3

Copy/paste this entire file into ONE Kaggle code cell.

This is designed for the "no internet" Kaggle runtime where:

- Weights are already mounted under:
    /kaggle/input/datasets/yliu95/ltx23-offline-assets
- Test inputs are already mounted under:
    /kaggle/input/datasets/yliu95/s2v-test-data

It also handles Kaggle images that do NOT include PyAV (`import av`) by:

- patching audio duration to use stdlib `wave` / `ffprobe`
- patching audio decode + mp4 encode to use `ffmpeg` when PyAV is missing

No downloads and no installs. Only output videos/configs go to /kaggle/working.
"""

import contextlib
import os
import shutil
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# 0) Keep caches out of /kaggle/working (only outputs should go there)
# -----------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/kaggle/temp/hf/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/kaggle/temp/hf/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/kaggle/temp/xdg-cache")
os.environ.setdefault("TORCH_HOME", "/kaggle/temp/torch")

# Recommended for large models: reduces CUDA memory fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Hard-offline (prevents any network calls).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# -----------------------------------------------------------------------------
# 1) Paths (your Kaggle inputs)
# -----------------------------------------------------------------------------
ASSETS_ROOT = "/kaggle/input/datasets/yliu95/ltx23-offline-assets"
TEST_DATA_DIR = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data"

AUDIO_PATH = f"{TEST_DATA_DIR}/4s_mandrain_Chinese.wav"
IMAGE_PATH = f"{TEST_DATA_DIR}/female secretary 512x763.png"
AUDIO_TEXT_FILE = f"{TEST_DATA_DIR}/4s_mandrian_Chinese_text.txt"  # optional
PROMPT_FILE = f"{TEST_DATA_DIR}/I2V prompt.txt"

# Repo code is bundled inside your assets dataset:
REPO_ROOT = Path(ASSETS_ROOT) / "repo_ltx-2.3" / "ltx-2.3"
RUN_S2V_PY = REPO_ROOT / "kaggle_s2v" / "run_s2v.py"

if not Path(ASSETS_ROOT).exists():
    raise FileNotFoundError(f"ASSETS_ROOT not found: {ASSETS_ROOT}")
if not Path(TEST_DATA_DIR).exists():
    raise FileNotFoundError(f"TEST_DATA_DIR not found: {TEST_DATA_DIR}")
if not RUN_S2V_PY.exists():
    raise FileNotFoundError(f"Runner not found: {RUN_S2V_PY}")


# -----------------------------------------------------------------------------
# 2) Make an "assets_root" that works with BOTH layouts (flat vs ltx/gemma dirs)
# -----------------------------------------------------------------------------
# Some older code expects:
#   assets_root/ltx/* and assets_root/gemma/*
# Your dataset is flat (files at ASSETS_ROOT). We create a tiny symlinked tree
# in /kaggle/temp and pass that as --assets_root.
OFFLINE_ASSETS_ROOT = Path("/kaggle/temp/ltx23_assets_root")
LTX_DIR = OFFLINE_ASSETS_ROOT / "ltx"
GEMMA_DIR = OFFLINE_ASSETS_ROOT / "gemma"


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_assets_tree() -> None:
    if (LTX_DIR / "ltx-2.3-22b-distilled.safetensors").exists() and (GEMMA_DIR / "tokenizer.model").exists():
        return

    root = Path(ASSETS_ROOT)
    LTX_DIR.mkdir(parents=True, exist_ok=True)
    GEMMA_DIR.mkdir(parents=True, exist_ok=True)

    # LTX weights
    for name in (
        "ltx-2.3-22b-distilled.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    ):
        src = root / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required LTX file: {src}")
        _safe_link_or_copy(src, LTX_DIR / name)

    # Gemma assets
    for name in (
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        src = root / name
        if src.exists():
            _safe_link_or_copy(src, GEMMA_DIR / name)

    shards = sorted(root.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"Missing Gemma shard files under: {root} (expected model*.safetensors)")
    for src in shards:
        _safe_link_or_copy(src, GEMMA_DIR / src.name)


_prepare_assets_tree()


# -----------------------------------------------------------------------------
# 3) Ensure imports work without installing anything
# -----------------------------------------------------------------------------
sys.path.insert(0, str(REPO_ROOT / "kaggle_s2v"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-core/src"))
sys.path.insert(0, str(REPO_ROOT / "packages/ltx-pipelines/src"))


# -----------------------------------------------------------------------------
# 4) PyAV is often missing on Kaggle. Provide a tiny stub so older code imports.
# -----------------------------------------------------------------------------
try:
    import av as _av  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    av_stub = types.ModuleType("av")

    # Minimal submodules/classes referenced in type annotations in some versions.
    container_mod = types.ModuleType("av.container")
    audio_mod = types.ModuleType("av.audio")
    video_mod = types.ModuleType("av.video")
    resampler_mod = types.ModuleType("av.audio.resampler")

    class _Stub:  # noqa: D401
        """Placeholder for PyAV classes when PyAV is not installed."""

    container_mod.Container = _Stub  # type: ignore[attr-defined]
    audio_mod.AudioStream = _Stub  # type: ignore[attr-defined]
    resampler_mod.AudioResampler = _Stub  # type: ignore[attr-defined]
    av_stub.container = container_mod  # type: ignore[attr-defined]
    av_stub.audio = audio_mod  # type: ignore[attr-defined]
    av_stub.video = video_mod  # type: ignore[attr-defined]
    audio_mod.resampler = resampler_mod  # type: ignore[attr-defined]
    av_stub.AudioFrame = _Stub  # type: ignore[attr-defined]
    av_stub.VideoFrame = _Stub  # type: ignore[attr-defined]

    def _no_pyav(*_a, **_kw):  # noqa: ANN001
        raise ModuleNotFoundError("PyAV ('av') is not installed in this Kaggle image.")

    av_stub.open = _no_pyav  # type: ignore[attr-defined]

    sys.modules["av"] = av_stub
    sys.modules["av.container"] = container_mod
    sys.modules["av.audio"] = audio_mod
    sys.modules["av.video"] = video_mod
    sys.modules["av.audio.resampler"] = resampler_mod

    print("[S2V] PyAV not found. Using ffmpeg/wave fallbacks for audio/video I/O.")


# -----------------------------------------------------------------------------
# 5) Monkeypatch: audio duration without PyAV (WAV -> ffprobe -> (optional) av)
# -----------------------------------------------------------------------------
import importlib.util

spec = importlib.util.spec_from_file_location("ltx_run_s2v", str(RUN_S2V_PY))
assert spec and spec.loader
run_s2v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_s2v)  # type: ignore[arg-type]


def _audio_duration_seconds_no_pyav(audio_path: str) -> float:
    """Return audio duration without requiring PyAV.

    Prefer:
      1) stdlib `wave` for .wav files (fast, no deps)
      2) `ffprobe` if available
    """
    import subprocess
    import wave

    # Fast-path for WAV.
    try:
        with contextlib.closing(wave.open(audio_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                raise ValueError("Invalid WAV sample rate")
            return float(frames) / float(rate)
    except (wave.Error, EOFError, OSError):
        pass

    # Fallback to ffprobe (usually present on Kaggle images).
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(r.stdout.strip())


# Patch even if the repo snapshot is old.
run_s2v._audio_duration_seconds = _audio_duration_seconds_no_pyav  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 6) Monkeypatch: media I/O fallback (decode audio + encode mp4 via ffmpeg)
# -----------------------------------------------------------------------------
import numpy as np
import torch
from tqdm.auto import tqdm


def _require_ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise RuntimeError(
            "ffmpeg is required for offline mp4 encoding when PyAV is not installed. "
            "This Kaggle image seems to be missing ffmpeg."
        )
    return p


def _write_wav_file(path: str, *, waveform: torch.Tensor, sampling_rate: int) -> None:
    import wave

    wf = waveform
    if wf.ndim == 3:
        wf = wf[0]
    if wf.ndim == 1:
        wf = wf.unsqueeze(0)

    # Ensure stereo for consistent muxing. If mono, duplicate.
    if wf.shape[0] == 1:
        wf = wf.repeat(2, 1)
    elif wf.shape[0] > 2:
        wf = wf[:2]

    if wf.dtype != torch.int16:
        wf = torch.clip(wf.to(torch.float32), -1.0, 1.0)
        wf = (wf * 32767.0).to(torch.int16)

    pcm = wf.t().contiguous().cpu().numpy()  # (samples, channels) int16
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(int(sampling_rate))
        f.writeframes(pcm.tobytes())


def _decode_audio_wav_or_ffmpeg(path: str, device: torch.device, start_time: float = 0.0, max_duration: float | None = None):
    from ltx_core.types import Audio  # imported here so sys.path is already set

    # 1) WAV via stdlib.
    try:
        import wave

        with contextlib.closing(wave.open(path, "rb")) as wf:
            sample_rate = int(wf.getframerate())
            channels = int(wf.getnchannels())
            sampwidth = int(wf.getsampwidth())
            n_frames = int(wf.getnframes())

            start_frame = max(0, int(round(float(start_time) * sample_rate)))
            start_frame = min(start_frame, n_frames)
            wf.setpos(start_frame)

            if max_duration is None:
                frames_to_read = n_frames - start_frame
            else:
                frames_to_read = min(n_frames - start_frame, int(round(float(max_duration) * sample_rate)))

            raw = wf.readframes(frames_to_read)

        if not raw:
            return None

        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            arr = (arr - 128.0) / 128.0
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        audio_np = arr.reshape(-1, channels).T  # (channels, samples)
        waveform = torch.from_numpy(audio_np).to(device).unsqueeze(0)  # (1, C, samples)
        return Audio(waveform=waveform, sampling_rate=sample_rate)
    except Exception:
        pass

    # 2) Fallback: ffmpeg decode to s16le stereo @ 16kHz.
    _require_ffmpeg()
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start_time and float(start_time) > 0:
        cmd += ["-ss", str(float(start_time))]
    cmd += ["-i", path]
    if max_duration is not None:
        cmd += ["-t", str(float(max_duration))]
    cmd += ["-f", "s16le", "-ac", "2", "-ar", "16000", "pipe:1"]

    pcm = subprocess.check_output(cmd)
    if not pcm:
        return None
    audio_i16 = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2).T
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_f32).to(device).unsqueeze(0)
    return Audio(waveform=waveform, sampling_rate=16000)


def _encode_video_ffmpeg(
    video,  # torch.Tensor | Iterator[torch.Tensor]
    *,
    fps: int,
    audio,
    output_path: str,
    expected_num_frames: int | None,
) -> None:
    _require_ffmpeg()

    # Materialize iterator
    if isinstance(video, torch.Tensor):
        video_iter = iter([video])
    else:
        video_iter = iter(video)

    first_chunk = next(video_iter)
    if first_chunk.dtype != torch.uint8:
        first_chunk = first_chunk.to(dtype=torch.uint8)

    _, height, width, channels = first_chunk.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames with 3 channels, got {channels}.")

    wav_path = None
    if audio is not None:
        tmp_dir = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path.cwd()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav_path = str(tmp_dir / f"ltx_audio_{os.getpid()}.wav")
        _write_wav_file(wav_path, waveform=audio.waveform, sampling_rate=int(audio.sampling_rate))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
    ]
    if wav_path is not None:
        cmd += ["-i", wav_path]

    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18"]
    if wav_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [output_path]

    pbar = tqdm(total=expected_num_frames, desc="Encode mp4 (ffmpeg)", unit="frame", dynamic_ncols=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    def _all_chunks():
        yield first_chunk
        yield from video_iter

    try:
        for chunk in _all_chunks():
            chunk_u8 = chunk.to(device="cpu", dtype=torch.uint8).contiguous()
            proc.stdin.write(chunk_u8.numpy().tobytes())
            pbar.update(int(chunk_u8.shape[0]))
            pbar.set_postfix(shape=f"{int(chunk_u8.shape[0])}f", refresh=False)
    finally:
        pbar.close()
        with contextlib.suppress(Exception):
            proc.stdin.close()
        rc = proc.wait()
        if wav_path is not None:
            with contextlib.suppress(Exception):
                os.remove(wav_path)

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


# Apply monkeypatches if the installed repo snapshot doesn't already support no-PyAV.
import ltx_pipelines.utils.media_io as media_io  # noqa: E402

media_io.decode_audio_from_file = _decode_audio_wav_or_ffmpeg  # type: ignore[assignment]


def _encode_video_patched(video, fps: int, audio, output_path: str, video_chunks_number: int, expected_num_frames: int | None = None) -> None:  # noqa: ARG001
    _encode_video_ffmpeg(video, fps=int(fps), audio=audio, output_path=output_path, expected_num_frames=expected_num_frames)


media_io.encode_video = _encode_video_patched  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# 6.5) Ensure tqdm denoising progress (speed/ETA + sigma + VRAM) if missing
# -----------------------------------------------------------------------------
# Some repo snapshots may not include the richer tqdm bars. If the sampler
# already supports the `progress_*` kwargs, we swap in a tqdm+VRAM version.
try:
    import inspect
    from dataclasses import replace

    import ltx_pipelines.utils.samplers as samplers  # noqa: E402
    from ltx_pipelines.utils.helpers import post_process_latent  # noqa: E402

    _sig = inspect.signature(samplers.euler_denoising_loop)
    if "progress" in _sig.parameters and "progress_vram" in _sig.parameters:

        def _cuda_vram_stats_gb(device: torch.device) -> dict[str, str]:
            if device.type != "cuda" or not torch.cuda.is_available():
                return {}
            idx = device.index if device.index is not None else torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(idx) / (1024**3)
            peak = torch.cuda.max_memory_allocated(idx) / (1024**3)
            return {"vram": f"{allocated:.1f}G", "resv": f"{reserved:.1f}G", "peak": f"{peak:.1f}G"}

        def _euler_denoising_loop_tqdm(  # noqa: PLR0913
            sigmas,
            video_state,
            audio_state,
            stepper,
            denoise_fn,
            *,
            progress: bool = True,
            progress_desc: str | None = None,
            progress_leave: bool = True,
            progress_position: int = 0,
            progress_vram: bool = True,
            progress_vram_every: int = 1,
            progress_bar_format: str | None = None,
            **_ignored,
        ):
            iterable = sigmas[:-1]
            pbar = (
                tqdm(
                    iterable,
                    desc=progress_desc,
                    leave=progress_leave,
                    position=progress_position,
                    dynamic_ncols=True,
                    bar_format=progress_bar_format,
                )
                if progress
                else iterable
            )

            for step_idx, _ in enumerate(pbar):
                denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
                denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
                denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
                video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
                audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))

                if progress and progress_vram and hasattr(pbar, "set_postfix") and progress_vram_every > 0:
                    if step_idx % progress_vram_every == 0 or step_idx == len(sigmas) - 2:
                        postfix = {"sigma": f"{float(sigmas[step_idx].item()):.4f}", **_cuda_vram_stats_gb(video_state.latent.device)}
                        pbar.set_postfix(postfix, refresh=False)

            return (video_state, audio_state)

        samplers.euler_denoising_loop = _euler_denoising_loop_tqdm  # type: ignore[assignment]

        # Some pipelines import `euler_denoising_loop` into their module namespace.
        with contextlib.suppress(Exception):
            import ltx_pipelines.distilled_a2v as distilled_a2v  # noqa: E402

            distilled_a2v.euler_denoising_loop = _euler_denoising_loop_tqdm  # type: ignore[attr-defined]
except Exception as e:
    print("[S2V] (warning) Could not patch tqdm denoising loop:", repr(e))


# -----------------------------------------------------------------------------
# 7) Run (edit these settings if you want)
# -----------------------------------------------------------------------------
# Notes on the non-beginner settings below:
#
# QUANTIZATION:
#   - "fp8-cast": stores transformer weights in FP8 and upcasts during matmul. Lower VRAM, good default.
#   - "fp8-scaled-mm": uses TensorRT-LLM scaled FP8 matmul ops (only if available in the environment).
#   - "none": full precision (highest VRAM).
#
# CLEANUP:
#   - "aggressive": unload/reload big modules between stages to minimize peak VRAM (slowest, safest).
#   - "balanced": unload some modules between stages (middle ground).
#   - "none": keep everything loaded (fastest, highest peak VRAM).
#
# VAE_TILING:
#   - "default": decodes the video in tiles (safer for long/high-res videos and lower VRAM).
#   - "none": decode in one go (faster but uses more VRAM).
#
# PROGRESS_VRAM_EVERY:
#   Update VRAM stats in the tqdm postfix every N denoising steps. 1 = most detailed, slight overhead.

PIPELINE = "distilled-a2v"
QUANTIZATION = "fp8-cast"
CLEANUP = "aggressive"
VAE_TILING = "default"

SEED = 10
FPS = 24.0
PROGRESS_VRAM_EVERY = 1

# Optional overrides (leave as None to auto-derive from audio/image)
NUM_FRAMES = None  # default: round(audio_duration * FPS), snapped to 8k+1
WIDTH = None  # default: pad image width to multiple-of-64
HEIGHT = None  # default: pad image height to multiple-of-64

RUN_NAME = f"s2v_offline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


argv = [
    "run_s2v.py",
    "--assets_mode",
    "offline",
    "--assets_root",
    str(OFFLINE_ASSETS_ROOT),
    "--pipeline",
    PIPELINE,
    "--audio_path",
    AUDIO_PATH,
    "--image_path",
    IMAGE_PATH,
    "--prompt_file",
    PROMPT_FILE,
    "--audio_text_file",
    AUDIO_TEXT_FILE,
    "--quantization",
    QUANTIZATION,
    "--cleanup",
    CLEANUP,
    "--vae_tiling",
    VAE_TILING,
    "--seed",
    str(int(SEED)),
    "--fps",
    str(float(FPS)),
    "--progress",
    "--progress_vram",
    "--progress_vram_every",
    str(int(PROGRESS_VRAM_EVERY)),
    "--run_name",
    RUN_NAME,
]

if NUM_FRAMES is not None:
    argv += ["--num_frames", str(int(NUM_FRAMES))]
if WIDTH is not None:
    argv += ["--width", str(int(WIDTH))]
if HEIGHT is not None:
    argv += ["--height", str(int(HEIGHT))]

print("[S2V] Launch:", " ".join(argv))
sys.argv = argv
run_s2v.main()
print(f"[S2V] Output: /kaggle/working/s2v/outputs/{RUN_NAME}/video.mp4")
