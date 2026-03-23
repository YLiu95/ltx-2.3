from __future__ import annotations

import contextlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO
from pathlib import Path

try:
    import av  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    av = None  # type: ignore
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from ltx_core.types import Audio
from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF

logger = logging.getLogger(__name__)
_WARNED_NO_AV = False


def _warn_no_av_once(feature: str) -> None:
    global _WARNED_NO_AV
    if _WARNED_NO_AV:
        return
    _WARNED_NO_AV = True
    logger.warning(
        "PyAV ('av') is not installed; %s will fall back to a simpler implementation. "
        "For best results, install PyAV in your environment.",
        feature,
    )


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_latent(latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (latent / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_image_conditioning(
    image_path: str,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    crf: int = DEFAULT_IMAGE_CRF,
) -> torch.Tensor:
    """
    Loads an image from a path and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    image = decode_image(image_path=image_path)
    image = preprocess(image=image, crf=crf)
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image = resize_and_center_crop(image, height, width)
    image = normalize_latent(image, device, dtype)
    return image


def load_video_conditioning(
    video_path: str, height: int, width: int, frame_cap: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Loads a video from a path and preprocesses it for conditioning.
    Note: The video is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
    result = None
    for f in frames:
        frame = resize_and_center_crop(f.to(torch.float32), height, width)
        frame = normalize_latent(frame, device, dtype)
        result = frame if result is None else torch.cat([result, frame], dim=2)
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]
    return np_array


def _write_audio(container: av.container.Container, audio_stream: av.audio.AudioStream, audio: Audio) -> None:
    samples = audio.waveform
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio.sampling_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def _require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is required for video/audio I/O when PyAV is not available. "
            "Install ffmpeg (system package) or install PyAV ('av')."
        )
    return ffmpeg_path


def _write_wav_file(path: str, audio: Audio) -> None:
    import wave

    waveform = audio.waveform
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Ensure stereo for consistent muxing. If mono, duplicate.
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    # Convert to int16 PCM.
    if waveform.dtype != torch.int16:
        waveform = torch.clip(waveform.to(torch.float32), -1.0, 1.0)
        waveform = (waveform * 32767.0).to(torch.int16)

    # Interleave to (samples, channels).
    pcm = waveform.t().contiguous().cpu().numpy()

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(audio.sampling_rate))
        wf.writeframes(pcm.tobytes())


def _encode_video_ffmpeg(
    video_chunks: Iterator[torch.Tensor],
    *,
    fps: int,
    audio: Audio | None,
    output_path: str,
    expected_num_frames: int | None,
) -> None:
    _require_ffmpeg()

    first_chunk = next(video_chunks)
    if first_chunk.dtype != torch.uint8:
        first_chunk = first_chunk.to(dtype=torch.uint8)

    _, height, width, channels = first_chunk.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames with 3 channels, got {channels}.")

    wav_path = None
    if audio is not None:
        tmp_dir = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        wav_path = str(tmp_dir / f"ltx_audio_{os.getpid()}.wav")
        _write_wav_file(wav_path, audio)

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

    def all_chunks() -> Iterator[torch.Tensor]:
        yield first_chunk
        yield from video_chunks

    try:
        for chunk in all_chunks():
            chunk_u8 = chunk.to(device="cpu", dtype=torch.uint8).contiguous()
            proc.stdin.write(chunk_u8.numpy().tobytes())
            pbar.update(int(chunk_u8.shape[0]))
            pbar.set_postfix(shape=f"{int(chunk_u8.shape[0])}f", refresh=False)
    finally:
        pbar.close()
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait()
        if wav_path is not None:
            with contextlib.suppress(Exception):
                os.remove(wav_path)

    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def encode_video(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: Audio | None,
    output_path: str,
    video_chunks_number: int,
    expected_num_frames: int | None = None,
) -> None:
    if isinstance(video, torch.Tensor):
        video = iter([video])

    if av is None:
        _warn_no_av_once("video encoding/decoding")
        _encode_video_ffmpeg(video_chunks=video, fps=fps, audio=audio, output_path=output_path, expected_num_frames=expected_num_frames)
        logger.info("Video saved to %s", output_path)
        return

    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        audio_stream = _prepare_audio_stream(container, audio.sampling_rate)

    def all_chunks(first: torch.Tensor, rest: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
        yield first
        yield from rest

    pbar = tqdm(total=expected_num_frames, desc="Encode mp4", unit="frame", dynamic_ncols=True)
    try:
        for video_chunk in all_chunks(first_chunk, video):
            video_chunk_cpu = video_chunk.to("cpu").numpy()
            for frame_array in video_chunk_cpu:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
                pbar.update(1)
    finally:
        pbar.close()

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio)

    container.close()
    logger.info("Video saved to %s", output_path)


_INT_FORMAT_MAX: dict[str, float] = {
    "u8": 128.0,
    "u8p": 128.0,
    "s16": 32768.0,
    "s16p": 32768.0,
    "s32": 2147483648.0,
    "s32p": 2147483648.0,
}


def _audio_frame_to_float(frame: av.AudioFrame) -> np.ndarray:
    """Convert an audio frame to a float32 ndarray with values in [-1, 1] and shape (channels, samples)."""
    fmt = frame.format.name
    arr = frame.to_ndarray().astype(np.float32)
    if fmt in _INT_FORMAT_MAX:
        arr = arr / _INT_FORMAT_MAX[fmt]
    if not frame.format.is_planar:
        # Interleaved formats have shape (1, samples * channels) — reshape to (channels, samples).
        channels = len(frame.layout.channels)
        arr = arr.reshape(-1, channels).T
    return arr


def get_videostream_metadata(path: str) -> tuple[float, int, int, int]:
    """Read video stream metadata: (fps, num_frames, width, height).
    If frame count is missing in the container, decodes the stream to count frames.
    """
    if av is None:
        raise RuntimeError("get_videostream_metadata requires PyAV ('av').")
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        fps = float(video_stream.average_rate)
        num_frames = video_stream.frames or 0
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        width = video_stream.codec_context.width
        height = video_stream.codec_context.height
        return fps, num_frames, width, height
    finally:
        container.close()


def decode_audio_from_file(
    path: str, device: torch.device, start_time: float = 0.0, max_duration: float | None = None
) -> Audio | None:
    """Decodes audio from a file, optionally seeking to a start time and limiting duration.
    Args:
        path: Path to the audio/video file containing an audio stream.
        device: Device to place the resulting tensor on.
        start_time: Start time in seconds to begin reading audio from.
        max_duration: Maximum audio duration in seconds. If None, reads to end of stream.
    Returns:
        An Audio object with waveform of shape (1, channels, samples), or None if no audio stream.
    """
    if av is None:
        _warn_no_av_once("audio decoding")

        # 1) WAV via stdlib (fast, no external deps)
        try:
            import wave

            with contextlib.closing(wave.open(path, "rb")) as wf:
                sample_rate = int(wf.getframerate())
                channels = int(wf.getnchannels())
                sampwidth = int(wf.getsampwidth())
                n_frames = int(wf.getnframes())

                start_frame = max(0, int(round(start_time * sample_rate)))
                start_frame = min(start_frame, n_frames)
                wf.setpos(start_frame)

                if max_duration is None:
                    frames_to_read = n_frames - start_frame
                else:
                    frames_to_read = min(n_frames - start_frame, int(round(max_duration * sample_rate)))

                raw = wf.readframes(frames_to_read)

            if not raw:
                return None

            if sampwidth == 1:
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif sampwidth == 2:
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 3:
                b = np.frombuffer(raw, dtype=np.uint8)
                b = b.reshape(-1, 3)
                vals = b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16)
                # sign extend 24-bit
                vals = vals - ((vals & 0x800000) << 1)
                arr = vals.astype(np.float32) / 8388608.0
            elif sampwidth == 4:
                arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

            audio_np = arr.reshape(-1, channels).T
            waveform = torch.from_numpy(audio_np).to(device).unsqueeze(0)
            return Audio(waveform=waveform, sampling_rate=sample_rate)
        except Exception:
            pass

        # 2) Fallback: ffmpeg decode to s16le stereo @ 16kHz
        _require_ffmpeg()
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if start_time > 0:
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

    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        return None

    sample_rate = audio_stream.rate
    start_pts = int(start_time / audio_stream.time_base)
    end_time = start_time + max_duration if max_duration else audio_stream.duration * audio_stream.time_base
    container.seek(start_pts, stream=audio_stream)

    samples = []
    first_frame_time = None
    for frame in container.decode(audio=0):
        if frame.pts is None:
            continue
        frame_time = float(frame.pts * audio_stream.time_base)
        frame_end = frame_time + frame.samples / frame.sample_rate
        if frame_end < start_time:
            continue
        if frame_time > end_time:
            break
        if first_frame_time is None:
            first_frame_time = frame_time
        samples.append(_audio_frame_to_float(frame))

    container.close()

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Trim samples that fall outside the requested [start_time, start_time + max_duration] window.
    # Audio codecs decode in fixed-size frames whose boundaries may not align with the requested
    # time range, so the first frame can start before start_time and the last frame can end after
    # start_time + max_duration.
    skip_samples = round((start_time - first_frame_time) * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    if max_duration is not None:
        max_samples = round(max_duration * sample_rate)
        audio = audio[..., :max_samples]

    waveform = torch.from_numpy(audio).to(device).unsqueeze(0)

    return Audio(waveform=waveform, sampling_rate=sample_rate)


def decode_video_from_file(path: str, frame_cap: int, device: DeviceLikeType) -> Generator[torch.Tensor]:
    if av is None:
        raise RuntimeError("decode_video_from_file requires PyAV ('av').")
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    if av is None:
        raise RuntimeError("encode_single_frame requires PyAV ('av').")
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    if av is None:
        raise RuntimeError("decode_single_frame requires PyAV ('av').")
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image
    if av is None:
        # In offline Kaggle environments, PyAV may not be installed. The CRF
        # preprocessing is only meant to mimic video compression artifacts; the
        # pipelines still work without it.
        _warn_no_av_once("image CRF preprocessing")
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
