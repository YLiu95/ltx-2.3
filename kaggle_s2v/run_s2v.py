from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm.auto import tqdm

from hf_assets import (
    DistilledA2VAssets,
    TwoStageA2VAssets,
    configure_kaggle_cache_dirs,
    ensure_distilled_a2v_assets,
    ensure_two_stage_a2v_assets,
)
from image_prep import ceil_to_multiple, pad_image_to_size


def _read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def _vram_gb() -> tuple[float, float] | None:
    if not torch.cuda.is_available():
        return None
    idx = torch.cuda.current_device()
    return (torch.cuda.memory_allocated(idx) / (1024**3), torch.cuda.max_memory_allocated(idx) / (1024**3))


def _vram_str() -> str:
    v = _vram_gb()
    if v is None:
        return "VRAM: n/a"
    return f"VRAM: {v[0]:.1f}G (peak {v[1]:.1f}G)"


class _StageTimer:
    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        tqdm.write(f"[S2V] {self.name}… ({_vram_str()})")
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
        dt = time.perf_counter() - self.start
        suffix = "OK" if exc is None else f"FAILED ({exc_type.__name__})"
        tqdm.write(f"[S2V] {self.name}: {dt:.1f}s — {suffix} ({_vram_str()})")
        return False


def _audio_duration_seconds(audio_path: str) -> float:
    import av

    container = av.open(audio_path)
    try:
        stream = next(s for s in container.streams if s.type == "audio")
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        total_samples = 0
        for frame in container.decode(stream):
            total_samples += frame.samples
        return float(total_samples) / float(stream.rate)
    finally:
        container.close()


def _snap_num_frames(num_frames: int, *, time_scale: int = 8) -> int:
    if num_frames <= 1:
        return 1
    k = round((num_frames - 1) / time_scale)
    snapped = int(k) * time_scale + 1
    return max(snapped, 1)


def _quantization_policy(name: str):
    from ltx_core.quantization import QuantizationPolicy

    if name == "none":
        return None
    if name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if name == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"Unknown quantization policy: {name}")


def _default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/s2v/outputs")
    return Path("./outputs")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LTX-2.3 Speech-to-Video (image + audio) Kaggle runner")

    p.add_argument(
        "--pipeline",
        choices=("distilled-a2v", "a2vid-two-stage"),
        default="distilled-a2v",
        help="Which pipeline to run. 'distilled-a2v' is recommended for FP8 distilled inference.",
    )

    p.add_argument("--audio_path", type=str, required=True)
    p.add_argument("--image_path", type=str, required=True)

    prompt_group = p.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Prompt string (quoted).")
    prompt_group.add_argument("--prompt_file", type=str, help="Path to a .txt file containing the prompt.")

    p.add_argument(
        "--audio_text_file",
        type=str,
        default=None,
        help="Optional .txt transcript of the audio. If provided, it is appended to the prompt.",
    )

    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--num_frames", type=int, default=None, help="Defaults to audio_duration * fps, snapped to 8k+1.")

    # If omitted, width/height are derived from the (padded) input image size.
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument(
        "--image_fit",
        choices=("pad64",),
        default="pad64",
        help="How to adapt the input image to model constraints. 'pad64' pads to multiples of 64 (no crop).",
    )

    p.add_argument("--image_strength", type=float, default=1.0, help="Conditioning strength (0..1).")
    p.add_argument("--image_frame_idx", type=int, default=0, help="Which frame index the image conditions.")

    p.add_argument("--audio_start_time", type=float, default=0.0)
    p.add_argument("--audio_max_duration", type=float, default=None, help="Defaults to num_frames/fps.")

    p.add_argument(
        "--quantization",
        choices=("fp8-cast", "fp8-scaled-mm", "none"),
        default="fp8-cast",
        help="FP8 mode. 'fp8-cast' is broadly supported. 'fp8-scaled-mm' requires TensorRT-LLM ops.",
    )
    p.add_argument(
        "--cleanup",
        choices=("aggressive", "balanced", "none"),
        default="aggressive",
        help="VRAM strategy: aggressive unload/reload between stages (lowest peak VRAM, slowest).",
    )

    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress_vram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress_vram_every", type=int, default=1)

    p.add_argument(
        "--vae_tiling",
        choices=("default", "none"),
        default="default",
        help="VAE decode tiling. 'default' is safer for long/high-res videos; 'none' can be faster if VRAM allows.",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_default_output_dir()),
        help="Output root directory. On Kaggle, defaults to /kaggle/working/s2v/outputs.",
    )

    p.add_argument("--run_name", type=str, default=None, help="Folder name under output_dir (defaults to timestamp).")
    p.add_argument("--dry_run", action="store_true", help="Validate inputs and print resolved config without running.")

    # Advanced (only used by a2vid-two-stage)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--negative_prompt_file", type=str, default=None)
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Only for a2vid-two-stage. Higher is slower but can improve adherence/quality.",
    )
    p.add_argument("--video_cfg_scale", type=float, default=None)
    p.add_argument("--video_stg_scale", type=float, default=None)
    p.add_argument("--video_rescale_scale", type=float, default=None)
    p.add_argument("--a2v_guidance_scale", type=float, default=None)
    p.add_argument("--video_skip_step", type=int, default=None)
    p.add_argument(
        "--video_stg_blocks",
        type=int,
        nargs="*",
        default=None,
        help="Only for a2vid-two-stage. Which transformer blocks to perturb for STG.",
    )

    return p


def main() -> None:  # noqa: C901
    logging_level = os.environ.get("LTX_LOG_LEVEL", "INFO").upper()
    import logging

    logging.getLogger().setLevel(logging_level)

    args = build_arg_parser().parse_args()

    # Ensure no HF downloads land in /kaggle/working.
    configure_kaggle_cache_dirs("/kaggle/temp")

    hf_token = os.environ.get("HF_TOKEN", None)

    prompt = args.prompt if args.prompt is not None else _read_text_file(args.prompt_file)
    if args.audio_text_file:
        audio_text = _read_text_file(args.audio_text_file)
        if audio_text:
            prompt = (
                f"{prompt}\n\n"
                f"Spoken content in the audio (for timing/lip-sync alignment; do not literally render as subtitles):\n"
                f"{audio_text}"
            )

    with _StageTimer("Inspect inputs"):
        audio_dur = _audio_duration_seconds(args.audio_path)
        if args.num_frames is None:
            raw_frames = int(round(audio_dur * float(args.fps)))
            args.num_frames = _snap_num_frames(raw_frames, time_scale=8)
        if args.audio_max_duration is None:
            args.audio_max_duration = float(args.num_frames) / float(args.fps)

    with _StageTimer("Prepare image (pad to multiples-of-64)"):
        from PIL import Image

        im = Image.open(args.image_path)
        src_w, src_h = im.size
        target_w = int(args.width) if args.width is not None else ceil_to_multiple(src_w, 64)
        target_h = int(args.height) if args.height is not None else ceil_to_multiple(src_h, 64)
        if target_w % 64 != 0 or target_h % 64 != 0:
            raise ValueError(f"--width/--height must be multiples of 64 for two-stage pipelines. Got {target_w}x{target_h}.")
        padded_image_path = pad_image_to_size(
            args.image_path,
            target_width=target_w,
            target_height=target_h,
            out_dir="/kaggle/temp/s2v/padded",
        )

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    output_video_path = out_root / "video.mp4"
    output_config_path = out_root / "run_config.json"

    resolved = {
        "pipeline": args.pipeline,
        "audio_path": args.audio_path,
        "image_path": args.image_path,
        "padded_image_path": padded_image_path,
        "prompt": prompt,
        "seed": args.seed,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "width": target_w,
        "height": target_h,
        "audio_start_time": args.audio_start_time,
        "audio_max_duration": args.audio_max_duration,
        "quantization": args.quantization,
        "cleanup": args.cleanup,
        "progress": args.progress,
        "progress_vram": args.progress_vram,
        "progress_vram_every": args.progress_vram_every,
        "vae_tiling": args.vae_tiling,
        "output_video_path": str(output_video_path),
    }
    output_config_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    if args.dry_run:
        tqdm.write(f"[S2V] dry_run: wrote {output_config_path}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LTX inference. No GPU detected.")

    with _StageTimer("Download model assets to /kaggle/temp (HF cache)"):
        if args.pipeline == "distilled-a2v":
            assets: DistilledA2VAssets = ensure_distilled_a2v_assets(hf_token, cache_root="/kaggle/temp")
        else:
            assets2: TwoStageA2VAssets = ensure_two_stage_a2v_assets(hf_token, cache_root="/kaggle/temp")

    with _StageTimer("Build pipeline (load on GPU)"):
        quant = _quantization_policy(args.quantization)

        if args.pipeline == "distilled-a2v":
            from ltx_pipelines import DistilledA2VPipeline

            pipeline = DistilledA2VPipeline(
                distilled_checkpoint_path=assets.distilled_checkpoint_path,
                gemma_root=assets.gemma_root,
                spatial_upsampler_path=assets.spatial_upsampler_path,
                quantization=quant,
                cleanup_mode=args.cleanup,
                device=torch.device("cuda"),
            )
        else:
            from ltx_core.components.guiders import MultiModalGuiderParams
            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            from ltx_pipelines import A2VidPipelineTwoStage
            from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params

            params = detect_params(assets2.checkpoint_path)
            # Override defaults only if explicitly passed.
            guider = params.video_guider_params
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=args.video_cfg_scale if args.video_cfg_scale is not None else guider.cfg_scale,
                stg_scale=args.video_stg_scale if args.video_stg_scale is not None else guider.stg_scale,
                rescale_scale=args.video_rescale_scale if args.video_rescale_scale is not None else guider.rescale_scale,
                modality_scale=args.a2v_guidance_scale if args.a2v_guidance_scale is not None else guider.modality_scale,
                skip_step=args.video_skip_step if args.video_skip_step is not None else guider.skip_step,
                stg_blocks=args.video_stg_blocks if args.video_stg_blocks is not None else guider.stg_blocks,
            )
            steps = args.num_inference_steps if args.num_inference_steps is not None else params.num_inference_steps

            neg = DEFAULT_NEGATIVE_PROMPT
            if args.negative_prompt_file:
                neg = _read_text_file(args.negative_prompt_file)
            if args.negative_prompt:
                neg = args.negative_prompt

            distilled_lora = [LoraPathStrengthAndSDOps(assets2.distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
            pipeline = A2VidPipelineTwoStage(
                checkpoint_path=assets2.checkpoint_path,
                distilled_lora=distilled_lora,
                spatial_upsampler_path=assets2.spatial_upsampler_path,
                gemma_root=assets2.gemma_root,
                loras=(),
                device=torch.device("cuda"),
                quantization=quant,
            )

    # Decode tiling
    if args.vae_tiling == "default":
        from ltx_core.model.video_vae import TilingConfig

        tiling_config = TilingConfig.default()
    else:
        tiling_config = None

    images = [(padded_image_path, int(args.image_frame_idx), float(args.image_strength))]

    with _StageTimer("Run inference"):
        if args.pipeline == "distilled-a2v":
            video_iter, audio = pipeline(
                prompt=prompt,
                seed=int(args.seed),
                height=int(target_h),
                width=int(target_w),
                num_frames=int(args.num_frames),
                frame_rate=float(args.fps),
                images=images,
                audio_path=args.audio_path,
                audio_start_time=float(args.audio_start_time),
                audio_max_duration=float(args.audio_max_duration),
                tiling_config=tiling_config,
                progress=bool(args.progress),
                progress_vram=bool(args.progress_vram),
                progress_vram_every=int(args.progress_vram_every),
            )
        else:
            video_iter, audio = pipeline(
                prompt=prompt,
                negative_prompt=neg,
                seed=int(args.seed),
                height=int(target_h),
                width=int(target_w),
                num_frames=int(args.num_frames),
                frame_rate=float(args.fps),
                num_inference_steps=int(steps),
                video_guider_params=video_guider_params,
                images=images,
                tiling_config=tiling_config,
                audio_path=args.audio_path,
                audio_start_time=float(args.audio_start_time),
                audio_max_duration=float(args.audio_max_duration),
                progress=bool(args.progress),
                progress_vram=bool(args.progress_vram),
                progress_vram_every=int(args.progress_vram_every),
            )

    with _StageTimer("Encode mp4 to /kaggle/working"):
        from ltx_core.model.video_vae import get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        chunks = int(get_video_chunks_number(int(args.num_frames), tiling_config))
        encode_video(video=video_iter, fps=int(args.fps), audio=audio, output_path=str(output_video_path), video_chunks_number=chunks)

    tqdm.write(f"[S2V] Done: {output_video_path}")


if __name__ == "__main__":
    main()
