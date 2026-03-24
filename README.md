# LTX-2.3 (Kaggle S2V)

This is a Kaggle-focused fork of `Lightricks/LTX-2` that adds:

- `kaggle_s2v/run_s2v.py`: a Kaggle runner for **speech/audio-conditioned image-to-video** (S2V/A2V).
- `packages/ltx-pipelines/src/ltx_pipelines/distilled_a2v.py`: `DistilledA2VPipeline` (uses `ltx-2.3-22b-distilled.safetensors`).
- Rich tqdm progress bars during denoising (speed/ETA + sigma + VRAM telemetry).

Key Kaggle constraints this repo follows:

- **No CPU/disk offloading** during inference (models are built on GPU; VRAM is managed via load/run/unload + `cleanup_memory()`).
- **No model downloads to `/kaggle/working`**: all HF downloads/caches go to `/kaggle/temp`, and only output videos/configs are written to `/kaggle/working`.

## Kaggle Quickstart (Speech-to-Video)

Copy/paste `kaggle_s2v/kaggle_cell_example.md` into a **single Kaggle code cell**, or use one of the minimal versions below.

### Option A (recommended): RTX PRO 6000 runtime with **no internet**

1) In a separate **internet-enabled** Kaggle notebook, build an offline assets dataset by running the single cell in:

- `kaggle_s2v/build_offline_assets_dataset.md`

2) In the same (internet-enabled) notebook, build a tiny **code** dataset (so you can update code without re-uploading ~67GB of weights):

- `kaggle_s2v/build_offline_code_dataset.md`

3) In the **offline** notebook, attach:

- your offline assets dataset (weights)
- your offline code dataset (repo code only)
- the input dataset with your audio/image/prompt files

Then run:

```python
import os
import glob

# Keep caches out of /kaggle/working (only outputs should go there)
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hard-offline (prevents any network calls)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Note: some Kaggle images don't ship with PyAV (`import av`). This repo falls
# back to `ffmpeg` for audio/video I/O when PyAV is missing.

def _find_dataset_root(slug: str) -> str:
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{slug}",
        f"/kaggle/input/datasets/yliu95/{slug}",  # common case for this project
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    matches = [p for p in glob.glob(f"/kaggle/input/**/{slug}", recursive=True) if os.path.isdir(p)]
    if matches:
        return sorted(matches, key=len)[0]
    raise FileNotFoundError(f"Could not find dataset folder for slug: {slug}")

# Weights dataset (built by `kaggle_s2v/build_offline_assets_dataset.md`)
ASSETS_ROOT = _find_dataset_root("ltx23-offline-assets")

# Code dataset (built by `kaggle_s2v/build_offline_code_dataset.md`)
CODE_ROOT = _find_dataset_root("ltx23-offline-code")

# Prefer code dataset (so code updates don't touch the weights dataset).
repo_candidates = [
    f"{CODE_ROOT}/repo_ltx-2.3/ltx-2.3",
    f"{CODE_ROOT}/repo_ltx-2.3",
    f"{CODE_ROOT}/repo/ltx-2.3",
    # Fallback: some builds bundled code inside the weights dataset too.
    f"{ASSETS_ROOT}/repo_ltx-2.3/ltx-2.3",
    f"{ASSETS_ROOT}/repo_ltx-2.3",
    f"{ASSETS_ROOT}/repo/ltx-2.3",
]
REPO_DIR = next((d for d in repo_candidates if os.path.exists(os.path.join(d, "kaggle_s2v", "run_s2v.py"))), None)
if REPO_DIR is None:
    raise FileNotFoundError("Could not find `kaggle_s2v/run_s2v.py` under:\n" + "\n".join(repo_candidates))

# Sanity check: if you see `ModuleNotFoundError: No module named 'av'`, you're running an old snapshot.
with open(os.path.join(REPO_DIR, "kaggle_s2v", "run_s2v.py"), "r", encoding="utf-8") as f:
    if "ffprobe" not in f.read():
        raise RuntimeError(
            "This repo snapshot is too old (it still hard-requires PyAV). "
            "Update the `ltx23-offline-code` dataset to the latest version and re-attach it."
        )

AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

!python {REPO_DIR}/kaggle_s2v/run_s2v.py \
  --assets_mode offline \
  --assets_root "{ASSETS_ROOT}" \
  --pipeline distilled-a2v \
  --audio_path "{AUDIO_PATH}" \
  --image_path "{IMAGE_PATH}" \
  --prompt_file "{PROMPT_FILE}" \
  --audio_text_file "{AUDIO_TEXT_FILE}" \
  --quantization fp8-cast \
  --cleanup aggressive \
  --vae_tiling default \
  --progress --progress_vram --progress_vram_every 1
```

If your `ltx23-offline-assets` dataset already includes the repo under
`repo_ltx-2.3/ltx-2.3/` (like in your directory listing), you can also use the
ready-to-copy single-cell script:

- `kaggle_s2v/kaggle_offline_single_cell.py`

It runs directly from `/kaggle/input/.../repo_ltx-2.3/ltx-2.3`, creates a
compat `assets_root` under `/kaggle/temp`, and patches common offline issues
like missing PyAV (`import av`).

> Important: Gemma is gated and may have redistribution restrictions. Uploading model weights to a public Kaggle dataset may violate terms. Prefer a **private** dataset.

### Option B: Internet-enabled notebook (downloads weights to `/kaggle/temp`)

> Note: the Gemma repo used by LTX is gated; make sure your Hugging Face account has accepted the model terms.

```python
import os
from kaggle_secrets import UserSecretsClient

# Keep caches out of /kaggle/working
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hugging Face token (Kaggle Secret)
user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")

# Clone code to /kaggle/temp (NOT /kaggle/working)
REPO_DIR = "/kaggle/temp/ltx-2.3"
if not os.path.exists(REPO_DIR):
    !git clone --depth 1 https://github.com/YLiu95/ltx-2.3 {REPO_DIR}

# Inputs
AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

!python {REPO_DIR}/kaggle_s2v/run_s2v.py \
  --assets_mode download \
  --pipeline distilled-a2v \
  --audio_path "{AUDIO_PATH}" \
  --image_path "{IMAGE_PATH}" \
  --prompt_file "{PROMPT_FILE}" \
  --audio_text_file "{AUDIO_TEXT_FILE}" \
  --quantization fp8-cast \
  --cleanup aggressive \
  --vae_tiling default \
  --progress --progress_vram --progress_vram_every 1
```

### Output layout (Kaggle)

- Outputs: `/kaggle/working/s2v/outputs/<run_name>/video.mp4`
- Run config (for reproducibility): `/kaggle/working/s2v/outputs/<run_name>/run_config.json`
- HF caches/models: `/kaggle/temp/hf/...`
- Intermediate padded image: `/kaggle/temp/s2v/padded/...`

## Runner settings (explained)

All settings are passed to `kaggle_s2v/run_s2v.py`.

### Assets (download vs offline)

- `--assets_mode`:
  - `download` (default): download from Hugging Face into `/kaggle/temp` (requires internet + `HF_TOKEN`).
  - `offline`: load from a Kaggle dataset mounted under `/kaggle/input` (no internet).
- `--assets_root`: required when `--assets_mode offline`. Points at the dataset root containing the required LTX + Gemma files.
  - Supported layouts: **flat** (all files at root) or **subdirs** (`ltx/`, `gemma/`).

### Core inputs

- `--audio_path`: Audio file to condition generation (wav/mp3/video-with-audio).
- `--image_path`: Image used for conditioning (first frame).
- `--prompt` / `--prompt_file`: Main prompt (required).
- `--audio_text_file`: Optional transcript. The runner appends it to the prompt to help semantic alignment.

### Resolution & frames

- `--width`, `--height`: Output resolution. **Must be multiples of 64** (two-stage requirement).
  - If omitted, the runner reads the input image size and **pads to the next multiple of 64** (e.g. 512x763 → 512x768).
- `--fps`: Output fps (default 24).
- `--num_frames`: If omitted, the runner computes `round(audio_duration * fps)` and snaps to **8k+1** (model constraint).
- `--audio_max_duration`: If omitted, defaults to `num_frames / fps` so the muxed audio doesn’t extend past the video.

### VRAM / speed knobs (no offloading)

- `--quantization`:
  - `fp8-cast` (default): lower VRAM by storing transformer weights in FP8 and upcasting during matmul.
  - `fp8-scaled-mm`: uses TensorRT-LLM FP8 scaled matmul ops (only if available in the environment).
  - `none`: full-precision weights.
- `--cleanup`:
  - `aggressive` (default): unload/reload big modules between stages (lowest peak VRAM, slowest).
  - `balanced`: fewer reloads (middle ground).
  - `none`: keep everything loaded (fastest, highest peak VRAM).
- `--vae_tiling`:
  - `default` (default): decode video in tiles (safer for long/high-res videos).
  - `none`: decode in one go (faster if you have plenty of VRAM).

### Progress / monitoring (tqdm)

During denoising you get tqdm bars showing:

- speed + ETA (tqdm default)
- postfix fields: `sigma`, `vram`, `resv`, `peak`

Controls:

- `--progress` / `--no-progress`: enable/disable progress bars
- `--progress_vram` / `--no-progress_vram`: enable/disable VRAM postfix
- `--progress_vram_every N`: update VRAM postfix every N denoising steps (1 = most detailed)

### Advanced pipeline (optional): `a2vid-two-stage`

Set `--pipeline a2vid-two-stage` to use the upstream `A2VidPipelineTwoStage` (dev checkpoint + distilled LoRA).
This exposes more “guidance” knobs, but downloads more weights.

Common knobs:

- `--num_inference_steps`: diffusion steps for Stage 1 (higher = slower, often better adherence)
- `--negative_prompt` / `--negative_prompt_file`: what to avoid
- `--video_cfg_scale`: classifier-free guidance strength
- `--video_stg_scale`, `--video_stg_blocks`: spatio-temporal guidance parameters
- `--a2v_guidance_scale`: audio-to-video guidance strength (can help lip-sync)

[![Website](https://img.shields.io/badge/Website-LTX-181717?logo=google-chrome)](https://ltx.io)
[![Model](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/Lightricks/LTX-2.3)
[![Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?logo=vercel)](https://app.ltx.studio/ltx-2-playground/i2v)
[![Paper](https://img.shields.io/badge/Paper-PDF-EC1C24?logo=adobeacrobatreader&logoColor=white)](https://arxiv.org/abs/2601.03233)
[![Discord](https://img.shields.io/badge/Join-Discord-5865F2?logo=discord)](https://discord.gg/ltxplatform)

**LTX-2** is the first DiT-based audio-video foundation model that contains all core capabilities of modern video generation in one model: synchronized audio and video, high fidelity, multiple performance modes, production-ready outputs, API access, and open access.

<div align="center">
  <video src="https://github.com/user-attachments/assets/4414adc0-086c-43de-b367-9362eeb20228" width="70%" poster=""> </video>
</div>

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2

# Set up the environment
uv sync --frozen
source .venv/bin/activate
```

### Required Models

Download the following models from the [LTX-2.3 HuggingFace repository](https://huggingface.co/Lightricks/LTX-2.3):

**LTX-2.3 Model Checkpoint** (choose and download one of the following)
  * [`ltx-2.3-22b-dev.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-dev.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-dev.safetensors)
  * [`ltx-2.3-22b-distilled.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled.safetensors)

**Spatial Upscaler** - Required for current two-stage pipeline implementations in this repository
  * [`ltx-2.3-spatial-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors)
  * [`ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors)

**Temporal Upscaler** - Supported by the model and will be required for future pipeline implementations
  * [`ltx-2.3-temporal-upscaler-x2-1.0.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-temporal-upscaler-x2-1.0.safetensors)

**Distilled LoRA** - Required for current two-stage pipeline implementations in this repository (except DistilledPipeline and ICLoraPipeline)
  * [`ltx-2.3-22b-distilled-lora-384.safetensors`](https://huggingface.co/Lightricks/LTX-2.3/blob/main/ltx-2.3-22b-distilled-lora-384.safetensors) - [Download](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors)

**Gemma Text Encoder** (download all assets from the repository)
  * [`Gemma 3`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main)

**LoRAs**
  * [`LTX-2.3-22b-IC-LoRA-Union-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors)
  * [`LTX-2.3-22b-IC-LoRA-Motion-Track-Control`](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control) - [Download](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/resolve/main/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors)
  * [`LTX-2-19b-IC-LoRA-Detailer`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors)
  * [`LTX-2-19b-IC-LoRA-Pose-Control`](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-In`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Left`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Out`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Dolly-Right`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Down`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Jib-Up`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors)
  * [`LTX-2-19b-LoRA-Camera-Control-Static`](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static) - [Download](https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors)

### Available Pipelines

* **[TI2VidTwoStagesPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py)** - Production-quality text/image-to-video with 2x upsampling (recommended)
* **[TI2VidTwoStagesHQPipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages_hq.py)** - Same two-stage flow as above but uses the res_2s second-order sampler (fewer steps, better quality)
* **[TI2VidOneStagePipeline](packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py)** - Single-stage generation for quick prototyping
* **[DistilledPipeline](packages/ltx-pipelines/src/ltx_pipelines/distilled.py)** - Fastest inference with 8 predefined sigmas
* **[ICLoraPipeline](packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py)** - Video-to-video and image-to-video transformations (uses distilled model.)
* **[KeyframeInterpolationPipeline](packages/ltx-pipelines/src/ltx_pipelines/keyframe_interpolation.py)** - Interpolate between keyframe images
* **[A2VidPipelineTwoStage](packages/ltx-pipelines/src/ltx_pipelines/a2vid_two_stage.py)** - Audio-to-video generation conditioned on an input audio file
* **[RetakePipeline](packages/ltx-pipelines/src/ltx_pipelines/retake.py)** - Regenerate a specific time region of an existing video

### ⚡ Optimization Tips

* **Use DistilledPipeline** - Fastest inference with only 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)
* **Enable FP8 quantization** - Enables lower memory footprint: `--quantization fp8-cast` (CLI) or `quantization=QuantizationPolicy.fp8_cast()` (Python). For Hopper GPUs with TensorRT-LLM, use `--quantization fp8-scaled-mm` for FP8 scaled matrix multiplication.
* **Install attention optimizations** - Use xFormers (`uv sync --extra xformers`) or [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) for Hopper GPUs
* **Use gradient estimation** - Reduce inference steps from 40 to 20-30 while maintaining quality (see [pipeline documentation](packages/ltx-pipelines/README.md#denoising-loop-optimization))
* **Skip memory cleanup** - If you have sufficient VRAM, disable automatic memory cleanup between stages for faster processing
* **Choose single-stage pipeline** - Use `TI2VidOneStagePipeline` for faster generation when high resolution isn't required

## ✍️ Prompting for LTX-2

When writing prompts, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. Think like a cinematographer describing a shot list. Keep within 200 words. For best results, build your prompts using this structure:

- Start with main action in a single sentence
- Add specific details about movements and gestures
- Describe character/object appearances precisely
- Include background and environment details
- Specify camera angles and movements
- Describe lighting and colors
- Note any changes or sudden events

For additional guidance on writing a prompt please refer to <https://ltx.video/blog/how-to-prompt-for-ltx-2>

### Automatic Prompt Enhancement

LTX-2 pipelines support automatic prompt enhancement via an `enhance_prompt` parameter.

## 🔌 ComfyUI Integration

To use our model with ComfyUI, please follow the instructions at <https://github.com/Lightricks/ComfyUI-LTXVideo/>.

## 📦 Packages

This repository is organized as a monorepo with three main packages:

* **[ltx-core](packages/ltx-core/)** - Core model implementation, inference stack, and utilities
* **[ltx-pipelines](packages/ltx-pipelines/)** - High-level pipeline implementations for text-to-video, image-to-video, and other generation modes
* **[ltx-trainer](packages/ltx-trainer/)** - Training and fine-tuning tools for LoRA, full fine-tuning, and IC-LoRA

Each package has its own README and documentation. See the [Documentation](#-documentation) section below.

## 📚 Documentation

Each package includes comprehensive documentation:

* **[LTX-Core README](packages/ltx-core/README.md)** - Core model implementation, inference stack, and utilities
* **[LTX-Pipelines README](packages/ltx-pipelines/README.md)** - High-level pipeline implementations and usage guides
* **[LTX-Trainer README](packages/ltx-trainer/README.md)** - Training and fine-tuning documentation with detailed guides
