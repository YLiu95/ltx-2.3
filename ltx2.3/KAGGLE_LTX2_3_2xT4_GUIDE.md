# How to run LTX-2.3 on Kaggle (2× T4 GPUs)

This guide is based on the official Lightricks code repo (`Lightricks/LTX-2`) and the LTX-2.3 model files on Hugging Face.

## What you are trying to do

You want to generate an MP4 video using LTX-2.3, using **both** Kaggle T4 GPUs.

LTX-2.3 can generate:
- **Text → video (+ audio)**
- **Image + text → video (+ audio)**
- **Audio (+ optional image + optional text) → video** (the output MP4 keeps your input audio)

## Important notes (simple and honest)

- LTX-2.3 is a **very large** model. Two T4 GPUs have 16 GB memory each.
- The official scripts run on **one GPU per process**.
- To use **both GPUs**, the simplest approach is to run **two jobs at the same time** (one job on GPU 0 and one job on GPU 1). This gives the best “both GPUs busy” result without rewriting the model.
- The repo has an **8-bit (FP8) weight storage** option called `fp8-cast`. This is the closest “8-bit mode” provided by the official pipelines.

If you still get “out of memory” errors even with smaller settings, you may need:
- smaller width/height
- fewer frames
- fewer steps
- or a bigger GPU than T4

## Files you need (download once)

From `Lightricks/LTX-2.3` on Hugging Face, download these files:
- `ltx-2.3-22b-dev.safetensors` **or** `ltx-2.3-22b-distilled.safetensors`
- `ltx-2.3-22b-distilled-lora-384.safetensors` (used by the two‑stage pipelines)
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (or `x2-1.0`)

You also need the **Gemma text encoder folder** (a local folder that contains at least `tokenizer.model`, `preprocessor_config.json`, and one or more `model*.safetensors`).

## Suggested folder layout on Kaggle

Put everything under one folder so paths are easy:

```
/kaggle/working/
  LTX-2/                      # code
  models/
    ltx2_3/
      ltx-2.3-22b-dev.safetensors
      ltx-2.3-22b-distilled-lora-384.safetensors
      ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    gemma/
      ... (Gemma files here)
```

## Install the code (Kaggle cells)

1) Enable **Internet** and set Accelerator to **GPU (2× T4)**.

2) Clone the repo:

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

3) Install Python packages.

The repo is designed around `uv`, but on Kaggle you can usually do:

```bash
pip install -U pip
pip install -e packages/ltx-core -e packages/ltx-pipelines
```

If you see missing packages, install them as pip suggests.

## Pick safe “first run” settings (works better on T4)

Start small and scale up only if it fits:
- width/height: **768×512** (final) or smaller
- frames: **97** (that is 4 seconds at 24 fps)
- steps: **20** to start

Rules from the official code:
- For **two-stage** pipelines, width and height must be **multiples of 64**.
- Frame count must be **(8 × K) + 1** (examples: 49, 97, 121, 193).

## 8-bit mode (FP8) and memory setting

In the official pipelines, “8-bit mode” is:
- `--quantization fp8-cast`

Also set this memory option:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If `fp8-cast` crashes on your setup, remove it and try again.

## 1) Text → video (+ audio)

Use the two-stage text/image pipeline without images:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --prompt "A dog running on the beach, sunny day, wide shot" \
  --output-path /kaggle/working/out_t2v.mp4 \
  --width 768 --height 512 \
  --num-frames 97 --frame-rate 24 \
  --num-inference-steps 20 \
  --quantization fp8-cast
```

## 2) Image + text → video (+ audio)

Add one or more `--image` inputs.

Format:
- `--image PATH FRAME_INDEX STRENGTH [CRF]`

Notes:
- `FRAME_INDEX 0` means “use this as the first frame”.
- `STRENGTH` around `0.6–1.0` is a normal starting range.
- `CRF 0` keeps the image sharp (no extra compression).

Example:

```bash
python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --prompt "A cinematic shot of a mountain lake at sunrise" \
  --image /kaggle/input/mydata/start.png 0 0.85 0 \
  --output-path /kaggle/working/out_i2v.mp4 \
  --width 768 --height 512 \
  --num-frames 97 --frame-rate 24 \
  --num-inference-steps 20 \
  --quantization fp8-cast
```

## 3) Image + audio → video (keeps your input audio)

Use the audio-to-video pipeline:

```bash
python -m ltx_pipelines.a2vid_two_stage \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --audio-path /kaggle/input/mydata/song.wav \
  --image /kaggle/input/mydata/start.png 0 0.85 0 \
  --prompt "" \
  --output-path /kaggle/working/out_i_plus_a.mp4 \
  --width 768 --height 512 \
  --num-frames 97 --frame-rate 24 \
  --num-inference-steps 20 \
  --quantization fp8-cast
```

Tip: even if you want “no text”, the command needs `--prompt`. Use an empty string or a short neutral sentence.

## 4) Image + audio + text → video

Same as the previous command, but add a real prompt:

```bash
python -m ltx_pipelines.a2vid_two_stage \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --audio-path /kaggle/input/mydata/speech.wav \
  --image /kaggle/input/mydata/start.png 0 0.9 0 \
  --prompt "A person speaking on stage, camera slowly moves closer" \
  --output-path /kaggle/working/out_i_plus_a_plus_t.mp4 \
  --width 768 --height 512 \
  --num-frames 97 --frame-rate 24 \
  --num-inference-steps 20 \
  --quantization fp8-cast
```

## Settings you can change (plain meaning)

- `--seed`: changes randomness. Same seed = more repeatable results.
- `--width`, `--height`: output size. Bigger uses more GPU memory.
- `--num-frames`: video length. More frames uses more GPU memory.
- `--frame-rate`: frames per second. Together with `num-frames` this controls seconds.
- `--num-inference-steps`: more steps = better quality but slower.
- `--enhance-prompt`: lets the model rewrite your prompt (sometimes helps).

There are also “prompt strength” settings in the CLI (video/audio guidance scales). If you keep them at defaults, that’s a good start.

## Load → run → offload (to keep GPU memory under control)

The official pipelines already do this pattern:
- load the text encoder → encode prompt → free it
- load the image encoder → encode your image(s) → free it
- load the main model → run stage 1 → free it
- load the upscaler + main model → run stage 2 → free it
- load decoders → write MP4

This reduces peak memory and helps avoid crashes.

## MUST use both GPUs (2× T4)

The official scripts use one GPU per process. To use both GPUs, run **two commands at the same time**, one on each GPU.

Example (two text-to-video jobs at once):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --prompt "A fast drone shot above a forest" \
  --output-path /kaggle/working/out_gpu0.mp4 \
  --width 768 --height 512 --num-frames 97 --frame-rate 24 --num-inference-steps 20 \
  --quantization fp8-cast &

CUDA_VISIBLE_DEVICES=1 python -m ltx_pipelines.ti2vid_two_stages \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --prompt "Close-up of a flower opening in time-lapse" \
  --output-path /kaggle/working/out_gpu1.mp4 \
  --width 768 --height 512 --num-frames 97 --frame-rate 24 --num-inference-steps 20 \
  --quantization fp8-cast &

wait
```

This is the cleanest way to keep both GPUs busy on Kaggle.

## Troubleshooting

- **Out of memory**: reduce width/height, reduce frames, reduce steps. Close other notebooks.
- **`fp8-cast` error**: remove `--quantization fp8-cast` and retry.
- **Gemma not found**: make sure your `--gemma-root` folder contains `tokenizer.model`, `preprocessor_config.json`, and `model*.safetensors`.
- **Wrong frame count**: use 49, 97, 121, 193, ... (must be 8×K + 1).
