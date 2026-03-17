# How to run LTX-2.3 on Kaggle (2× T4 GPUs)

This guide is based on the official Lightricks code repo (`Lightricks/LTX-2`) and the LTX-2.3 model files on Hugging Face.

## What you are trying to do

You want to generate an **MP4 video** with LTX-2.3 on Kaggle, and you want to use **both** T4 GPUs.

LTX-2.3 can do:
- **Text → video (+ sound)**
- **Image + text → video (+ sound)**
- **Audio (+ optional image + optional text) → video** (the output keeps your input audio)

## Important notes (simple and honest)

- LTX-2.3 is **very large**.
- A Kaggle T4 GPU has **16 GB** of GPU memory.
- The official LTX commands run on **one GPU per run**.

So, to use **both GPUs**, the simplest and most reliable way is:
- Run **two separate jobs at the same time** (one job on GPU 0, one job on GPU 1).

This keeps both GPUs busy without needing to rewrite the model.

## “8-bit mode”

The official LTX commands include an option that stores the main model weights in an **8-bit format** to save GPU memory.

In the command line, that option is:
- `--quantization fp8-cast`

(Yes, the flag name says `quantization`. You can just think of it as “8-bit mode”.)

If `fp8-cast` crashes on your setup, remove it and try again.

## Files you need (download once)

From `Lightricks/LTX-2.3` on Hugging Face, download:

1) **One main model file** (pick one)
- `ltx-2.3-22b-dev.safetensors` (full quality, slower)
- `ltx-2.3-22b-distilled.safetensors` (faster)

2) **One stage‑2 helper file** (used by the two‑stage commands)
- `ltx-2.3-22b-distilled-lora-384.safetensors`

3) **One upscaler file** (used by the two‑stage commands)
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (or `x2-1.0`)

4) **Gemma text encoder folder**

This must be a local folder that contains at least:
- `tokenizer.model`
- `preprocessor_config.json`
- one or more `model*.safetensors`

## Suggested folder layout on Kaggle

Keep everything together so paths are easy:

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

1) Enable **Internet** and choose **GPU (2× T4)**.

2) Clone the official repo:

```bash
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

3) Install Python packages:

```bash
pip install -U pip
pip install -e packages/ltx-core -e packages/ltx-pipelines
```

If pip says something is missing, install what it asks for.

## Pick safe “first run” settings (works better on T4)

Start small and scale up only if it fits:
- width/height: **768×512** (final output)
- frames: **97** (about 4 seconds at 24 fps)
- steps: **20** to start

Rules from the official code:
- For these **two‑stage** commands, width and height must be **multiples of 64**.
- Frame count must be **(8 × K) + 1**.
  - Examples: 49, 97, 121, 193

## Helpful memory setting

Set this once before running commands:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1) Text → video (+ sound)

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

## 2) Image + text → video (+ sound)

Add one or more `--image` inputs.

Format:
- `--image PATH FRAME_INDEX STRENGTH [CRF]`

What those mean:
- `FRAME_INDEX 0` means “use this as the first frame”.
- `STRENGTH` controls how strongly the image is followed (try `0.6` to `1.0`).
- `CRF 0` keeps the image sharp.

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

```bash
python -m ltx_pipelines.a2vid_two_stage \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --audio-path /kaggle/input/mydata/song.wav \
  --image /kaggle/input/mydata/start.png 0 0.85 0 \
  --prompt "" \
  --output-path /kaggle/working/out_image_plus_audio.mp4 \
  --width 768 --height 512 \
  --num-frames 97 --frame-rate 24 \
  --num-inference-steps 20 \
  --quantization fp8-cast
```

Tip: even if you want “no text”, the command still needs `--prompt`. Use an empty string or a short neutral sentence.

## 4) Image + audio + text → video

Same as above, but add a real prompt:

```bash
python -m ltx_pipelines.a2vid_two_stage \
  --checkpoint-path /kaggle/working/models/ltx2_3/ltx-2.3-22b-dev.safetensors \
  --distilled-lora /kaggle/working/models/ltx2_3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
  --spatial-upsampler-path /kaggle/working/models/ltx2_3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /kaggle/working/models/gemma \
  --audio-path /kaggle/input/mydata/speech.wav \
  --image /kaggle/input/mydata/start.png 0 0.9 0 \
  --prompt "A person speaking on stage, camera slowly moves closer" \
  --output-path /kaggle/working/out_image_audio_text.mp4 \
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
- `--num-inference-steps`: “how many steps it thinks”. More steps usually looks better but is slower.
- `--enhance-prompt`: lets the model rewrite your prompt (sometimes helps).

There are also extra “how strongly to follow the prompt” knobs in the CLI. If you leave them at the defaults, that is a good start.

## Load → run → unload (to use GPU memory well)

The official code already follows this pattern:
- load the text encoder → encode prompt → unload it
- load the image encoder → encode your image(s) → unload it
- load the main model → run stage 1 → unload it
- load the upscaler + main model → run stage 2 → unload it

This helps reduce peak GPU memory.

## MUST use both GPUs (2× T4)

Because each run uses one GPU, the simplest way to use **both GPUs** is to run **two jobs at once**, one per GPU.

You can confirm it is working by running:

```bash
nvidia-smi
```

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

## Troubleshooting

- **Out of memory**: reduce width/height, reduce frames, reduce steps.
- **`fp8-cast` error**: remove `--quantization fp8-cast` and retry.
- **Gemma not found**: your `--gemma-root` folder must contain `tokenizer.model`, `preprocessor_config.json`, and `model*.safetensors`.
- **Wrong frame count**: use 49, 97, 121, 193, ... (must be 8×K + 1).
