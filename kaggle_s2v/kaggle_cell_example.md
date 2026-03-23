# Kaggle: single-cell S2V (image + audio → video)

Choose **one** of the following single-cell options:

- **A) Offline notebook (NO internet)**: run from a Kaggle Dataset that already contains weights + repo code.
- **B) Internet-enabled notebook**: download weights from Hugging Face into `/kaggle/temp`.

## A) Offline (NO internet) notebook

Attach:

- `ltx23-offline-assets` (weights)
- `ltx23-offline-code` (this repo code; lets you update code without re-uploading ~67GB)

```python
import os

# --- 0) Keep caches out of /kaggle/working -----------------------------------
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"

# Recommended for large models: reduces CUDA memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Hard-offline (prevents any network calls).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Note: some Kaggle images don't include PyAV (`import av`). The runner falls
# back to `ffmpeg` for audio/video I/O when PyAV is missing.

# --- 1) Locate datasets ------------------------------------------------------
# Kaggle sometimes mounts datasets as:
# - /kaggle/input/<slug>
# - /kaggle/input/datasets/<owner>/<slug>
import glob

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

# --- 2) Locate repo code -----------------------------------------------------
# Prefer code dataset (so you can update code without touching the ~67GB weights dataset).
# Supported layouts in CODE_ROOT:
# - {CODE_ROOT}/repo_ltx-2.3/ltx-2.3/...
# - {CODE_ROOT}/repo_ltx-2.3/...
# - {CODE_ROOT}/repo/ltx-2.3/...
repo_candidates = [
    f"{CODE_ROOT}/repo_ltx-2.3/ltx-2.3",
    f"{CODE_ROOT}/repo_ltx-2.3",
    f"{CODE_ROOT}/repo/ltx-2.3",
    # Fallback: some people bundled code inside the weights dataset too.
    f"{ASSETS_ROOT}/repo_ltx-2.3/ltx-2.3",
    f"{ASSETS_ROOT}/repo_ltx-2.3",
    f"{ASSETS_ROOT}/repo/ltx-2.3",
]
REPO_DIR = next((d for d in repo_candidates if os.path.exists(os.path.join(d, "kaggle_s2v", "run_s2v.py"))), None)
if REPO_DIR is None:
    raise FileNotFoundError(
        "Could not find repo code. Looked for `kaggle_s2v/run_s2v.py` under:\n" + "\n".join(repo_candidates)
    )

# Sanity check: if you see `ModuleNotFoundError: No module named 'av'`, you're running an old snapshot.
with open(os.path.join(REPO_DIR, "kaggle_s2v", "run_s2v.py"), "r", encoding="utf-8") as f:
    if "ffprobe" not in f.read():
        raise RuntimeError(
            "This repo snapshot is too old (it still hard-requires PyAV). "
            "Update the `ltx23-offline-code` dataset to the latest version and re-attach it."
        )

# --- 3) Inputs ---------------------------------------------------------------
AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

# --- 4) Run (recommended pipeline) ------------------------------------------
# Notes on the non-beginner settings below:
#
# --quantization:
#   - fp8-cast: stores transformer weights in FP8 and upcasts during matmul. Lower VRAM, good default.
#   - fp8-scaled-mm: uses TensorRT-LLM scaled FP8 matmul ops (only if the environment provides them).
#
# --cleanup:
#   - aggressive: unload/reload big modules between stages to minimize peak VRAM (slowest, safest).
#   - balanced: unload some modules between stages (middle ground).
#   - none: keep everything loaded (fastest, highest peak VRAM).
#
# --vae_tiling:
#   - default: decodes the video in tiles (safer for long/high-res videos and lower VRAM).
#   - none: decode in one go (faster but uses more VRAM).
#
# --progress_vram_every:
#   How often (in denoising steps) to query VRAM for tqdm postfix. 1 = most detailed, slight overhead.
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

# Output video(s) will be under:
#   /kaggle/working/s2v/outputs/<run_name>/video.mp4
```

## B) Internet-enabled notebook (downloads to `/kaggle/temp`)

> Note: the Gemma repo used by LTX is gated; make sure your Hugging Face account has accepted the model terms.

```python
import os

# --- 0) Keep caches out of /kaggle/working -----------------------------------
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 1) Secrets (HF token) ---------------------------------------------------
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")

# --- 2) Get code (clone to /kaggle/temp, NOT /kaggle/working) -----------------
REPO_DIR = "/kaggle/temp/ltx-2.3"
if not os.path.exists(REPO_DIR):
    !git clone --depth 1 https://github.com/YLiu95/ltx-2.3 {REPO_DIR}

# --- 3) Inputs ---------------------------------------------------------------
AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

# --- 4) Run ------------------------------------------------------------------
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
