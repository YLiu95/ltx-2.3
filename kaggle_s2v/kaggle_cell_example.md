# Kaggle: single-cell S2V (image + audio → video)

Choose **one** of the following single-cell options:

- **A) Offline notebook (NO internet)**: run from a Kaggle Dataset that already contains weights + repo code.
- **B) Internet-enabled notebook**: download weights from Hugging Face into `/kaggle/temp`.

## A) Offline (NO internet) notebook

Attach your offline assets dataset (built from `kaggle_s2v/build_offline_assets_dataset.md`) as a Kaggle input.

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

# --- 1) Point to the *dataset root* ------------------------------------------
# This dataset is built by `kaggle_s2v/build_offline_assets_dataset.md`.
#
# Expected *flat* layout under ASSETS_ROOT (no directories):
#   - ltx-2.3-22b-distilled.safetensors
#   - ltx-2.3-spatial-upscaler-x2-1.0.safetensors
#   - tokenizer.model + preprocessor_config.json + model*.safetensors (Gemma)
#   - repo_ltx-2.3.zip  (repo code bundle)
ASSETS_ROOT = "/kaggle/input/ltx23-offline-assets"  # <-- change to your dataset folder name

# Extract the repo zip into /kaggle/temp so we can run it without internet.
import zipfile

# Prefer running directly from a repo folder if the dataset contains one
# (older dataset versions used ASSETS_ROOT/repo/ltx-2.3/).
REPO_DIR = f"{ASSETS_ROOT}/repo/ltx-2.3"
if not os.path.isdir(REPO_DIR):
    # Newer dataset versions store the repo as a single zip: ASSETS_ROOT/repo_ltx-2.3.zip
    REPO_DIR = "/kaggle/temp/ltx-2.3"
    REPO_ZIP = f"{ASSETS_ROOT}/repo_ltx-2.3.zip"
    if not os.path.exists(REPO_DIR):
        if not os.path.exists(REPO_ZIP):
            raise FileNotFoundError(
                "Could not find repo code in the assets dataset. Expected either:\n"
                f"- {ASSETS_ROOT}/repo/ltx-2.3/\n"
                f"- {REPO_ZIP}\n"
            )
        with zipfile.ZipFile(REPO_ZIP, "r") as zf:
            zf.extractall("/kaggle/temp")

# --- 2) Inputs ---------------------------------------------------------------
AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

# --- 3) Run (recommended pipeline) ------------------------------------------
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
