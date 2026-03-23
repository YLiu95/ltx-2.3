# Kaggle: single-cell S2V (image + audio → video)

Paste the following into **one Kaggle notebook cell**.

```python
# --- 0) Keep *all* caches out of /kaggle/working -----------------------------
import os

os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"

# Reduces CUDA memory fragmentation on large models (recommended for LTX).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 1) Secrets (HF token) ---------------------------------------------------
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")

# --- 2) Get code (clone to /kaggle/temp, NOT /kaggle/working) -----------------
REPO_DIR = "/kaggle/temp/ltx-2.3"
if not os.path.exists(REPO_DIR):
    !git clone --depth 1 https://github.com/YLiu95/ltx-2.3 {REPO_DIR}

# --- 3) Install the repo as a package (installs into the Python env, not /kaggle/working)
!pip -q install -e {REPO_DIR}

# --- 4) Inputs ---------------------------------------------------------------
AUDIO_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrain_Chinese.wav"
IMAGE_PATH = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/female secretary 512x763.png"
AUDIO_TEXT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/4s_mandrian_Chinese_text.txt"
PROMPT_FILE = "/kaggle/input/datasets/yliu95/s2v-test-data/S2V data/I2V prompt.txt"

# --- 5) Run (recommended pipeline) ------------------------------------------
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
#   How often (in denoising steps) to query VRAM for tqdm postfix. 1 = most detailed, slightly more overhead.
!
!python {REPO_DIR}/kaggle_s2v/run_s2v.py \
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
#   /kaggle/working/s2v/outputs/<timestamp>/video.mp4
```

