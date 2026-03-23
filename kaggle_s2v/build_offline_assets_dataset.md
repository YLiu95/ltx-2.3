# Kaggle (internet-enabled): build an offline **assets dataset** for the RTX PRO 6000 (no-internet) notebook

This creates a Kaggle Dataset that contains:

- LTX-2.3 distilled checkpoint + spatial upscaler
- Gemma 3 weights required by LTX’s text encoder
- a copy of this repo (so the offline notebook doesn’t need `git clone` / `pip install -e`)

Then, in the **offline** notebook, you attach the dataset as an input and run `kaggle_s2v/run_s2v.py` with `--assets_mode offline`.

> Important: Gemma is gated and may have redistribution restrictions. Uploading model weights to a **public** Kaggle dataset may violate terms. Consider using a **private** dataset.

## Single Kaggle cell

Paste this into **one** Kaggle notebook cell **with internet enabled**:

```python
import json
import os
import shutil
import subprocess
from pathlib import Path

# -----------------------
# 0) Secrets
# -----------------------
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")

# -----------------------
# 1) Kaggle API auth (store creds under /kaggle/temp)
# -----------------------
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = KAGGLE_KEY

kaggle_cfg_dir = Path("/kaggle/temp/.kaggle")
kaggle_cfg_dir.mkdir(parents=True, exist_ok=True)
Path(kaggle_cfg_dir / "kaggle.json").write_text(
    json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}),
    encoding="utf-8",
)
os.chmod(kaggle_cfg_dir / "kaggle.json", 0o600)
os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_cfg_dir)

# Install CLI if needed (installs into env; keep pip cache out of /kaggle/working)
os.environ["PIP_CACHE_DIR"] = "/kaggle/temp/pip-cache"
subprocess.run(["python", "-m", "pip", "-q", "install", "kaggle", "huggingface_hub"], check=True)

# -----------------------
# 2) Dataset config
# -----------------------
# Choose the dataset slug you want to create/update under your account:
DATASET_SLUG = "ltx23-offline-assets"
DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

# This folder's contents will become /kaggle/input/<dataset>/...
# Use /kaggle/temp so we don't stage huge model files under /kaggle/working.
DATASET_DIR = Path("/kaggle/temp/ltx23_offline_assets")
if DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

(DATASET_DIR / "ltx").mkdir()
(DATASET_DIR / "gemma").mkdir()
(DATASET_DIR / "repo").mkdir()

# -----------------------
# 3) Download model files into the dataset folder
# -----------------------
from huggingface_hub import hf_hub_download, snapshot_download

LTX_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

print("Downloading LTX weights...")
hf_hub_download(
    repo_id=LTX_REPO_ID,
    filename="ltx-2.3-22b-distilled.safetensors",
    token=HF_TOKEN,
    local_dir=str(DATASET_DIR / "ltx"),
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id=LTX_REPO_ID,
    filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    token=HF_TOKEN,
    local_dir=str(DATASET_DIR / "ltx"),
    local_dir_use_symlinks=False,
)

print("Downloading Gemma assets (tokenizer + processor + all model*.safetensors shards)...")
snapshot_download(
    repo_id=GEMMA_REPO_ID,
    token=HF_TOKEN,
    local_dir=str(DATASET_DIR / "gemma"),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    ],
)

# -----------------------
# 4) Bundle repo code (no .git folder)
# -----------------------
REPO_URL = "https://github.com/YLiu95/ltx-2.3.git"
REPO_DST = DATASET_DIR / "repo" / "ltx-2.3"
print("Cloning repo code...")
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DST)], check=True)
shutil.rmtree(REPO_DST / ".git", ignore_errors=True)

# -----------------------
# 5) Create dataset metadata
# -----------------------
metadata = {
    "title": "LTX-2.3 Offline Assets (S2V)",
    "id": DATASET_ID,
    "licenses": [{"name": "CC0-1.0"}],
}
(DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print("Staged files:")
subprocess.run(["bash", "-lc", f"ls -la '{DATASET_DIR}' && du -sh '{DATASET_DIR}'"], check=True)

# -----------------------
# 6) Upload (create or version)
# -----------------------
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False, text=True, capture_output=True)

create = run(["kaggle", "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "zip"])
if create.returncode == 0:
    print("Created:", DATASET_ID)
else:
    # If it already exists, create a new version instead.
    print("Create failed (likely already exists). Output:")
    print(create.stdout[-2000:])
    print(create.stderr[-2000:])
    version = run(["kaggle", "datasets", "version", "-p", str(DATASET_DIR), "-m", "Update offline assets", "--dir-mode", "zip"])
    if version.returncode != 0:
        print(version.stdout[-2000:])
        print(version.stderr[-2000:])
        raise RuntimeError("Failed to create/update dataset.")
    print("Updated:", DATASET_ID)

print("\nNext (offline notebook): attach this dataset as input, then set:")
print(f"ASSETS_ROOT = '/kaggle/input/{DATASET_SLUG}'  # or the dataset folder name shown in the 'Data' panel")
```
