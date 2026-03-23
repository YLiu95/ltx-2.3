# Kaggle (internet-enabled): build an offline **assets dataset** for the RTX PRO 6000 (no-internet) notebook

This creates a Kaggle Dataset that contains:

- LTX-2.3 distilled checkpoint + spatial upscaler
- Gemma 3 weights required by LTX’s text encoder
- (optional) a zip of this repo (so the offline notebook doesn’t need `git clone` / internet)

Then, in the **offline** notebook, you attach the dataset as an input and run `kaggle_s2v/run_s2v.py` with `--assets_mode offline`.

> Important: Gemma is gated and may have redistribution restrictions. Uploading model weights to a **public** Kaggle dataset may violate terms. Consider using a **private** dataset.

## Single Kaggle cell

Why the dataset is **flat** (no subdirectories):

- `kaggle datasets create --dir-mode zip` will **zip each folder** before uploading it.
- Zipping huge `.safetensors` files is very slow and uses a lot of extra temporary disk space.
- A flat layout lets us upload with `--dir-mode skip` (no zipping/tarring on Kaggle).

Paste this into **one** Kaggle notebook cell **with internet enabled**:

```python
import json
import os
import shutil
import subprocess
from pathlib import Path
from tqdm.auto import tqdm

# -----------------------
# 0) Secrets
# -----------------------
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")

# -----------------------
# 0.5) Keep caches out of /kaggle/working + enable progress bars
# -----------------------
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/temp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf/transformers"
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["TORCH_HOME"] = "/kaggle/temp/torch"
os.environ["PIP_CACHE_DIR"] = "/kaggle/temp/pip-cache"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")  # show HF download tqdm

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

# Install CLI if needed (installs into env; cache goes to /kaggle/temp)
subprocess.run(["python", "-m", "pip", "-q", "install", "kaggle", "huggingface_hub"], check=True)

# -----------------------
# 2) Dataset config
# -----------------------
# Choose the dataset slug you want to create/update under your account:
DATASET_SLUG = "ltx23-offline-assets"
DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

# Set this True if you *really* want a public dataset (not recommended for gated models like Gemma).
MAKE_PUBLIC = False

# Recommended: keep repo code in a separate tiny dataset (`ltx23-offline-code`)
# so you can update code without re-uploading ~67GB of weights.
INCLUDE_REPO_CODE = False

# This folder's contents will become /kaggle/input/<dataset>/...
# Use /kaggle/temp so we don't stage huge model files under /kaggle/working.
#
# IMPORTANT: keep this folder *flat* (no subdirectories) so we can upload with
# --dir-mode skip and avoid zipping huge folders.
DATASET_DIR = Path("/kaggle/temp/ltx23_offline_assets")
# If you already downloaded the files in a previous run, keep them and reuse.
# Set CLEAN_STAGING=True only when you want to force a full rebuild.
CLEAN_STAGING = False
if CLEAN_STAGING and DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# 3) Prepare / reuse files (no re-download if already present)
# -----------------------
from huggingface_hub import hf_hub_download, snapshot_download

LTX_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

def _move_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dst)
    except OSError:
        shutil.move(str(src), str(dst))


def _flatten_if_needed() -> None:
    # Supports older layouts from previous versions of this guide:
    #   DATASET_DIR/ltx/*, DATASET_DIR/gemma/*, DATASET_DIR/repo/ltx-2.3/*
    for sub in ("ltx", "gemma"):
        d = DATASET_DIR / sub
        if not d.exists():
            continue
        files = [p for p in d.rglob("*") if p.is_file()]
        for p in tqdm(files, desc=f"Flatten {sub}", unit="file", dynamic_ncols=True):
            dst = DATASET_DIR / p.name
            if dst.exists():
                continue
            _move_file(p, dst)
        shutil.rmtree(d, ignore_errors=True)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except FileExistsError:
        return
    except OSError:
        shutil.copy2(src, dst)


_flatten_if_needed()


def _ensure_gemma_files() -> None:
    required = [
        DATASET_DIR / "tokenizer.model",
        DATASET_DIR / "preprocessor_config.json",
        DATASET_DIR / "config.json",
    ]
    has_weights = any(DATASET_DIR.glob("model*.safetensors"))
    if all(p.exists() for p in required) and has_weights:
        return

    allow_patterns = [
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json",
        "generation_config.json",
        "model*.safetensors",
        "model.safetensors.index.json",
    ]

    # Prefer local cache if present; otherwise download once and cache it.
    try:
        snap_dir = snapshot_download(
            repo_id=GEMMA_REPO_ID,
            token=HF_TOKEN,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
    except Exception:
        snap_dir = snapshot_download(
            repo_id=GEMMA_REPO_ID,
            token=HF_TOKEN,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            allow_patterns=allow_patterns,
        )

    snap = Path(snap_dir)
    wanted: list[Path] = []
    for name in (
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        p = snap / name
        if p.exists():
            wanted.append(p)
    wanted.extend(sorted(snap.glob("model*.safetensors")))

    if not wanted:
        raise RuntimeError(f"Gemma snapshot seems empty at: {snap_dir}")

    total = sum(p.stat().st_size for p in wanted)
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Stage Gemma", dynamic_ncols=True) as pbar:
        for p in wanted:
            dst = DATASET_DIR / p.name
            if not dst.exists():
                _safe_symlink_or_copy(p, dst)
            pbar.update(p.stat().st_size)
            pbar.set_postfix(file=p.name, refresh=False)


print("Ensuring LTX weights (reuse cache if available)...")
ltx_needed: list[tuple[Path, Path]] = []
for fname in ("ltx-2.3-22b-distilled.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"):
    dst = DATASET_DIR / fname
    if dst.exists():
        continue
    try:
        cached = hf_hub_download(
            repo_id=LTX_REPO_ID,
            filename=fname,
            token=HF_TOKEN,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            local_files_only=True,
        )
    except Exception:
        cached = hf_hub_download(
            repo_id=LTX_REPO_ID,
            filename=fname,
            token=HF_TOKEN,
            cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
        )
    ltx_needed.append((Path(cached), dst))

total = sum(src.stat().st_size for src, _ in ltx_needed)
with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Stage LTX", dynamic_ncols=True) as pbar:
    for src, dst in ltx_needed:
        _safe_symlink_or_copy(src, dst)
        pbar.update(src.stat().st_size)
        pbar.set_postfix(file=dst.name, refresh=False)

print("Ensuring Gemma assets (reuse cache if available)...")
_ensure_gemma_files()

# -----------------------
# 4) (Optional) Bundle repo code (no .git folder)
# -----------------------
if INCLUDE_REPO_CODE:
    REPO_URL = "https://github.com/YLiu95/ltx-2.3.git"
    REPO_TMP_PARENT = Path("/kaggle/temp/ltx23_repo_clone")
    REPO_DST = REPO_TMP_PARENT / "ltx-2.3"
    REPO_ZIP = DATASET_DIR / "repo_ltx-2.3.zip"
    if not REPO_ZIP.exists():
        old_repo_dir = DATASET_DIR / "repo" / "ltx-2.3"
        if old_repo_dir.exists():
            print("Reusing previously-downloaded repo folder and zipping it...")
            repo_zip_base = DATASET_DIR / "repo_ltx-2.3"
            shutil.make_archive(str(repo_zip_base), "zip", root_dir=str(DATASET_DIR / "repo"), base_dir="ltx-2.3")
            shutil.rmtree(DATASET_DIR / "repo", ignore_errors=True)
        else:
            print("Cloning repo code...")
            shutil.rmtree(REPO_TMP_PARENT, ignore_errors=True)
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DST)], check=True)
            shutil.rmtree(REPO_DST / ".git", ignore_errors=True)

            # Zip the repo into a single file so the dataset stays flat (no directories).
            print("Zipping repo code into repo_ltx-2.3.zip ...")
            repo_zip_base = DATASET_DIR / "repo_ltx-2.3"
            shutil.make_archive(str(repo_zip_base), "zip", root_dir=str(REPO_TMP_PARENT), base_dir="ltx-2.3")
            shutil.rmtree(REPO_TMP_PARENT, ignore_errors=True)

# -----------------------
# 5) Create dataset metadata
# -----------------------
metadata = {
    "title": "LTX-2.3 Offline Assets (S2V)",
    "id": DATASET_ID,
    "licenses": [{"name": "CC0-1.0"}],
}
(DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

files = sorted([p for p in DATASET_DIR.iterdir() if p.is_file()])
total_bytes = sum(p.stat().st_size for p in files)
print(f"Staged {len(files)} files, total {total_bytes/1024**3:.1f} GiB in: {DATASET_DIR}")
for p in files[:10]:
    print(" -", p.name, f"({p.stat().st_size/1024**3:.2f} GiB)")
if len(files) > 10:
    print(f" - ... ({len(files)-10} more files)")

# -----------------------
# 6) Upload (create or version)
# -----------------------
def run_live(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode

create_cmd = ["kaggle", "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "skip"]
if MAKE_PUBLIC:
    create_cmd.insert(3, "-u")

rc = run_live(create_cmd)
if rc == 0:
    print("Created:", DATASET_ID)
else:
    print("Create failed (likely already exists). Trying version upload...")
    version_cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(DATASET_DIR),
        "-m",
        "Update offline assets",
        "--dir-mode",
        "skip",
    ]
    rc2 = run_live(version_cmd)
    if rc2 != 0:
        raise RuntimeError("Failed to create/update dataset.")
    print("Updated:", DATASET_ID)

print("\nNext (offline notebook): attach this dataset as input, plus `ltx23-offline-code`, then set:")
print(f"ASSETS_ROOT = '/kaggle/input/{DATASET_SLUG}'  # or the dataset folder name shown in the 'Data' panel")
```
