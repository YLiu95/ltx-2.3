# Kaggle (internet-enabled): build/update a tiny **offline code dataset** (no weights)

This creates a small Kaggle Dataset that contains **only this repo’s code**.

Why this exists:

- Your `ltx23-offline-assets` dataset is ~67GB (weights). Re-uploading it just to update code is painful.
- Keeping code in a separate dataset lets the **offline** RTX PRO 6000 notebook stay fully offline, while still letting you iterate on code quickly.

In the **offline** notebook you’ll attach:

- `ltx23-offline-assets` (weights)
- `ltx23-offline-code` (repo code)

and run `kaggle_s2v/run_s2v.py` from the code dataset.

## Single Kaggle cell

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

KAGGLE_USERNAME = user_secrets.get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = user_secrets.get_secret("KAGGLE_KEY")

# -----------------------
# 0.5) Keep caches out of /kaggle/working
# -----------------------
os.environ["XDG_CACHE_HOME"] = "/kaggle/temp/xdg-cache"
os.environ["PIP_CACHE_DIR"] = "/kaggle/temp/pip-cache"

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
subprocess.run(["python", "-m", "pip", "-q", "install", "kaggle"], check=True)

# -----------------------
# 2) Dataset config
# -----------------------
DATASET_SLUG = "ltx23-offline-code"
DATASET_ID = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

MAKE_PUBLIC = True  # code-only is safe to keep public

DATASET_DIR = Path("/kaggle/temp/ltx23_offline_code")
CLEAN_STAGING = True
if CLEAN_STAGING and DATASET_DIR.exists():
    shutil.rmtree(DATASET_DIR)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# 3) Stage repo code into dataset folder
# -----------------------
REPO_URL = "https://github.com/YLiu95/ltx-2.3.git"
REPO_DST = DATASET_DIR / "repo_ltx-2.3" / "ltx-2.3"
REPO_DST.parent.mkdir(parents=True, exist_ok=True)

print("Cloning repo code...")
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DST)], check=True)
shutil.rmtree(REPO_DST / ".git", ignore_errors=True)

# Quick sanity checks: verify this snapshot includes the "no PyAV required" fixes.
run_py = REPO_DST / "kaggle_s2v" / "run_s2v.py"
media_io_py = REPO_DST / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "utils" / "media_io.py"
if not run_py.exists():
    raise RuntimeError(f"Missing runner: {run_py}")
if not media_io_py.exists():
    raise RuntimeError(f"Missing media I/O module: {media_io_py}")

txt = run_py.read_text(encoding="utf-8")
if "ffprobe" not in txt:
    raise RuntimeError(
        "This repo snapshot looks too old (it still hard-requires PyAV). "
        "Make sure you're cloning the right repo/branch."
    )

# -----------------------
# 4) Dataset metadata
# -----------------------
metadata = {
    "title": "LTX-2.3 Offline Code (S2V)",
    "id": DATASET_ID,
    "licenses": [{"name": "Apache 2.0"}],
}
(DATASET_DIR / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

# Small progress summary
files = [p for p in DATASET_DIR.rglob("*") if p.is_file()]
total_bytes = sum(p.stat().st_size for p in files)
print(f"Staged {len(files)} files, total {total_bytes/1024**2:.1f} MiB in: {DATASET_DIR}")

# -----------------------
# 5) Upload (version first, then create)
# -----------------------
def run_live(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode

version_cmd = [
    "kaggle",
    "datasets",
    "version",
    "-p",
    str(DATASET_DIR),
    "-m",
    "Update offline code",
    "--dir-mode",
    "skip",
]
rc = run_live(version_cmd)
if rc == 0:
    print("Updated:", DATASET_ID)
else:
    create_cmd = ["kaggle", "datasets", "create", "-p", str(DATASET_DIR), "--dir-mode", "skip"]
    if MAKE_PUBLIC:
        create_cmd.insert(3, "-u")
    rc2 = run_live(create_cmd)
    if rc2 != 0:
        raise RuntimeError("Failed to create/update dataset.")
    print("Created:", DATASET_ID)

print("\nNext (offline notebook): attach this dataset as input, then point CODE_ROOT to it (see `kaggle_s2v/kaggle_cell_example.md`).")
```

