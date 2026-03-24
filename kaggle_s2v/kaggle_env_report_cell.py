"""
Kaggle: environment report (offline RTX PRO 6000 notebook)

Copy/paste this entire file into ONE Kaggle code cell, run it, then save the
cell output to a local file and share it back.

This prints:
- Python/OS info
- GPU + CUDA + torch info
- key package versions
- whether imports succeed (with full tracebacks on failure)
- ffmpeg/ffprobe presence + versions
"""

import json
import os
import platform
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
        out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
        out = out.strip()
        return f"$ {' '.join(cmd)}\n(rc={r.returncode})\n{out}".strip()
    except Exception:
        return f"$ {' '.join(cmd)}\n(EXCEPTION)\n{traceback.format_exc()}".strip()


def _pkg_version(dist_name: str) -> str:
    try:
        import importlib.metadata as im

        return im.version(dist_name)
    except Exception:
        return "<not installed>"


def _try_import(label: str, code: str) -> str:
    try:
        g: dict = {}
        exec(code, g, g)
        return f"[OK] {label}"
    except Exception:
        return f"[FAIL] {label}\n{traceback.format_exc()}"


lines: list[str] = []
lines.append("=== LTX Kaggle RTX PRO 6000 Env Report ===")
lines.append(f"Timestamp (UTC): {datetime.utcnow().isoformat()}Z")
lines.append("")

lines.append("== Python / OS ==")
lines.append(f"sys.executable: {sys.executable}")
lines.append(f"sys.version: {sys.version.replace(os.linesep, ' ')}")
lines.append(f"platform: {platform.platform()}")
lines.append(f"uname: {platform.uname()}")
lines.append("")

lines.append("== Environment vars (selected) ==")
for k in (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_URL_BASE",
    "KAGGLE_DATA_PROXY_URL",
    "KAGGLE_WORKING_DIR",
    "KAGGLE_TMP",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
):
    v = os.environ.get(k)
    if v is not None:
        lines.append(f"{k}={v}")
lines.append("")

lines.append("== GPU / CUDA ==")
lines.append(_run(["nvidia-smi", "-L"]))
lines.append("")
lines.append(_run(["nvidia-smi"]))
lines.append("")

lines.append("== ffmpeg/ffprobe ==")
lines.append(_run(["which", "ffmpeg"]))
lines.append(_run(["which", "ffprobe"]))
lines.append("")
lines.append(_run(["ffmpeg", "-version"]))
lines.append("")
lines.append(_run(["ffprobe", "-version"]))
lines.append("")

lines.append("== Key package versions (pip dist names) ==")
for pkg in (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
    "diffusers",
    "einops",
    "numpy",
    "Pillow",
    "sentencepiece",
    "opencv-python",
    "pyav",
):
    lines.append(f"{pkg}: {_pkg_version(pkg)}")
lines.append("")

lines.append("== Import checks ==")
checks = [
    ("import torch; torch.__version__", "import torch; print(torch.__version__)"),
    ("torch cuda availability", "import torch; print('cuda:', torch.cuda.is_available()); print('cuda ver:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"),
    ("import torchvision", "import torchvision; print(torchvision.__version__)"),
    ("import torchaudio", "import torchaudio; print(torchaudio.__version__)"),
    ("import transformers", "import transformers; print(transformers.__version__); print(transformers.__file__)"),
    ("from transformers import AutoImageProcessor", "from transformers import AutoImageProcessor; print(AutoImageProcessor)"),
    ("import transformers.models.auto.image_processing_auto", "import transformers.models.auto.image_processing_auto as m; print(getattr(m,'AutoImageProcessor',None))"),
    ("from transformers import AutoFeatureExtractor (fallback)", "from transformers import AutoFeatureExtractor; print(AutoFeatureExtractor)"),
    ("from transformers import AutoTokenizer", "from transformers import AutoTokenizer; print(AutoTokenizer)"),
    ("import PIL", "from PIL import Image; import PIL; print(PIL.__version__)"),
    ("import sentencepiece", "import sentencepiece as spm; print(spm.__version__)"),
    ("import safetensors", "import safetensors; print(safetensors.__version__)"),
    ("import einops", "import einops; print(einops.__version__)"),
    ("import av (PyAV)", "import av; print(av.__version__)"),
]
for label, code in checks:
    lines.append(_try_import(label, code))
    lines.append("")

lines.append("== transformers optional deps info ==")
lines.append(_try_import("transformers.utils.import_utils.is_vision_available()", "from transformers.utils.import_utils import is_vision_available; print('is_vision_available:', is_vision_available())"))
lines.append("")
lines.append(_try_import("transformers.image_processing_utils import", "import transformers.image_processing_utils as m; print(m)"))
lines.append("")

print("\n".join(lines))

