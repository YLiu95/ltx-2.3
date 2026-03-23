from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LTX_REPO_ID = "Lightricks/LTX-2.3"
GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"

LTX_DISTILLED_CHECKPOINT = "ltx-2.3-22b-distilled.safetensors"
LTX_DEV_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"
LTX_SPATIAL_UPSCALER_X2 = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
LTX_DISTILLED_LORA_384 = "ltx-2.3-22b-distilled-lora-384.safetensors"


@dataclass(frozen=True)
class DistilledA2VAssets:
    distilled_checkpoint_path: str
    spatial_upsampler_path: str
    gemma_root: str


@dataclass(frozen=True)
class TwoStageA2VAssets:
    checkpoint_path: str
    distilled_lora_path: str
    spatial_upsampler_path: str
    gemma_root: str


class OfflineAssetsError(RuntimeError):
    pass


def resolve_distilled_a2v_assets_from_dir(assets_root: str) -> DistilledA2VAssets:
    """Resolve model file paths from an *offline* Kaggle dataset mount.

    Expected layout under ``assets_root``:

    - ltx/
      - ltx-2.3-22b-distilled.safetensors
      - ltx-2.3-spatial-upscaler-x2-1.0.safetensors
    - gemma/
      - tokenizer.model
      - preprocessor_config.json
      - model*.safetensors (+ index json if sharded)
    - repo/ltx-2.3/ (optional; for running without git clone)
    """
    root = Path(assets_root)
    ltx_dir = root / "ltx"
    gemma_dir = root / "gemma"

    distilled_checkpoint_path = ltx_dir / LTX_DISTILLED_CHECKPOINT
    spatial_upsampler_path = ltx_dir / LTX_SPATIAL_UPSCALER_X2

    required = [
        distilled_checkpoint_path,
        spatial_upsampler_path,
        gemma_dir / "tokenizer.model",
        gemma_dir / "preprocessor_config.json",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise OfflineAssetsError(
            "Offline assets are missing required files:\n"
            + "\n".join(f"- {p}" for p in missing)
            + f"\n\nassets_root was: {assets_root}"
        )

    # Gemma weights can be sharded; just require at least one shard.
    if not list(gemma_dir.rglob("model*.safetensors")):
        raise OfflineAssetsError(
            "Offline assets are missing Gemma weights (expected at least one 'model*.safetensors' under "
            f"{gemma_dir})."
        )

    return DistilledA2VAssets(
        distilled_checkpoint_path=str(distilled_checkpoint_path),
        spatial_upsampler_path=str(spatial_upsampler_path),
        gemma_root=str(gemma_dir),
    )


def configure_kaggle_cache_dirs(cache_root: str = "/kaggle/temp") -> Path:
    """Route all HF/Transformers/Torch caches away from /kaggle/working.

    Kaggle best practice:
    - /kaggle/input is read-only
    - /kaggle/working is for *outputs*
    - /kaggle/temp is for ephemeral downloads / caches
    """
    root = Path(cache_root)
    hf_home = root / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / "xdg-cache"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))

    # Keep Kaggle logs clean + reduce metadata traffic.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    return hf_home


def _hf_hub_download(repo_id: str, filename: str, token: str | None, cache_dir: Path) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=str(cache_dir),
    )


def _hf_snapshot_download(repo_id: str, token: str | None, cache_dir: Path, allow_patterns: list[str]) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        token=token,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    )


def ensure_gemma_root(hf_token: str | None, cache_root: str = "/kaggle/temp") -> str:
    """Download Gemma assets required by LTX's GemmaTextEncoder loader.

    LTX's ModelLedger expects:
    - tokenizer.model (SentencePiece)
    - preprocessor_config.json (for AutoImageProcessor)
    - model*.safetensors shards + index json
    """
    cache_dir = configure_kaggle_cache_dirs(cache_root)
    allow_patterns = [
        "**/tokenizer.model",
        "**/tokenizer_config.json",
        "**/special_tokens_map.json",
        "**/preprocessor_config.json",
        "**/config.json",
        "**/generation_config.json",
        "**/model*.safetensors",
        "**/model.safetensors.index.json",
    ]
    return _hf_snapshot_download(GEMMA_REPO_ID, hf_token, cache_dir=cache_dir, allow_patterns=allow_patterns)


def ensure_distilled_a2v_assets(hf_token: str | None, cache_root: str = "/kaggle/temp") -> DistilledA2VAssets:
    cache_dir = configure_kaggle_cache_dirs(cache_root)
    distilled_checkpoint_path = _hf_hub_download(LTX_REPO_ID, LTX_DISTILLED_CHECKPOINT, hf_token, cache_dir=cache_dir)
    spatial_upsampler_path = _hf_hub_download(LTX_REPO_ID, LTX_SPATIAL_UPSCALER_X2, hf_token, cache_dir=cache_dir)
    gemma_root = ensure_gemma_root(hf_token, cache_root=cache_root)
    return DistilledA2VAssets(
        distilled_checkpoint_path=distilled_checkpoint_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
    )


def ensure_two_stage_a2v_assets(hf_token: str | None, cache_root: str = "/kaggle/temp") -> TwoStageA2VAssets:
    cache_dir = configure_kaggle_cache_dirs(cache_root)
    checkpoint_path = _hf_hub_download(LTX_REPO_ID, LTX_DEV_CHECKPOINT, hf_token, cache_dir=cache_dir)
    distilled_lora_path = _hf_hub_download(LTX_REPO_ID, LTX_DISTILLED_LORA_384, hf_token, cache_dir=cache_dir)
    spatial_upsampler_path = _hf_hub_download(LTX_REPO_ID, LTX_SPATIAL_UPSCALER_X2, hf_token, cache_dir=cache_dir)
    gemma_root = ensure_gemma_root(hf_token, cache_root=cache_root)
    return TwoStageA2VAssets(
        checkpoint_path=checkpoint_path,
        distilled_lora_path=distilled_lora_path,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
    )
