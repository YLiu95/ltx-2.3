from __future__ import annotations

from pathlib import Path

from PIL import Image


def ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def pad_image_to_size(
    image_path: str,
    *,
    target_width: int,
    target_height: int,
    fill: tuple[int, int, int] = (0, 0, 0),
    out_dir: str = "/kaggle/temp/s2v/padded",
) -> str:
    """Pad an image to an explicit target size (no crop, no stretch)."""
    src = Path(image_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    image = Image.open(src).convert("RGB")
    width, height = image.size
    if target_width < width or target_height < height:
        raise ValueError(
            f"Target size {target_width}x{target_height} is smaller than source {width}x{height}. "
            "Padding can only increase size."
        )

    if target_width == width and target_height == height:
        out_path = out_root / f"{src.stem}_padded.png"
        image.save(out_path)
        return str(out_path)

    padded = Image.new("RGB", (target_width, target_height), fill)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    padded.paste(image, (left, top))

    out_path = out_root / f"{src.stem}_padded_{target_width}x{target_height}.png"
    padded.save(out_path)
    return str(out_path)


def pad_image_to_multiple(
    image_path: str,
    *,
    multiple: int = 64,
    fill: tuple[int, int, int] = (0, 0, 0),
    out_dir: str = "/kaggle/temp/s2v/padded",
) -> str:
    """Pad an image to the nearest (multiple x multiple) size (no crop, no stretch).

    This is useful because LTX two-stage pipelines require width/height to be multiples of 64.
    Padding avoids the center-crop done by LTX's internal preprocessing when the input doesn't
    already match the target aspect/resolution.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    target_w = ceil_to_multiple(width, multiple)
    target_h = ceil_to_multiple(height, multiple)
    return pad_image_to_size(
        image_path=image_path,
        target_width=target_w,
        target_height=target_h,
        fill=fill,
        out_dir=out_dir,
    )
