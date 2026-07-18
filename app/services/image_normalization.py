from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def normalize_source_image(
    source_path: str,
    output_path: str,
    *,
    max_side: int = 2048,
) -> str:
    source = Path(source_path)
    output = Path(output_path)

    if not source.exists():
        raise ValueError(
            f"Source image does not exist: {source}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened)
            image = image.convert("RGB")

            width, height = image.size
            longest_side = max(width, height)

            if longest_side > max_side:
                scale = max_side / longest_side

                new_size = (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                )

                image = image.resize(
                    new_size,
                    Image.Resampling.LANCZOS,
                )

            image.save(
                output,
                format="PNG",
                optimize=True,
            )

    except Exception as exc:
        raise ValueError(
            f"Could not normalize image: {exc}"
        ) from exc

    return str(output)