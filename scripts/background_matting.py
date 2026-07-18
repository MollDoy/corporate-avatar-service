from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove


def _parse_hex_color(value: str) -> tuple[int, int, int]:
    normalized = value.strip().lstrip("#")

    if len(normalized) != 6:
        raise ValueError(
            "Background color must contain six hexadecimal "
            "characters."
        )

    try:
        red = int(normalized[0:2], 16)
        green = int(normalized[2:4], 16)
        blue = int(normalized[4:6], 16)
    except ValueError as exc:
        raise ValueError(
            "Background color contains invalid hexadecimal "
            "characters."
        ) from exc

    return red, green, blue


def _load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(
            f"Input portrait does not exist: {path}"
        )

    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _prepare_mask(
    mask: Image.Image,
    *,
    expected_size: tuple[int, int],
    alpha_gamma: float,
) -> tuple[Image.Image, dict[str, float]]:
    prepared = mask.convert("L")

    if prepared.size != expected_size:
        prepared = prepared.resize(
            expected_size,
            resample=Image.Resampling.LANCZOS,
        )

    array = np.asarray(
        prepared,
        dtype=np.float32,
    ) / 255.0

    if not np.isfinite(array).all():
        raise RuntimeError(
            "BiRefNet mask contains NaN or infinity."
        )

    if alpha_gamma <= 0:
        raise ValueError(
            "Alpha gamma must be greater than zero."
        )

    if abs(alpha_gamma - 1.0) > 1e-6:
        array = np.power(
            np.clip(array, 0.0, 1.0),
            alpha_gamma,
        )

    array = np.clip(array, 0.0, 1.0)

    statistics = {
        "foreground_mean": float(array.mean()),
        "foreground_hard_ratio": float(
            (array >= 0.5).mean()
        ),
        "alpha_min": float(array.min()),
        "alpha_max": float(array.max()),
        "alpha_std": float(array.std()),
    }

    uint8_mask = np.round(
        array * 255.0
    ).astype(np.uint8)

    return Image.fromarray(uint8_mask, mode="L"), statistics


def process_background(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    background_hex: str,
    alpha_gamma: float,
    min_foreground_ratio: float,
    max_foreground_ratio: float,
    mask_output_path: Path | None,
    stats_output_path: Path | None,
) -> None:
    image = _load_rgb(input_path)

    session = new_session(
        model_name,
        providers=["CPUExecutionProvider"],
    )

    predicted_mask = remove(
        image,
        session=session,
        only_mask=True,
        post_process_mask=False,
    )

    if not isinstance(predicted_mask, Image.Image):
        predicted_mask = Image.fromarray(
            np.asarray(predicted_mask)
        )

    mask, statistics = _prepare_mask(
        predicted_mask,
        expected_size=image.size,
        alpha_gamma=alpha_gamma,
    )

    hard_ratio = statistics["foreground_hard_ratio"]

    if not (
        min_foreground_ratio
        <= hard_ratio
        <= max_foreground_ratio
    ):
        raise RuntimeError(
            "BiRefNet produced an implausible foreground ratio: "
            f"{hard_ratio:.4f}; expected between "
            f"{min_foreground_ratio:.4f} and "
            f"{max_foreground_ratio:.4f}."
        )

    background_color = _parse_hex_color(
        background_hex
    )

    background = Image.new(
        "RGB",
        image.size,
        background_color,
    )

    composite = Image.composite(
        image,
        background,
        mask,
    ).convert("RGB")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    temporary_output.unlink(missing_ok=True)
    composite.save(
        temporary_output,
        format="PNG",
        optimize=True,
    )
    temporary_output.replace(output_path)

    if mask_output_path is not None:
        mask_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        mask.save(
            mask_output_path,
            format="PNG",
            optimize=True,
        )

    statistics.update(
        {
            "model": model_name,
            "background_hex": (
                background_hex.strip().lstrip("#").upper()
            ),
            "width": image.width,
            "height": image.height,
            "provider": "CPUExecutionProvider",
            "u2net_home": os.getenv("U2NET_HOME", ""),
        }
    )

    if stats_output_path is not None:
        stats_output_path.write_text(
            json.dumps(
                statistics,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            statistics,
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply BiRefNet portrait matting and a fixed "
            "corporate background."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default="birefnet-portrait",
    )
    parser.add_argument(
        "--background-hex",
        required=True,
    )
    parser.add_argument(
        "--alpha-gamma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-foreground-ratio",
        type=float,
        default=0.98,
    )
    parser.add_argument("--mask-output")
    parser.add_argument("--stats-output")

    arguments = parser.parse_args()

    process_background(
        input_path=Path(arguments.input).resolve(),
        output_path=Path(arguments.output).resolve(),
        model_name=arguments.model,
        background_hex=arguments.background_hex,
        alpha_gamma=arguments.alpha_gamma,
        min_foreground_ratio=(
            arguments.min_foreground_ratio
        ),
        max_foreground_ratio=(
            arguments.max_foreground_ratio
        ),
        mask_output_path=(
            Path(arguments.mask_output).resolve()
            if arguments.mask_output
            else None
        ),
        stats_output_path=(
            Path(arguments.stats_output).resolve()
            if arguments.stats_output
            else None
        ),
    )


if __name__ == "__main__":
    main()