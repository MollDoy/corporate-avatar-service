import ctypes
import gc
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from app.core.config import settings
from app.services.avatar_styles import get_avatar_style
from app.services.image_storage import save_job_image, save_result_image


_sessions: dict[str, Any] = {}


def _configure_rembg_model_dir() -> None:
    """
    rembg reads U2NET_HOME from environment variables.
    We set it explicitly so downloaded models are stored in /app/models/rembg.
    """
    os.environ["U2NET_HOME"] = settings.u2net_home
    os.environ["OMP_NUM_THREADS"] = str(settings.rembg_omp_num_threads)

    Path(settings.u2net_home).mkdir(parents=True, exist_ok=True)


def _get_rembg_session() -> Any:
    """
    Creating a rembg session is relatively expensive,
    so we cache it at module level.
    """
    _configure_rembg_model_dir()

    model_name = settings.rembg_model_name

    if model_name not in _sessions:
        from rembg import new_session

        _sessions[model_name] = new_session(model_name)

    return _sessions[model_name]

def clear_rembg_sessions() -> None:
    """
    Удаляет ссылки на ONNX-сессии и пытается вернуть освобождённую
    память процессу операционной системы.
    """
    _sessions.clear()
    gc.collect()

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        # Оптимизация предназначена для Linux/glibc.
        pass

def _create_gradient_background(
    width: int,
    height: int,
    top_color: tuple[int, int, int],
    bottom_color: tuple[int, int, int],
) -> Image.Image:
    background = Image.new("RGB", (width, height), top_color)
    pixels = background.load()

    for y in range(height):
        ratio = y / max(height - 1, 1)

        r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)

        for x in range(width):
            pixels[x, y] = (r, g, b)

    return background


def _feather_alpha(foreground: Image.Image) -> Image.Image:
    """
    Slightly smooths alpha edges after background removal.
    This does not fix bad masks completely, but reduces harsh blocky borders.
    """
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    red, green, blue, alpha = foreground.split()

    radius = max(0.0, settings.mask_feather_radius)

    if radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))

    return Image.merge("RGBA", (red, green, blue, alpha))


def _trim_transparent_edges(
    foreground: Image.Image,
    padding_ratio: float = 0.04,
) -> Image.Image:
    """
    Trims transparent area around the segmented person.

    Important:
    We add padding so that hair/head edges are not cut too tightly.
    """
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    alpha = foreground.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        return foreground

    width, height = foreground.size
    left, top, right, bottom = bbox

    padding = int(max(width, height) * padding_ratio)

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)

    return foreground.crop((left, top, right, bottom))


def _fit_foreground_to_avatar_canvas(
    foreground: Image.Image,
    output_size: int,
) -> Image.Image:
    """
    Fits the full segmented person into a square avatar canvas.

    Unlike center-crop, this function does not cut the head.
    It scales the person to fit inside the canvas with margins.
    """
    if foreground.mode != "RGBA":
        foreground = foreground.convert("RGBA")

    horizontal_margin = int(output_size * 0.05)
    top_margin = int(output_size * 0.04)
    bottom_margin = int(output_size * 0.00)

    max_width = output_size - 2 * horizontal_margin
    max_height = output_size - top_margin - bottom_margin

    width, height = foreground.size

    if width <= 0 or height <= 0:
        raise ValueError("Foreground image has invalid size")

    scale = min(max_width / width, max_height / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = foreground.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGBA", (output_size, output_size), (0, 0, 0, 0))

    x = (output_size - new_width) // 2
    y = output_size - bottom_margin - new_height

    if y < top_margin:
        y = top_margin

    canvas.alpha_composite(resized, dest=(x, y))

    return canvas

def _resize_for_rembg(
    image: Image.Image,
) -> Image.Image:
    max_size = settings.rembg_max_input_size

    if max_size <= 0:
        return image.copy()

    width, height = image.size
    longest_side = max(width, height)

    if longest_side <= max_size:
        return image.copy()

    scale = max_size / longest_side

    resized_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    return image.resize(
        resized_size,
        Image.Resampling.LANCZOS,
    )

def generate_basic_corporate_avatar(
    job_id: str,
    source_image_path: str,
    style_id: str = "default_business",
) -> str:
    """
    Удаляет фон на уменьшенной рабочей копии изображения,
    помещает человека на корпоративный фон и сохраняет результат 512x512.
    """
    from rembg import remove

    output_size = settings.avatar_output_size
    style = get_avatar_style(style_id)

    with Image.open(source_image_path) as opened_source:
        source = opened_source.convert("RGB")

    working_source = _resize_for_rembg(source)
    source.close()

    session = _get_rembg_session()

    foreground: Image.Image | None = None
    foreground_canvas: Image.Image | None = None
    person_mask: Image.Image | None = None
    background: Image.Image | None = None
    result: Image.Image | None = None
    segmentation_mask: Image.Image | None = None

    try:
        if settings.rembg_alpha_matting:
            foreground = remove(
                working_source,
                session=session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=3,
            ).convert("RGBA")
        else:
            segmentation_mask = remove(
                working_source,
                session=session,
                only_mask=True,
            ).convert("L")

            foreground = working_source.convert("RGBA")
            foreground.putalpha(segmentation_mask)

        working_source.close()
        working_source = None

        if segmentation_mask is not None:
            segmentation_mask.close()
            segmentation_mask = None

        smoothed_foreground = _feather_alpha(foreground)

        if smoothed_foreground is not foreground:
            foreground.close()

        foreground = smoothed_foreground

        trimmed_foreground = _trim_transparent_edges(foreground)

        if trimmed_foreground is not foreground:
            foreground.close()

        foreground = trimmed_foreground

        foreground_canvas = _fit_foreground_to_avatar_canvas(
            foreground=foreground,
            output_size=output_size,
        )

        if foreground_canvas is not foreground:
            foreground.close()
            foreground = None

        person_mask = foreground_canvas.getchannel("A")
        save_job_image(job_id, person_mask, "person_mask.png")

        background = _create_gradient_background(
            width=output_size,
            height=output_size,
            top_color=style.top_color,
            bottom_color=style.bottom_color,
        ).convert("RGBA")

        background.alpha_composite(foreground_canvas)

        result = background.convert("RGB")

        return save_result_image(job_id, result)

    finally:
        images = (
            source,
            working_source,
            segmentation_mask,
            foreground,
            foreground_canvas,
            person_mask,
            background,
            result,
        )

        for image in images:
            if image is not None:
                try:
                    image.close()
                except Exception:
                    pass