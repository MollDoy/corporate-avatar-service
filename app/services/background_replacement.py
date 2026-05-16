import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from app.core.config import settings
from app.services.image_storage import save_result_image


_sessions: dict[str, Any] = {}


def _configure_rembg_model_dir() -> None:
    """
    rembg reads U2NET_HOME from environment variables.
    We set it explicitly so downloaded models are stored in /app/models/rembg.
    """
    os.environ["U2NET_HOME"] = settings.u2net_home
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


def _create_gradient_background(width: int, height: int) -> Image.Image:
    """
    Creates a simple corporate light gradient background.
    Later we can replace colors with brand colors from DB or .env.
    """
    top_color = (245, 248, 255)
    bottom_color = (218, 229, 245)

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


def _resize_foreground(foreground: Image.Image, output_size: int) -> Image.Image:
    """
    Keeps the whole person visible and scales the foreground into the target canvas.
    """
    result = foreground.copy()
    result.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    return result


def generate_basic_corporate_avatar(job_id: str, source_image_path: str) -> str:
    """
    Step 4 MVP:
    - takes source image
    - removes background
    - places person on a clean corporate gradient
    - saves result.png
    """
    from rembg import remove

    output_size = settings.avatar_output_size

    source = Image.open(source_image_path).convert("RGB")

    session = _get_rembg_session()

    foreground = remove(
        source,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    ).convert("RGBA")

    foreground = _resize_foreground(foreground, output_size)

    background = _create_gradient_background(output_size, output_size).convert("RGBA")
    background = background.filter(ImageFilter.GaussianBlur(radius=0.3))

    x = (output_size - foreground.width) // 2
    y = output_size - foreground.height

    background.alpha_composite(foreground, dest=(x, y))

    result = background.convert("RGB")

    return save_result_image(job_id, result)