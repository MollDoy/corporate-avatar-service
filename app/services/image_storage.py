import base64
import binascii
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, status
from PIL import Image

from app.core.config import settings


def _remove_base64_prefix(image_base64: str) -> str:
    """
    Supports both pure base64 and data URL format:
    data:image/jpeg;base64,...
    """
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        return image_base64.split(",", 1)[1]

    return image_base64


def decode_image_from_base64(image_base64: str) -> Image.Image:
    cleaned_base64 = _remove_base64_prefix(image_base64)

    try:
        image_bytes = base64.b64decode(cleaned_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image",
        ) from exc

    max_bytes = settings.max_image_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is too large. Max size is {settings.max_image_mb} MB",
        )

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image",
        ) from exc

    return image.convert("RGB")


def save_source_image(job_id: str, image: Image.Image) -> str:
    storage_dir = Path(settings.storage_dir)
    job_dir = storage_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    image_path = job_dir / "source.png"
    image.save(image_path, format="PNG")

    return str(image_path)