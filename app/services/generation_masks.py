from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.services.face_detection import FaceBox, validate_single_face
from app.services.image_storage import save_job_image


@dataclass(frozen=True)
class GenerationMaskPaths:
    person_mask_path: str
    background_mask_path: str
    face_protection_mask_path: str
    clothes_mask_path: str
    ai_inpaint_mask_path: str


def _load_grayscale_mask(mask_path: str) -> Image.Image:
    path = Path(mask_path)

    if not path.exists():
        raise ValueError(f"Mask file does not exist: {mask_path}")

    return Image.open(path).convert("L")


def _binarize_mask(mask: Image.Image, threshold: int = 16) -> Image.Image:
    return mask.point(lambda value: 255 if value > threshold else 0).convert("L")


def _create_face_protection_mask(
    size: tuple[int, int],
    face: FaceBox,
) -> Image.Image:
    """
    White = protected area.

    We protect not only the strict face rectangle, but also extra head/hair area.
    This is important before diffusion inpainting: the generator must not redraw identity.
    """
    width, height = size

    expansion_x = int(face.width * 0.65)
    expansion_top = int(face.height * 0.85)
    expansion_bottom = int(face.height * 0.65)

    left = max(0, face.x - expansion_x)
    top = max(0, face.y - expansion_top)
    right = min(width, face.x + face.width + expansion_x)
    bottom = min(height, face.y + face.height + expansion_bottom)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse((left, top, right, bottom), fill=255)

    # Slight blur makes future inpainting transition softer.
    return mask.filter(ImageFilter.GaussianBlur(radius=3))


def _create_clothes_mask(
    person_mask: Image.Image,
    face: FaceBox,
    face_protection_mask: Image.Image,
) -> Image.Image:
    """
    White = likely clothes/body area.

    We take the person mask and keep mostly the lower part of the portrait,
    excluding protected face/head area.
    """
    width, height = person_mask.size

    y_start = int(face.y + face.height * 0.85)
    y_start = max(0, min(height, y_start))

    lower_body_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(lower_body_mask)
    draw.rectangle((0, y_start, width, height), fill=255)

    clothes_mask = ImageChops.multiply(person_mask, lower_body_mask)

    inverted_face_protection = ImageChops.invert(face_protection_mask)
    clothes_mask = ImageChops.multiply(clothes_mask, inverted_face_protection)

    return _binarize_mask(clothes_mask, threshold=32)


def create_generation_masks(
    job_id: str,
    result_image_path: str,
    person_mask_path: str,
) -> GenerationMaskPaths:
    """
    Creates technical masks for future diffusion/inpainting generation.

    White on ai_inpaint_mask means:
    - this area may be changed by the generator.

    Black means:
    - keep unchanged, especially face/head/identity area.
    """
    person_mask = _load_grayscale_mask(person_mask_path)
    person_mask = _binarize_mask(person_mask)

    detection = validate_single_face(result_image_path)
    face = detection.faces[0]

    face_protection_mask = _create_face_protection_mask(
        size=person_mask.size,
        face=face,
    )

    background_mask = ImageChops.invert(person_mask)
    background_mask = _binarize_mask(background_mask)

    clothes_mask = _create_clothes_mask(
        person_mask=person_mask,
        face=face,
        face_protection_mask=face_protection_mask,
    )

    ai_inpaint_mask = ImageChops.lighter(background_mask, clothes_mask)

    inverted_face_protection = ImageChops.invert(face_protection_mask)
    ai_inpaint_mask = ImageChops.multiply(ai_inpaint_mask, inverted_face_protection)
    ai_inpaint_mask = _binarize_mask(ai_inpaint_mask, threshold=32)

    background_mask_path = save_job_image(
        job_id,
        background_mask,
        "background_mask.png",
    )
    face_protection_mask_path = save_job_image(
        job_id,
        face_protection_mask,
        "face_protection_mask.png",
    )
    clothes_mask_path = save_job_image(
        job_id,
        clothes_mask,
        "clothes_mask.png",
    )
    ai_inpaint_mask_path = save_job_image(
        job_id,
        ai_inpaint_mask,
        "ai_inpaint_mask.png",
    )

    return GenerationMaskPaths(
        person_mask_path=person_mask_path,
        background_mask_path=background_mask_path,
        face_protection_mask_path=face_protection_mask_path,
        clothes_mask_path=clothes_mask_path,
        ai_inpaint_mask_path=ai_inpaint_mask_path,
    )