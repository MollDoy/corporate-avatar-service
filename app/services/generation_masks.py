from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.services.face_detection import FaceBox, detect_faces
from app.services.image_storage import save_job_image


@dataclass(frozen=True)
class GenerationMaskPaths:
    person_mask_path: str
    background_mask_path: str
    face_protection_mask_path: str
    face_restore_mask_path: str
    clothes_mask_path: str
    ai_inpaint_mask_path: str


def _load_grayscale_mask(mask_path: str) -> Image.Image:
    path = Path(mask_path)

    if not path.exists():
        raise ValueError(f"Mask file does not exist: {mask_path}")

    return Image.open(path).convert("L")


def _binarize_mask(mask: Image.Image, threshold: int = 16) -> Image.Image:
    return mask.point(lambda value: 255 if value > threshold else 0).convert("L")

def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    bbox = mask.getbbox()
    if bbox is None:
        return None
    return bbox

def _create_face_protection_mask(
    size: tuple[int, int],
    face: FaceBox,
) -> Image.Image:
    """
    White = protected identity area.

    This mask protects face/head/hair, but should not cover the collar,
    shoulders and upper clothes. Otherwise inpainting cannot create
    a business shirt or suit jacket.
    """
    width, height = size

    expansion_x = int(face.width * 0.55)
    expansion_top = int(face.height * 0.80)
    expansion_bottom = int(face.height * 0.25)

    left = max(0, face.x - expansion_x)
    top = max(0, face.y - expansion_top)
    right = min(width, face.x + face.width + expansion_x)
    bottom = min(height, face.y + face.height + expansion_bottom)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse((left, top, right, bottom), fill=255)

    return mask.filter(ImageFilter.GaussianBlur(radius=2))

def _create_face_restore_mask(
    size: tuple[int, int],
    face: FaceBox,
) -> Image.Image:
    """
    White = face area restored from pre-AI result after inpainting.

    This mask must be smaller than face_protection_mask.
    It should restore identity, eyes, nose, mouth and most of the face,
    but should not overwrite shirt collar, tie or upper clothing.
    """
    width, height = size

    expansion_x = int(face.width * 0.22)
    expansion_top = int(face.height * 0.32)
    expansion_bottom = int(face.height * 0.06)

    left = max(0, face.x - expansion_x)
    top = max(0, face.y - expansion_top)
    right = min(width, face.x + face.width + expansion_x)
    bottom = min(height, face.y + face.height + expansion_bottom)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    draw.ellipse((left, top, right, bottom), fill=255)

    return mask.filter(ImageFilter.GaussianBlur(radius=0.8))

def _create_clothes_mask(
    person_mask: Image.Image,
    face: FaceBox,
    face_protection_mask: Image.Image,
) -> Image.Image:
    """
    White = area for business clothing inpainting.

    The mask includes upper body, shoulders and torso,
    but removes protected face/head area.
    """
    width, height = person_mask.size

    y_start = int(face.y + face.height * 0.45)
    y_start = max(0, min(height, y_start))

    upper_body_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(upper_body_mask)
    draw.rectangle((0, y_start, width, height), fill=255)

    clothes_mask = ImageChops.multiply(person_mask, upper_body_mask)

    inverted_face_protection = ImageChops.invert(face_protection_mask)
    clothes_mask = ImageChops.multiply(clothes_mask, inverted_face_protection)

    clothes_mask = clothes_mask.filter(ImageFilter.MaxFilter(size=7))
    clothes_mask = clothes_mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    return _binarize_mask(clothes_mask, threshold=24)


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

    detection = detect_faces(result_image_path)

    if detection.face_count == 0:
        raise ValueError(
            "No face detected on processed avatar image. "
            "Could not create generation masks."
        )

    face = detection.faces[0]

    face_protection_mask = _create_face_protection_mask(
        size=person_mask.size,
        face=face,
    )

    face_restore_mask = _create_face_restore_mask(
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
    face_restore_mask_path = save_job_image(
        job_id,
        face_restore_mask,
        "face_restore_mask.png",
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
        face_restore_mask_path=face_restore_mask_path,
        clothes_mask_path=clothes_mask_path,
        ai_inpaint_mask_path=ai_inpaint_mask_path,
    )