from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.services.face_detection import FaceBox, validate_single_face


@dataclass(frozen=True)
class FaceSimilarityResult:
    score: float
    source_face: FaceBox
    result_face: FaceBox


def _read_image_bgr(image_path: str) -> np.ndarray:
    path = Path(image_path)

    if not path.exists():
        raise ValueError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    return image


def _crop_face_with_padding(
    image: np.ndarray,
    face: FaceBox,
    padding_ratio: float = 0.35,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]

    padding_x = int(face.width * padding_ratio)
    padding_y = int(face.height * padding_ratio)

    x1 = max(face.x - padding_x, 0)
    y1 = max(face.y - padding_y, 0)
    x2 = min(face.x + face.width + padding_x, image_width)
    y2 = min(face.y + face.height + padding_y, image_height)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Face crop is empty")

    return crop


def _prepare_face_crop(face_crop: np.ndarray) -> np.ndarray:
    crop_size = settings.face_similarity_crop_size

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    resized = cv2.resize(
        gray,
        (crop_size, crop_size),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype("float32") / 255.0


def _calculate_pixel_similarity(source_face: np.ndarray, result_face: np.ndarray) -> float:
    """
    Simple MVP similarity:
    1. Normalize both face crops to same size.
    2. Calculate average absolute difference.
    3. Convert difference to similarity in [0, 1].

    This is not a biometric embedding. Later it can be replaced with InsightFace.
    """
    difference = np.mean(np.abs(source_face - result_face))
    similarity = 1.0 - float(difference)

    return max(0.0, min(1.0, similarity))


def _calculate_histogram_similarity(
    source_crop_bgr: np.ndarray,
    result_crop_bgr: np.ndarray,
) -> float:
    source_hsv = cv2.cvtColor(source_crop_bgr, cv2.COLOR_BGR2HSV)
    result_hsv = cv2.cvtColor(result_crop_bgr, cv2.COLOR_BGR2HSV)

    source_hist = cv2.calcHist(
        [source_hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )
    result_hist = cv2.calcHist(
        [result_hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    cv2.normalize(source_hist, source_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(result_hist, result_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    correlation = cv2.compareHist(source_hist, result_hist, cv2.HISTCMP_CORREL)

    # HISTCMP_CORREL can return [-1, 1]. Convert to [0, 1].
    similarity = (float(correlation) + 1.0) / 2.0

    return max(0.0, min(1.0, similarity))


def calculate_face_similarity(
    source_image_path: str,
    result_image_path: str,
) -> FaceSimilarityResult:
    source_detection = validate_single_face(source_image_path)
    result_detection = validate_single_face(result_image_path)

    source_face = source_detection.faces[0]
    result_face = result_detection.faces[0]

    source_image = _read_image_bgr(source_image_path)
    result_image = _read_image_bgr(result_image_path)

    source_crop_bgr = _crop_face_with_padding(source_image, source_face)
    result_crop_bgr = _crop_face_with_padding(result_image, result_face)

    source_prepared = _prepare_face_crop(source_crop_bgr)
    result_prepared = _prepare_face_crop(result_crop_bgr)

    pixel_similarity = _calculate_pixel_similarity(
        source_prepared,
        result_prepared,
    )
    histogram_similarity = _calculate_histogram_similarity(
        source_crop_bgr,
        result_crop_bgr,
    )

    score = 0.75 * pixel_similarity + 0.25 * histogram_similarity
    score = round(max(0.0, min(1.0, score)), 4)

    return FaceSimilarityResult(
        score=score,
        source_face=source_face,
        result_face=result_face,
    )