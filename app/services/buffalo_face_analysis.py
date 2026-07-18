from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings


@dataclass(frozen=True)
class BuffaloFaceResult:
    raw_face: object
    normalized_embedding: np.ndarray
    box: tuple[int, int, int, int]
    detection_score: float
    sharpness: float


@lru_cache(maxsize=1)
def get_buffalo_face_analyzer() -> FaceAnalysis:
    analyzer = FaceAnalysis(
        name=settings.insightface_swap_model_name,
        root=settings.insightface_root,
        providers=["CPUExecutionProvider"],
    )

    analyzer.prepare(
        ctx_id=-1,
        det_size=(
            settings.face_detection_width,
            settings.face_detection_height,
        ),
        det_thresh=settings.face_detection_threshold,
    )

    return analyzer


def release_buffalo_face_analyzer() -> None:
    get_buffalo_face_analyzer.cache_clear()


def read_bgr_image(image_path: str) -> np.ndarray:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"Could not decode image: {image_path}"
        )

    return image


def _primary_face_score(
    face: object,
    *,
    image_width: int,
    image_height: int,
) -> float:
    bbox = np.asarray(
        face.bbox,
        dtype=np.float32,
    )

    width = max(
        0.0,
        float(bbox[2] - bbox[0]),
    )

    height = max(
        0.0,
        float(bbox[3] - bbox[1]),
    )

    area = width * height

    center_x = float(
        (bbox[0] + bbox[2]) / 2.0
    )

    center_y = float(
        (bbox[1] + bbox[3]) / 2.0
    )

    dx = abs(
        center_x - image_width / 2.0
    ) / max(1.0, image_width / 2.0)

    dy = abs(
        center_y - image_height / 2.0
    ) / max(1.0, image_height / 2.0)

    center_weight = 1.0 - min(
        0.45,
        (dx + dy) * 0.20,
    )

    detection_score = float(
        getattr(face, "det_score", 1.0)
    )

    return (
        area
        * center_weight
        * detection_score
    )


def _face_sharpness(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = box

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return 0.0

    resized = cv2.resize(
        crop,
        (256, 256),
        interpolation=cv2.INTER_LANCZOS4,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def analyze_buffalo_face_bgr(
    image: np.ndarray,
) -> BuffaloFaceResult:
    faces = list(
        get_buffalo_face_analyzer().get(
            image
        )
    )

    if not faces:
        raise ValueError(
            "No face was detected by buffalo_l."
        )

    image_height, image_width = (
        image.shape[:2]
    )

    face = max(
        faces,
        key=lambda item: _primary_face_score(
            item,
            image_width=image_width,
            image_height=image_height,
        ),
    )

    bbox = np.asarray(
        face.bbox,
        dtype=np.float32,
    )

    x1 = max(
        0,
        int(round(float(bbox[0]))),
    )

    y1 = max(
        0,
        int(round(float(bbox[1]))),
    )

    x2 = min(
        image_width,
        int(round(float(bbox[2]))),
    )

    y2 = min(
        image_height,
        int(round(float(bbox[3]))),
    )

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            "buffalo_l returned an invalid "
            "face bounding box."
        )

    embedding = np.asarray(
        face.normed_embedding,
        dtype=np.float32,
    )

    if (
        embedding.ndim != 1
        or embedding.size == 0
        or not np.isfinite(embedding).all()
    ):
        raise ValueError(
            "buffalo_l returned an invalid "
            "face embedding."
        )

    norm = float(
        np.linalg.norm(embedding)
    )

    if norm <= 0:
        raise ValueError(
            "buffalo_l face embedding "
            "has zero norm."
        )

    embedding = embedding / norm

    box = (x1, y1, x2, y2)

    return BuffaloFaceResult(
        raw_face=face,
        normalized_embedding=embedding,
        box=box,
        detection_score=float(
            face.det_score
        ),
        sharpness=_face_sharpness(
            image,
            box,
        ),
    )


def analyze_buffalo_face(
    image_path: str,
) -> BuffaloFaceResult:
    return analyze_buffalo_face_bgr(
        read_bgr_image(image_path)
    )