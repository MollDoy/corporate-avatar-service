from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.core.config import settings


@dataclass(frozen=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(
            0,
            self.x2 - self.x1,
        )

    @property
    def height(self) -> int:
        return max(
            0,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> int:
        return (
            self.width
            * self.height
        )


@dataclass(frozen=True)
class AnalyzedFace:
    box: FaceBox
    detection_score: float
    normalized_embedding: np.ndarray
    landmarks: np.ndarray
    area_ratio: float
    gender: int | None
    age: int | None


@dataclass(frozen=True)
class FaceAnalysisResult:
    image_width: int
    image_height: int
    primary_face: AnalyzedFace
    detected_face_count: int
    warnings: tuple[str, ...]


@lru_cache(maxsize=1)
def get_face_analyzer() -> FaceAnalysis:
    analyzer = FaceAnalysis(
        name=(
            settings
            .insightface_model_name
        ),
        root=settings.insightface_root,
        providers=[
            "CPUExecutionProvider",
        ],
    )

    analyzer.prepare(
        ctx_id=-1,
        det_size=(
            settings
            .face_detection_width,
            settings
            .face_detection_height,
        ),
        det_thresh=(
            settings
            .face_detection_threshold
        ),
    )

    return analyzer


def release_face_analyzer() -> None:
    get_face_analyzer.cache_clear()


def _read_image(
    image_path: str,
) -> np.ndarray:
    path = Path(image_path)

    if not path.is_file():
        raise ValueError(
            "Image file does not exist: "
            f"{image_path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode image: "
            f"{image_path}"
        )

    return image


def _enhance_image(
    image: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    lightness, channel_a, channel_b = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    lightness = clahe.apply(
        lightness
    )

    enhanced = cv2.merge(
        (
            lightness,
            channel_a,
            channel_b,
        )
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=1.0,
    )

    return cv2.addWeighted(
        enhanced,
        1.20,
        blurred,
        -0.20,
        0,
    )


def _convert_face(
    raw_face: object,
    *,
    image_width: int,
    image_height: int,
) -> AnalyzedFace:
    bbox = np.asarray(
        raw_face.bbox,
        dtype=np.float32,
    )

    if bbox.shape != (4,):
        raise ValueError(
            "Invalid face bounding box."
        )

    x1 = max(
        0,
        round(float(bbox[0])),
    )
    y1 = max(
        0,
        round(float(bbox[1])),
    )
    x2 = min(
        image_width,
        round(float(bbox[2])),
    )
    y2 = min(
        image_height,
        round(float(bbox[3])),
    )

    box = FaceBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )

    embedding = np.asarray(
        raw_face.normed_embedding,
        dtype=np.float32,
    )

    if (
        embedding.ndim != 1
        or embedding.size == 0
    ):
        raise ValueError(
            "Invalid face embedding."
        )

    landmarks = np.asarray(
        raw_face.kps,
        dtype=np.float32,
    )

    if landmarks.shape != (5, 2):
        raise ValueError(
            "Invalid face landmarks."
        )

    image_area = max(
        1,
        image_width * image_height,
    )

    raw_gender = getattr(
        raw_face,
        "gender",
        None,
    )

    gender = (
        int(raw_gender)
        if raw_gender is not None
        else None
    )

    raw_age = getattr(
        raw_face,
        "age",
        None,
    )

    age = (
        int(raw_age)
        if raw_age is not None
        else None
    )

    return AnalyzedFace(
        box=box,
        detection_score=float(
            raw_face.det_score
        ),
        normalized_embedding=(
            embedding
        ),
        landmarks=landmarks,
        area_ratio=(
            box.area / image_area
        ),
        gender=gender,
        age=age,
    )


def _primary_score(
    face: AnalyzedFace,
    *,
    image_width: int,
    image_height: int,
) -> float:
    center_x = (
        face.box.x1
        + face.box.x2
    ) / 2

    center_y = (
        face.box.y1
        + face.box.y2
    ) / 2

    dx = abs(
        center_x
        - image_width / 2
    ) / max(
        1,
        image_width / 2,
    )

    dy = abs(
        center_y
        - image_height / 2
    ) / max(
        1,
        image_height / 2,
    )

    center_weight = (
        1.0
        - min(
            0.30,
            (dx + dy) * 0.15,
        )
    )

    area_weight = np.sqrt(
        max(
            face.area_ratio,
            0.000001,
        )
    )

    return float(
        face.detection_score
        * area_weight
        * center_weight
    )


def _quality_warnings(
    image: np.ndarray,
    face: AnalyzedFace,
) -> list[str]:
    warnings: list[str] = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(
        gray.mean()
    )

    if blur_score < 35:
        warnings.append(
            "source_image_blurred"
        )

    if brightness < 45:
        warnings.append(
            "source_image_dark"
        )

    if brightness > 220:
        warnings.append(
            "source_image_overexposed"
        )

    margin_x = image.shape[1] * 0.01
    margin_y = image.shape[0] * 0.01

    if (
        face.box.x1 <= margin_x
        or face.box.y1 <= margin_y
        or face.box.x2
        >= image.shape[1] - margin_x
        or face.box.y2
        >= image.shape[0] - margin_y
    ):
        warnings.append(
            "source_face_near_border"
        )

    return warnings


def analyze_faces(
    image_path: str,
    *,
    allow_enhancement: bool = True,
) -> FaceAnalysisResult:
    image = _read_image(
        image_path
    )

    height, width = image.shape[:2]

    analyzer = get_face_analyzer()

    raw_faces = list(
        analyzer.get(image)
    )

    enhancement_used = False

    if (
        not raw_faces
        and allow_enhancement
    ):
        enhanced_image = (
            _enhance_image(image)
        )

        raw_faces = list(
            analyzer.get(
                enhanced_image
            )
        )

        enhancement_used = bool(
            raw_faces
        )

    if not raw_faces:
        raise ValueError(
            "No usable face detected."
        )

    faces = [
        _convert_face(
            raw_face,
            image_width=width,
            image_height=height,
        )
        for raw_face in raw_faces
    ]

    faces = [
        face
        for face in faces
        if (
            face.box.width > 0
            and face.box.height > 0
        )
    ]

    if not faces:
        raise ValueError(
            "Face detector returned "
            "invalid face boxes."
        )

    faces.sort(
        key=lambda face: _primary_score(
            face,
            image_width=width,
            image_height=height,
        ),
        reverse=True,
    )

    primary = faces[0]

    if (
        primary.area_ratio
        < settings.face_min_area_ratio
    ):
        raise ValueError(
            "Detected face is too small. "
            f"Area ratio="
            f"{primary.area_ratio:.4f}; "
            f"minimum="
            f"{settings.face_min_area_ratio:.4f}."
        )

    significant_secondary = [
        face
        for face in faces[1:]
        if (
            face.detection_score
            >= settings
            .secondary_face_min_score
            and face.box.area
            >= (
                primary.box.area
                * settings
                .secondary_face_min_area_ratio
            )
        )
    ]

    if significant_secondary:
        raise ValueError(
            "Multiple significant people "
            "were detected."
        )

    warnings = _quality_warnings(
        image,
        primary,
    )

    if enhancement_used:
        warnings.append(
            "face_detected_after_enhancement"
        )

    return FaceAnalysisResult(
        image_width=width,
        image_height=height,
        primary_face=primary,
        detected_face_count=(
            len(faces)
        ),
        warnings=tuple(warnings),
    )