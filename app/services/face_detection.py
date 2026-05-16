from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import settings


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class FaceDetectionResult:
    image_width: int
    image_height: int
    faces: list[FaceBox]

    @property
    def face_count(self) -> int:
        return len(self.faces)


def detect_faces(image_path: str) -> FaceDetectionResult:
    """
    Detects frontal faces using OpenCV Haar Cascade.

    This is a lightweight MVP-level check:
    - good enough to reject obvious invalid input;
    - later can be replaced or supplemented with InsightFace/MediaPipe.
    """
    path = Path(image_path)

    if not path.exists():
        raise ValueError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    image_height, image_width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

    if not cascade_path.exists():
        raise RuntimeError(f"OpenCV Haar cascade file not found: {cascade_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")

    min_side = min(image_width, image_height)
    min_face_size = max(24, int(min_side * settings.face_min_size_ratio))

    raw_faces = detector.detectMultiScale(
        gray,
        scaleFactor=settings.face_detection_scale_factor,
        minNeighbors=settings.face_detection_min_neighbors,
        minSize=(min_face_size, min_face_size),
    )

    faces = [
        FaceBox(
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
        )
        for x, y, width, height in raw_faces
    ]

    faces.sort(key=lambda face: face.area, reverse=True)

    return FaceDetectionResult(
        image_width=image_width,
        image_height=image_height,
        faces=faces,
    )


def validate_single_face(image_path: str) -> FaceDetectionResult:
    """
    Validates that image contains exactly one reasonably large face.
    Raises ValueError with user-readable messages if validation fails.
    """
    result = detect_faces(image_path)

    if result.face_count == 0:
        raise ValueError(
            "No face detected. Please upload a clear frontal portrait with one employee."
        )

    if result.face_count > 1:
        raise ValueError(
            f"Multiple faces detected: {result.face_count}. "
            "Please upload a portrait with only one employee."
        )

    face = result.faces[0]

    min_side = min(result.image_width, result.image_height)
    min_required_size = int(min_side * settings.face_min_size_ratio)

    if face.width < min_required_size or face.height < min_required_size:
        raise ValueError(
            "Detected face is too small. Please upload a closer portrait photo."
        )

    return result