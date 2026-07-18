from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceConditioningAssets:
    identity_reference_path: str
    face_embedding_path: str

    @property
    def identity_reference_name(self) -> str:
        return Path(self.identity_reference_path).name

    @property
    def face_embedding_name(self) -> str:
        return Path(self.face_embedding_path).name


def _pad_square_crop(
    image: np.ndarray,
    *,
    face_box: tuple[int, int, int, int],
    output_size: int,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = face_box

    face_width = max(1.0, float(x2 - x1))
    face_height = max(1.0, float(y2 - y1))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # The reference is intentionally face-centric. It keeps hair, ears,
    # jaw and some neck, but removes most of the source background and body.
    crop_side = max(
        face_width * 2.35,
        face_height * 2.25,
    )

    crop_center_y = center_y + face_height * 0.12

    left = int(round(center_x - crop_side / 2.0))
    top = int(round(crop_center_y - crop_side / 2.0))
    right = int(round(left + crop_side))
    bottom = int(round(top + crop_side))

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image_width)
    pad_bottom = max(0, bottom - image_height)

    clipped_left = max(0, left)
    clipped_top = max(0, top)
    clipped_right = min(image_width, right)
    clipped_bottom = min(image_height, bottom)

    crop = image[
        clipped_top:clipped_bottom,
        clipped_left:clipped_right,
    ]

    if crop.size == 0:
        raise ValueError(
            "Could not create identity reference crop from source image."
        )

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        border_color = np.median(
            image.reshape(-1, 3),
            axis=0,
        )

        border_value = tuple(
            int(round(float(channel)))
            for channel in border_color
        )

        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_value,
        )

    interpolation = (
        cv2.INTER_CUBIC
        if max(crop.shape[:2]) < output_size
        else cv2.INTER_AREA
    )

    return cv2.resize(
        crop,
        (output_size, output_size),
        interpolation=interpolation,
    )


def create_face_conditioning_assets(
    *,
    source_image_path: str,
    face_box: tuple[int, int, int, int],
    normalized_embedding: np.ndarray,
    job_directory: Path,
    reference_size: int = 512,
) -> FaceConditioningAssets:
    image = cv2.imread(
        source_image_path,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode source image: "
            f"{source_image_path}"
        )

    embedding = np.asarray(
        normalized_embedding,
        dtype=np.float32,
    )

    if embedding.shape != (512,):
        raise ValueError(
            "Invalid face embedding shape: "
            f"{embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Face embedding contains NaN or infinity."
        )

    embedding_norm = float(np.linalg.norm(embedding))

    if embedding_norm <= 0:
        raise ValueError(
            "Face embedding has zero norm."
        )

    embedding = embedding / embedding_norm

    identity_reference = _pad_square_crop(
        image,
        face_box=face_box,
        output_size=reference_size,
    )

    job_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    identity_reference_path = (
        job_directory / "identity_reference.png"
    )

    face_embedding_path = (
        job_directory / "face_embedding.npy"
    )

    if not cv2.imwrite(
        str(identity_reference_path),
        identity_reference,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    ):
        raise RuntimeError(
            "Could not save identity reference: "
            f"{identity_reference_path}"
        )

    np.save(
        face_embedding_path,
        embedding,
        allow_pickle=False,
    )

    return FaceConditioningAssets(
        identity_reference_path=str(identity_reference_path),
        face_embedding_path=str(face_embedding_path),
    )