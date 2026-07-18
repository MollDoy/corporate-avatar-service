from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.utils import face_align

from app.core.config import settings
from app.services.buffalo_face_analysis import (
    analyze_buffalo_face_bgr,
    read_bgr_image,
)
from app.services.portrait_generation_client import (
    GeneratedCandidate,
)


@lru_cache(maxsize=1)
def get_face_swapper() -> object:
    model_path = (
        Path(settings.insightface_root)
        / "models"
        / settings.inswapper_model_name
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            "InSwapper model does not exist: "
            f"{model_path}"
        )

    swapper = insightface.model_zoo.get_model(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    if swapper is None:
        raise RuntimeError(
            "InsightFace could not initialize "
            "the InSwapper model: "
            f"{model_path}"
        )

    return swapper


def release_face_swapper() -> None:
    get_face_swapper.cache_clear()


def _validate_pixel_boost_size(
    *,
    boost_size: int,
    model_width: int,
    model_height: int,
) -> int:
    if boost_size <= 0:
        raise ValueError(
            "FACE_SWAP_PIXEL_BOOST_SIZE must be positive."
        )

    if model_width != model_height:
        raise ValueError(
            "Only square InSwapper inputs are supported."
        )

    if boost_size < model_width:
        raise ValueError(
            "Pixel boost size cannot be smaller than "
            f"the model input size ({model_width})."
        )

    if boost_size % model_width != 0:
        raise ValueError(
            "Pixel boost size must be divisible by "
            f"the InSwapper input size ({model_width})."
        )

    return boost_size // model_width


def _implode_pixel_boost(
    crop: np.ndarray,
    *,
    pixel_boost_total: int,
    model_width: int,
    model_height: int,
) -> list[np.ndarray]:
    expected_height = (
        model_height * pixel_boost_total
    )
    expected_width = (
        model_width * pixel_boost_total
    )

    if crop.shape[:2] != (
        expected_height,
        expected_width,
    ):
        raise ValueError(
            "Unexpected aligned face size for pixel boost: "
            f"{crop.shape[:2]}."
        )

    frames = crop.reshape(
        model_height,
        pixel_boost_total,
        model_width,
        pixel_boost_total,
        3,
    )

    frames = frames.transpose(
        1,
        3,
        0,
        2,
        4,
    )

    frames = frames.reshape(
        pixel_boost_total ** 2,
        model_height,
        model_width,
        3,
    )

    return [
        np.ascontiguousarray(frame)
        for frame in frames
    ]


def _explode_pixel_boost(
    frames: list[np.ndarray],
    *,
    pixel_boost_total: int,
    model_width: int,
    model_height: int,
    boost_size: int,
) -> np.ndarray:
    expected_count = pixel_boost_total ** 2

    if len(frames) != expected_count:
        raise ValueError(
            "Unexpected number of pixel-boost frames: "
            f"{len(frames)}; expected {expected_count}."
        )

    crop = np.stack(frames).reshape(
        pixel_boost_total,
        pixel_boost_total,
        model_height,
        model_width,
        3,
    )

    crop = crop.transpose(
        2,
        0,
        3,
        1,
        4,
    )

    return crop.reshape(
        boost_size,
        boost_size,
        3,
    )


def _prepare_source_latent(
    swapper: object,
    source_face: object,
) -> np.ndarray:
    source_embedding = np.asarray(
        source_face.normed_embedding,
        dtype=np.float32,
    ).reshape(1, -1)

    latent = np.dot(
        source_embedding,
        swapper.emap,
    ).astype(np.float32)

    latent_norm = float(
        np.linalg.norm(latent)
    )

    if latent_norm <= 0:
        raise ValueError(
            "InSwapper source latent has zero norm."
        )

    return latent / latent_norm


def _run_inswapper_patch(
    *,
    swapper: object,
    patch: np.ndarray,
    latent: np.ndarray,
    model_width: int,
    model_height: int,
) -> np.ndarray:
    blob = cv2.dnn.blobFromImage(
        patch,
        scalefactor=1.0 / 255.0,
        size=(model_width, model_height),
        mean=(0.0, 0.0, 0.0),
        swapRB=True,
    )

    prediction = swapper.session.run(
        swapper.output_names,
        {
            swapper.input_names[0]: blob,
            swapper.input_names[1]: latent,
        },
    )[0]

    prediction = prediction.transpose(
        0,
        2,
        3,
        1,
    )[0]

    return np.clip(
        prediction[:, :, ::-1] * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)


def _transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    return cv2.transform(
        points.reshape(1, -1, 2),
        matrix,
    ).reshape(-1, 2)


def _create_ellipse_mask(
    *,
    aligned_kps: np.ndarray,
    boost_size: int,
) -> np.ndarray:
    left_eye = aligned_kps[0]
    right_eye = aligned_kps[1]
    nose = aligned_kps[2]
    left_mouth = aligned_kps[3]
    right_mouth = aligned_kps[4]

    eye_center = (
        left_eye + right_eye
    ) / 2.0

    mouth_center = (
        left_mouth + right_mouth
    ) / 2.0

    eye_distance = float(
        np.linalg.norm(
            right_eye - left_eye
        )
    )

    center = (
        eye_center * 0.42
        + nose * 0.18
        + mouth_center * 0.40
    )

    axis_x = max(
        1,
        int(round(eye_distance * 1.28)),
    )

    axis_y = max(
        1,
        int(round(eye_distance * 1.70)),
    )

    mask = np.zeros(
        (boost_size, boost_size),
        dtype=np.float32,
    )

    cv2.ellipse(
        mask,
        (
            int(round(float(center[0]))),
            int(round(float(center[1]))),
        ),
        (axis_x, axis_y),
        0.0,
        0.0,
        360.0,
        1.0,
        thickness=-1,
    )

    return mask


def _create_landmark_mask(
    *,
    target_face: object,
    affine_matrix: np.ndarray,
    boost_size: int,
) -> np.ndarray:
    aligned_kps = _transform_points(
        np.asarray(
            target_face.kps,
            dtype=np.float32,
        ),
        affine_matrix,
    )

    landmarks_106 = getattr(
        target_face,
        "landmark_2d_106",
        None,
    )

    mask: np.ndarray

    if landmarks_106 is not None:
        landmarks_array = np.asarray(
            landmarks_106,
            dtype=np.float32,
        )

        if (
            landmarks_array.ndim == 2
            and landmarks_array.shape[1] == 2
            and landmarks_array.shape[0] >= 20
        ):
            aligned_landmarks = _transform_points(
                landmarks_array,
                affine_matrix,
            )

            hull = cv2.convexHull(
                np.round(
                    aligned_landmarks
                ).astype(np.int32)
            )

            mask = np.zeros(
                (boost_size, boost_size),
                dtype=np.float32,
            )

            cv2.fillConvexPoly(
                mask,
                hull,
                1.0,
            )

        else:
            mask = _create_ellipse_mask(
                aligned_kps=aligned_kps,
                boost_size=boost_size,
            )

    else:
        mask = _create_ellipse_mask(
            aligned_kps=aligned_kps,
            boost_size=boost_size,
        )

    dilation_size = max(
        1,
        int(round(
            boost_size
            * settings.face_swap_mask_dilation_ratio
        )),
    )

    if dilation_size % 2 == 0:
        dilation_size += 1

    if dilation_size > 1:
        mask = cv2.dilate(
            mask,
            np.ones(
                (dilation_size, dilation_size),
                dtype=np.uint8,
            ),
            iterations=1,
        )

    blur_size = max(
        3,
        int(round(
            boost_size
            * settings.face_swap_mask_blur_ratio
        )),
    )

    if blur_size % 2 == 0:
        blur_size += 1

    mask = cv2.GaussianBlur(
        mask,
        (blur_size, blur_size),
        sigmaX=0,
    )

    return np.clip(
        mask,
        0.0,
        1.0,
    )

def _match_face_color(
    *,
    swapped_crop: np.ndarray,
    target_crop: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    strength = float(
        np.clip(
            settings.face_swap_color_match_strength,
            0.0,
            1.0,
        )
    )

    if strength <= 0:
        return swapped_crop

    selected = mask > 0.45

    if int(selected.sum()) < 64:
        return swapped_crop

    swapped_lab = cv2.cvtColor(
        swapped_crop,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    target_lab = cv2.cvtColor(
        target_crop,
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)

    corrected = swapped_lab.copy()

    for channel in range(3):
        source_values = swapped_lab[
            :, :, channel
        ][selected]

        target_values = target_lab[
            :, :, channel
        ][selected]

        source_mean = float(
            source_values.mean()
        )
        source_std = float(
            source_values.std()
        )
        target_mean = float(
            target_values.mean()
        )
        target_std = float(
            target_values.std()
        )

        scale = target_std / max(
            source_std,
            1.0,
        )

        scale = float(
            np.clip(scale, 0.65, 1.55)
        )

        corrected[:, :, channel] = (
            swapped_lab[:, :, channel]
            - source_mean
        ) * scale + target_mean

    corrected = np.clip(
        corrected,
        0.0,
        255.0,
    ).astype(np.uint8)

    corrected_bgr = cv2.cvtColor(
        corrected,
        cv2.COLOR_LAB2BGR,
    )

    return np.clip(
        swapped_crop.astype(np.float32)
        * (1.0 - strength)
        + corrected_bgr.astype(np.float32)
        * strength,
        0.0,
        255.0,
    ).astype(np.uint8)


def _paste_back(
    *,
    target_image: np.ndarray,
    swapped_crop: np.ndarray,
    crop_mask: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    image_height, image_width = (
        target_image.shape[:2]
    )

    inverse_matrix = cv2.invertAffineTransform(
        affine_matrix
    )

    restored_face = cv2.warpAffine(
        swapped_crop,
        inverse_matrix,
        (image_width, image_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    restored_mask = cv2.warpAffine(
        crop_mask,
        inverse_matrix,
        (image_width, image_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    alpha = np.clip(
        restored_mask,
        0.0,
        1.0,
    )[..., None]

    return np.clip(
        restored_face.astype(np.float32)
        * alpha
        + target_image.astype(np.float32)
        * (1.0 - alpha),
        0.0,
        255.0,
    ).astype(np.uint8)


def _swap_single_candidate(
    *,
    source_face: object,
    candidate: GeneratedCandidate,
) -> GeneratedCandidate:
    target_image = read_bgr_image(
        candidate.image_path
    )

    target_analysis = analyze_buffalo_face_bgr(
        target_image
    )

    target_face = target_analysis.raw_face
    swapper = get_face_swapper()

    model_width, model_height = (
        int(swapper.input_size[0]),
        int(swapper.input_size[1]),
    )

    boost_size = int(
        settings.face_swap_pixel_boost_size
    )

    pixel_boost_total = (
        _validate_pixel_boost_size(
            boost_size=boost_size,
            model_width=model_width,
            model_height=model_height,
        )
    )

    target_crop, affine_matrix = (
        face_align.norm_crop2(
            target_image,
            target_face.kps,
            boost_size,
        )
    )

    source_latent = _prepare_source_latent(
        swapper,
        source_face,
    )

    boosted_frames = _implode_pixel_boost(
        target_crop,
        pixel_boost_total=(
            pixel_boost_total
        ),
        model_width=model_width,
        model_height=model_height,
    )

    swapped_frames = [
        _run_inswapper_patch(
            swapper=swapper,
            patch=frame,
            latent=source_latent,
            model_width=model_width,
            model_height=model_height,
        )
        for frame in boosted_frames
    ]

    swapped_crop = _explode_pixel_boost(
        swapped_frames,
        pixel_boost_total=(
            pixel_boost_total
        ),
        model_width=model_width,
        model_height=model_height,
        boost_size=boost_size,
    )

    crop_mask = _create_landmark_mask(
        target_face=target_face,
        affine_matrix=affine_matrix,
        boost_size=boost_size,
    )

    swapped_crop = _match_face_color(
        swapped_crop=swapped_crop,
        target_crop=target_crop,
        mask=crop_mask,
    )

    swapped_image = _paste_back(
        target_image=target_image,
        swapped_crop=swapped_crop,
        crop_mask=crop_mask,
        affine_matrix=affine_matrix,
    )

    candidate_path = Path(
        candidate.image_path
    )

    output_path = candidate_path.with_name(
        candidate_path.stem
        + "_swapped.png"
    )

    if not cv2.imwrite(
        str(output_path),
        swapped_image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    ):
        raise RuntimeError(
            "Could not save face-swapped candidate: "
            f"{output_path}"
        )

    return GeneratedCandidate(
        image_path=str(output_path),
        seed=candidate.seed,
        attempt_number=(
            candidate.attempt_number
        ),
    )


def swap_faces_in_candidates(
    *,
    source_image_path: str,
    candidates: list[GeneratedCandidate],
) -> list[GeneratedCandidate]:
    if not settings.face_swap_enabled:
        return []

    source_image = read_bgr_image(
        source_image_path
    )

    source_face = analyze_buffalo_face_bgr(
        source_image
    ).raw_face

    swapped_candidates: list[
        GeneratedCandidate
    ] = []

    for candidate in candidates:
        try:
            swapped_candidates.append(
                _swap_single_candidate(
                    source_face=source_face,
                    candidate=candidate,
                )
            )

        except Exception as exc:
            print(
                "[worker] pixel-boost face swap "
                f"failed for {candidate.image_path}: "
                f"{exc}",
                flush=True,
            )

    return swapped_candidates