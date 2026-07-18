from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.services.buffalo_face_analysis import analyze_buffalo_face
from app.services.face_analysis import analyze_faces
from app.services.face_similarity import cosine_similarity
from app.services.portrait_generation_client import GeneratedCandidate


@dataclass(frozen=True)
class RankedCandidate:
    candidate: GeneratedCandidate

    antelope_similarity: float
    buffalo_similarity: float
    conservative_identity: float
    mean_identity: float

    face_sharpness: float
    detected_face_count: int

    face_area_ratio: float
    face_center_x_ratio: float
    face_center_y_ratio: float
    headroom_ratio: float
    torso_space_ratio: float
    framing_score: float

    passed_antelope: bool
    passed_buffalo: bool
    passed_mean: bool

    is_swapped: bool

    @property
    def passed(self) -> bool:
        return (
            self.passed_antelope
            and self.passed_buffalo
            and self.passed_mean
            and self.detected_face_count == 1
        )


def candidate_key(candidate: GeneratedCandidate) -> tuple[int, int]:
    return candidate.seed, candidate.attempt_number


def _is_swapped_candidate(candidate: GeneratedCandidate) -> bool:
    return Path(candidate.image_path).stem.endswith("_swapped")


def _trapezoid_score(
    value: float,
    *,
    hard_low: float,
    ideal_low: float,
    ideal_high: float,
    hard_high: float,
) -> float:
    if value <= hard_low or value >= hard_high:
        return 0.0

    if ideal_low <= value <= ideal_high:
        return 1.0

    if value < ideal_low:
        return float(
            np.clip(
                (value - hard_low)
                / max(ideal_low - hard_low, 1e-6),
                0.0,
                1.0,
            )
        )

    return float(
        np.clip(
            (hard_high - value)
            / max(hard_high - ideal_high, 1e-6),
            0.0,
            1.0,
        )
    )


def _framing_score(
    *,
    face_area_ratio: float,
    face_center_x_ratio: float,
    face_center_y_ratio: float,
    headroom_ratio: float,
    torso_space_ratio: float,
) -> float:
    area_score = _trapezoid_score(
        face_area_ratio,
        hard_low=0.045,
        ideal_low=0.105,
        ideal_high=0.225,
        hard_high=0.340,
    )

    x_score = 1.0 - min(
        abs(face_center_x_ratio - 0.50) / 0.24,
        1.0,
    )

    y_score = 1.0 - min(
        abs(face_center_y_ratio - 0.34) / 0.25,
        1.0,
    )

    headroom_score = _trapezoid_score(
        headroom_ratio,
        hard_low=0.005,
        ideal_low=0.055,
        ideal_high=0.170,
        hard_high=0.300,
    )

    torso_score = _trapezoid_score(
        torso_space_ratio,
        hard_low=0.22,
        ideal_low=0.36,
        ideal_high=0.64,
        hard_high=0.82,
    )

    return float(
        np.clip(
            area_score * 0.34
            + x_score * 0.14
            + y_score * 0.18
            + headroom_score * 0.18
            + torso_score * 0.16,
            0.0,
            1.0,
        )
    )


def rank_candidates(
    *,
    source_antelope_embedding: np.ndarray,
    source_buffalo_embedding: np.ndarray,
    candidates: list[GeneratedCandidate],
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []

    for candidate in candidates:
        candidate_path = Path(candidate.image_path)

        if not candidate_path.is_file():
            continue

        try:
            antelope_result = analyze_faces(
                str(candidate_path),
                allow_enhancement=True,
            )

            buffalo_result = analyze_buffalo_face(
                str(candidate_path)
            )

            antelope_similarity = cosine_similarity(
                source_antelope_embedding,
                antelope_result.primary_face.normalized_embedding,
            )

            buffalo_similarity = cosine_similarity(
                source_buffalo_embedding,
                buffalo_result.normalized_embedding,
            )

        except Exception as exc:
            print(
                "[worker] candidate ranking failed for "
                f"{candidate_path}: {exc}",
                flush=True,
            )
            continue

        primary_face = antelope_result.primary_face
        image_width = max(1, antelope_result.image_width)
        image_height = max(1, antelope_result.image_height)

        center_x_ratio = (
            primary_face.box.x1 + primary_face.box.x2
        ) / (2.0 * image_width)

        center_y_ratio = (
            primary_face.box.y1 + primary_face.box.y2
        ) / (2.0 * image_height)

        headroom_ratio = primary_face.box.y1 / image_height

        torso_space_ratio = (
            image_height - primary_face.box.y2
        ) / image_height

        framing_score = _framing_score(
            face_area_ratio=primary_face.area_ratio,
            face_center_x_ratio=center_x_ratio,
            face_center_y_ratio=center_y_ratio,
            headroom_ratio=headroom_ratio,
            torso_space_ratio=torso_space_ratio,
        )

        conservative_identity = min(
            antelope_similarity,
            buffalo_similarity,
        )

        mean_identity = (
            antelope_similarity + buffalo_similarity
        ) / 2.0

        ranked.append(
            RankedCandidate(
                candidate=candidate,
                antelope_similarity=antelope_similarity,
                buffalo_similarity=buffalo_similarity,
                conservative_identity=round(
                    conservative_identity,
                    4,
                ),
                mean_identity=round(
                    mean_identity,
                    4,
                ),
                face_sharpness=round(
                    buffalo_result.sharpness,
                    2,
                ),
                detected_face_count=(
                    antelope_result.detected_face_count
                ),
                face_area_ratio=round(
                    primary_face.area_ratio,
                    4,
                ),
                face_center_x_ratio=round(
                    center_x_ratio,
                    4,
                ),
                face_center_y_ratio=round(
                    center_y_ratio,
                    4,
                ),
                headroom_ratio=round(
                    headroom_ratio,
                    4,
                ),
                torso_space_ratio=round(
                    torso_space_ratio,
                    4,
                ),
                framing_score=round(
                    framing_score,
                    4,
                ),
                passed_antelope=(
                    antelope_similarity
                    >= settings.face_identity_antelope_threshold
                ),
                passed_buffalo=(
                    buffalo_similarity
                    >= settings.face_identity_buffalo_threshold
                ),
                passed_mean=(
                    mean_identity
                    >= settings.face_identity_mean_threshold
                ),
                is_swapped=_is_swapped_candidate(candidate),
            )
        )

    ranked.sort(
        key=lambda item: (
            item.passed,
            item.conservative_identity,
            item.mean_identity,
            item.face_sharpness,
            item.framing_score,
            not item.is_swapped,
        ),
        reverse=True,
    )

    return ranked


def _filter_harmful_swaps(
    ranked: list[RankedCandidate],
) -> list[RankedCandidate]:
    generated_by_key = {
        candidate_key(item.candidate): item
        for item in ranked
        if not item.is_swapped
    }

    filtered: list[RankedCandidate] = []

    for item in ranked:
        if not item.is_swapped:
            filtered.append(item)
            continue

        generated = generated_by_key.get(
            candidate_key(item.candidate)
        )

        if generated is None:
            continue

        sharpness_ratio = (
            item.face_sharpness
            / max(generated.face_sharpness, 1.0)
        )

        identity_gain = (
            item.conservative_identity
            - generated.conservative_identity
        )

        antelope_drop = (
            generated.antelope_similarity
            - item.antelope_similarity
        )

        buffalo_drop = (
            generated.buffalo_similarity
            - item.buffalo_similarity
        )

        if (
            sharpness_ratio
            < settings.face_swap_min_sharpness_ratio
        ):
            continue

        if identity_gain < settings.face_swap_min_identity_gain:
            continue

        if antelope_drop > settings.face_swap_max_identity_drop:
            continue

        if buffalo_drop > settings.face_swap_max_identity_drop:
            continue

        filtered.append(item)

    return filtered


def select_best_candidate(
    ranked: list[RankedCandidate],
    *,
    require_passed: bool,
) -> RankedCandidate | None:
    ranked = _filter_harmful_swaps(ranked)

    if require_passed:
        pool = [
            item
            for item in ranked
            if item.passed
        ]
    else:
        pool = list(ranked)

    if not pool:
        return None

    best_identity = max(
        item.conservative_identity
        for item in pool
    )

    near_best = [
        item
        for item in pool
        if item.conservative_identity
        >= (
            best_identity
            - settings.identity_quality_tie_margin
        )
    ]

    # Identity is always the primary criterion. Framing is only a soft
    # tiebreaker and can never reject a candidate.
    return max(
        near_best,
        key=lambda item: (
            item.mean_identity,
            item.face_sharpness,
            item.framing_score,
            not item.is_swapped,
        ),
    )