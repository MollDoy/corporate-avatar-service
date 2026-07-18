from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import AvatarJob, AvatarJobStatus
from app.services.avatar_styles import get_avatar_style
from app.services.background_matting_client import (
    apply_birefnet_portrait_background,
)
from app.services.buffalo_face_analysis import analyze_buffalo_face
from app.services.candidate_ranking import (
    RankedCandidate,
    rank_candidates,
    select_best_candidate,
)
from app.services.face_analysis import (
    AnalyzedFace,
    analyze_faces,
)
from app.services.face_conditioning import (
    FaceConditioningAssets,
    create_face_conditioning_assets,
)
from app.services.face_swap import swap_faces_in_candidates
from app.services.image_normalization import normalize_source_image
from app.services.job_artifacts import publish_job_image_artifacts
from app.services.object_storage import ObjectStorageError
from app.services.portrait_generation_client import (
    GeneratedCandidate,
    generate_portrait_candidates,
)
from app.services.runtime_cleanup import (
    release_worker_runtime_memory,
)


def _mark_failed(
    job: AvatarJob,
    db: Session,
    message: str,
) -> None:
    job.status = AvatarJobStatus.failed
    job.error_message = message

    db.add(job)
    db.commit()


def _release_worker_models(
    *,
    stage: str | None = None,
) -> None:
    release_worker_runtime_memory(
        stage=stage
    )


def _build_generation_prompts(
    *,
    style_prompt: str,
    style_negative_prompt: str,
    source_face: AnalyzedFace,
) -> tuple[str, str]:
    if source_face.gender == 1:
        subject_prompt = "man, "
        demographic_negative = "woman, feminine, makeup, "

    elif source_face.gender == 0:
        subject_prompt = "woman, "
        demographic_negative = "man, beard, mustache, "

    else:
        subject_prompt = "person, "
        demographic_negative = ""

    return (
        subject_prompt + style_prompt,
        demographic_negative
        + style_negative_prompt,
    )


def _parse_candidate_seeds() -> list[int]:
    seeds: list[int] = []

    for raw_value in (
        settings.ai_candidate_seeds.split(",")
    ):
        value = raw_value.strip()

        if not value:
            continue

        try:
            seed = int(value)
        except ValueError as exc:
            raise ValueError(
                "AI_CANDIDATE_SEEDS contains a non-integer value: "
                f"{value}"
            ) from exc

        if (
            seed < 0
            or seed > 2_147_483_647
        ):
            raise ValueError(
                "AI_CANDIDATE_SEEDS contains an out-of-range seed: "
                f"{seed}"
            )

        if seed not in seeds:
            seeds.append(seed)

    if not seeds:
        raise ValueError(
            "AI_CANDIDATE_SEEDS must contain at least one seed."
        )

    return seeds


def _generate_candidates(
    *,
    job: AvatarJob,
    conditioning: FaceConditioningAssets,
    prompt: str,
    negative_prompt: str,
    seeds: list[int],
    attempt_number_start: int = 1,
) -> list[GeneratedCandidate]:
    return generate_portrait_candidates(
        job_id=job.id,
        identity_reference_name=(
            conditioning.identity_reference_name
        ),
        face_embedding_name=(
            conditioning.face_embedding_name
        ),
        prompt=prompt,
        negative_prompt=negative_prompt,
        seeds=seeds,
        attempt_number_start=(
            attempt_number_start
        ),
    )


def _rank_candidates(
    *,
    source_antelope_embedding: np.ndarray,
    source_buffalo_embedding: np.ndarray,
    candidates: list[GeneratedCandidate],
) -> list[RankedCandidate]:
    return rank_candidates(
        source_antelope_embedding=(
            source_antelope_embedding
        ),
        source_buffalo_embedding=(
            source_buffalo_embedding
        ),
        candidates=candidates,
    )


def _select_candidates_for_swap(
    ranked_generated: list[RankedCandidate],
) -> list[GeneratedCandidate]:
    usable = [
        item
        for item in ranked_generated
        if item.detected_face_count == 1
    ]

    if not usable:
        usable = list(
            ranked_generated
        )

    usable.sort(
        key=lambda item: (
            item.conservative_identity,
            item.mean_identity,
            item.face_sharpness,
            item.framing_score,
        ),
        reverse=True,
    )

    return [
        item.candidate
        for item in usable[
            : settings.face_swap_top_k
        ]
    ]


def _candidate_diagnostic(
    ranked: RankedCandidate,
) -> str:
    variant = (
        "swapped"
        if ranked.is_swapped
        else "generated"
    )

    return (
        "candidate="
        f"{Path(ranked.candidate.image_path).name};"
        f"variant={variant};"
        f"antelope={ranked.antelope_similarity:.4f};"
        f"buffalo={ranked.buffalo_similarity:.4f};"
        f"min={ranked.conservative_identity:.4f};"
        f"mean={ranked.mean_identity:.4f};"
        f"faces={ranked.detected_face_count};"
        f"face_area={ranked.face_area_ratio:.4f};"
        f"face_center_x={ranked.face_center_x_ratio:.4f};"
        f"face_center_y={ranked.face_center_y_ratio:.4f};"
        f"headroom={ranked.headroom_ratio:.4f};"
        f"torso_space={ranked.torso_space_ratio:.4f};"
        f"framing={ranked.framing_score:.4f};"
        f"sharpness={ranked.face_sharpness:.2f};"
        f"passed={ranked.passed}"
    )


def _failure_message(
    best: RankedCandidate | None,
) -> str:
    if best is None:
        return (
            "No candidate contained a usable face for strict "
            "identity checking."
        )

    return (
        "Could not generate a portrait that passed strict identity "
        "checks. "
        f"Antelope={best.antelope_similarity:.4f} "
        f"(required {settings.face_identity_antelope_threshold:.4f}), "
        f"buffalo_l={best.buffalo_similarity:.4f} "
        f"(required {settings.face_identity_buffalo_threshold:.4f}), "
        f"mean={best.mean_identity:.4f} "
        f"(required {settings.face_identity_mean_threshold:.4f})."
    )


def _remove_candidate_files(
    candidates: list[GeneratedCandidate],
) -> None:
    if settings.keep_candidate_files:
        return

    for image_path in {
        candidate.image_path
        for candidate in candidates
    }:
        try:
            Path(image_path).unlink(
                missing_ok=True
            )
        except OSError:
            pass


def process_avatar_job(
    job_id: str,
    db: Session,
) -> None:
    job = db.get(
        AvatarJob,
        job_id,
    )

    if job is None:
        return

    if job.status in {
        AvatarJobStatus.done,
        AvatarJobStatus.failed,
    }:
        return

    if not job.source_image_path:
        _mark_failed(
            job,
            db,
            "Source image path is empty.",
        )
        return

    try:
        style = get_avatar_style(
            job.style_id
        )

        job.status = (
            AvatarJobStatus.processing
        )
        job.error_message = None
        job.result_image_path = None
        job.face_similarity_score = None
        job.identity_similarity = None
        job.source_face_detection_score = None
        job.source_face_area_ratio = None
        job.generation_attempts = 0
        job.generation_seed = None
        job.pipeline_version = (
            settings.pipeline_version
        )
        job.warnings_json = None

        db.add(job)
        db.commit()

        job_directory = (
            Path(settings.storage_dir)
            / "jobs"
            / job.id
        )

        job_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalized_path = (
            job_directory
            / "source_normalized.png"
        )

        normalize_source_image(
            source_path=(
                job.source_image_path
            ),
            output_path=str(
                normalized_path
            ),
            max_side=(
                settings
                .normalized_image_max_side
            ),
        )

        source_analysis = analyze_faces(
            str(normalized_path),
            allow_enhancement=True,
        )

        source_face = (
            source_analysis.primary_face
        )

        source_buffalo = (
            analyze_buffalo_face(
                str(normalized_path)
            )
        )

        source_antelope_embedding = (
            np.asarray(
                source_face.normalized_embedding,
                dtype=np.float32,
            ).copy()
        )

        source_buffalo_embedding = (
            np.asarray(
                source_buffalo.normalized_embedding,
                dtype=np.float32,
            ).copy()
        )

        source_detection_score = (
            source_face.detection_score
        )

        source_area_ratio = (
            source_face.area_ratio
        )

        source_gender = (
            source_face.gender
        )

        source_age = (
            source_face.age
        )

        source_warnings = list(
            source_analysis.warnings
        )

        conditioning = (
            create_face_conditioning_assets(
                source_image_path=str(
                    normalized_path
                ),
                face_box=(
                    source_face.box.x1,
                    source_face.box.y1,
                    source_face.box.x2,
                    source_face.box.y2,
                ),
                normalized_embedding=(
                    source_buffalo_embedding
                ),
                job_directory=(
                    job_directory
                ),
                reference_size=(
                    settings.ai_output_size
                ),
            )
        )

        prompt, negative_prompt = (
            _build_generation_prompts(
                style_prompt=(
                    style.prompt
                ),
                style_negative_prompt=(
                    style.negative_prompt
                ),
                source_face=source_face,
            )
        )

        seeds = (
            _parse_candidate_seeds()
        )

        job.source_face_detection_score = (
            source_detection_score
        )

        job.source_face_area_ratio = (
            source_area_ratio
        )

        initial_warnings = list(
            source_warnings
        )

        if source_gender is not None:
            initial_warnings.append(
                "source_gender="
                + (
                    "male"
                    if source_gender == 1
                    else "female"
                )
            )

        if source_age is not None:
            initial_warnings.append(
                "source_age_estimate="
                f"{source_age}"
            )

        initial_warnings.extend(
            [
                "generation_model=ConsistentID-v1",
                "base_model=Realistic_Vision_V6.0_B1_noVAE",
                "generation_mode=official_staged_fp32_subprocess",
                "consistentid_face_embedding=buffalo_l",
                "candidate_seeds="
                + ",".join(
                    str(seed)
                    for seed in seeds
                ),
                "generation_strategy=sequential_seed_fallback",
                "identity_reference_mode=original_face_centric_crop",
                "face_swap_mode="
                f"pixel_boost_{settings.face_swap_pixel_boost_size}",
                "background_mode=ConsistentID_prompt_plus_"
                "BiRefNet_portrait_final",
                "pose_control=official_ControlNet_v11_OpenPose_"
                "centered_body_template",
                "resource_mode=sequential_process_isolation",
            ]
        )

        job.warnings_json = json.dumps(
            initial_warnings,
            ensure_ascii=False,
        )

        db.add(job)
        db.commit()

        del source_analysis
        del source_face
        del source_buffalo

        _release_worker_models(
            stage="after_source_analysis"
        )

        generated_candidates: list[
            GeneratedCandidate
        ] = []

        swapped_candidates: list[
            GeneratedCandidate
        ] = []

        ranked_all: list[
            RankedCandidate
        ] = []

        best: (
            RankedCandidate | None
        ) = None

        for (
            attempt_number,
            seed,
        ) in enumerate(
            seeds,
            start=1,
        ):
            generated_for_attempt = (
                _generate_candidates(
                    job=job,
                    conditioning=conditioning,
                    prompt=prompt,
                    negative_prompt=(
                        negative_prompt
                    ),
                    seeds=[seed],
                    attempt_number_start=(
                        attempt_number
                    ),
                )
            )

            generated_candidates.extend(
                generated_for_attempt
            )

            ranked_generated = (
                _rank_candidates(
                    source_antelope_embedding=(
                        source_antelope_embedding
                    ),
                    source_buffalo_embedding=(
                        source_buffalo_embedding
                    ),
                    candidates=(
                        generated_for_attempt
                    ),
                )
            )

            candidates_for_swap = (
                _select_candidates_for_swap(
                    ranked_generated
                )
            )

            if not candidates_for_swap:
                candidates_for_swap = list(
                    generated_for_attempt
                )

            swapped_for_attempt = (
                swap_faces_in_candidates(
                    source_image_path=str(
                        normalized_path
                    ),
                    candidates=(
                        candidates_for_swap
                    ),
                )
            )

            swapped_candidates.extend(
                swapped_for_attempt
            )

            ranked_attempt = (
                _rank_candidates(
                    source_antelope_embedding=(
                        source_antelope_embedding
                    ),
                    source_buffalo_embedding=(
                        source_buffalo_embedding
                    ),
                    candidates=(
                        generated_for_attempt
                        + swapped_for_attempt
                    ),
                )
            )

            ranked_all.extend(
                ranked_attempt
            )

            job.generation_attempts = (
                attempt_number
            )

            db.add(job)
            db.commit()

            best = select_best_candidate(
                ranked_attempt,
                require_passed=True,
            )

            if best is not None:
                break

            _release_worker_models(
                stage=f"after_failed_seed_{seed}"
            )

        if best is None:
            diagnostic_best = (
                select_best_candidate(
                    ranked_all,
                    require_passed=False,
                )
            )

            raise ValueError(
                _failure_message(
                    diagnostic_best
                )
            )

        selected_image_path = (
            best.candidate.image_path
        )

        final_path = (
            job_directory
            / "result.png"
        )

        _release_worker_models(
            stage="before_background_matting"
        )

        if (
            settings
            .background_matting_enabled
        ):
            apply_birefnet_portrait_background(
                input_path=(
                    selected_image_path
                ),
                output_path=str(
                    final_path
                ),
                job_directory=(
                    job_directory
                ),
            )

        elif (
            settings
            .background_matting_required
        ):
            raise RuntimeError(
                "Background matting is required but disabled."
            )

        else:
            final_path.write_bytes(
                Path(
                    selected_image_path
                ).read_bytes()
            )

        final_warnings = list(
            initial_warnings
        )

        final_warnings.extend(
            [
                "selected_variant="
                + (
                    "swapped"
                    if best.is_swapped
                    else "generated"
                ),
                "identity_antelope="
                f"{best.antelope_similarity:.4f}",
                "identity_buffalo="
                f"{best.buffalo_similarity:.4f}",
                "identity_conservative="
                f"{best.conservative_identity:.4f}",
                "identity_mean="
                f"{best.mean_identity:.4f}",
                "face_area_ratio="
                f"{best.face_area_ratio:.4f}",
                "face_center_x_ratio="
                f"{best.face_center_x_ratio:.4f}",
                "face_center_y_ratio="
                f"{best.face_center_y_ratio:.4f}",
                "headroom_ratio="
                f"{best.headroom_ratio:.4f}",
                "torso_space_ratio="
                f"{best.torso_space_ratio:.4f}",
                "framing_score="
                f"{best.framing_score:.4f}",
                "face_sharpness="
                f"{best.face_sharpness:.2f}",
                "background_model="
                f"{settings.background_matting_model}",
                "corporate_background_hex="
                f"{settings.corporate_background_hex.upper()}",
            ]
        )

        final_warnings.extend(
            _candidate_diagnostic(
                item
            )
            for item in ranked_all
        )

        job.identity_similarity = (
            best.conservative_identity
        )

        job.face_similarity_score = (
            best.conservative_identity
        )

        job.generation_seed = (
            best.candidate.seed
        )

        job.result_image_path = str(
            final_path
        )

        job.warnings_json = json.dumps(
            final_warnings,
            ensure_ascii=False,
        )

        # Persist all business image artifacts before the job is marked done.
        # This changes only storage orchestration; model execution is untouched.
        publish_job_image_artifacts(
            db=db,
            job_id=job.id,
        )

        job.status = (
            AvatarJobStatus.done
        )

        job.error_message = None

        db.add(job)
        db.commit()

        _remove_candidate_files(
            generated_candidates
            + swapped_candidates
        )

    except ObjectStorageError:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        failed_job = db.get(
            AvatarJob,
            job_id,
        )

        if failed_job is not None:
            _mark_failed(
                failed_job,
                db,
                str(exc),
            )

    finally:
        _release_worker_models(
            stage="job_finally"
        )