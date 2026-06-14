from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import AvatarJob, AvatarJobStatus
from app.services.ai_inpainting_client import run_ai_inpainting
from app.services.background_replacement import (
    clear_rembg_sessions,
    generate_basic_corporate_avatar,
)
from app.services.face_detection import validate_single_face
from app.services.face_similarity import calculate_face_similarity
from app.services.generation_masks import create_generation_masks


def _mark_failed(
    job: AvatarJob,
    db: Session,
    message: str,
) -> None:
    job.status = AvatarJobStatus.failed
    job.error_message = message

    db.add(job)
    db.commit()


def process_avatar_job(
    job_id: str,
    db: Session,
) -> None:
    job = db.get(AvatarJob, job_id)

    if job is None:
        return

    # Защита от повторной доставки уже завершённого сообщения.
    if job.status in {
        AvatarJobStatus.done,
        AvatarJobStatus.failed,
    }:
        return

    if not job.source_image_path:
        _mark_failed(
            job,
            db,
            "Source image path is empty",
        )
        return

    try:
        job.status = AvatarJobStatus.processing
        job.error_message = None
        job.face_similarity_score = None

        db.add(job)
        db.commit()

        validate_single_face(job.source_image_path)

        try:
            result_image_path = generate_basic_corporate_avatar(
                job_id=job.id,
                source_image_path=job.source_image_path,
                style_id=job.style_id,
            )
        finally:
            # BiRefNet освобождается до запуска DreamShaper,
            # чтобы модели не занимали RAM одновременно.
            clear_rembg_sessions()

        person_mask_path = str(
            Path(result_image_path).parent
            / "person_mask.png"
        )

        create_generation_masks(
            job_id=job.id,
            result_image_path=result_image_path,
            person_mask_path=person_mask_path,
        )

        final_result_image_path = result_image_path

        if job.style_id == "ai_business":
            job_dir = str(
                Path(result_image_path).parent
            )

            final_result_image_path = run_ai_inpainting(
                job_dir=job_dir,
                input_name="result.png",
                mask_name="clothes_mask.png",
                output_name=settings.ai_output_name,
                prompt=(
                    "professional corporate ID portrait, "
                    "formal business headshot, "
                    "wearing a light blue dress shirt "
                    "with a dark tie, clean collar, "
                    "neat office clothing, "
                    "realistic corporate portrait, "
                    "studio lighting, high quality, "
                    "sharp details, natural hands"
                ),
                negative_prompt=(
                    "changed face, distorted face, "
                    "changed eyes, distorted eyes, "
                    "deformed mouth, bad anatomy, "
                    "extra fingers, missing fingers, "
                    "fused fingers, broken hands, "
                    "extra limbs, low quality, blurry, "
                    "artifacts, cartoon, t-shirt, "
                    "casual shirt, hoodie, sweater, "
                    "sportswear, watch, jewelry"
                ),
                steps=18,
                guidance_scale=6.5,
                strength=0.80,
                seed=43,
            )

        similarity_result = calculate_face_similarity(
            source_image_path=job.source_image_path,
            result_image_path=final_result_image_path,
        )

        job.result_image_path = final_result_image_path
        job.face_similarity_score = (
            similarity_result.score
        )

        if (
            similarity_result.score
            < settings.face_similarity_threshold
        ):
            job.status = AvatarJobStatus.failed
            job.error_message = (
                "Face similarity check failed. "
                f"Score={similarity_result.score}, "
                f"threshold="
                f"{settings.face_similarity_threshold}."
            )
        else:
            job.status = AvatarJobStatus.done
            job.error_message = None

        db.add(job)
        db.commit()

    except Exception as exc:
        db.rollback()

        failed_job = db.get(AvatarJob, job_id)

        if failed_job is not None:
            _mark_failed(
                failed_job,
                db,
                str(exc),
            )