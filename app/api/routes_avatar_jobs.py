from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import verify_api_key
from app.db.dependencies import get_db
from app.db.session import SessionLocal
from app.db.models import AvatarJob, AvatarJobStatus
from app.schemas.avatar import (
    AvatarJobCreateRequest,
    AvatarJobCreateResponse,
    AvatarJobResultResponse,
    AvatarJobStatusResponse,
)
from app.services.background_replacement import (
    clear_rembg_sessions,
    generate_basic_corporate_avatar,
)
from app.services.face_detection import validate_single_face
from app.services.face_similarity import calculate_face_similarity
from app.services.generation_masks import create_generation_masks
from app.services.ai_inpainting_client import run_ai_inpainting
from app.services.image_storage import (
    decode_image_from_base64,
    encode_file_to_base64,
    save_source_image,
)


router = APIRouter(
    prefix="/api/v1/avatar-jobs",
    tags=["avatar-jobs"],
    dependencies=[Depends(verify_api_key)],
)


def _serialize_job(job: AvatarJob) -> AvatarJobStatusResponse:
    return AvatarJobStatusResponse(
        job_id=job.id,
        employee_id=job.employee_id,
        style_id=job.style_id,
        status=job.status.value,
        source_image_path=job.source_image_path,
        result_image_path=job.result_image_path,
        error_message=job.error_message,
        face_similarity_score=job.face_similarity_score,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _process_job(job: AvatarJob, db: Session) -> None:
    if not job.source_image_path:
        job.status = AvatarJobStatus.failed
        job.error_message = "Source image path is empty"
        db.add(job)
        db.commit()
        return

    try:
        job.status = AvatarJobStatus.processing
        job.error_message = None
        job.face_similarity_score = None
        db.add(job)
        db.commit()
        db.refresh(job)

        validate_single_face(job.source_image_path)

        result_image_path = generate_basic_corporate_avatar(
            job_id=job.id,
            source_image_path=job.source_image_path,
            style_id=job.style_id,
        )

        person_mask_path = str(Path(result_image_path).parent / "person_mask.png")

        create_generation_masks(
            job_id=job.id,
            result_image_path=result_image_path,
            person_mask_path=person_mask_path,
        )

        final_result_image_path = result_image_path

        if job.style_id == "ai_business":
            clear_rembg_sessions()
            
            job_dir = str(Path(result_image_path).parent)

            final_result_image_path = run_ai_inpainting(
                job_dir=job_dir,
                input_name="result.png",
                mask_name="clothes_mask.png",
                output_name=settings.ai_output_name,
                prompt=(
                    "professional corporate ID portrait, formal business headshot, "
                    "wearing a light blue dress shirt with a dark tie, clean collar, "
                    "neat office clothing, realistic corporate portrait, studio lighting, "
                    "high quality, sharp details, natural hands"
                ),
                negative_prompt=(
                    "changed face, distorted face, changed eyes, distorted eyes, deformed mouth, "
                    "bad anatomy, extra fingers, missing fingers, fused fingers, broken hands, "
                    "extra limbs, low quality, blurry, artifacts, cartoon, "
                    "t-shirt, casual shirt, hoodie, sweater, sportswear, watch, jewelry"
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
        job.face_similarity_score = similarity_result.score

        if similarity_result.score < settings.face_similarity_threshold:
            job.status = AvatarJobStatus.failed
            job.error_message = (
                "Face similarity check failed. "
                f"Score={similarity_result.score}, "
                f"threshold={settings.face_similarity_threshold}."
            )
        else:
            job.status = AvatarJobStatus.done
            job.error_message = None

        db.add(job)
        db.commit()
        db.refresh(job)

    except Exception as exc:
        job.status = AvatarJobStatus.failed
        job.error_message = str(exc)

        db.add(job)
        db.commit()
        db.refresh(job)

def _process_job_by_id(job_id: str) -> None:
    """
    Processes avatar job in a separate background task.

    A new DB session is created here because SQLAlchemy sessions
    from request handlers must not be reused in background tasks.
    """
    db = SessionLocal()

    try:
        job = db.get(AvatarJob, job_id)

        if job is None:
            return

        _process_job(job, db)

    finally:
        db.close()

@router.post(
    "",
    response_model=AvatarJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_avatar_job(
    payload: AvatarJobCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AvatarJobCreateResponse:
    image = decode_image_from_base64(payload.image_base64)

    job = AvatarJob(
        employee_id=payload.employee_id,
        style_id=payload.style_id,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    source_image_path = save_source_image(job.id, image)

    job.source_image_path = source_image_path

    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_process_job_by_id, job.id)

    return AvatarJobCreateResponse(
        job_id=job.id,
        status=job.status.value,
        face_similarity_score=job.face_similarity_score,
    )


@router.get(
    "/{job_id}",
    response_model=AvatarJobStatusResponse,
)
def get_avatar_job(
    job_id: str,
    db: Session = Depends(get_db),
) -> AvatarJobStatusResponse:
    job = db.get(AvatarJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar job not found",
        )

    return _serialize_job(job)


@router.get(
    "/{job_id}/result",
    response_model=AvatarJobResultResponse,
)
def get_avatar_job_result(
    job_id: str,
    db: Session = Depends(get_db),
) -> AvatarJobResultResponse:
    job = db.get(AvatarJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar job not found",
        )

    if job.status != AvatarJobStatus.done:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Avatar job is not done. Current status: {job.status.value}",
        )

    if not job.result_image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result image path is empty",
        )

    return AvatarJobResultResponse(
        job_id=job.id,
        image_base64=encode_file_to_base64(job.result_image_path),
    )