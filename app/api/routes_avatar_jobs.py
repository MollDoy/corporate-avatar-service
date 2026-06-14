from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from app.core.security import verify_api_key
from app.db.dependencies import get_db
from app.db.models import AvatarJob, AvatarJobStatus
from app.schemas.avatar import (
    AvatarJobCreateRequest,
    AvatarJobCreateResponse,
    AvatarJobResultResponse,
    AvatarJobStatusResponse,
)
from app.services.image_storage import (
    decode_image_from_base64,
    encode_file_to_base64,
    save_source_image,
)
from app.tasks.avatar_jobs import process_avatar_job


router = APIRouter(
    prefix="/api/v1/avatar-jobs",
    tags=["avatar-jobs"],
    dependencies=[Depends(verify_api_key)],
)


def _serialize_job(
    job: AvatarJob,
) -> AvatarJobStatusResponse:
    return AvatarJobStatusResponse(
        job_id=job.id,
        employee_id=job.employee_id,
        style_id=job.style_id,
        status=job.status.value,
        source_image_path=job.source_image_path,
        result_image_path=job.result_image_path,
        error_message=job.error_message,
        face_similarity_score=(
            job.face_similarity_score
        ),
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post(
    "",
    response_model=AvatarJobCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_avatar_job(
    payload: AvatarJobCreateRequest,
    db: Session = Depends(get_db),
) -> AvatarJobCreateResponse:
    image = decode_image_from_base64(
        payload.image_base64
    )

    job = AvatarJob(
        employee_id=payload.employee_id,
        style_id=payload.style_id,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        source_image_path = save_source_image(
            job.id,
            image,
        )
    except Exception as exc:
        job.status = AvatarJobStatus.failed
        job.error_message = (
            f"Could not save source image: {exc}"
        )

        db.add(job)
        db.commit()

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=job.error_message,
        ) from exc
    finally:
        image.close()

    job.source_image_path = source_image_path

    db.add(job)
    db.commit()
    db.refresh(job)

    try:
        process_avatar_job.delay(job.id)
    except Exception as exc:
        job.status = AvatarJobStatus.failed
        job.error_message = (
            f"Task queue is unavailable: {exc}"
        )

        db.add(job)
        db.commit()

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=job.error_message,
        ) from exc

    return AvatarJobCreateResponse(
        job_id=job.id,
        status=job.status.value,
        face_similarity_score=(
            job.face_similarity_score
        ),
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
            detail=(
                "Avatar job is not done. "
                f"Current status: {job.status.value}"
            ),
        )

    if not job.result_image_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result image path is empty",
        )

    return AvatarJobResultResponse(
        job_id=job.id,
        image_base64=encode_file_to_base64(
            job.result_image_path
        ),
    )