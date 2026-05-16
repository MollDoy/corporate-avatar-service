from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import verify_api_key
from app.db.dependencies import get_db
from app.db.models import AvatarJob
from app.schemas.avatar import (
    AvatarJobCreateRequest,
    AvatarJobCreateResponse,
    AvatarJobStatusResponse,
)
from app.services.image_storage import decode_image_from_base64, save_source_image


router = APIRouter(
    prefix="/api/v1/avatar-jobs",
    tags=["avatar-jobs"],
    dependencies=[Depends(verify_api_key)],
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
    job = AvatarJob(
        employee_id=payload.employee_id,
        style_id=payload.style_id,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    image = decode_image_from_base64(payload.image_base64)
    source_image_path = save_source_image(job.id, image)

    job.source_image_path = source_image_path

    db.add(job)
    db.commit()
    db.refresh(job)

    return AvatarJobCreateResponse(
        job_id=job.id,
        status=job.status.value,
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

    return AvatarJobStatusResponse(
        job_id=job.id,
        employee_id=job.employee_id,
        style_id=job.style_id,
        status=job.status.value,
        source_image_path=job.source_image_path,
        result_image_path=job.result_image_path,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )