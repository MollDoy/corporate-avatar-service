import base64
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import verify_api_key
from app.db.dependencies import get_db
from app.db.models import (
    AvatarJob,
    AvatarJobStatus,
)
from app.schemas.avatar import (
    AvatarArtifactResponse,
    AvatarJobArtifactsResponse,
    AvatarJobCreateRequest,
    AvatarJobCreateResponse,
    AvatarJobResultResponse,
    AvatarJobStatusResponse,
)
from app.services.avatar_styles import get_avatar_style
from app.services.image_storage import (
    decode_image_from_base64,
    save_source_image,
)
from app.services.job_artifacts import (
    artifact_to_view,
    get_job_artifacts,
    publish_artifact_file,
    read_result_image_bytes,
)
from app.services.object_storage import ObjectStorageError
from app.tasks.avatar_jobs import process_avatar_job


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
        source_face_detection_score=(
            job.source_face_detection_score
        ),
        source_face_area_ratio=job.source_face_area_ratio,
        identity_similarity=job.identity_similarity,
        generation_attempts=job.generation_attempts,
        generation_seed=job.generation_seed,
        pipeline_version=job.pipeline_version,
        warnings_json=job.warnings_json,
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
    try:
        get_avatar_style(payload.style_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    image = decode_image_from_base64(payload.image_base64)

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

        job.source_image_path = source_image_path
        db.add(job)
        db.commit()
        db.refresh(job)

        publish_artifact_file(
            db=db,
            job_id=job.id,
            local_path=Path(source_image_path),
        )

    except (ObjectStorageError, OSError, ValueError) as exc:
        db.rollback()

        job.status = AvatarJobStatus.failed
        job.error_message = (
            "Could not persist source image: "
            f"{exc}"
        )

        db.add(job)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=job.error_message,
        ) from exc

    finally:
        image.close()

    try:
        process_avatar_job.delay(job.id)
    except Exception as exc:
        job.status = AvatarJobStatus.failed
        job.error_message = (
            "Task queue is unavailable: "
            f"{exc}"
        )

        db.add(job)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=job.error_message,
        ) from exc

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
            detail=(
                "Avatar job is not done. "
                f"Current status: {job.status.value}"
            ),
        )

    try:
        result_bytes = read_result_image_bytes(
            db=db,
            job=job,
        )
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return AvatarJobResultResponse(
        job_id=job.id,
        image_base64=base64.b64encode(
            result_bytes
        ).decode("utf-8"),
    )


@router.get(
    "/{job_id}/artifacts",
    response_model=AvatarJobArtifactsResponse,
)
def list_avatar_job_artifacts(
    job_id: str,
    db: Session = Depends(get_db),
) -> AvatarJobArtifactsResponse:
    job = db.get(AvatarJob, job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Avatar job not found",
        )

    try:
        views = [
            artifact_to_view(artifact)
            for artifact in get_job_artifacts(
                db=db,
                job_id=job_id,
            )
        ]
    except ObjectStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return AvatarJobArtifactsResponse(
        job_id=job_id,
        storage_backend=(
            settings.object_storage_backend
            .strip()
            .lower()
        ),
        artifacts=[
            AvatarArtifactResponse(
                artifact_id=view.id,
                artifact_type=view.artifact_type,
                file_name=view.file_name,
                seed=view.seed,
                attempt_number=view.attempt_number,
                content_type=view.content_type,
                size_bytes=view.size_bytes,
                sha256=view.sha256,
                object_key=view.object_key,
                download_url=view.download_url,
            )
            for view in views
        ],
    )