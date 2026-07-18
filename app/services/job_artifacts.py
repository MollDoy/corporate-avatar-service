from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import AvatarJob, AvatarJobArtifact
from app.services.object_storage import (
    ObjectStorageError,
    build_object_key,
    download_file,
    generate_presigned_download_url,
    object_storage_is_s3,
    read_object_bytes,
    upload_file,
)


_CANDIDATE_PATTERN = re.compile(
    r"^candidate_(?P<attempt>\d+)_seed_(?P<seed>\d+)(?P<swapped>_swapped)?\.png$"
)


@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_type: str
    file_name: str
    seed: int | None = None
    attempt_number: int | None = None


@dataclass(frozen=True)
class ArtifactView:
    id: str
    artifact_type: str
    file_name: str
    seed: int | None
    attempt_number: int | None
    content_type: str
    size_bytes: int
    sha256: str
    object_key: str | None
    download_url: str | None


def classify_artifact_file(file_name: str) -> ArtifactDescriptor | None:
    if file_name == "source.png":
        return ArtifactDescriptor(
            artifact_type="source",
            file_name=file_name,
        )

    if file_name == "result.png":
        return ArtifactDescriptor(
            artifact_type="result",
            file_name=file_name,
        )

    match = _CANDIDATE_PATTERN.fullmatch(file_name)

    if match is None:
        return None

    return ArtifactDescriptor(
        artifact_type=(
            "swapped"
            if match.group("swapped")
            else "candidate"
        ),
        file_name=file_name,
        seed=int(match.group("seed")),
        attempt_number=int(match.group("attempt")),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _artifact_record(
    *,
    db: Session,
    job_id: str,
    file_name: str,
) -> AvatarJobArtifact | None:
    return db.scalar(
        select(AvatarJobArtifact).where(
            AvatarJobArtifact.job_id == job_id,
            AvatarJobArtifact.file_name == file_name,
        )
    )


def publish_artifact_file(
    *,
    db: Session,
    job_id: str,
    local_path: Path,
) -> AvatarJobArtifact:
    descriptor = classify_artifact_file(local_path.name)

    if descriptor is None:
        raise ValueError(
            "The file is not a publishable avatar artifact: "
            f"{local_path.name}"
        )

    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    size_bytes = local_path.stat().st_size
    sha256 = _sha256_file(local_path)
    existing = _artifact_record(
        db=db,
        job_id=job_id,
        file_name=descriptor.file_name,
    )

    object_key: str | None = None

    if object_storage_is_s3():
        object_key = build_object_key(
            job_id=job_id,
            artifact_type=descriptor.artifact_type,
            file_name=descriptor.file_name,
        )

        unchanged = (
            existing is not None
            and existing.sha256 == sha256
            and existing.object_key == object_key
        )

        if not unchanged:
            metadata = {
                "job-id": job_id,
                "artifact-type": descriptor.artifact_type,
                "sha256": sha256,
            }

            if descriptor.seed is not None:
                metadata["seed"] = str(descriptor.seed)

            if descriptor.attempt_number is not None:
                metadata["attempt-number"] = str(
                    descriptor.attempt_number
                )

            upload_file(
                local_path=local_path,
                object_key=object_key,
                content_type="image/png",
                metadata=metadata,
            )

    if existing is None:
        existing = AvatarJobArtifact(
            job_id=job_id,
            artifact_type=descriptor.artifact_type,
            file_name=descriptor.file_name,
            seed=descriptor.seed,
            attempt_number=descriptor.attempt_number,
            local_path=str(local_path),
            object_key=object_key,
            content_type="image/png",
            size_bytes=size_bytes,
            sha256=sha256,
        )
    else:
        existing.artifact_type = descriptor.artifact_type
        existing.seed = descriptor.seed
        existing.attempt_number = descriptor.attempt_number
        existing.local_path = str(local_path)
        existing.object_key = object_key
        existing.content_type = "image/png"
        existing.size_bytes = size_bytes
        existing.sha256 = sha256

    db.add(existing)
    db.commit()
    db.refresh(existing)

    return existing


def publish_job_image_artifacts(
    *,
    db: Session,
    job_id: str,
) -> list[AvatarJobArtifact]:
    job_directory = (
        Path(settings.storage_dir)
        / "jobs"
        / job_id
    )

    if not job_directory.is_dir():
        return []

    publishable: list[Path] = []

    for path in job_directory.iterdir():
        if not path.is_file():
            continue

        if classify_artifact_file(path.name) is not None:
            publishable.append(path)

    type_order = {
        "source": 0,
        "candidate": 1,
        "swapped": 2,
        "result": 3,
    }

    publishable.sort(
        key=lambda path: (
            type_order[
                classify_artifact_file(path.name).artifact_type  # type: ignore[union-attr]
            ],
            path.name,
        )
    )

    return [
        publish_artifact_file(
            db=db,
            job_id=job_id,
            local_path=path,
        )
        for path in publishable
    ]


def get_job_artifacts(
    *,
    db: Session,
    job_id: str,
) -> list[AvatarJobArtifact]:
    return list(
        db.scalars(
            select(AvatarJobArtifact)
            .where(AvatarJobArtifact.job_id == job_id)
            .order_by(
                AvatarJobArtifact.created_at.asc(),
                AvatarJobArtifact.file_name.asc(),
            )
        )
    )


def get_job_artifact_by_type(
    *,
    db: Session,
    job_id: str,
    artifact_type: str,
) -> AvatarJobArtifact | None:
    return db.scalar(
        select(AvatarJobArtifact)
        .where(
            AvatarJobArtifact.job_id == job_id,
            AvatarJobArtifact.artifact_type == artifact_type,
        )
        .order_by(AvatarJobArtifact.updated_at.desc())
        .limit(1)
    )


def ensure_source_image_local(
    *,
    db: Session,
    job: AvatarJob,
) -> Path:
    expected_path = (
        Path(settings.storage_dir)
        / "jobs"
        / job.id
        / "source.png"
    )

    if expected_path.is_file():
        if job.source_image_path != str(expected_path):
            job.source_image_path = str(expected_path)
            db.add(job)
            db.commit()
        return expected_path

    source_artifact = get_job_artifact_by_type(
        db=db,
        job_id=job.id,
        artifact_type="source",
    )

    if source_artifact is None or not source_artifact.object_key:
        raise ObjectStorageError(
            "Source image is missing locally and no S3 source artifact exists."
        )

    download_file(
        object_key=source_artifact.object_key,
        local_path=expected_path,
    )

    job.source_image_path = str(expected_path)
    source_artifact.local_path = str(expected_path)
    db.add_all([job, source_artifact])
    db.commit()

    return expected_path


def read_result_image_bytes(
    *,
    db: Session,
    job: AvatarJob,
) -> bytes:
    result_artifact = get_job_artifact_by_type(
        db=db,
        job_id=job.id,
        artifact_type="result",
    )

    if (
        object_storage_is_s3()
        and result_artifact is not None
        and result_artifact.object_key
    ):
        return read_object_bytes(result_artifact.object_key)

    candidate_paths: list[Path] = []

    if result_artifact is not None and result_artifact.local_path:
        candidate_paths.append(Path(result_artifact.local_path))

    if job.result_image_path:
        candidate_paths.append(Path(job.result_image_path))

    for path in candidate_paths:
        if path.is_file():
            return path.read_bytes()

    raise FileNotFoundError(
        f"Result image is unavailable for job {job.id}."
    )


def artifact_to_view(artifact: AvatarJobArtifact) -> ArtifactView:
    download_url = None

    if artifact.object_key:
        download_url = generate_presigned_download_url(
            artifact.object_key
        )

    return ArtifactView(
        id=artifact.id,
        artifact_type=artifact.artifact_type,
        file_name=artifact.file_name,
        seed=artifact.seed,
        attempt_number=artifact.attempt_number,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        object_key=artifact.object_key,
        download_url=download_url,
    )