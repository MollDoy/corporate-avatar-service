import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from app.db.base import Base


class AvatarJobStatus(
    str,
    enum.Enum,
):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class AvatarJob(Base):
    __tablename__ = "avatar_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    employee_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    style_id: Mapped[str] = mapped_column(
        String(100),
        default="ai_business",
        nullable=False,
    )

    status: Mapped[AvatarJobStatus] = mapped_column(
        Enum(AvatarJobStatus),
        default=AvatarJobStatus.queued,
        nullable=False,
    )

    source_image_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    result_image_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    face_similarity_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source_face_detection_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    source_face_area_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    identity_similarity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    generation_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    generation_seed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    pipeline_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    warnings_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    artifacts: Mapped[list["AvatarJobArtifact"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AvatarJobArtifact(Base):
    __tablename__ = "avatar_job_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "file_name",
            name="uq_avatar_job_artifacts_job_file",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "avatar_jobs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
    )

    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    seed: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    attempt_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    local_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    object_key: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        unique=True,
    )

    content_type: Mapped[str] = mapped_column(
        String(100),
        default="image/png",
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    job: Mapped[AvatarJob] = relationship(
        back_populates="artifacts",
    )