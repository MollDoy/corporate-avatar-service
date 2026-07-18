import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
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
        default=lambda: str(
            uuid.uuid4()
        ),
    )

    employee_id: Mapped[str] = (
        mapped_column(
            String(100),
            nullable=False,
        )
    )

    style_id: Mapped[str] = (
        mapped_column(
            String(100),
            default="ai_business",
            nullable=False,
        )
    )

    status: Mapped[
        AvatarJobStatus
    ] = mapped_column(
        Enum(AvatarJobStatus),
        default=AvatarJobStatus.queued,
        nullable=False,
    )

    source_image_path: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    result_image_path: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    face_similarity_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    source_face_detection_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    source_face_area_ratio: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    identity_similarity: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    generation_attempts: Mapped[
        int
    ] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    generation_seed: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    pipeline_version: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    warnings_json: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )