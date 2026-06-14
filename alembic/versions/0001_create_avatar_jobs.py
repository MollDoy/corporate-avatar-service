"""Create avatar_jobs table.

Revision ID: 0001_create_avatar_jobs
Revises:
Create Date: 2026-06-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0001_create_avatar_jobs"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


avatar_job_status_enum = postgresql.ENUM(
    "queued",
    "processing",
    "done",
    "failed",
    name="avatarjobstatus",
    create_type=False,
)


def upgrade() -> None:
    """Create PostgreSQL enum and avatar_jobs table."""

    bind = op.get_bind()

    postgresql.ENUM(
        "queued",
        "processing",
        "done",
        "failed",
        name="avatarjobstatus",
    ).create(bind, checkfirst=True)

    op.create_table(
        "avatar_jobs",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "employee_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "style_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "status",
            avatar_job_status_enum,
            nullable=False,
        ),
        sa.Column(
            "source_image_path",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "result_image_path",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "face_similarity_score",
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop avatar_jobs table and PostgreSQL enum."""

    op.drop_table("avatar_jobs")

    postgresql.ENUM(
        "queued",
        "processing",
        "done",
        "failed",
        name="avatarjobstatus",
    ).drop(op.get_bind(), checkfirst=True)