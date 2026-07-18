"""Add identity-generation fields.

Revision ID: 0002_identity_generation
Revises: 0001_create_avatar_jobs
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_identity_generation"
down_revision = "0001_create_avatar_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "avatar_jobs",
        sa.Column(
            "source_face_detection_score",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "source_face_area_ratio",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "identity_similarity",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "generation_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "generation_seed",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "pipeline_version",
            sa.String(length=100),
            nullable=True,
        ),
    )

    op.add_column(
        "avatar_jobs",
        sa.Column(
            "warnings_json",
            sa.Text(),
            nullable=True,
        ),
    )

    op.alter_column(
        "avatar_jobs",
        "generation_attempts",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_column(
        "avatar_jobs",
        "warnings_json",
    )

    op.drop_column(
        "avatar_jobs",
        "pipeline_version",
    )

    op.drop_column(
        "avatar_jobs",
        "generation_seed",
    )

    op.drop_column(
        "avatar_jobs",
        "generation_attempts",
    )

    op.drop_column(
        "avatar_jobs",
        "identity_similarity",
    )

    op.drop_column(
        "avatar_jobs",
        "source_face_area_ratio",
    )

    op.drop_column(
        "avatar_jobs",
        "source_face_detection_score",
    )