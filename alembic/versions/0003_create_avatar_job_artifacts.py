"""Create avatar job artifacts table.

Revision ID: 0003_avatar_job_artifacts
Revises: 0002_identity_generation
"""

from alembic import op
import sqlalchemy as sa


revision = "0003_avatar_job_artifacts"
down_revision = "0002_identity_generation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "avatar_job_artifacts",
        sa.Column(
            "id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "artifact_type",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "file_name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "seed",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "attempt_number",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "local_path",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "object_key",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "content_type",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "sha256",
            sa.String(length=64),
            nullable=False,
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
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["avatar_jobs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            "file_name",
            name="uq_avatar_job_artifacts_job_file",
        ),
        sa.UniqueConstraint("object_key"),
    )

    op.create_index(
        "ix_avatar_job_artifacts_job_id",
        "avatar_job_artifacts",
        ["job_id"],
        unique=False,
    )

    op.create_index(
        "ix_avatar_job_artifacts_artifact_type",
        "avatar_job_artifacts",
        ["artifact_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_avatar_job_artifacts_artifact_type",
        table_name="avatar_job_artifacts",
    )

    op.drop_index(
        "ix_avatar_job_artifacts_job_id",
        table_name="avatar_job_artifacts",
    )

    op.drop_table("avatar_job_artifacts")