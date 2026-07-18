from datetime import datetime

from pydantic import BaseModel, Field


class AvatarJobCreateRequest(BaseModel):
    employee_id: str = Field(
        min_length=1,
        max_length=100,
        examples=["employee_001"],
    )

    style_id: str = Field(
        default="ai_business",
        min_length=1,
        max_length=100,
        examples=["ai_business"],
    )

    image_base64: str = Field(
        min_length=1,
        description=(
            "Source employee portrait "
            "encoded as base64 string"
        ),
    )


class AvatarJobCreateResponse(BaseModel):
    job_id: str
    status: str

    face_similarity_score: (
        float | None
    ) = None


class AvatarJobStatusResponse(BaseModel):
    job_id: str
    employee_id: str
    style_id: str
    status: str

    source_image_path: str | None
    result_image_path: str | None
    error_message: str | None

    face_similarity_score: (
        float | None
    )

    source_face_detection_score: (
        float | None
    )

    source_face_area_ratio: (
        float | None
    )

    identity_similarity: (
        float | None
    )

    generation_attempts: int
    generation_seed: int | None

    pipeline_version: str | None
    warnings_json: str | None

    created_at: datetime
    updated_at: datetime


class AvatarJobResultResponse(BaseModel):
    job_id: str
    image_base64: str
    mime_type: str = "image/png"


class AvatarStyleResponse(BaseModel):
    style_id: str
    title: str
    description: str