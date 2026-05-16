from datetime import datetime

from pydantic import BaseModel, Field


class AvatarJobCreateRequest(BaseModel):
    employee_id: str = Field(
        min_length=1,
        max_length=100,
        examples=["employee_001"],
    )

    style_id: str = Field(
        default="default_business",
        min_length=1,
        max_length=100,
        examples=["default_business"],
    )

    image_base64: str = Field(
        min_length=1,
        description="Source employee portrait encoded as base64 string",
    )


class AvatarJobCreateResponse(BaseModel):
    job_id: str
    status: str


class AvatarJobStatusResponse(BaseModel):
    job_id: str
    employee_id: str
    style_id: str
    status: str
    source_image_path: str | None
    result_image_path: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime