from fastapi import APIRouter, Depends

from app.core.security import verify_api_key
from app.schemas.avatar import AvatarStyleResponse
from app.services.avatar_styles import list_avatar_styles


router = APIRouter(
    prefix="/api/v1/styles",
    tags=["styles"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("", response_model=list[AvatarStyleResponse])
def get_styles() -> list[AvatarStyleResponse]:
    styles = list_avatar_styles()

    return [
        AvatarStyleResponse(
            style_id=style.style_id,
            title=style.title,
            description=style.description,
        )
        for style in styles
    ]