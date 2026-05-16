from dataclasses import dataclass
from typing import Literal


StyleId = Literal[
    "default_business",
    "blue_business",
    "warm_office",
    "gray_minimal",
    "ai_business",
]


@dataclass(frozen=True)
class AvatarStyle:
    style_id: str
    title: str
    description: str
    top_color: tuple[int, int, int]
    bottom_color: tuple[int, int, int]


AVATAR_STYLES: dict[str, AvatarStyle] = {
    "default_business": AvatarStyle(
        style_id="default_business",
        title="Default Business",
        description="Light blue-gray corporate gradient",
        top_color=(245, 248, 255),
        bottom_color=(218, 229, 245),
    ),
    "blue_business": AvatarStyle(
        style_id="blue_business",
        title="Blue Business",
        description="Cold blue corporate background",
        top_color=(238, 245, 255),
        bottom_color=(190, 213, 240),
    ),
    "warm_office": AvatarStyle(
        style_id="warm_office",
        title="Warm Office",
        description="Warm beige office-like background",
        top_color=(250, 246, 238),
        bottom_color=(226, 214, 195),
    ),
    "gray_minimal": AvatarStyle(
        style_id="gray_minimal",
        title="Gray Minimal",
        description="Neutral minimal gray background",
        top_color=(248, 248, 248),
        bottom_color=(220, 224, 229),
    ),
    "ai_business": AvatarStyle(
        style_id="ai_business",
        title="AI Business",
        description="Experimental Stable Diffusion inpainting mode for background and clothes",
        top_color=(245, 248, 255),
        bottom_color=(218, 229, 245),
    ),
}


def get_avatar_style(style_id: str) -> AvatarStyle:
    return AVATAR_STYLES.get(style_id, AVATAR_STYLES["default_business"])


def list_avatar_styles() -> list[AvatarStyle]:
    return list(AVATAR_STYLES.values())