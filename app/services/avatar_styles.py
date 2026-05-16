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
        description="Clear light blue corporate gradient",
        top_color=(235, 242, 255),
        bottom_color=(185, 210, 240),
    ),
    "blue_business": AvatarStyle(
        style_id="blue_business",
        title="Blue Business",
        description="More visible cold blue corporate background",
        top_color=(225, 238, 255),
        bottom_color=(155, 190, 230),
    ),
    "warm_office": AvatarStyle(
        style_id="warm_office",
        title="Warm Office",
        description="Warm beige office-like background",
        top_color=(250, 244, 232),
        bottom_color=(218, 199, 168),
    ),
    "gray_minimal": AvatarStyle(
        style_id="gray_minimal",
        title="Gray Minimal",
        description="Neutral minimal gray background",
        top_color=(242, 245, 249),
        bottom_color=(198, 207, 219),
    ),
    "ai_business": AvatarStyle(
        style_id="ai_business",
        title="AI Business",
        description="AI business portrait with clear light blue corporate gradient",
        top_color=(235, 242, 255),
        bottom_color=(185, 210, 240),
    ),
}

def get_avatar_style(style_id: str) -> AvatarStyle:
    return AVATAR_STYLES.get(style_id, AVATAR_STYLES["default_business"])


def list_avatar_styles() -> list[AvatarStyle]:
    return list(AVATAR_STYLES.values())