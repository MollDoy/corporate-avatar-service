from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AvatarStyle:
    style_id: str
    title: str
    description: str
    prompt: str
    negative_prompt: str


COMMON_NEGATIVE_PROMPT = (
    "lowres, blurry, deformed face, deformed clothing, extra limbs, "
    "cropped head, multiple people, text, logo, watermark, cartoon, CGI"
)


AVATAR_STYLES: dict[str, AvatarStyle] = {
    "ai_business": AvatarStyle(
        style_id="ai_business",
        title="AI Business",
        description=(
            "Identity-preserving photorealistic corporate portrait"
        ),
        prompt=(
            "photorealistic, realistic skin, corporate chest-up portrait, "
            "centered, dark business suit, light blue shirt, direct gaze, "
            "neutral expression, soft even studio light, plain blue-gray background"
        ),
        negative_prompt=COMMON_NEGATIVE_PROMPT,
    ),
}


def get_avatar_style(style_id: str) -> AvatarStyle:
    try:
        return AVATAR_STYLES[style_id]
    except KeyError as exc:
        raise ValueError(
            "Unknown avatar style: "
            f"{style_id}"
        ) from exc


def list_avatar_styles() -> list[AvatarStyle]:
    return list(AVATAR_STYLES.values())