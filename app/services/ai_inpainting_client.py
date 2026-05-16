from pathlib import Path

import httpx

from app.core.config import settings


DEFAULT_PROMPT = (
    "professional corporate ID portrait, formal business headshot, "
    "wearing a dark business blazer over a light dress shirt with a dark tie, "
    "clean collar, neat office clothing, realistic corporate portrait, "
    "studio lighting, high quality, sharp details"
)

DEFAULT_NEGATIVE_PROMPT = (
    "changed face, distorted face, changed eyes, distorted eyes, deformed mouth, "
    "bad anatomy, extra fingers, missing fingers, fused fingers, broken hands, "
    "extra limbs, low quality, blurry, artifacts, cartoon, "
    "t-shirt, casual shirt, hoodie, sweater, sportswear, watch, jewelry"
)


def run_ai_inpainting(
    job_dir: str,
    input_name: str = "result.png",
    mask_name: str = "clothes_mask.png",
    output_name: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    strength: float | None = None,
) -> str:
    """
    Calls AI inference service and returns path to generated image.

    Important:
    job_dir must be the path visible inside Docker containers:
    /app/storage/jobs/<job_id>
    """
    if output_name is None:
        output_name = settings.ai_output_name

    payload = {
        "job_dir": job_dir,
        "input_name": input_name,
        "mask_name": mask_name,
        "face_mask_name": "face_protection_mask.png",
        "output_name": output_name,
        "model_id": settings.ai_model_id,
        "prompt": prompt or DEFAULT_PROMPT,
        "negative_prompt": negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        "steps": steps if steps is not None else settings.ai_default_steps,
        "guidance_scale": (
            guidance_scale
            if guidance_scale is not None
            else settings.ai_default_guidance_scale
        ),
        "strength": strength if strength is not None else settings.ai_default_strength,
        "seed": 42,
        "device": settings.ai_device,
        "dtype": settings.ai_dtype,
        "low_vram": settings.ai_low_vram,
        "restore_face": settings.ai_restore_face_after_inpaint,
    }

    url = settings.ai_service_url.rstrip("/") + "/inpaint"

    try:
        with httpx.Client(timeout=settings.ai_inpaint_timeout_seconds) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"AI inpainting service request failed: {exc}") from exc

    data = response.json()
    output_path = data.get("output_path")

    if not output_path:
        raise RuntimeError("AI inpainting service did not return output_path")

    if not Path(output_path).exists():
        raise RuntimeError(f"AI output file does not exist: {output_path}")

    return output_path