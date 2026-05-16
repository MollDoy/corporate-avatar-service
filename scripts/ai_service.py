import os
from pathlib import Path
from typing import Literal

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageFilter


app = FastAPI(title="avatar-ai-inpainting-service")


class InpaintRequest(BaseModel):
    job_dir: str
    input_name: str = "result.png"
    mask_name: str = "clothes_mask.png"
    face_mask_name: str = "face_restore_mask.png"
    output_name: str = "ai_result.png"

    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting"

    prompt: str = Field(
        default=(
            "professional corporate ID portrait, formal business headshot, "
            "wearing a dark business blazer over a light dress shirt with a dark tie, "
            "clean collar, neat office clothing, realistic corporate portrait, "
            "studio lighting, high quality, sharp details"
        )
    )
    negative_prompt: str = Field(
        default=(
            "changed face, distorted face, changed eyes, distorted eyes, deformed mouth, "
            "bad anatomy, extra fingers, missing fingers, fused fingers, broken hands, "
            "extra limbs, low quality, blurry, artifacts, cartoon, "
            "t-shirt, casual shirt, hoodie, sweater, sportswear, watch, jewelry"
        )
    )

    steps: int = 12
    guidance_scale: float = 7.0
    strength: float = 0.75
    seed: int = 42

    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["float16", "float32"] = "float32"
    low_vram: bool = True
    restore_face: bool = True


class InpaintResponse(BaseModel):
    output_path: str


_PIPELINE_CACHE: dict[str, StableDiffusionInpaintPipeline] = {}


def _cache_key(model_id: str, dtype: str, device: str, low_vram: bool) -> str:
    return f"{model_id}|{dtype}|{device}|{low_vram}"


def _load_image(path: Path, mode: str) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    image = Image.open(path).convert(mode)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    return image


def _get_torch_dtype(dtype: str, use_cuda: bool) -> torch.dtype:
    if dtype == "float16" and use_cuda:
        return torch.float16

    return torch.float32


def _get_pipeline(request: InpaintRequest) -> StableDiffusionInpaintPipeline:
    use_cuda = request.device == "cuda" and torch.cuda.is_available()
    torch_dtype = _get_torch_dtype(request.dtype, use_cuda)

    key = _cache_key(
        model_id=request.model_id,
        dtype=str(torch_dtype),
        device="cuda" if use_cuda else "cpu",
        low_vram=request.low_vram,
    )

    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    token = os.getenv("HF_TOKEN") or None

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        request.model_id,
        torch_dtype=torch_dtype,
        token=token,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if use_cuda:
        if request.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    _PIPELINE_CACHE[key] = pipe
    return pipe


def _restore_protected_face(
    original_image: Image.Image,
    generated_image: Image.Image,
    face_mask: Image.Image,
) -> Image.Image:
    """
    Restores protected face/head region from original image after AI generation.

    White pixels on face_mask are copied from original_image.
    Black pixels remain from generated_image.
    """
    original = original_image.convert("RGB")
    generated = generated_image.convert("RGB")

    mask = face_mask.convert("L")
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.8))

    return Image.composite(original, generated, mask)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/inpaint", response_model=InpaintResponse)
def inpaint(request: InpaintRequest) -> InpaintResponse:
    try:
        job_dir = Path(request.job_dir)

        input_path = job_dir / request.input_name
        mask_path = job_dir / request.mask_name
        face_mask_path = job_dir / request.face_mask_name
        output_path = job_dir / request.output_name

        image = _load_image(input_path, "RGB")
        mask = _load_image(mask_path, "L")

        pipe = _get_pipeline(request)

        use_cuda = request.device == "cuda" and torch.cuda.is_available()
        generator_device = "cuda" if use_cuda and not request.low_vram else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(request.seed)

        with torch.inference_mode():
            result = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                strength=request.strength,
                generator=generator,
            ).images[0]

        if request.restore_face and face_mask_path.exists():
            face_mask = _load_image(face_mask_path, "L")
            result = _restore_protected_face(
                original_image=image,
                generated_image=result,
                face_mask=face_mask,
            )

        result.save(output_path)

        return InpaintResponse(output_path=str(output_path))

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc