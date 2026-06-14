import os
import gc
from pathlib import Path
from typing import Literal

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageFilter

from threading import Lock


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

_INFERENCE_LOCK = Lock()

def _log_cuda_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        return

    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    max_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    print(
        f"[CUDA][{stage}] "
        f"allocated={allocated_mb:.0f} MiB, "
        f"reserved={reserved_mb:.0f} MiB, "
        f"max_allocated={max_allocated_mb:.0f} MiB, "
        f"max_reserved={max_reserved_mb:.0f} MiB",
        flush=True,
    )

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
        low_cpu_mem_usage=True,
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
def inpaint(
    request: InpaintRequest,
) -> InpaintResponse:
    with _INFERENCE_LOCK:
        return _inpaint_locked(request)

def _inpaint_locked(request: InpaintRequest) -> InpaintResponse:
    opened_images: list[Image.Image] = []
    pipe: StableDiffusionInpaintPipeline | None = None

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        _log_cuda_memory("before")

        job_dir = Path(request.job_dir)

        input_path = job_dir / request.input_name
        mask_path = job_dir / request.mask_name
        face_mask_path = job_dir / request.face_mask_name
        output_path = job_dir / request.output_name

        image = _load_image(input_path, "RGB")
        opened_images.append(image)

        mask = _load_image(mask_path, "L")
        opened_images.append(mask)

        pipe = _get_pipeline(request)

        use_cuda = request.device == "cuda" and torch.cuda.is_available()

        generator_device = (
            "cuda"
            if use_cuda and not request.low_vram
            else "cpu"
        )

        generator = torch.Generator(
            device=generator_device
        ).manual_seed(request.seed)

        with torch.inference_mode():
            generated_image = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance_scale,
                strength=request.strength,
                generator=generator,
            ).images[0]

        opened_images.append(generated_image)

        _log_cuda_memory("after_generation")

        result = generated_image

        if request.restore_face and face_mask_path.exists():
            face_mask = _load_image(face_mask_path, "L")
            opened_images.append(face_mask)

            restored_result = _restore_protected_face(
                original_image=image,
                generated_image=generated_image,
                face_mask=face_mask,
            )

            opened_images.append(restored_result)
            result = restored_result

        result.save(output_path)

        return InpaintResponse(output_path=str(output_path))

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    finally:
        # При model CPU offload явно просим pipeline выгрузить
        # активные компоненты обратно на CPU.
        if pipe is not None and request.low_vram:
            try:
                pipe.maybe_free_model_hooks()
            except Exception as exc:
                print(
                    f"Could not free model hooks: {exc}",
                    flush=True,
                )

        closed_ids: set[int] = set()

        for image_to_close in opened_images:
            image_id = id(image_to_close)

            if image_id in closed_ids:
                continue

            closed_ids.add(image_id)

            try:
                image_to_close.close()
            except Exception:
                pass

        opened_images.clear()

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            torch.cuda.empty_cache()

        _log_cuda_memory("after_cleanup")