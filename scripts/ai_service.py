from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
)
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from pipline_StableDiffusion_ConsistentID import (
    ConsistentIDStableDiffusionPipeline,
)


STORAGE_ROOT = Path(
    os.getenv(
        "STORAGE_ROOT",
        "/app/storage/jobs",
    )
).resolve()

AI_BASE_MODEL_DIR = Path(
    os.getenv(
        "AI_BASE_MODEL_DIR",
        "/models/base_model",
    )
).resolve()

AI_VAE_MODEL_DIR = Path(
    os.getenv(
        "AI_VAE_MODEL_DIR",
        "/models/vae",
    )
).resolve()

AI_CLIP_MODEL_DIR = Path(
    os.getenv(
        "AI_CLIP_MODEL_DIR",
        "/models/clip_vision",
    )
).resolve()

CONSISTENTID_MODEL_DIR = Path(
    os.getenv(
        "CONSISTENTID_MODEL_DIR",
        "/models/consistentid",
    )
).resolve()

CONSISTENTID_WEIGHT_NAME = os.getenv(
    "CONSISTENTID_WEIGHT_NAME",
    "ConsistentID-v1.bin",
)

CONSISTENTID_FACE_PARSING_NAME = os.getenv(
    "CONSISTENTID_FACE_PARSING_NAME",
    "face_parsing.pth",
)

AI_CONTROLNET_MODEL_DIR = Path(
    os.getenv(
        "AI_CONTROLNET_MODEL_DIR",
        "/models/controlnet_openpose",
    )
).resolve()

AI_CONTROLNET_ENABLED = os.getenv(
    "AI_CONTROLNET_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

AI_CONTROLNET_CONDITIONING_SCALE = float(
    os.getenv(
        "AI_CONTROLNET_CONDITIONING_SCALE",
        "0.35",
    )
)

AI_CONTROLNET_GUIDANCE_START = float(
    os.getenv(
        "AI_CONTROLNET_GUIDANCE_START",
        "0.0",
    )
)

AI_CONTROLNET_GUIDANCE_END = float(
    os.getenv(
        "AI_CONTROLNET_GUIDANCE_END",
        "0.62",
    )
)

AI_OUTPUT_SIZE = int(
    os.getenv(
        "AI_OUTPUT_SIZE",
        "512",
    )
)

AI_CPU_THREADS = int(
    os.getenv(
        "AI_CPU_THREADS",
        "8",
    )
)

AI_CUDA_DEVICE_INDEX = int(
    os.getenv(
        "AI_CUDA_DEVICE_INDEX",
        "0",
    )
)

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{4}-"
    r"[0-9a-f]{12}$"
)

COMPUTE_DTYPE = torch.float32
CONTROLNET_DTYPE = torch.float16
CPU_DEVICE = torch.device("cpu")
CUDA_DEVICE = torch.device(
    f"cuda:{AI_CUDA_DEVICE_INDEX}"
)

SERVICE_VERSION = (
    "sd15-consistentid-v1-official-fp32-openpose-birefnet-v17"
)


try:
    import xformers  # noqa: F401

    XFORMERS_AVAILABLE = True
except Exception:
    XFORMERS_AVAILABLE = False


torch.set_num_threads(max(1, AI_CPU_THREADS))
torch.set_num_interop_threads(1)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


app = FastAPI(
    title="corporate-avatar-ai-portrait"
)

INFERENCE_LOCK = Lock()
_RUNTIME_BUSY = False
_RUNTIME: ConsistentIDRuntime | None = None


class PortraitRequest(BaseModel):
    job_id: str = Field(
        min_length=36,
        max_length=36,
    )

    identity_reference_name: str = Field(
        default="identity_reference.png",
        min_length=1,
        max_length=200,
    )

    face_embedding_name: str = Field(
        default="face_embedding.npy",
        min_length=1,
        max_length=200,
    )

    output_name: str = Field(
        min_length=1,
        max_length=200,
    )

    prompt: str = Field(
        min_length=1,
        max_length=1000,
    )

    negative_prompt: str = Field(
        default="",
        max_length=1000,
    )

    seed: int = Field(
        ge=0,
        le=2_147_483_647,
    )

    num_inference_steps: int = Field(
        default=50,
        ge=20,
        le=80,
    )

    guidance_scale: float = Field(
        default=5.0,
        ge=1.0,
        le=12.0,
    )

    adapter_scale: float = Field(
        default=1.0,
        ge=0.20,
        le=2.00,
    )

    start_merge_step: int = Field(
        default=30,
        ge=0,
        le=79,
    )


class PortraitResponse(BaseModel):
    output_path: str
    seed: int


@dataclass
class ConsistentIDRuntime:
    pipeline: ConsistentIDStableDiffusionPipeline
    controlnet: ControlNetModel | None


@dataclass(frozen=True)
class ConsistentIDConditioning:
    null_prompt_embeds: torch.Tensor
    augmented_prompt_embeds: torch.Tensor
    text_prompt_embeds: torch.Tensor


class GenerationStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        cause: Exception,
    ) -> None:
        self.stage = stage
        self.cause = cause

        super().__init__(
            f"stage={stage}; "
            f"type={type(cause).__name__}; "
            f"message={cause}"
        )


def _validate_filename(
    filename: str,
    allowed_suffixes: set[str],
) -> str:
    path = Path(filename)

    if path.name != filename:
        raise ValueError(
            "Nested paths are not allowed."
        )

    if path.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            "Unsupported file type: "
            f"{path.suffix}"
        )

    return filename


def _job_directory(job_id: str) -> Path:
    if not UUID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job_id.")

    directory = (
        STORAGE_ROOT / job_id
    ).resolve()

    if directory.parent != STORAGE_ROOT:
        raise ValueError(
            "Invalid job directory."
        )

    if not directory.is_dir():
        raise FileNotFoundError(
            "Job directory does not exist: "
            f"{directory}"
        )

    return directory


def _load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(
            "Image does not exist: "
            f"{path}"
        )

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(
            opened
        ).convert("RGB")

    return ImageOps.fit(
        image,
        (AI_OUTPUT_SIZE, AI_OUTPUT_SIZE),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.46),
    )


def _load_face_embedding(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            "Face embedding does not exist: "
            f"{path}"
        )

    embedding = np.load(
        path,
        allow_pickle=False,
    ).astype(
        np.float32,
        copy=False,
    )

    if embedding.shape != (512,):
        raise ValueError(
            "Invalid face embedding shape: "
            f"{embedding.shape}"
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "Face embedding contains NaN or infinity."
        )

    norm = float(np.linalg.norm(embedding))

    if norm <= 0:
        raise ValueError(
            "Face embedding has zero norm."
        )

    return torch.from_numpy(
        embedding / norm
    ).unsqueeze(0)


def _current_rss_mb() -> float | None:
    try:
        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None

    return None


def _trim_cpu_memory() -> None:
    gc.collect(2)

    try:
        ctypes.CDLL(
            "libc.so.6"
        ).malloc_trim(0)
    except Exception:
        pass


def _release_cuda_memory() -> None:
    _trim_cpu_memory()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(
                AI_CUDA_DEVICE_INDEX
            )
        except Exception:
            pass

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    _trim_cpu_memory()


def _log_memory(stage: str) -> None:
    rss = _current_rss_mb()

    if torch.cuda.is_available():
        allocated = (
            torch.cuda.memory_allocated(
                AI_CUDA_DEVICE_INDEX
            )
            / 1024 ** 2
        )
        reserved = (
            torch.cuda.memory_reserved(
                AI_CUDA_DEVICE_INDEX
            )
            / 1024 ** 2
        )
    else:
        allocated = 0.0
        reserved = 0.0

    rss_text = (
        f"{rss:.1f} MiB"
        if rss is not None
        else "unknown"
    )

    print(
        "[ai-memory] "
        f"stage={stage}; rss={rss_text}; "
        f"cuda_allocated={allocated:.1f} MiB; "
        f"cuda_reserved={reserved:.1f} MiB",
        flush=True,
    )


def _release_conditioning_modules(
    pipeline: ConsistentIDStableDiffusionPipeline,
) -> None:
    """Drop modules that are no longer needed after prompt conditioning."""

    pipeline.external_faceid_embeds = None

    for attribute_name in (
        "text_encoder",
        "image_encoder",
        "image_proj_model",
        "FacialEncoder",
        "bise_net",
    ):
        if hasattr(pipeline, attribute_name):
            setattr(
                pipeline,
                attribute_name,
                None,
            )

    _release_cuda_memory()
    _log_memory("conditioning_modules_released")


def _release_denoising_modules(
    runtime: ConsistentIDRuntime,
) -> None:
    runtime.pipeline.unet = None
    runtime.controlnet = None
    _release_cuda_memory()
    _log_memory("denoising_modules_released")


def _drop_runtime() -> None:
    global _RUNTIME

    runtime = _RUNTIME
    _RUNTIME = None

    if runtime is None:
        _release_cuda_memory()
        return

    pipeline = runtime.pipeline

    for attribute_name in (
        "text_encoder",
        "unet",
        "vae",
        "image_encoder",
        "image_proj_model",
        "FacialEncoder",
        "bise_net",
    ):
        if hasattr(pipeline, attribute_name):
            setattr(
                pipeline,
                attribute_name,
                None,
            )

    pipeline.external_faceid_embeds = None
    runtime.controlnet = None

    del pipeline
    del runtime

    _release_cuda_memory()
    _log_memory("runtime_dropped")


def _move_module_with_dtype(
    module: torch.nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if module is None:
        return

    module.to(
        device=device,
        dtype=dtype,
    )


def _move_module(
    module: torch.nn.Module | None,
    device: torch.device,
) -> None:
    _move_module_with_dtype(
        module,
        device,
        COMPUTE_DTYPE,
    )


def _move_modules_to_cpu(
    *modules: torch.nn.Module | None,
) -> None:
    for module in modules:
        _move_module_with_dtype(
            module,
            CPU_DEVICE,
            COMPUTE_DTYPE,
        )

    _release_cuda_memory()


def _move_controlnet_to_cpu(
    controlnet: ControlNetModel | None,
) -> None:
    if controlnet is None:
        return

    _move_module_with_dtype(
        controlnet,
        CPU_DEVICE,
        CONTROLNET_DTYPE,
    )
    _release_cuda_memory()


def _file_ready(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def _any_file_ready(paths: tuple[Path, ...]) -> bool:
    return any(
        _file_ready(path)
        for path in paths
    )


def _check_model_files() -> dict[str, bool]:
    return {
        "base_model": _file_ready(
            AI_BASE_MODEL_DIR / "model_index.json"
        ),
        "base_text_encoder": _any_file_ready(
            (
                AI_BASE_MODEL_DIR
                / "text_encoder"
                / "model.safetensors",
                AI_BASE_MODEL_DIR
                / "text_encoder"
                / "pytorch_model.bin",
            )
        ),
        "base_unet": _any_file_ready(
            (
                AI_BASE_MODEL_DIR
                / "unet"
                / "diffusion_pytorch_model.safetensors",
                AI_BASE_MODEL_DIR
                / "unet"
                / "diffusion_pytorch_model.bin",
            )
        ),
        "vae": _any_file_ready(
            (
                AI_VAE_MODEL_DIR
                / "diffusion_pytorch_model.safetensors",
                AI_VAE_MODEL_DIR
                / "diffusion_pytorch_model.bin",
            )
        ),
        "clip_vision": _any_file_ready(
            (
                AI_CLIP_MODEL_DIR / "model.safetensors",
                AI_CLIP_MODEL_DIR / "pytorch_model.bin",
            )
        ),
        "consistentid": _file_ready(
            CONSISTENTID_MODEL_DIR
            / CONSISTENTID_WEIGHT_NAME
        ),
        "face_parsing": _file_ready(
            CONSISTENTID_MODEL_DIR
            / CONSISTENTID_FACE_PARSING_NAME
        ),
        "controlnet_openpose": (
            not AI_CONTROLNET_ENABLED
            or (
                _file_ready(
                    AI_CONTROLNET_MODEL_DIR / "config.json"
                )
                and _file_ready(
                    AI_CONTROLNET_MODEL_DIR
                    / "diffusion_pytorch_model.fp16.safetensors"
                )
            )
        ),
    }


def _load_runtime() -> ConsistentIDRuntime:
    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    checks = _check_model_files()

    if not all(checks.values()):
        missing = [
            name
            for name, ready in checks.items()
            if not ready
        ]

        raise FileNotFoundError(
            "Required ConsistentID model files are missing: "
            + ", ".join(missing)
        )

    vae = AutoencoderKL.from_pretrained(
        str(AI_VAE_MODEL_DIR),
        local_files_only=True,
        torch_dtype=COMPUTE_DTYPE,
        low_cpu_mem_usage=True,
    )

    pipeline = (
        ConsistentIDStableDiffusionPipeline
        .from_pretrained(
            str(AI_BASE_MODEL_DIR),
            local_files_only=True,
            torch_dtype=COMPUTE_DTYPE,
            low_cpu_mem_usage=True,
            vae=vae,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    )

    pipeline.load_ConsistentID_model(
        str(CONSISTENTID_MODEL_DIR),
        weight_name=CONSISTENTID_WEIGHT_NAME,
        image_encoder_path=str(
            AI_CLIP_MODEL_DIR
        ),
        bise_net_cp=str(
            CONSISTENTID_MODEL_DIR
            / CONSISTENTID_FACE_PARSING_NAME
        ),
        torch_dtype=COMPUTE_DTYPE,
        local_files_only=True,
    )

    pipeline.scheduler = (
        EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
    )

    pipeline.set_progress_bar_config(
        disable=False
    )

    controlnet: ControlNetModel | None = None

    if AI_CONTROLNET_ENABLED:
        if not (
            0.0
            <= AI_CONTROLNET_GUIDANCE_START
            < AI_CONTROLNET_GUIDANCE_END
            <= 1.0
        ):
            raise ValueError(
                "ControlNet guidance range must satisfy "
                "0 <= start < end <= 1."
            )

        if AI_CONTROLNET_CONDITIONING_SCALE <= 0.0:
            raise ValueError(
                "ControlNet conditioning scale must be greater "
                "than zero."
            )

        controlnet = ControlNetModel.from_pretrained(
            str(AI_CONTROLNET_MODEL_DIR),
            local_files_only=True,
            torch_dtype=CONTROLNET_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            variant="fp16",
        )

        _move_controlnet_to_cpu(controlnet)

    _move_modules_to_cpu(
        pipeline.text_encoder,
        pipeline.unet,
        pipeline.vae,
        pipeline.image_encoder,
        pipeline.image_proj_model,
        pipeline.FacialEncoder,
        pipeline.bise_net,
    )

    print(
        "[ai-portrait] Official ConsistentID SD 1.5 runtime "
        "loaded in FP32. Components are staged manually "
        "between CPU and CUDA without Accelerate hooks. "
        "Official ControlNet v1.1 OpenPose="
        f"{AI_CONTROLNET_ENABLED}.",
        flush=True,
    )

    _RUNTIME = ConsistentIDRuntime(
        pipeline=pipeline,
        controlnet=controlnet,
    )

    _log_memory("runtime_loaded")

    return _RUNTIME


def _encode_text_conditioning(
    *,
    pipeline: ConsistentIDStableDiffusionPipeline,
    prompt_text_only: str,
    clean_input_id: torch.Tensor,
    negative_prompt: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _move_module(
        pipeline.text_encoder,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            encoder_hidden_states = (
                pipeline.text_encoder(
                    clean_input_id.to(
                        CUDA_DEVICE
                    )
                )[0]
            )

            prompt_embeds = pipeline._encode_prompt(
                prompt_text_only,
                device=CUDA_DEVICE,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            negative_text_states = (
                prompt_embeds[0:1]
            )

            positive_text_states = (
                prompt_embeds[1:]
            )

            return (
                encoder_hidden_states.float().cpu(),
                negative_text_states.float().cpu(),
                positive_text_states.float().cpu(),
            )

    finally:
        _move_modules_to_cpu(
            pipeline.text_encoder
        )


def _encode_identity_conditioning(
    *,
    pipeline: ConsistentIDStableDiffusionPipeline,
    identity_reference: Image.Image,
    face_embedding: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    negative_text_states: torch.Tensor,
    facial_clip_image: torch.Tensor,
    facial_token_mask: torch.Tensor,
    facial_token_idx_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the official ConsistentID conditioning math in FP32.

    Only component placement is changed for the 6 GB GTX 1660:
    CLIP Vision, the FaceID projection, and the FacialEncoder are
    placed on CUDA one after another. Facial crops are encoded as
    micro-batches of one. CLIP Vision is an eval-only transformer,
    so this batching change preserves the same per-image operation
    while avoiding simultaneous FP32 residency and large activations.
    """

    _move_module(
        pipeline.image_encoder,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            clip_image = (
                pipeline.clip_image_processor(
                    images=identity_reference,
                    return_tensors="pt",
                )
                .pixel_values
                .to(
                    CUDA_DEVICE,
                    dtype=COMPUTE_DTYPE,
                )
            )

            clip_image_embeds = (
                pipeline.image_encoder(
                    clip_image,
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .float()
                .cpu()
            )

            uncond_clip_image_embeds = (
                pipeline.image_encoder(
                    torch.zeros_like(clip_image),
                    output_hidden_states=True,
                )
                .hidden_states[-2]
                .float()
                .cpu()
            )

            hidden_states: list[torch.Tensor] = []
            uncond_hidden_states: list[torch.Tensor] = []

            for facial_part in facial_clip_image:
                facial_part_batch = (
                    facial_part
                    .unsqueeze(0)
                    .to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    )
                )

                hidden_states.append(
                    pipeline.image_encoder(
                        facial_part_batch,
                        output_hidden_states=True,
                    )
                    .hidden_states[-2]
                    .float()
                    .cpu()
                )

                uncond_hidden_states.append(
                    pipeline.image_encoder(
                        torch.zeros_like(
                            facial_part_batch
                        ),
                        output_hidden_states=True,
                    )
                    .hidden_states[-2]
                    .float()
                    .cpu()
                )

                del facial_part_batch

            multi_facial_embeds = torch.stack(
                hidden_states,
                dim=1,
            )

            uncond_multi_facial_embeds = torch.stack(
                uncond_hidden_states,
                dim=1,
            )

            del clip_image
            del hidden_states
            del uncond_hidden_states

    finally:
        _move_modules_to_cpu(
            pipeline.image_encoder
        )

    _move_module(
        pipeline.image_proj_model,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            faceid_cuda = face_embedding.to(
                CUDA_DEVICE,
                dtype=COMPUTE_DTYPE,
            )

            prompt_tokens_faceid = (
                pipeline.image_proj_model(
                    faceid_cuda,
                    clip_image_embeds.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    shortcut=False,
                    scale=1.0,
                )
                .float()
                .cpu()
            )

            uncond_prompt_tokens_faceid = (
                pipeline.image_proj_model(
                    torch.zeros_like(faceid_cuda),
                    uncond_clip_image_embeds.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    shortcut=False,
                    scale=1.0,
                )
                .float()
                .cpu()
            )

            del faceid_cuda

    finally:
        _move_modules_to_cpu(
            pipeline.image_proj_model
        )

    _move_module(
        pipeline.FacialEncoder,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            facial_token_mask_cuda = (
                facial_token_mask.to(
                    CUDA_DEVICE
                )
            )

            facial_token_idx_mask_cuda = (
                facial_token_idx_mask.to(
                    CUDA_DEVICE
                )
            )

            prompt_embeds_facial = (
                pipeline.FacialEncoder(
                    encoder_hidden_states.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    multi_facial_embeds.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    facial_token_mask_cuda,
                    facial_token_idx_mask_cuda,
                )
                .float()
                .cpu()
            )

            uncond_prompt_embeds_facial = (
                pipeline.FacialEncoder(
                    negative_text_states.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    uncond_multi_facial_embeds.to(
                        CUDA_DEVICE,
                        dtype=COMPUTE_DTYPE,
                    ),
                    facial_token_mask_cuda,
                    facial_token_idx_mask_cuda,
                )
                .float()
                .cpu()
            )

    finally:
        _move_modules_to_cpu(
            pipeline.FacialEncoder
        )

    return (
        prompt_embeds_facial,
        uncond_prompt_embeds_facial,
        prompt_tokens_faceid,
        uncond_prompt_tokens_faceid,
    )

def _prepare_official_conditioning(
    *,
    pipeline: ConsistentIDStableDiffusionPipeline,
    identity_reference: Image.Image,
    face_embedding: torch.Tensor,
    prompt: str,
    negative_prompt: str,
    adapter_scale: float,
) -> ConsistentIDConditioning:
    pipeline.external_faceid_embeds = (
        face_embedding
    )
    pipeline.set_scale(adapter_scale)

    _move_module(
        pipeline.bise_net,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            (
                key_parsing_mask_list,
                _vis_parsing_anno_color,
            ) = pipeline.get_prepare_facemask(
                identity_reference
            )
    finally:
        _move_modules_to_cpu(
            pipeline.bise_net
        )

    face_caption = (
        pipeline.get_prepare_llva_caption(
            identity_reference
        )
    )

    (
        prompt_text_only,
        clean_input_id,
        key_parsing_mask_list_align,
        facial_token_mask,
        _facial_token_idx,
        facial_token_idx_mask,
    ) = pipeline.encode_prompt_with_trigger_word(
        prompt=prompt,
        face_caption=face_caption,
        key_parsing_mask_list=(
            key_parsing_mask_list
        ),
        device=CPU_DEVICE,
        max_num_facials=5,
        num_id_images=1,
    )

    (
        facial_clip_image,
        _facial_mask,
    ) = pipeline.get_prepare_clip_image(
        identity_reference,
        key_parsing_mask_list_align,
        image_size=512,
        max_num_facials=5,
    )

    (
        encoder_hidden_states,
        negative_text_states,
        positive_text_states,
    ) = _encode_text_conditioning(
        pipeline=pipeline,
        prompt_text_only=prompt_text_only,
        clean_input_id=clean_input_id,
        negative_prompt=negative_prompt,
    )

    (
        prompt_embeds_facial,
        uncond_prompt_embeds_facial,
        prompt_tokens_faceid,
        uncond_prompt_tokens_faceid,
    ) = _encode_identity_conditioning(
        pipeline=pipeline,
        identity_reference=identity_reference,
        face_embedding=face_embedding,
        encoder_hidden_states=(
            encoder_hidden_states
        ),
        negative_text_states=(
            negative_text_states
        ),
        facial_clip_image=facial_clip_image,
        facial_token_mask=facial_token_mask,
        facial_token_idx_mask=(
            facial_token_idx_mask
        ),
    )

    prompt_embeds_augmented = torch.cat(
        [
            prompt_embeds_facial,
            prompt_tokens_faceid,
        ],
        dim=1,
    )

    negative_prompt_embeds_augmented = torch.cat(
        [
            uncond_prompt_embeds_facial,
            uncond_prompt_tokens_faceid,
        ],
        dim=1,
    )

    with torch.inference_mode():
        prompt_embeds = pipeline._encode_prompt(
            prompt,
            device=CPU_DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            prompt_embeds=(
                prompt_embeds_augmented
            ),
            negative_prompt_embeds=(
                negative_prompt_embeds_augmented
            ),
        )

    prompt_embeds_text_only = torch.cat(
        [
            positive_text_states,
            prompt_tokens_faceid,
        ],
        dim=1,
    )

    prompt_embeds = torch.cat(
        [
            prompt_embeds,
            prompt_embeds_text_only,
        ],
        dim=0,
    )

    (
        null_prompt_embeds,
        augmented_prompt_embeds,
        text_prompt_embeds,
    ) = prompt_embeds.chunk(3)

    return ConsistentIDConditioning(
        null_prompt_embeds=(
            null_prompt_embeds
            .contiguous()
            .float()
            .cpu()
        ),
        augmented_prompt_embeds=(
            augmented_prompt_embeds
            .contiguous()
            .float()
            .cpu()
        ),
        text_prompt_embeds=(
            text_prompt_embeds
            .contiguous()
            .float()
            .cpu()
        ),
    )


OPENPOSE_LIMB_SEQUENCE = (
    (2, 3),
    (2, 6),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (2, 12),
    (12, 13),
    (13, 14),
    (2, 1),
    (1, 15),
    (15, 17),
    (1, 16),
    (16, 18),
)

OPENPOSE_COLORS = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


def _create_centered_openpose_control_image() -> Image.Image:
    """Create an upper-body OpenPose condition in the official format.

    Face, eye, ear and hand keypoints are intentionally omitted. The
    condition controls only the centered torso, shoulders and relaxed
    upper arms, leaving ConsistentID responsible for facial identity and
    head details.
    """

    size = AI_OUTPUT_SIZE
    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8,
    )

    keypoints = {
        2: (0.50, 0.24),
        3: (0.36, 0.31),
        4: (0.29, 0.56),
        6: (0.64, 0.31),
        7: (0.71, 0.56),
        9: (0.43, 0.67),
        12: (0.57, 0.67),
    }

    pixel_keypoints = {
        index: (
            int(round(x_ratio * size)),
            int(round(y_ratio * size)),
        )
        for index, (x_ratio, y_ratio) in keypoints.items()
    }

    point_radius = 4
    stick_width = 5

    for index, point in pixel_keypoints.items():
        cv2.circle(
            canvas,
            point,
            point_radius,
            OPENPOSE_COLORS[index - 1],
            thickness=-1,
        )

    for limb_index, limb in enumerate(
        OPENPOSE_LIMB_SEQUENCE
    ):
        first = pixel_keypoints.get(limb[0])
        second = pixel_keypoints.get(limb[1])

        if first is None or second is None:
            continue

        x_values = np.asarray(
            [first[0], second[0]],
            dtype=np.float32,
        )
        y_values = np.asarray(
            [first[1], second[1]],
            dtype=np.float32,
        )

        center_x = float(x_values.mean())
        center_y = float(y_values.mean())
        length = float(
            np.hypot(
                x_values[0] - x_values[1],
                y_values[0] - y_values[1],
            )
        )
        angle = math.degrees(
            math.atan2(
                y_values[0] - y_values[1],
                x_values[0] - x_values[1],
            )
        )

        polygon = cv2.ellipse2Poly(
            (int(round(center_x)), int(round(center_y))),
            (int(round(length / 2.0)), stick_width),
            int(round(angle)),
            0,
            360,
            1,
        )

        limb_canvas = canvas.copy()
        cv2.fillConvexPoly(
            limb_canvas,
            polygon,
            OPENPOSE_COLORS[limb_index],
        )
        canvas = cv2.addWeighted(
            canvas,
            0.4,
            limb_canvas,
            0.6,
            0.0,
        )

    return Image.fromarray(
        canvas,
        mode="RGB",
    )


def _prepare_control_image_tensor(
    image: Image.Image,
) -> torch.Tensor:
    array = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1)
    ).unsqueeze(0)

    return torch.cat(
        [tensor, tensor],
        dim=0,
    )


def _controlnet_is_active(
    step_index: int,
    total_steps: int,
) -> bool:
    if total_steps <= 0:
        return False

    progress_start = step_index / total_steps
    progress_end = (step_index + 1) / total_steps

    return (
        progress_start >= AI_CONTROLNET_GUIDANCE_START
        and progress_end <= AI_CONTROLNET_GUIDANCE_END
    )


def _guard_latents(
    step_index: int,
    timestep: int,
    latents: torch.Tensor,
) -> None:
    if bool(
        torch.isfinite(latents).all().item()
    ):
        return

    nan_count = int(
        torch.isnan(latents).sum().item()
    )

    inf_count = int(
        torch.isinf(latents).sum().item()
    )

    raise RuntimeError(
        "Denoising produced non-finite latents in the "
        "official-style ConsistentID SD 1.5 FP32 profile. "
        f"step={step_index}, "
        f"timestep={int(timestep)}, "
        f"nan={nan_count}, "
        f"inf={inf_count}."
    )


def _run_official_denoising(
    *,
    pipeline: ConsistentIDStableDiffusionPipeline,
    controlnet: ControlNetModel | None,
    control_image: Image.Image | None,
    conditioning: ConsistentIDConditioning,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    start_merge_step: int,
) -> torch.Tensor:
    if start_merge_step >= num_inference_steps:
        raise ValueError(
            "start_merge_step must be smaller than "
            "num_inference_steps."
        )

    if (controlnet is None) != (control_image is None):
        raise ValueError(
            "ControlNet and its control image must either both be "
            "provided or both be omitted."
        )

    _move_module(
        pipeline.unet,
        CUDA_DEVICE,
    )

    control_image_tensor: torch.Tensor | None = None
    controlnet_on_cuda = False

    if controlnet is not None and control_image is not None:
        _move_module_with_dtype(
            controlnet,
            CUDA_DEVICE,
            CONTROLNET_DTYPE,
        )

        controlnet_on_cuda = True

        control_image_tensor = (
            _prepare_control_image_tensor(
                control_image
            )
            .to(
                CUDA_DEVICE,
                dtype=CONTROLNET_DTYPE,
            )
        )

    null_prompt_embeds = (
        conditioning.null_prompt_embeds.to(
            CUDA_DEVICE,
            dtype=COMPUTE_DTYPE,
        )
    )

    augmented_prompt_embeds = (
        conditioning.augmented_prompt_embeds.to(
            CUDA_DEVICE,
            dtype=COMPUTE_DTYPE,
        )
    )

    text_prompt_embeds = (
        conditioning.text_prompt_embeds.to(
            CUDA_DEVICE,
            dtype=COMPUTE_DTYPE,
        )
    )

    generator = torch.Generator(
        device=CUDA_DEVICE
    ).manual_seed(seed)

    try:
        pipeline.scheduler.set_timesteps(
            num_inference_steps,
            device=CUDA_DEVICE,
        )

        timesteps = pipeline.scheduler.timesteps

        latents = pipeline.prepare_latents(
            1,
            pipeline.unet.config.in_channels,
            AI_OUTPUT_SIZE,
            AI_OUTPUT_SIZE,
            COMPUTE_DTYPE,
            CUDA_DEVICE,
            generator,
            None,
        )

        extra_step_kwargs = (
            pipeline.prepare_extra_step_kwargs(
                generator,
                0.0,
            )
        )

        num_warmup_steps = (
            len(timesteps)
            - num_inference_steps
            * pipeline.scheduler.order
        )

        with (
            torch.inference_mode(),
            pipeline.progress_bar(
                total=num_inference_steps
            ) as progress_bar,
        ):
            for index, timestep in enumerate(
                timesteps
            ):
                latent_model_input = torch.cat(
                    [latents] * 2
                )

                latent_model_input = (
                    pipeline.scheduler
                    .scale_model_input(
                        latent_model_input,
                        timestep,
                    )
                )

                if index <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [
                            null_prompt_embeds,
                            text_prompt_embeds,
                        ],
                        dim=0,
                    )
                else:
                    current_prompt_embeds = torch.cat(
                        [
                            null_prompt_embeds,
                            augmented_prompt_embeds,
                        ],
                        dim=0,
                    )

                down_block_residuals: (
                    list[torch.Tensor] | None
                ) = None
                mid_block_residual: torch.Tensor | None = None

                controlnet_active = _controlnet_is_active(
                    index,
                    len(timesteps),
                )

                if (
                    controlnet is not None
                    and control_image_tensor is not None
                    and controlnet_active
                ):
                    (
                        down_block_residuals_raw,
                        mid_block_residual_raw,
                    ) = controlnet(
                        latent_model_input.to(
                            dtype=CONTROLNET_DTYPE
                        ),
                        timestep,
                        encoder_hidden_states=(
                            current_prompt_embeds.to(
                                dtype=CONTROLNET_DTYPE
                            )
                        ),
                        controlnet_cond=(
                            control_image_tensor
                        ),
                        conditioning_scale=(
                            AI_CONTROLNET_CONDITIONING_SCALE
                        ),
                        guess_mode=False,
                        return_dict=False,
                    )

                    if not all(
                        bool(torch.isfinite(residual).all().item())
                        for residual in down_block_residuals_raw
                    ) or not bool(
                        torch.isfinite(
                            mid_block_residual_raw
                        ).all().item()
                    ):
                        raise RuntimeError(
                            "ControlNet OpenPose produced non-finite "
                            "residuals. Disable AI_CONTROLNET_ENABLED "
                            "to return to the exact v15 generation path."
                        )

                    down_block_residuals = [
                        residual.to(
                            dtype=COMPUTE_DTYPE
                        )
                        for residual in (
                            down_block_residuals_raw
                        )
                    ]

                    mid_block_residual = (
                        mid_block_residual_raw.to(
                            dtype=COMPUTE_DTYPE
                        )
                    )

                    del down_block_residuals_raw
                    del mid_block_residual_raw

                elif (
                    controlnet is not None
                    and controlnet_on_cuda
                    and (
                        index / max(len(timesteps), 1)
                        >= AI_CONTROLNET_GUIDANCE_END
                    )
                ):
                    controlnet.to(
                        device=CPU_DEVICE,
                        dtype=CONTROLNET_DTYPE,
                    )
                    control_image_tensor = None
                    controlnet_on_cuda = False
                    _release_cuda_memory()

                unet_kwargs: dict[str, object] = {
                    "encoder_hidden_states": (
                        current_prompt_embeds
                    ),
                    "cross_attention_kwargs": {},
                }

                if down_block_residuals is not None:
                    unet_kwargs[
                        "down_block_additional_residuals"
                    ] = down_block_residuals

                if mid_block_residual is not None:
                    unet_kwargs[
                        "mid_block_additional_residual"
                    ] = mid_block_residual

                noise_pred = pipeline.unet(
                    latent_model_input,
                    timestep,
                    **unet_kwargs,
                ).sample

                (
                    noise_pred_uncond,
                    noise_pred_text,
                ) = noise_pred.chunk(2)

                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale
                    * (
                        noise_pred_text
                        - noise_pred_uncond
                    )
                )

                latents = pipeline.scheduler.step(
                    noise_pred,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                ).prev_sample

                _guard_latents(
                    index,
                    int(timestep),
                    latents,
                )

                del down_block_residuals
                del mid_block_residual

                if (
                    index == len(timesteps) - 1
                    or (
                        index + 1 > num_warmup_steps
                        and (
                            index + 1
                        )
                        % pipeline.scheduler.order
                        == 0
                    )
                ):
                    progress_bar.update()

        return latents

    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "GTX 1660 ran out of VRAM during the FP32 UNet "
            "and mixed-precision ControlNet OpenPose denoising "
            "stage."
        ) from exc

    finally:
        _release_cuda_memory()


def _decode_latents(
    *,
    pipeline: ConsistentIDStableDiffusionPipeline,
    latents: torch.Tensor,
) -> Image.Image:
    _move_module(
        pipeline.vae,
        CUDA_DEVICE,
    )

    try:
        with torch.inference_mode():
            image_array = pipeline.decode_latents(
                latents.to(
                    CUDA_DEVICE,
                    dtype=COMPUTE_DTYPE,
                )
            )

        return pipeline.numpy_to_pil(
            image_array
        )[0]

    finally:
        pipeline.vae = None
        _release_cuda_memory()
        _log_memory("vae_released")


def _validate_generated_image(
    image: Image.Image,
) -> None:
    array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    if array.size == 0:
        raise RuntimeError(
            "Generated image is empty."
        )

    minimum = int(array.min())
    maximum = int(array.max())
    deviation = float(array.std())

    if (
        maximum <= 8
        or (
            maximum - minimum <= 3
            and deviation < 1.0
        )
    ):
        raise RuntimeError(
            "Generated image is blank or black. "
            f"min={minimum}, "
            f"max={maximum}, "
            f"std={deviation:.4f}."
        )


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service_version": SERVICE_VERSION,
        "cuda_available": (
            torch.cuda.is_available()
        ),
        "cuda_device": (
            torch.cuda.get_device_name(
                AI_CUDA_DEVICE_INDEX
            )
            if torch.cuda.is_available()
            else None
        ),
        "compute_dtype": "float32",
        "output_size": AI_OUTPUT_SIZE,
        "generation_mode": (
            "official-consistentid-v1-staged"
        ),
        "memory_mode": (
            "manual-component-staging-no-accelerate-hooks"
        ),
        "xformers_available": (
            XFORMERS_AVAILABLE
        ),
        "busy": _RUNTIME_BUSY,
        "models_loaded": _RUNTIME is not None,
        "models_cached_between_requests": True,
        "controlnet_openpose_enabled": AI_CONTROLNET_ENABLED,
        "controlnet_conditioning_scale": (
            AI_CONTROLNET_CONDITIONING_SCALE
        ),
        "controlnet_guidance_start": (
            AI_CONTROLNET_GUIDANCE_START
        ),
        "controlnet_guidance_end": (
            AI_CONTROLNET_GUIDANCE_END
        ),
    }


@app.get("/ready")
def ready() -> dict[str, object]:
    checks = {
        "cuda": torch.cuda.is_available(),
        "xformers": XFORMERS_AVAILABLE,
        **_check_model_files(),
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail=checks,
        )

    return {
        "status": "ready",
        "checks": checks,
    }


@app.post(
    "/v1/portraits/generate",
    response_model=PortraitResponse,
)
def generate_portrait(
    request: PortraitRequest,
) -> PortraitResponse:
    global _RUNTIME_BUSY

    with INFERENCE_LOCK:
        _RUNTIME_BUSY = True
        stage = "request_validation"

        try:
            job_directory = _job_directory(
                request.job_id
            )

            identity_reference_name = (
                _validate_filename(
                    request.identity_reference_name,
                    {".png", ".jpg", ".jpeg", ".webp"},
                )
            )

            embedding_name = _validate_filename(
                request.face_embedding_name,
                {".npy"},
            )

            output_name = _validate_filename(
                request.output_name,
                {".png"},
            )

            stage = "load_identity_reference"

            identity_reference = _load_rgb(
                job_directory
                / identity_reference_name
            )

            stage = "load_face_embedding"

            face_embedding = _load_face_embedding(
                job_directory / embedding_name
            )

            stage = "load_consistentid_runtime"

            runtime = _load_runtime()
            pipeline = runtime.pipeline

            control_image: Image.Image | None = None

            if runtime.controlnet is not None:
                stage = "prepare_centered_openpose_control"

                control_image = (
                    _create_centered_openpose_control_image()
                )

                control_image.save(
                    job_directory / "control_pose.png",
                    format="PNG",
                    optimize=True,
                )

            stage = "official_conditioning"

            conditioning = (
                _prepare_official_conditioning(
                    pipeline=pipeline,
                    identity_reference=(
                        identity_reference
                    ),
                    face_embedding=face_embedding,
                    prompt=request.prompt,
                    negative_prompt=(
                        request.negative_prompt
                    ),
                    adapter_scale=(
                        request.adapter_scale
                    ),
                )
            )

            del identity_reference
            del face_embedding

            _release_conditioning_modules(
                pipeline
            )

            stage = "official_denoising"

            latents = _run_official_denoising(
                pipeline=pipeline,
                controlnet=runtime.controlnet,
                control_image=control_image,
                conditioning=conditioning,
                seed=request.seed,
                num_inference_steps=(
                    request.num_inference_steps
                ),
                guidance_scale=(
                    request.guidance_scale
                ),
                start_merge_step=(
                    request.start_merge_step
                ),
            )

            del conditioning
            del control_image

            _release_denoising_modules(
                runtime
            )

            stage = "vae_decode"

            result = _decode_latents(
                pipeline=pipeline,
                latents=latents,
            ).convert("RGB")

            del latents

            stage = "validate_generated_image"

            _validate_generated_image(result)

            stage = "save_result"

            output_path = (
                job_directory / output_name
            )

            result.save(
                output_path,
                format="PNG",
                optimize=True,
            )

            return PortraitResponse(
                output_path=str(output_path),
                seed=request.seed,
            )

        except HTTPException:
            raise

        except Exception as exc:
            traceback.print_exc()

            if isinstance(
                exc,
                GenerationStageError,
            ):
                detail = str(exc)
            else:
                detail = (
                    f"stage={stage}; "
                    f"type={type(exc).__name__}; "
                    f"message={exc}"
                )

            raise HTTPException(
                status_code=500,
                detail=detail,
            ) from exc

        finally:
            _RUNTIME_BUSY = False
            _drop_runtime()


def _run_batch_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Batch manifest does not exist: "
            f"{manifest_path}"
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Could not read ConsistentID batch manifest."
        ) from exc

    raw_requests = manifest.get("requests")

    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError(
            "ConsistentID batch manifest must contain a non-empty "
            "requests list."
        )

    response_path_raw = manifest.get("response_path")

    if not isinstance(response_path_raw, str) or not response_path_raw:
        raise ValueError(
            "ConsistentID batch manifest must contain response_path."
        )

    response_path = Path(response_path_raw).resolve()
    results: list[dict[str, object]] = []

    for request_index, raw_request in enumerate(
        raw_requests,
        start=1,
    ):
        if not isinstance(raw_request, dict):
            raise ValueError(
                "ConsistentID batch request "
                f"{request_index} is not an object."
            )

        request = PortraitRequest(**raw_request)

        print(
            "[consistentid-batch] generating "
            f"seed={request.seed}; "
            f"output={request.output_name}",
            flush=True,
        )

        try:
            response = generate_portrait(request)
        except HTTPException as exc:
            detail = exc.detail
            raise RuntimeError(
                "ConsistentID batch request failed: "
                f"seed={request.seed}; detail={detail}"
            ) from exc

        results.append(response.model_dump())

    response_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_response = response_path.with_suffix(
        response_path.suffix + ".tmp"
    )

    temporary_response.write_text(
        json.dumps(
            {"results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_response.replace(response_path)

    print(
        "[consistentid-batch] completed "
        f"count={len(results)}",
        flush=True,
    )


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official staged FP32 ConsistentID pipeline "
            "for all requests in one manifest."
        )
    )
    parser.add_argument(
        "--batch-manifest",
        required=True,
    )
    arguments = parser.parse_args()

    try:
        _run_batch_manifest(
            Path(arguments.batch_manifest).resolve()
        )
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    _cli_main()