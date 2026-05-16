import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image


DEFAULT_PROMPT = (
    "professional corporate headshot, formal business portrait, "
    "person wearing a dark navy business suit jacket and white collared dress shirt, "
    "conservative office clothing, clean corporate style, studio lighting, "
    "realistic photo, high quality, sharp details"
)

DEFAULT_NEGATIVE_PROMPT = (
    "changed face, distorted face, changed eyes, distorted eyes, deformed mouth, "
    "bad anatomy, extra limbs, low quality, blurry, artifacts, cartoon, "
    "t-shirt, casual shirt, sportswear, hoodie, fantasy armor, medieval costume, "
    "robe, dress, exposed shoulders, cleavage, bare chest, muscular arms, bodybuilder"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental Stable Diffusion inpainting for corporate avatars."
    )

    parser.add_argument(
        "--job-dir",
        type=str,
        default="",
        help="Path to job directory, for example /app/storage/jobs/<job_id>",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="result.png",
        help="Input image filename inside job-dir.",
    )
    parser.add_argument(
        "--mask-name",
        type=str,
        default="clothes_mask.png",
        help="Mask filename inside job-dir. White pixels are repainted.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=os.getenv("AI_OUTPUT_NAME", "ai_result.png"),
        help="Output image filename inside job-dir.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv(
            "AI_MODEL_ID",
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
        ),
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("AI_DEVICE", "cuda"),
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=os.getenv("AI_DTYPE", "float32"),
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--low-vram",
        action="store_true",
        default=os.getenv("AI_LOW_VRAM", "true").lower() == "true",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check PyTorch/CUDA availability.",
    )

    return parser.parse_args()


def print_cuda_info() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"cuda device count: {torch.cuda.device_count()}")
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"cuda total memory MiB: {total_memory // 1024 // 1024}")


def load_image(path: Path, mode: str) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    image = Image.open(path).convert(mode)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)

    return image


def main() -> None:
    args = parse_args()

    print_cuda_info()

    if args.check:
        return

    if not args.job_dir:
        raise ValueError("Argument --job-dir is required unless --check is used.")

    job_dir = Path(args.job_dir)

    input_path = job_dir / args.input_name
    mask_path = job_dir / args.mask_name
    output_path = job_dir / args.output_name

    image = load_image(input_path, "RGB")
    mask = load_image(mask_path, "L")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    if args.dtype == "float16" and use_cuda:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"model id: {args.model_id}")
    print(f"input image: {input_path}")
    print(f"mask image: {mask_path}")
    print(f"output image: {output_path}")
    print(f"device: {'cuda' if use_cuda else 'cpu'}")
    print(f"dtype: {torch_dtype}")
    print(f"low_vram: {args.low_vram}")

    token = os.getenv("HF_TOKEN") or None

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        token=token,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if use_cuda:
        if args.low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    generator_device = "cuda" if use_cuda and not args.low_vram else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            generator=generator,
        ).images[0]

    result.save(output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()