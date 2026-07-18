from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


MODELS_ROOT = Path(
    os.getenv(
        "MODELS_ROOT",
        "/models",
    )
).resolve()

AI_BASE_MODEL_DIR = Path(
    os.getenv(
        "AI_BASE_MODEL_DIR",
        str(MODELS_ROOT / "base_model"),
    )
).resolve()

AI_VAE_MODEL_DIR = Path(
    os.getenv(
        "AI_VAE_MODEL_DIR",
        str(MODELS_ROOT / "vae"),
    )
).resolve()

AI_CLIP_MODEL_DIR = Path(
    os.getenv(
        "AI_CLIP_MODEL_DIR",
        str(MODELS_ROOT / "clip_vision"),
    )
).resolve()

CONSISTENTID_MODEL_DIR = Path(
    os.getenv(
        "CONSISTENTID_MODEL_DIR",
        str(MODELS_ROOT / "consistentid"),
    )
).resolve()

AI_CONTROLNET_MODEL_DIR = Path(
    os.getenv(
        "AI_CONTROLNET_MODEL_DIR",
        str(MODELS_ROOT / "controlnet_openpose"),
    )
).resolve()

REMBG_MODEL_DIR = Path(
    os.getenv(
        "REMBG_MODEL_DIR",
        str(MODELS_ROOT / "rembg"),
    )
).resolve()

INSIGHTFACE_ROOT = Path(
    os.getenv(
        "INSIGHTFACE_ROOT",
        str(MODELS_ROOT / "insightface"),
    )
).resolve()

AI_BASE_MODEL_ID = os.getenv(
    "AI_BASE_MODEL_ID",
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
)

AI_BASE_MODEL_REVISION = os.getenv(
    "AI_BASE_MODEL_REVISION",
    "main",
)

AI_VAE_MODEL_ID = os.getenv(
    "AI_VAE_MODEL_ID",
    "stabilityai/sd-vae-ft-mse",
)

AI_VAE_MODEL_REVISION = os.getenv(
    "AI_VAE_MODEL_REVISION",
    "main",
)

AI_CLIP_MODEL_ID = os.getenv(
    "AI_CLIP_MODEL_ID",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
)

AI_CLIP_MODEL_REVISION = os.getenv(
    "AI_CLIP_MODEL_REVISION",
    "main",
)

CONSISTENTID_REPO_ID = os.getenv(
    "CONSISTENTID_REPO_ID",
    "JackAILab/ConsistentID",
)

CONSISTENTID_REVISION = os.getenv(
    "CONSISTENTID_REVISION",
    "main",
)

CONSISTENTID_WEIGHT_NAME = os.getenv(
    "CONSISTENTID_WEIGHT_NAME",
    "ConsistentID-v1.bin",
)

CONSISTENTID_FACE_PARSING_NAME = os.getenv(
    "CONSISTENTID_FACE_PARSING_NAME",
    "face_parsing.pth",
)

AI_CONTROLNET_MODEL_ID = os.getenv(
    "AI_CONTROLNET_MODEL_ID",
    "lllyasviel/control_v11p_sd15_openpose",
)

AI_CONTROLNET_REVISION = os.getenv(
    "AI_CONTROLNET_REVISION",
    "9ae9f970358db89e211b87c915f9535c6686d5ba",
)

REMBG_MODEL_NAME = os.getenv(
    "REMBG_MODEL_NAME",
    "birefnet-portrait.onnx",
)

REMBG_MODEL_URL = os.getenv(
    "REMBG_MODEL_URL",
    (
        "https://github.com/danielgatis/rembg/releases/"
        "download/v0.0.0/BiRefNet-portrait-epoch_150.onnx"
    ),
)

REMBG_MODEL_MD5 = os.getenv(
    "REMBG_MODEL_MD5",
    "c3a64a6abf20250d090cd055f12a3b67",
).lower()

INSWAPPER_MODEL_NAME = os.getenv(
    "INSWAPPER_MODEL_NAME",
    "inswapper_128.onnx",
)

INSWAPPER_REPO_ID = os.getenv(
    "INSWAPPER_REPO_ID",
    "ezioruan/inswapper_128.onnx",
)

INSWAPPER_REVISION = os.getenv(
    "INSWAPPER_REVISION",
    "6ffdf0e83c5996cc425e77b59913fc48d79441be",
)

INSWAPPER_SHA256 = os.getenv(
    "INSWAPPER_SHA256",
    "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
).lower()

HF_TOKEN = os.getenv("HF_TOKEN") or None

MODEL_CLEANUP_ENABLED = os.getenv(
    "MODEL_CLEANUP_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ANTELOPEV2_URL = os.getenv(
    "ANTELOPEV2_URL",
    (
        "https://github.com/deepinsight/"
        "insightface/releases/download/"
        "v0.7/antelopev2.zip"
    ),
)

BUFFALO_L_URL = os.getenv(
    "BUFFALO_L_URL",
    (
        "https://github.com/deepinsight/"
        "insightface/releases/download/"
        "v0.7/buffalo_l.zip"
    ),
)

EXPECTED_ANTELOPE_FILES = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
    "glintr100.onnx",
    "scrfd_10g_bnkps.onnx",
}

EXPECTED_BUFFALO_L_FILES = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
}

BASE_MODEL_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "text_encoder/config.json",
    "unet/config.json",
)

VAE_REQUIRED_FILES = (
    "config.json",
)

CLIP_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
)

CONTROLNET_REQUIRED_FILES = (
    "config.json",
    "diffusion_pytorch_model.fp16.safetensors",
)

DEPRECATED_MODEL_DIRECTORIES = (
    MODELS_ROOT / "huggingface",
    MODELS_ROOT / "ip_adapter_faceid",
)


def log(message: str) -> None:
    print(
        f"[models-init] {message}",
        flush=True,
    )


def ensure_directory(path: Path) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def file_is_ready(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        while True:
            chunk = model_file.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def calculate_md5(path: Path) -> str:
    digest = hashlib.md5()

    with path.open("rb") as model_file:
        while True:
            chunk = model_file.read(8 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def file_has_expected_md5(
    path: Path,
    expected_md5: str,
) -> bool:
    if not file_is_ready(path):
        return False

    return calculate_md5(path) == expected_md5.lower()


def file_has_expected_sha256(
    path: Path,
    expected_sha256: str,
) -> bool:
    if not file_is_ready(path):
        return False

    return (
        calculate_sha256(path)
        == expected_sha256.lower()
    )


def _marker_path(directory: Path) -> Path:
    return directory / ".model_source"


def _expected_marker(
    *,
    repo_id: str,
    revision: str,
) -> str:
    return f"{repo_id}@{revision}\n"


def _prepare_local_model_directory(
    *,
    directory: Path,
    repo_id: str,
    revision: str,
) -> None:
    expected_marker = _expected_marker(
        repo_id=repo_id,
        revision=revision,
    )

    marker = _marker_path(directory)

    if directory.exists() and marker.is_file():
        current_marker = marker.read_text(
            encoding="utf-8"
        )

        if current_marker != expected_marker:
            log(
                "Removing model directory with a different source: "
                f"{directory}"
            )
            shutil.rmtree(directory)

    ensure_directory(directory)


def _write_model_marker(
    *,
    directory: Path,
    repo_id: str,
    revision: str,
) -> None:
    _marker_path(directory).write_text(
        _expected_marker(
            repo_id=repo_id,
            revision=revision,
        ),
        encoding="utf-8",
    )


def _find_weight_file(
    directory: Path,
    candidates: tuple[str, ...],
) -> Path | None:
    for relative_name in candidates:
        candidate = directory / relative_name

        if file_is_ready(candidate):
            return candidate

    return None


def cleanup_deprecated_models() -> None:
    if not MODEL_CLEANUP_ENABLED:
        log("Deprecated model cleanup is disabled.")
        return

    for directory in DEPRECATED_MODEL_DIRECTORIES:
        if not directory.exists():
            continue

        resolved = directory.resolve()

        if resolved == INSIGHTFACE_ROOT:
            raise RuntimeError(
                "Refusing to delete the active InsightFace directory."
            )

        if MODELS_ROOT not in resolved.parents:
            raise RuntimeError(
                "Refusing to delete a path outside MODELS_ROOT: "
                f"{resolved}"
            )

        log(
            "Removing deprecated model directory: "
            f"{resolved}"
        )
        shutil.rmtree(resolved)


def download_base_model() -> Path:
    _prepare_local_model_directory(
        directory=AI_BASE_MODEL_DIR,
        repo_id=AI_BASE_MODEL_ID,
        revision=AI_BASE_MODEL_REVISION,
    )

    log(
        "Preparing Realistic Vision V6 Diffusers components: "
        f"{AI_BASE_MODEL_ID}@{AI_BASE_MODEL_REVISION}"
    )

    snapshot_download(
        repo_id=AI_BASE_MODEL_ID,
        revision=AI_BASE_MODEL_REVISION,
        local_dir=str(AI_BASE_MODEL_DIR),
        token=HF_TOKEN,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "unet/*",
        ],
        ignore_patterns=[
            "*.msgpack",
            "*.onnx",
            "*.xml",
            "*.h5",
            "*.ot",
        ],
    )

    for relative_name in BASE_MODEL_REQUIRED_FILES:
        path = AI_BASE_MODEL_DIR / relative_name

        if not file_is_ready(path):
            raise RuntimeError(
                "Base model file is missing: "
                f"{path}"
            )

    text_encoder_weight = _find_weight_file(
        AI_BASE_MODEL_DIR,
        (
            "text_encoder/model.safetensors",
            "text_encoder/pytorch_model.bin",
        ),
    )

    unet_weight = _find_weight_file(
        AI_BASE_MODEL_DIR,
        (
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
        ),
    )

    if text_encoder_weight is None:
        raise RuntimeError(
            "Base model text encoder weight was not downloaded."
        )

    if unet_weight is None:
        raise RuntimeError(
            "Base model UNet weight was not downloaded."
        )

    _write_model_marker(
        directory=AI_BASE_MODEL_DIR,
        repo_id=AI_BASE_MODEL_ID,
        revision=AI_BASE_MODEL_REVISION,
    )

    log(
        "Realistic Vision V6 Diffusers model is ready: "
        f"{AI_BASE_MODEL_DIR}"
    )

    return AI_BASE_MODEL_DIR


def download_vae() -> Path:
    _prepare_local_model_directory(
        directory=AI_VAE_MODEL_DIR,
        repo_id=AI_VAE_MODEL_ID,
        revision=AI_VAE_MODEL_REVISION,
    )

    log(
        "Preparing SD 1.5 VAE: "
        f"{AI_VAE_MODEL_ID}@{AI_VAE_MODEL_REVISION}"
    )

    snapshot_download(
        repo_id=AI_VAE_MODEL_ID,
        revision=AI_VAE_MODEL_REVISION,
        local_dir=str(AI_VAE_MODEL_DIR),
        token=HF_TOKEN,
        allow_patterns=[
            "config.json",
            "diffusion_pytorch_model.*",
        ],
    )

    for relative_name in VAE_REQUIRED_FILES:
        path = AI_VAE_MODEL_DIR / relative_name

        if not file_is_ready(path):
            raise RuntimeError(
                "VAE file is missing: "
                f"{path}"
            )

    vae_weight = _find_weight_file(
        AI_VAE_MODEL_DIR,
        (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
        ),
    )

    if vae_weight is None:
        raise RuntimeError(
            "VAE weight was not downloaded."
        )

    _write_model_marker(
        directory=AI_VAE_MODEL_DIR,
        repo_id=AI_VAE_MODEL_ID,
        revision=AI_VAE_MODEL_REVISION,
    )

    log(
        "SD 1.5 VAE is ready: "
        f"{AI_VAE_MODEL_DIR}"
    )

    return AI_VAE_MODEL_DIR


def download_clip_vision() -> Path:
    _prepare_local_model_directory(
        directory=AI_CLIP_MODEL_DIR,
        repo_id=AI_CLIP_MODEL_ID,
        revision=AI_CLIP_MODEL_REVISION,
    )

    log(
        "Preparing CLIP ViT-H encoder: "
        f"{AI_CLIP_MODEL_ID}@{AI_CLIP_MODEL_REVISION}"
    )

    snapshot_download(
        repo_id=AI_CLIP_MODEL_ID,
        revision=AI_CLIP_MODEL_REVISION,
        local_dir=str(AI_CLIP_MODEL_DIR),
        token=HF_TOKEN,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ],
    )

    for relative_name in CLIP_REQUIRED_FILES:
        path = AI_CLIP_MODEL_DIR / relative_name

        if not file_is_ready(path):
            raise RuntimeError(
                "CLIP file is missing: "
                f"{path}"
            )

    clip_weight = _find_weight_file(
        AI_CLIP_MODEL_DIR,
        (
            "model.safetensors",
            "pytorch_model.bin",
        ),
    )

    if clip_weight is None:
        raise RuntimeError(
            "CLIP ViT-H weight was not downloaded."
        )

    _write_model_marker(
        directory=AI_CLIP_MODEL_DIR,
        repo_id=AI_CLIP_MODEL_ID,
        revision=AI_CLIP_MODEL_REVISION,
    )

    log(
        "CLIP ViT-H encoder is ready: "
        f"{AI_CLIP_MODEL_DIR}"
    )

    return AI_CLIP_MODEL_DIR


def download_consistentid() -> Path:
    _prepare_local_model_directory(
        directory=CONSISTENTID_MODEL_DIR,
        repo_id=CONSISTENTID_REPO_ID,
        revision=CONSISTENTID_REVISION,
    )

    log(
        "Preparing ConsistentID V1 weights: "
        f"{CONSISTENTID_REPO_ID}@{CONSISTENTID_REVISION}"
    )

    for filename in (
        CONSISTENTID_WEIGHT_NAME,
        CONSISTENTID_FACE_PARSING_NAME,
    ):
        downloaded_path = Path(
            hf_hub_download(
                repo_id=CONSISTENTID_REPO_ID,
                filename=filename,
                revision=CONSISTENTID_REVISION,
                local_dir=str(CONSISTENTID_MODEL_DIR),
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
        )

        if not file_is_ready(downloaded_path):
            raise RuntimeError(
                "ConsistentID file is missing or empty: "
                f"{downloaded_path}"
            )

    _write_model_marker(
        directory=CONSISTENTID_MODEL_DIR,
        repo_id=CONSISTENTID_REPO_ID,
        revision=CONSISTENTID_REVISION,
    )

    log(
        "ConsistentID V1 weights are ready: "
        f"{CONSISTENTID_MODEL_DIR}"
    )

    return CONSISTENTID_MODEL_DIR


def download_controlnet_openpose() -> Path:
    _prepare_local_model_directory(
        directory=AI_CONTROLNET_MODEL_DIR,
        repo_id=AI_CONTROLNET_MODEL_ID,
        revision=AI_CONTROLNET_REVISION,
    )

    log(
        "Preparing official ControlNet v1.1 OpenPose model: "
        f"{AI_CONTROLNET_MODEL_ID}@{AI_CONTROLNET_REVISION}"
    )

    snapshot_download(
        repo_id=AI_CONTROLNET_MODEL_ID,
        revision=AI_CONTROLNET_REVISION,
        local_dir=str(AI_CONTROLNET_MODEL_DIR),
        token=HF_TOKEN,
        allow_patterns=[
            "config.json",
            "diffusion_pytorch_model.fp16.safetensors",
        ],
    )

    for relative_name in CONTROLNET_REQUIRED_FILES:
        path = AI_CONTROLNET_MODEL_DIR / relative_name

        if not file_is_ready(path):
            raise RuntimeError(
                "ControlNet OpenPose file is missing: "
                f"{path}"
            )

    _write_model_marker(
        directory=AI_CONTROLNET_MODEL_DIR,
        repo_id=AI_CONTROLNET_MODEL_ID,
        revision=AI_CONTROLNET_REVISION,
    )

    log(
        "ControlNet v1.1 OpenPose is ready: "
        f"{AI_CONTROLNET_MODEL_DIR}"
    )

    return AI_CONTROLNET_MODEL_DIR


def download_url(
    url: str,
    destination: Path,
) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "corporate-avatar-service/"
                "models-init"
            )
        },
    )

    log(f"Downloading: {url}")

    with urllib.request.urlopen(
        request,
        timeout=1200,
    ) as response:
        status = getattr(
            response,
            "status",
            200,
        )

        if status != 200:
            raise RuntimeError(
                "Model download failed. "
                f"HTTP status={status}."
            )

        with destination.open("wb") as output:
            shutil.copyfileobj(
                response,
                output,
            )


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
) -> None:
    destination = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (
                destination / member.filename
            ).resolve()

            if (
                member_path != destination
                and destination not in member_path.parents
            ):
                raise RuntimeError(
                    "Unsafe path inside ZIP archive: "
                    f"{member.filename}"
                )

        archive.extractall(destination)


def model_pack_is_complete(
    model_directory: Path,
    expected_files: set[str],
) -> bool:
    if not model_directory.is_dir():
        return False

    actual_files = {
        path.name
        for path in model_directory.glob("*.onnx")
        if file_is_ready(path)
    }

    return expected_files.issubset(actual_files)


def find_model_pack_directory(
    extracted_root: Path,
    expected_files: set[str],
) -> Path:
    candidates = [extracted_root]
    candidates.extend(
        path
        for path in extracted_root.rglob("*")
        if path.is_dir()
    )

    for candidate in candidates:
        if model_pack_is_complete(
            candidate,
            expected_files,
        ):
            return candidate

    raise RuntimeError(
        "Downloaded InsightFace archive does not "
        "contain the expected ONNX files."
    )


def download_insightface_pack(
    *,
    pack_name: str,
    url: str,
    expected_files: set[str],
) -> Path:
    target_directory = (
        INSIGHTFACE_ROOT
        / "models"
        / pack_name
    )

    if model_pack_is_complete(
        target_directory,
        expected_files,
    ):
        log(
            f"{pack_name} already exists: "
            f"{target_directory}"
        )
        return target_directory

    ensure_directory(target_directory.parent)

    with tempfile.TemporaryDirectory(
        prefix=f"{pack_name}-"
    ) as temporary_directory_name:
        temporary_directory = Path(
            temporary_directory_name
        )

        archive_path = (
            temporary_directory
            / f"{pack_name}.zip"
        )

        extracted_directory = (
            temporary_directory
            / "extracted"
        )

        ensure_directory(extracted_directory)
        download_url(url, archive_path)

        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                f"Downloaded {pack_name} file is "
                "not a valid ZIP archive."
            )

        safe_extract_zip(
            archive_path,
            extracted_directory,
        )

        source_directory = (
            find_model_pack_directory(
                extracted_directory,
                expected_files,
            )
        )

        if target_directory.exists():
            shutil.rmtree(target_directory)

        shutil.copytree(
            source_directory,
            target_directory,
        )

    if not model_pack_is_complete(
        target_directory,
        expected_files,
    ):
        raise RuntimeError(
            f"{pack_name} installation failed."
        )

    log(
        f"{pack_name} installed: "
        f"{target_directory}"
    )

    return target_directory


def prepare_inswapper_model() -> Path:
    target_directory = (
        INSIGHTFACE_ROOT / "models"
    )

    target_path = (
        target_directory
        / INSWAPPER_MODEL_NAME
    )

    ensure_directory(target_directory)

    if file_has_expected_sha256(
        target_path,
        INSWAPPER_SHA256,
    ):
        log(
            "InSwapper already exists and passed SHA-256 validation: "
            f"{target_path}"
        )
        return target_path

    if target_path.exists():
        log(
            "Removing invalid or incomplete InSwapper model: "
            f"{target_path}"
        )
        target_path.unlink()

    downloaded_path = Path(
        hf_hub_download(
            repo_id=INSWAPPER_REPO_ID,
            filename=INSWAPPER_MODEL_NAME,
            revision=INSWAPPER_REVISION,
            local_dir=str(target_directory),
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
    )

    if not file_is_ready(downloaded_path):
        raise RuntimeError(
            "InSwapper download returned an empty file: "
            f"{downloaded_path}"
        )

    actual_sha256 = calculate_sha256(downloaded_path)

    if actual_sha256 != INSWAPPER_SHA256:
        downloaded_path.unlink(missing_ok=True)
        raise RuntimeError(
            "InSwapper SHA-256 mismatch. "
            f"Expected={INSWAPPER_SHA256}, "
            f"actual={actual_sha256}."
        )

    if downloaded_path.resolve() != target_path.resolve():
        temporary_path = target_path.with_suffix(
            target_path.suffix + ".part"
        )
        temporary_path.unlink(missing_ok=True)
        shutil.copyfile(downloaded_path, temporary_path)

        if not file_has_expected_sha256(
            temporary_path,
            INSWAPPER_SHA256,
        ):
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Copied InSwapper failed SHA-256 validation."
            )

        temporary_path.replace(target_path)

    if not file_has_expected_sha256(
        target_path,
        INSWAPPER_SHA256,
    ):
        raise RuntimeError(
            "InSwapper was not installed correctly: "
            f"{target_path}"
        )

    log(
        "InSwapper model is ready: "
        f"{target_path}"
    )

    return target_path


def prepare_birefnet_portrait_model() -> Path:
    ensure_directory(REMBG_MODEL_DIR)

    target_path = REMBG_MODEL_DIR / REMBG_MODEL_NAME

    if file_has_expected_md5(
        target_path,
        REMBG_MODEL_MD5,
    ):
        log(
            "BiRefNet-portrait already exists and passed "
            f"MD5 validation: {target_path}"
        )
        return target_path

    if target_path.exists():
        log(
            "Removing invalid or incomplete BiRefNet-portrait "
            f"model: {target_path}"
        )
        target_path.unlink()

    temporary_path = target_path.with_suffix(
        target_path.suffix + ".part"
    )
    temporary_path.unlink(missing_ok=True)

    download_url(
        REMBG_MODEL_URL,
        temporary_path,
    )

    actual_md5 = calculate_md5(temporary_path)

    if actual_md5 != REMBG_MODEL_MD5:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "BiRefNet-portrait MD5 mismatch. "
            f"Expected={REMBG_MODEL_MD5}, actual={actual_md5}."
        )

    temporary_path.replace(target_path)

    log(
        "BiRefNet-portrait model is ready: "
        f"{target_path}"
    )

    return target_path


def verify_models(
    *,
    base_model_directory: Path,
    vae_directory: Path,
    clip_directory: Path,
    consistentid_directory: Path,
    controlnet_directory: Path,
    antelopev2_directory: Path,
    buffalo_l_directory: Path,
    inswapper_path: Path,
    birefnet_path: Path,
) -> None:
    required_paths = [
        base_model_directory / "model_index.json",
        vae_directory / "config.json",
        clip_directory / "config.json",
        consistentid_directory / CONSISTENTID_WEIGHT_NAME,
        consistentid_directory / CONSISTENTID_FACE_PARSING_NAME,
        controlnet_directory / "config.json",
        controlnet_directory / "diffusion_pytorch_model.fp16.safetensors",
    ]

    missing_paths = [
        str(path)
        for path in required_paths
        if not file_is_ready(path)
    ]

    if not model_pack_is_complete(
        antelopev2_directory,
        EXPECTED_ANTELOPE_FILES,
    ):
        missing_paths.append(
            str(antelopev2_directory)
        )

    if not model_pack_is_complete(
        buffalo_l_directory,
        EXPECTED_BUFFALO_L_FILES,
    ):
        missing_paths.append(
            str(buffalo_l_directory)
        )

    if not file_has_expected_sha256(
        inswapper_path,
        INSWAPPER_SHA256,
    ):
        missing_paths.append(str(inswapper_path))

    if not file_has_expected_md5(
        birefnet_path,
        REMBG_MODEL_MD5,
    ):
        missing_paths.append(str(birefnet_path))

    if missing_paths:
        raise RuntimeError(
            "Required model files are missing: "
            + ", ".join(missing_paths)
        )

    log(
        "Realistic Vision V6, ConsistentID V1, CLIP ViT-H, "
        "ControlNet v1.1 OpenPose, antelopev2, buffalo_l, "
        "InSwapper and BiRefNet-portrait are ready."
    )


def main() -> None:
    ensure_directory(MODELS_ROOT)
    ensure_directory(INSIGHTFACE_ROOT)

    cleanup_deprecated_models()

    base_model_directory = download_base_model()
    vae_directory = download_vae()
    clip_directory = download_clip_vision()
    consistentid_directory = download_consistentid()
    controlnet_directory = download_controlnet_openpose()

    antelopev2_directory = download_insightface_pack(
        pack_name="antelopev2",
        url=ANTELOPEV2_URL,
        expected_files=EXPECTED_ANTELOPE_FILES,
    )

    buffalo_l_directory = download_insightface_pack(
        pack_name="buffalo_l",
        url=BUFFALO_L_URL,
        expected_files=EXPECTED_BUFFALO_L_FILES,
    )

    inswapper_path = prepare_inswapper_model()
    birefnet_path = prepare_birefnet_portrait_model()

    verify_models(
        base_model_directory=base_model_directory,
        vae_directory=vae_directory,
        clip_directory=clip_directory,
        consistentid_directory=consistentid_directory,
        controlnet_directory=controlnet_directory,
        antelopev2_directory=antelopev2_directory,
        buffalo_l_directory=buffalo_l_directory,
        inswapper_path=inswapper_path,
        birefnet_path=birefnet_path,
    )


if __name__ == "__main__":
    main()