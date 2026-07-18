from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings


class ObjectStorageError(RuntimeError):
    pass


def object_storage_is_s3() -> bool:
    return settings.object_storage_backend.strip().lower() == "s3"


def _normalized_prefix() -> str:
    return settings.s3_key_prefix.strip().strip("/")


def build_object_key(
    *,
    job_id: str,
    artifact_type: str,
    file_name: str,
) -> str:
    prefix = _normalized_prefix()

    if artifact_type == "source":
        relative_key = f"jobs/{job_id}/input/{file_name}"
    elif artifact_type == "result":
        relative_key = f"jobs/{job_id}/output/{file_name}"
    elif artifact_type in {"candidate", "swapped"}:
        relative_key = f"jobs/{job_id}/candidates/{file_name}"
    else:
        raise ValueError(
            "Unsupported artifact type for object storage: "
            f"{artifact_type}"
        )

    if prefix:
        return f"{prefix}/{relative_key}"

    return relative_key


def _validate_s3_settings() -> None:
    missing: list[str] = []

    if not settings.s3_bucket_name.strip():
        missing.append("S3_BUCKET_NAME")

    if not settings.s3_access_key_id.strip():
        missing.append("S3_ACCESS_KEY_ID")

    if not settings.s3_secret_access_key.strip():
        missing.append("S3_SECRET_ACCESS_KEY")

    if missing:
        raise ObjectStorageError(
            "S3 storage is enabled, but required settings are missing: "
            + ", ".join(missing)
        )


def _create_s3_client(*, endpoint_url_override: str | None = None):
    _validate_s3_settings()

    client_kwargs: dict[str, object] = {
        "service_name": "s3",
        "aws_access_key_id": settings.s3_access_key_id,
        "aws_secret_access_key": settings.s3_secret_access_key,
        "region_name": settings.s3_region_name,
        "verify": settings.s3_verify_ssl,
        "config": Config(
            signature_version=settings.s3_signature_version,
            connect_timeout=settings.s3_connect_timeout_seconds,
            read_timeout=settings.s3_read_timeout_seconds,
            retries={
                "max_attempts": settings.s3_max_attempts,
                "mode": "standard",
            },
            s3={
                "addressing_style": settings.s3_addressing_style,
            },
            max_pool_connections=4,
        ),
    }

    endpoint_url = (
        endpoint_url_override
        if endpoint_url_override is not None
        else settings.s3_endpoint_url
    ).strip()
    session_token = settings.s3_session_token.strip()

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    if session_token:
        client_kwargs["aws_session_token"] = session_token

    return boto3.client(**client_kwargs)


@lru_cache(maxsize=1)
def get_s3_client():
    return _create_s3_client()


@lru_cache(maxsize=1)
def get_s3_public_client():
    public_endpoint = settings.s3_public_endpoint_url.strip()

    if not public_endpoint:
        return get_s3_client()

    return _create_s3_client(
        endpoint_url_override=public_endpoint
    )


_TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=1,
    use_threads=False,
)


def _upload_extra_args(
    *,
    content_type: str,
    metadata: dict[str, str],
) -> dict[str, object]:
    extra_args: dict[str, object] = {
        "ContentType": content_type,
        "Metadata": metadata,
    }

    encryption = settings.s3_server_side_encryption.strip()

    if encryption:
        extra_args["ServerSideEncryption"] = encryption

    return extra_args


def upload_file(
    *,
    local_path: Path,
    object_key: str,
    content_type: str,
    metadata: dict[str, str],
) -> None:
    if not object_storage_is_s3():
        return

    if not local_path.is_file():
        raise ObjectStorageError(
            f"Cannot upload missing artifact: {local_path}"
        )

    try:
        get_s3_client().upload_file(
            Filename=str(local_path),
            Bucket=settings.s3_bucket_name,
            Key=object_key,
            ExtraArgs=_upload_extra_args(
                content_type=content_type,
                metadata=metadata,
            ),
            Config=_TRANSFER_CONFIG,
        )
    except (BotoCoreError, ClientError, OSError) as exc:
        raise ObjectStorageError(
            "Could not upload object to S3: "
            f"bucket={settings.s3_bucket_name}; "
            f"key={object_key}; error={exc}"
        ) from exc


def download_file(
    *,
    object_key: str,
    local_path: Path,
) -> None:
    if not object_storage_is_s3():
        raise ObjectStorageError(
            "Cannot download from S3 while OBJECT_STORAGE_BACKEND is not s3."
        )

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = local_path.with_name(
        local_path.name + ".part"
    )

    try:
        get_s3_client().download_file(
            Bucket=settings.s3_bucket_name,
            Key=object_key,
            Filename=str(temporary_path),
            Config=_TRANSFER_CONFIG,
        )
        temporary_path.replace(local_path)
    except (BotoCoreError, ClientError, OSError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise ObjectStorageError(
            "Could not download object from S3: "
            f"bucket={settings.s3_bucket_name}; "
            f"key={object_key}; error={exc}"
        ) from exc


def read_object_bytes(object_key: str) -> bytes:
    if not object_storage_is_s3():
        raise ObjectStorageError(
            "Cannot read from S3 while OBJECT_STORAGE_BACKEND is not s3."
        )

    try:
        response = get_s3_client().get_object(
            Bucket=settings.s3_bucket_name,
            Key=object_key,
        )
        body: BinaryIO = response["Body"]
        return body.read()
    except (BotoCoreError, ClientError, OSError) as exc:
        raise ObjectStorageError(
            "Could not read object from S3: "
            f"bucket={settings.s3_bucket_name}; "
            f"key={object_key}; error={exc}"
        ) from exc


def generate_presigned_download_url(object_key: str) -> str | None:
    if not object_storage_is_s3():
        return None

    ttl = settings.s3_presigned_url_ttl_seconds

    if ttl <= 0:
        return None

    try:
        return get_s3_public_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": settings.s3_bucket_name,
                "Key": object_key,
            },
            ExpiresIn=ttl,
        )
    except (BotoCoreError, ClientError) as exc:
        raise ObjectStorageError(
            "Could not create presigned S3 URL: "
            f"bucket={settings.s3_bucket_name}; "
            f"key={object_key}; error={exc}"
        ) from exc


def check_bucket_access() -> None:
    if not object_storage_is_s3():
        return

    try:
        get_s3_client().head_bucket(
            Bucket=settings.s3_bucket_name,
        )
    except (BotoCoreError, ClientError) as exc:
        raise ObjectStorageError(
            "S3 bucket is not accessible: "
            f"bucket={settings.s3_bucket_name}; error={exc}"
        ) from exc