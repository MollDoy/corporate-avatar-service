from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "corporate-avatar-service"
    app_env: str = "dev"
    api_key: str = "change_me"

    celery_broker_url: str = "redis://redis:6379/0"
    celery_queue_name: str = "avatar_jobs"
    celery_visibility_timeout: int = 7200

    database_url: str

    storage_dir: str = "/app/storage"
    max_image_mb: int = 10
    normalized_image_max_side: int = 2048

    object_storage_backend: str = "local"

    s3_endpoint_url: str = ""
    s3_public_endpoint_url: str = ""
    s3_region_name: str = "us-east-1"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_session_token: str = ""
    s3_bucket_name: str = ""
    s3_key_prefix: str = "corporate-avatar-service"
    s3_addressing_style: str = "path"
    s3_signature_version: str = "s3v4"
    s3_verify_ssl: bool = True
    s3_server_side_encryption: str = ""
    s3_connect_timeout_seconds: int = 10
    s3_read_timeout_seconds: int = 60
    s3_max_attempts: int = 4
    s3_presigned_url_ttl_seconds: int = 900
    s3_publish_max_retries: int = 5
    s3_publish_retry_delay_seconds: int = 30

    insightface_root: str = "/models/insightface"
    insightface_model_name: str = "antelopev2"
    insightface_swap_model_name: str = "buffalo_l"
    inswapper_model_name: str = "inswapper_128.onnx"

    face_detection_width: int = 640
    face_detection_height: int = 640
    face_detection_threshold: float = 0.35
    face_min_area_ratio: float = 0.008

    secondary_face_min_score: float = 0.65
    secondary_face_min_area_ratio: float = 0.25

    face_swap_enabled: bool = True
    face_swap_pixel_boost_size: int = 512
    face_swap_mask_blur_ratio: float = 0.025
    face_swap_mask_dilation_ratio: float = 0.015
    face_swap_color_match_strength: float = 0.45
    face_swap_min_sharpness_ratio: float = 0.72
    face_swap_min_identity_gain: float = 0.015
    face_swap_max_identity_drop: float = 0.020
    face_swap_top_k: int = 2

    face_identity_antelope_threshold: float = 0.60
    face_identity_buffalo_threshold: float = 0.58
    face_identity_mean_threshold: float = 0.61
    identity_quality_tie_margin: float = 0.012

    keep_candidate_files: bool = True

    ai_batch_script_path: str = "/app/scripts/ai_service.py"
    ai_timeout_seconds: int = 7200
    ai_candidate_seeds: str = "44,144,244"
    ai_num_inference_steps: int = 50
    ai_guidance_scale: float = 5.0
    ai_output_size: int = 512
    ai_cpu_threads: int = 8
    ai_consistentid_adapter_scale: float = 1.00
    ai_consistentid_start_merge_step: int = 30

    ai_controlnet_enabled: bool = True
    ai_controlnet_model_dir: str = "/models/controlnet_openpose"
    ai_controlnet_conditioning_scale: float = 0.35
    ai_controlnet_guidance_start: float = 0.0
    ai_controlnet_guidance_end: float = 0.62

    background_matting_enabled: bool = True
    background_matting_required: bool = True
    background_matting_script_path: str = (
        "/app/scripts/background_matting.py"
    )
    background_matting_timeout_seconds: int = 1800
    background_matting_model: str = "birefnet-portrait"
    background_matting_threads: int = 6
    background_matting_model_dir: str = "/models/rembg"
    corporate_background_hex: str = "D5E0E8"
    background_matting_alpha_gamma: float = 1.0
    background_matting_min_foreground_ratio: float = 0.10
    background_matting_max_foreground_ratio: float = 0.98
    keep_background_mask: bool = False

    pipeline_version: str = (
        "sd15-consistentid-v1-sequential-originalref-birefnet-v17"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()