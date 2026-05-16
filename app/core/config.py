from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "corporate-avatar-service"
    app_env: str = "dev"
    api_key: str = "change_me"

    database_url: str

    storage_dir: str = "/app/storage"
    max_image_mb: int = 10

    u2net_home: str = "/app/models/rembg"
    rembg_model_name: str = "u2net"
    avatar_output_size: int = 512
    mask_feather_radius: float = 0.8

    face_min_size_ratio: float = 0.12
    face_detection_scale_factor: float = 1.1
    face_detection_min_neighbors: int = 5

    face_similarity_threshold: float = 0.45
    face_similarity_crop_size: int = 160

    ai_model_id: str = "Lykon/dreamshaper-8-inpainting"
    ai_service_url: str = "http://ai_inpaint:8010"
    ai_device: str = "cuda"
    ai_dtype: str = "float32"
    ai_low_vram: bool = True
    ai_output_name: str = "ai_result.png"
    ai_inpaint_timeout_seconds: int = 900
    ai_default_steps: int = 16
    ai_default_guidance_scale: float = 8.0
    ai_default_strength: float = 0.85
    ai_restore_face_after_inpaint: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()