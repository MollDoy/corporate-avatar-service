from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "corporate-avatar-service"
    app_env: str = "dev"
    api_key: str = "change_me"

    database_url: str

    storage_dir: str = "/app/storage"
    max_image_mb: int = 10

    u2net_home: str = "/app/models/rembg"
    rembg_model_name: str = "u2netp"
    avatar_output_size: int = 512

    face_min_size_ratio: float = 0.12
    face_detection_scale_factor: float = 1.1
    face_detection_min_neighbors: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()