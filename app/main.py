from fastapi import FastAPI

from app.api.routes_avatar_jobs import router as avatar_jobs_router
from app.api.routes_styles import router as styles_router
from app.core.config import settings


app = FastAPI(title=settings.app_name)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


app.include_router(avatar_jobs_router)
app.include_router(styles_router)