from app.celery_app import celery_app
from app.db.session import SessionLocal
from app.services.avatar_job_processor import (
    process_avatar_job as process_job,
)


@celery_app.task(
    name="app.tasks.avatar_jobs.process_avatar_job",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_avatar_job(job_id: str) -> None:
    db = SessionLocal()

    try:
        process_job(job_id=job_id, db=db)
    finally:
        db.close()