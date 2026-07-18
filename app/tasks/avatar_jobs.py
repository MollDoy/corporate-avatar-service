from app.celery_app import celery_app
from app.core.config import settings
from app.db.models import AvatarJob
from app.db.session import SessionLocal
from app.services.avatar_job_processor import (
    process_avatar_job as process_job,
)
from app.services.job_artifacts import (
    ensure_source_image_local,
    publish_job_image_artifacts,
)
from app.services.object_storage import ObjectStorageError
from app.services.runtime_cleanup import (
    release_worker_runtime_memory,
)


@celery_app.task(
    bind=True,
    name="app.tasks.avatar_jobs.process_avatar_job",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=settings.s3_publish_max_retries,
)
def process_avatar_job(self, job_id: str) -> None:
    db = SessionLocal()

    try:
        job = db.get(AvatarJob, job_id)

        if job is None:
            return

        ensure_source_image_local(
            db=db,
            job=job,
        )

        process_job(
            job_id=job_id,
            db=db,
        )

        # Idempotent safety pass. Successful jobs already publish artifacts
        # before status=done; failed jobs can still publish partial candidates.
        publish_job_image_artifacts(
            db=db,
            job_id=job_id,
        )

    except ObjectStorageError as exc:
        db.rollback()

        raise self.retry(
            exc=exc,
            countdown=(
                settings.s3_publish_retry_delay_seconds
            ),
        )

    finally:
        db.close()
        release_worker_runtime_memory(
            stage="celery_task_finally"
        )