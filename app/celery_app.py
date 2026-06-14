from celery import Celery

from app.core.config import settings


celery_app = Celery(
    "corporate_avatar_service",
    broker=settings.celery_broker_url,
    include=["app.tasks.avatar_jobs"],
)

celery_app.conf.update(
    task_default_queue=settings.celery_queue_name,
    task_routes={
        "app.tasks.avatar_jobs.process_avatar_job": {
            "queue": settings.celery_queue_name,
        }
    },
    task_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,

    # Задание подтверждается только после завершения обработки.
    task_acks_late=True,

    # При аварийном завершении worker-процесса сообщение возвращается в очередь.
    task_reject_on_worker_lost=True,

    # Worker не забирает много тяжёлых заданий заранее.
    worker_prefetch_multiplier=1,

    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": settings.celery_visibility_timeout,
    },

    timezone="UTC",
    enable_utc=True,
)