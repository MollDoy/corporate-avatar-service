# corporate-avatar-service

Сервис генерации корпоративных аватаров сотрудников на основе портретной фотографии.

Проект принимает исходное фото сотрудника, формирует корпоративный аватар и возвращает PNG-результат:

- заменяет фон на единый светлый корпоративный градиент;
- дорисовывает / заменяет одежду на деловую;
- сохраняет лицо и проверяет похожесть результата на исходное фото;
- предоставляет HTTP API;
- обрабатывает длительные задания через очередь Redis и Celery worker;
- управляет схемой PostgreSQL через миграции Alembic;
- содержит расширение 1С:УНФ для вызова сервиса из интерфейса 1С.

---

## Содержание

- [Стек](#стек)
- [Архитектура](#архитектура)
- [Очередь задач](#очередь-задач)
- [Миграции базы данных](#миграции-базы-данных)
- [Структура проекта](#структура-проекта)
- [Требования к окружению](#требования-к-окружению)
- [Первый запуск проекта с нуля](#первый-запуск-проекта-с-нуля)
- [Быстрый запуск после первой сборки](#быстрый-запуск-после-первой-сборки)
- [Когда нужна пересборка](#когда-нужна-пересборка)
- [Остановка проекта](#остановка-проекта)
- [Настройка и описание .env](#настройка-env)
- [API](#api)
- [Тестирование API из WSL](#тестирование-api-из-wsl)
- [Проверка очереди](#проверка-очереди)
- [Интеграция с 1С:УНФ](#интеграция-с-1сунф)
- [Используемые модели](#используемые-модели)
- [Стили генерации](#стили-генерации)
- [Хранение и очистка файлов](#хранение-и-очистка-файлов)
- [Возможные проблемы](#возможные-проблемы)

---

## Стек

- Python 3.11
- FastAPI
- PostgreSQL 16
- SQLAlchemy
- Alembic
- Redis
- Celery
- Docker / Docker Compose
- rembg + `birefnet-portrait` для удаления фона
- OpenCV Haar Cascade для детекции лица
- DreamShaper inpainting `Lykon/dreamshaper-8-inpainting` для AI-дорисовки деловой одежды
- 1С:УНФ, расширение `.cfe`

---

## Архитектура

```text
1С:УНФ / curl / Swagger
        |
        | HTTP API
        v
FastAPI service (avatar_api)
        |
        | создаёт запись задания
        v
PostgreSQL (avatar_db)
        |
        | публикует job_id
        v
Redis broker (avatar_redis)
        |
        | Celery task
        v
Celery worker (avatar_worker, concurrency=1)
        |
        | проверка входного фото
        | удаление фона
        v
rembg / birefnet-portrait
        |
        | построение масок лица, одежды и фона
        v
AI inpainting service (avatar_ai_inpaint)
        |
        | восстановление защищённой области лица
        | проверка похожести лица
        v
PNG avatar result + обновление статуса в PostgreSQL
```

Контейнеры проекта:

| Контейнер | Назначение |
|---|---|
| `avatar_api` | HTTP API, валидация запроса, сохранение исходника и постановка задания в очередь |
| `avatar_worker` | Последовательная обработка тяжёлых CV/AI-заданий |
| `avatar_redis` | Брокер очереди Celery |
| `avatar_ai_inpaint` | Stable Diffusion inpainting на CPU/GPU |
| `avatar_db` | PostgreSQL со статусами и метаданными заданий |
| `avatar_migrate` | Одноразовое применение миграций Alembic при запуске |

Основной API не выполняет генерацию внутри HTTP-процесса. После создания записи он публикует идентификатор задания в Redis и сразу возвращает клиенту `job_id` со статусом `queued`.

Тяжёлая обработка выполняется отдельным Celery worker. В текущей конфигурации используется один рабочий процесс, чтобы несколько одновременных запросов не запускали несколько экземпляров BiRefNet и DreamShaper параллельно на ограниченных ресурсах.

AI inpainting вынесен в отдельный контейнер: PyTorch, diffusers и модели Stable Diffusion занимают много места и требуют GPU, поэтому они не смешиваются с API- и worker-контейнерами.

Изображения хранятся локально в `app/storage/jobs`. Для текущего MVP этого достаточно. Для распределённого развёртывания с несколькими worker-узлами локальное хранилище следует заменить на S3-совместимое хранилище, например MinIO или Amazon S3.

---

## Очередь задач

Очередь реализована через Redis и Celery.

При создании задания выполняется следующая последовательность:

```text
POST /api/v1/avatar-jobs
        ↓
создание записи PostgreSQL со статусом queued
        ↓
сохранение source.png
        ↓
публикация job_id в Redis
        ↓
Celery worker получает задачу
        ↓
status: processing
        ↓
генерация
        ↓
status: done или failed
```

Основные настройки worker в `docker-compose.yml`:

```text
--concurrency=1
--prefetch-multiplier=1
--max-tasks-per-child=1
```

Их назначение:

- `concurrency=1` — одновременно выполняется только одно тяжёлое задание;
- `prefetch-multiplier=1` — worker не резервирует большую пачку заданий заранее;
- `max-tasks-per-child=1` — дочерний процесс Celery завершается после каждого задания, что позволяет операционной системе освободить RAM, занятую ONNX Runtime и обработкой изображений.

В конфигурации Celery также включены:

- позднее подтверждение задания после завершения обработки;
- возврат сообщения в очередь при аварийном завершении worker-процесса;
- повторное подключение к Redis при старте;
- `visibility_timeout` для длительных задач.

Статус `queued` теперь означает, что задание не только записано в PostgreSQL, но и опубликовано в реальную очередь Redis.

---

## Миграции базы данных

Структура PostgreSQL управляется через Alembic.

В проекте больше не используется автоматическое создание таблиц через:

```python
Base.metadata.create_all(...)
```

При запуске Compose одноразовый контейнер:

```text
avatar_migrate
```

выполняет:

```bash
alembic upgrade head
```

API и worker запускаются только после успешного завершения миграций.

Начальная миграция:

```text
alembic/versions/0001_create_avatar_jobs.py
```

Она создаёт:

- PostgreSQL enum `avatarjobstatus`;
- таблицу `avatar_jobs`;
- служебную таблицу Alembic `alembic_version`.

Проверить текущую ревизию после запуска проекта:

```bash
docker compose exec api alembic current
```

Посмотреть историю миграций:

```bash
docker compose exec api alembic history
```

Проверить, совпадают ли SQLAlchemy-модели с последней миграцией:

```bash
docker compose run --rm migrate alembic check
```

После изменения моделей создать новую ревизию:

```bash
docker compose run --rm migrate \
  alembic revision --autogenerate -m "describe schema change"
```

Сгенерированный файл необходимо проверить вручную, затем применить:

```bash
docker compose run --rm migrate alembic upgrade head
```

Откатить одну ревизию:

```bash
docker compose run --rm migrate alembic downgrade -1
```

Изменения Redis, Celery и переменных окружения не требуют миграции, если структура таблиц PostgreSQL не меняется.

---

## Структура проекта

```text
alembic/
  env.py               конфигурация Alembic и подключение Base.metadata
  script.py.mako       шаблон новых ревизий
  versions/            файлы миграций

app/
  api/                 FastAPI routes
  core/                настройки и безопасность
  db/                  SQLAlchemy models/session
  schemas/             Pydantic-схемы API
  services/            CV/AI/image-processing логика
  tasks/               Celery-задачи
  storage/             локальное хранилище job-файлов, не коммитится
  celery_app.py        конфигурация Celery
  main.py              точка входа FastAPI

scripts/
  ai_service.py
                       отдельный FastAPI-сервис для Stable Diffusion inpainting

  create_avatar_request_from_image.py
                       генерация JSON-запроса с base64-изображением

  make_test_base64.py
                       тестовая генерация base64

  run_inpainting_experiment.py
                       экспериментальный запуск AI inpainting

onec_config/
  *.cfe
                       расширение 1С:УНФ

alembic.ini            настройки Alembic
Dockerfile             образ API, worker и migrate
Dockerfile.ai          AI-контейнер
docker-compose.yml     PostgreSQL + Redis + migrate + API + worker + AI profile
requirements.txt       зависимости API, worker и миграций
requirements-ai.txt    зависимости AI-сервиса
.env.example           пример переменных окружения
README.md              документация проекта
```

---

## Требования к окружению

Рекомендуемая среда:

- Windows 10/11;
- WSL2 Ubuntu;
- Docker Desktop;
- включённая WSL Integration в Docker Desktop;
- NVIDIA GPU с поддержкой CUDA для режима `ai_business`;
- установленный NVIDIA-драйвер;
- установленная 1С:Предприятие;
- учебная база 1С:УНФ для проверки расширения.

Проверка GPU внутри Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если GPU отображается, AI-контейнер сможет использовать CUDA.

Если GPU недоступен, можно переключить AI на CPU через `.env`:

```env
AI_DEVICE=cpu
AI_DTYPE=float32
```

Но генерация AI-одежды на CPU будет очень медленной.

Для сборки AI-образа требуется значительный объём диска. Рекомендуется иметь не менее 40–50 ГБ свободного места на диске, где расположен виртуальный диск Docker Desktop.

---

## Первый запуск проекта с нуля

### 1. Открыть Docker Desktop

Перед запуском проекта на Windows необходимо открыть Docker Desktop и дождаться, пока Docker Engine полностью запустится.

Проверьте Docker из WSL:

```bash
docker version
docker compose version
```

Если команды не работают, проверьте, что в Docker Desktop включена WSL Integration для вашей Ubuntu:

```text
Docker Desktop → Settings → Resources → WSL Integration
```

---

### 2. Перейти в папку проекта

```bash
cd ~/Projects/corporate-avatar-service
```

---

### 3. Создать `.env`

```bash
cp .env.example .env
```

Откройте `.env`:

```bash
code .env
```

Минимально важные параметры:

```env
API_KEY=change_me
DATABASE_URL=postgresql+psycopg://avatar_user:avatar_pass@db:5432/avatar_db
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_QUEUE_NAME=avatar_jobs
REMBG_MODEL_NAME=birefnet-portrait
AI_MODEL_ID=Lykon/dreamshaper-8-inpainting
AI_DEVICE=cuda
AI_DTYPE=float32
AI_LOW_VRAM=true
```

---

### 4. Проверить исходный код и Compose

Проверка синтаксиса Python:

```bash
python3 -m compileall -q app scripts
echo $?
```

Ожидаемый код возврата:

```text
0
```

Проверка итоговой конфигурации Compose:

```bash
docker compose --profile ai config
```

---

### 5. Проверить GPU в Docker

Для режима `ai_business` желательно наличие NVIDIA GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если команда показывает видеокарту, GPU-доступ из Docker работает.

---

### 6. Собрать основной образ

```bash
docker compose build --provenance=false api
```

Этот образ используется сразу тремя сервисами:

- `avatar_api`;
- `avatar_worker`;
- `avatar_migrate`.

Он содержит:

- FastAPI;
- SQLAlchemy;
- Alembic;
- Celery;
- Redis client;
- rembg;
- OpenCV;
- основную бизнес-логику сервиса.

---

### 7. Собрать AI-контейнер

```bash
docker compose build --provenance=false ai_inpaint
```

Этот шаг может выполняться долго, потому что устанавливаются:

- PyTorch;
- diffusers;
- transformers;
- accelerate;
- другие AI-зависимости.

Флаг `--provenance=false` отключает дополнительные build-attestations, которые не нужны для локального учебного проекта.

---

### 8. Запустить проект

Полный запуск PostgreSQL, Redis, миграций, API, worker и AI-сервиса:

```bash
docker compose --profile ai up -d
```

При первом запуске модели будут скачаны автоматически в каталоги:

```text
models/rembg
models/huggingface
```

---

### 9. Проверить контейнеры

```bash
docker compose --profile ai ps -a
```

Ожидаемое состояние:

```text
avatar_db           Up (healthy)
avatar_redis        Up (healthy)
avatar_migrate      Exited (0)
avatar_api          Up
avatar_ai_inpaint   Up (healthy)
avatar_worker       Up
```

Статус `Exited (0)` у `avatar_migrate` является нормальным: контейнер применил миграции и завершился без ошибки.

---

### 10. Проверить миграции

```bash
docker compose logs migrate
docker compose exec api alembic current
```

Ожидаемая ревизия:

```text
0001_create_avatar_jobs (head)
```

---

### 11. Проверить API

```bash
curl http://localhost:8000/health
```

Ожидаемый ответ:

```json
{
  "status": "ok",
  "service": "corporate-avatar-service",
  "environment": "dev"
}
```

---

### 12. Проверить Redis, worker и AI-сервис

Redis:

```bash
docker compose exec redis redis-cli ping
```

Ожидаемый ответ:

```text
PONG
```

Worker:

```bash
docker compose logs worker
```

В логе должны присутствовать сообщения:

```text
celery@... ready
app.tasks.avatar_jobs.process_avatar_job
```

AI-сервис:

```bash
docker exec -it avatar_api python -c "import httpx; print(httpx.get('http://ai_inpaint:8010/health', timeout=10).json())"
```

Пример успешного ответа:

```json
{
  "status": "ok",
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce GTX 1660"
}
```

Название видеокарты может отличаться.

---

## Быстрый запуск после первой сборки

Если образы уже собраны:

```bash
docker compose --profile ai up -d
```

Проверка:

```bash
docker compose --profile ai ps -a
curl http://localhost:8000/health
```

---

## Запуск со сборкой одной командой

Можно использовать:

```bash
docker compose --profile ai up -d --build
```

Но для первого запуска рекомендуется собирать тяжёлый AI-образ отдельно, чтобы проще диагностировать нехватку диска или сбой Docker Desktop:

```bash
docker compose build --provenance=false api
docker compose build --provenance=false ai_inpaint
docker compose --profile ai up -d
```

---

## Когда нужна пересборка

### Изменился код в `app/`

Например, изменены FastAPI routes, Celery-задачи, worker, сервисы обработки изображений или настройки:

```bash
docker compose build api
docker compose --profile ai up -d
```

После пересборки одного API-образа обновляются `api`, `worker` и `migrate`, поскольку они используют один и тот же image.

---

### Изменился `scripts/ai_service.py`

```bash
docker compose build ai_inpaint
docker compose --profile ai up -d
```

---

### Изменился `requirements.txt`

```bash
docker compose build api
docker compose --profile ai up -d
```

---

### Изменился `requirements-ai.txt`

```bash
docker compose build ai_inpaint
docker compose --profile ai up -d
```

---

### Изменилась SQLAlchemy-модель

Сначала создайте и проверьте новую миграцию:

```bash
docker compose run --rm migrate \
  alembic revision --autogenerate -m "describe schema change"
```

Затем пересоберите основной образ и запустите Compose:

```bash
docker compose build api
docker compose --profile ai up -d
```

---

### Изменился только `.env`

Обычно достаточно пересоздать контейнеры:

```bash
docker compose --profile ai up -d --force-recreate
```

---

## Остановка проекта

Остановить контейнеры:

```bash
docker compose --profile ai down
```

Остановить контейнеры и удалить именованные volumes PostgreSQL и Redis:

```bash
docker compose --profile ai down -v
```

Команду `down -v` используйте осторожно: она удалит базу заданий PostgreSQL и сохранённую очередь Redis.

Локальные bind-mounted файлы в:

```text
app/storage
models/rembg
models/huggingface
```

команда `down -v` не удаляет.

---

## Настройка `.env`

Создайте `.env` из шаблона:

```bash
cp .env.example .env
```

Пример `.env`:

```env
APP_NAME=corporate-avatar-service
APP_ENV=dev
API_KEY=change_me

CELERY_BROKER_URL=redis://redis:6379/0
CELERY_QUEUE_NAME=avatar_jobs
CELERY_VISIBILITY_TIMEOUT=3600

POSTGRES_DB=avatar_db
POSTGRES_USER=avatar_user
POSTGRES_PASSWORD=avatar_pass
POSTGRES_HOST=db
POSTGRES_PORT=5432

DATABASE_URL=postgresql+psycopg://avatar_user:avatar_pass@db:5432/avatar_db

STORAGE_DIR=/app/storage
MAX_IMAGE_MB=10

U2NET_HOME=/app/models/rembg
REMBG_MODEL_NAME=birefnet-portrait
REMBG_MAX_INPUT_SIZE=1280
REMBG_ALPHA_MATTING=false
REMBG_OMP_NUM_THREADS=2

AVATAR_OUTPUT_SIZE=512
MASK_FEATHER_RADIUS=0.8

FACE_MIN_SIZE_RATIO=0.12
FACE_DETECTION_SCALE_FACTOR=1.1
FACE_DETECTION_MIN_NEIGHBORS=5

FACE_SIMILARITY_THRESHOLD=0.45
FACE_SIMILARITY_CROP_SIZE=160

HF_TOKEN=
AI_MODEL_ID=Lykon/dreamshaper-8-inpainting
AI_SERVICE_URL=http://ai_inpaint:8010
AI_DEVICE=cuda
AI_DTYPE=float32
AI_LOW_VRAM=true
AI_OUTPUT_NAME=ai_result.png
AI_INPAINT_TIMEOUT_SECONDS=900
AI_DEFAULT_STEPS=16
AI_DEFAULT_GUIDANCE_SCALE=8.0
AI_DEFAULT_STRENGTH=0.85
AI_RESTORE_FACE_AFTER_INPAINT=true
```

---

## Описание переменных `.env`

| Переменная | Назначение |
|---|---|
| `APP_NAME` | Название приложения FastAPI |
| `APP_ENV` | Окружение приложения, например `dev` |
| `API_KEY` | API-ключ, который передаётся в HTTP-заголовке `x-api-key` |
| `CELERY_BROKER_URL` | URL Redis, используемого как брокер Celery |
| `CELERY_QUEUE_NAME` | Имя очереди тяжёлых заданий |
| `CELERY_VISIBILITY_TIMEOUT` | Срок, в течение которого взятое worker задание считается невидимым для других потребителей |
| `POSTGRES_DB` | Имя базы PostgreSQL |
| `POSTGRES_USER` | Пользователь PostgreSQL |
| `POSTGRES_PASSWORD` | Пароль PostgreSQL |
| `POSTGRES_HOST` | Host PostgreSQL внутри Docker Compose, обычно `db` |
| `POSTGRES_PORT` | Порт PostgreSQL |
| `DATABASE_URL` | SQLAlchemy DSN для подключения к PostgreSQL |
| `STORAGE_DIR` | Папка хранения изображений внутри контейнеров |
| `MAX_IMAGE_MB` | Максимальный размер входного изображения |
| `U2NET_HOME` | Папка хранения моделей rembg внутри worker-контейнера |
| `REMBG_MODEL_NAME` | Модель удаления фона. Рекомендуемый вариант: `birefnet-portrait` |
| `REMBG_MAX_INPUT_SIZE` | Максимальный размер длинной стороны рабочей копии перед BiRefNet |
| `REMBG_ALPHA_MATTING` | Включение ресурсоёмкого alpha matting после сегментации |
| `REMBG_OMP_NUM_THREADS` | Ограничение числа потоков ONNX Runtime |
| `AVATAR_OUTPUT_SIZE` | Размер итогового изображения, по умолчанию 512x512 |
| `MASK_FEATHER_RADIUS` | Радиус сглаживания маски при замене фона |
| `FACE_MIN_SIZE_RATIO` | Минимальный относительный размер лица для проверки входного фото |
| `FACE_DETECTION_SCALE_FACTOR` | Параметр OpenCV face detection |
| `FACE_DETECTION_MIN_NEIGHBORS` | Параметр OpenCV face detection |
| `FACE_SIMILARITY_THRESHOLD` | Минимальный score похожести лица |
| `FACE_SIMILARITY_CROP_SIZE` | Размер crop лица для сравнения |
| `HF_TOKEN` | Hugging Face token, если модель требует авторизации. Для текущей модели можно оставить пустым |
| `AI_MODEL_ID` | Hugging Face model id для inpainting |
| `AI_SERVICE_URL` | URL AI-сервиса внутри Docker Compose |
| `AI_DEVICE` | Устройство для AI: `cuda` или `cpu` |
| `AI_DTYPE` | Тип вычислений. В текущей конфигурации используется `float32` для стабильности на GTX 1660 |
| `AI_LOW_VRAM` | Режим экономии видеопамяти через model CPU offload |
| `AI_OUTPUT_NAME` | Имя итогового AI-файла в папке job |
| `AI_INPAINT_TIMEOUT_SECONDS` | Таймаут запроса worker к AI-сервису |
| `AI_DEFAULT_STEPS` | Значение по умолчанию для количества diffusion-шагов |
| `AI_DEFAULT_GUIDANCE_SCALE` | Значение по умолчанию для guidance scale |
| `AI_DEFAULT_STRENGTH` | Значение по умолчанию для strength |
| `AI_RESTORE_FACE_AFTER_INPAINT` | Возвращать ли защищённую область лица после генерации |

Для основного стиля `ai_business` параметры генерации зафиксированы в:

```text
app/services/avatar_job_processor.py
```

```python
steps=18
guidance_scale=6.5
strength=0.80
seed=43
```

Переменные `AI_DEFAULT_STEPS`, `AI_DEFAULT_GUIDANCE_SCALE` и `AI_DEFAULT_STRENGTH` используются как значения по умолчанию для вызовов `run_ai_inpainting`, если параметры не переданы явно.

---

## API

Swagger UI доступен по адресу:

```text
http://localhost:8000/docs
```

Все API-методы защищены заголовком:

```text
x-api-key: change_me
```

Значение должно совпадать с `API_KEY` из `.env`.

---

### Создать задачу генерации

```http
POST /api/v1/avatar-jobs
```

Тело запроса:

```json
{
  "employee_id": "employee_001",
  "style_id": "ai_business",
  "image_base64": "<base64 image>"
}
```

Сервис создаёт запись задания, сохраняет исходник, публикует `job_id` в очередь Celery и сразу возвращает ответ. Генерация выполняется отдельным worker.

Пример ответа:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "status": "queued",
  "face_similarity_score": null
}
```

Если Redis или очередь недоступны, API возвращает HTTP `503 Service Unavailable`, а задание получает статус `failed`.

---

### Получить статус задачи

```http
GET /api/v1/avatar-jobs/{job_id}
```

Пример ответа:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "employee_id": "employee_001",
  "style_id": "ai_business",
  "status": "processing",
  "source_image_path": "/app/storage/jobs/.../source.png",
  "result_image_path": null,
  "error_message": null,
  "face_similarity_score": null,
  "created_at": "2026-01-01T10:00:00",
  "updated_at": "2026-01-01T10:00:05"
}
```

Возможные статусы:

```text
queued
processing
done
failed
```

---

### Получить результат

```http
GET /api/v1/avatar-jobs/{job_id}/result
```

Если задача завершена успешно, сервис вернёт PNG в base64:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "image_base64": "...",
  "mime_type": "image/png"
}
```

Если задача ещё не завершена, будет HTTP `409 Conflict`.

---

## Тестирование API из WSL

Создание JSON-запроса из изображения:

```bash
python3 scripts/create_avatar_request_from_image.py \
  references/man.jpg \
  employee_001 \
  ai_business \
  > /tmp/avatar_job_request.json
```

Отправка запроса:

```bash
curl -X POST "http://localhost:8000/api/v1/avatar-jobs" \
  -H "accept: application/json" \
  -H "x-api-key: change_me" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/avatar_job_request.json
```

Пример ответа:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "status": "queued",
  "face_similarity_score": null
}
```

Сохраните `job_id` из ответа.

Проверка статуса:

```bash
curl -X GET "http://localhost:8000/api/v1/avatar-jobs/<JOB_ID>" \
  -H "accept: application/json" \
  -H "x-api-key: change_me"
```

Получение результата:

```bash
curl -X GET "http://localhost:8000/api/v1/avatar-jobs/<JOB_ID>/result" \
  -H "accept: application/json" \
  -H "x-api-key: change_me"
```

Результаты и промежуточные файлы сохраняются в:

```text
app/storage/jobs/<job_id>/
```

Обычно там есть:

```text
source.png
person_mask.png
background_mask.png
face_protection_mask.png
face_restore_mask.png
clothes_mask.png
ai_inpaint_mask.png
result.png
ai_result.png
```

---

## Проверка очереди

Посмотреть количество ожидающих сообщений в Redis:

```bash
docker compose exec redis redis-cli LLEN avatar_jobs
```

Посмотреть активные задачи Celery:

```bash
docker compose exec worker \
  celery -A app.celery_app.celery_app inspect active
```

Посмотреть зарезервированные задачи:

```bash
docker compose exec worker \
  celery -A app.celery_app.celery_app inspect reserved
```

Следить за логом worker:

```bash
docker compose logs -f worker
```

Проверить последние статусы PostgreSQL:

```bash
docker compose exec db \
  psql -U avatar_user -d avatar_db \
  -c "
  SELECT id, status, created_at, updated_at
  FROM avatar_jobs
  ORDER BY created_at DESC
  LIMIT 10;
  "
```

При одновременной отправке трёх заданий ожидаемая картина во время обработки первого:

```text
processing
queued
queued
```

---

## Интеграция с 1С:УНФ

В каталоге:

```text
onec_config/
```

находится расширение 1С:УНФ в формате `.cfe`.

Расширение позволяет:

- выбрать сотрудника справочника `Сотрудники`;
- выбрать исходную портретную фотографию;
- отправить изображение и UUID сотрудника в FastAPI-сервис;
- получить `job_id`;
- периодически опрашивать статус `queued`, `processing`, `done` или `failed`;
- получить готовый PNG;
- записать результат в регистр сведений `АватарыСотрудников`;
- хранить несколько вариантов аватара сотрудника;
- отмечать один вариант как активный;
- открывать историю аватаров из карточки сотрудника;
- просматривать ранее созданный аватар и менять активный вариант.

После успешной генерации результат также открывается стандартным просмотрщиком Windows.

---

### Подключение расширения в 1С:УНФ

1. Откройте информационную базу УНФ в режиме **Конфигуратор**.
2. Откройте меню:

   ```text
   Конфигурация → Расширения конфигурации
   ```

3. Добавьте `.cfe` из каталога:

   ```text
   onec_config/
   ```

4. Убедитесь, что расширение активно.
5. Обновите конфигурацию базы данных.
6. Запустите 1С в режиме предприятия.

---

### Использование обработки

В учебной / демо УНФ обработку можно открыть через подсистему **Корпоративные аватары ИИ**, а также через:

```text
Функции для технического специалиста
→ Обработки
→ Генерация корпоративного аватара
```

На форме обработки:

1. Проверьте параметры подключения:
   - сервер: `localhost`;
   - порт: `8000`;
   - API-ключ: `change_me`;
   - стиль: `ai_business`.

2. Выберите сотрудника.
3. Нажмите **Выбрать фото**.
4. Выберите портретное изображение.
5. Нажмите **Сгенерировать аватар**.
6. Обработка создаст задачу и будет периодически проверять её состояние.
7. После завершения PNG будет записан в историю сотрудника и открыт автоматически.

История открывается командой **История аватаров** из карточки конкретного сотрудника.

Перед запуском генерации из 1С сервис должен быть запущен:

```bash
docker compose --profile ai up -d
```

Если Python-сервис развёрнут на отдельном сервере, вместо `localhost` укажите IP-адрес или DNS-имя этого сервера. `localhost` всегда означает компьютер, на котором запущен конкретный клиент 1С.

---

## Используемые модели

### Удаление фона

Используется `rembg` с моделью:

```text
birefnet-portrait
```

Эта модель выбрана после сравнения с:

```text
u2net
isnet-general-use
u2net_human_seg
```

`birefnet-portrait` показала лучший результат на портретных фото, особенно на волосах.

В текущей конфигурации BiRefNet выполняется на CPU внутри `avatar_worker`. Перед сегментацией рабочая копия изображения ограничивается параметром:

```env
REMBG_MAX_INPUT_SIZE=1280
```

Ресурсоёмкий alpha matting по умолчанию отключён:

```env
REMBG_ALPHA_MATTING=false
```

Количество потоков ONNX Runtime ограничивается через:

```env
REMBG_OMP_NUM_THREADS=2
```

После этапа сегментации rembg-сессия освобождается до запуска DreamShaper. Дополнительно `max-tasks-per-child=1` завершает дочерний процесс Celery после задания и освобождает его оперативную память.

---

### Генерация деловой одежды

Используется inpainting-модель:

```text
Lykon/dreamshaper-8-inpainting
```

AI-сервис получает:

```text
result.png
clothes_mask.png
face_restore_mask.png
```

После inpainting лицо частично возвращается из исходного результата, чтобы сохранить индивидуальность.

В AI-сервисе применяется блокировка инференса, поэтому даже при прямых одновременных HTTP-вызовах GPU-генерация выполняется последовательно.

Для GTX 1660 используется:

```env
AI_DTYPE=float32
AI_LOW_VRAM=true
```

Режим `AI_LOW_VRAM=true` включает перенос компонентов модели между CPU и GPU, снижая требования к видеопамяти ценой дополнительного потребления RAM и времени передачи данных.

---

### Проверка сохранности лица

После генерации выполняется проверка похожести лица между исходным изображением и итоговым аватаром.

В текущей версии проекта не используется отдельная нейросетевая face-recognition модель вроде InsightFace, FaceNet или ArcFace. Проверка реализована как MVP-эвристика на OpenCV:

1. На исходном изображении и на итоговом аватаре находится лицо через OpenCV Haar Cascade:

   ```text
   haarcascade_frontalface_default.xml
   ```

2. Из обоих изображений вырезаются области лица с небольшим отступом.
3. Оба crop приводятся к одному размеру, переводятся в grayscale и нормализуются.
4. Считаются две метрики:
   - pixel similarity — средняя попиксельная близость подготовленных crop;
   - histogram similarity — похожесть HSV-гистограмм.
5. Итоговый score считается как взвешенная сумма:

   ```python
   score = 0.75 * pixel_similarity + 0.25 * histogram_similarity
   ```

Если итоговый score ниже порога:

```env
FACE_SIMILARITY_THRESHOLD=0.45
```

задача получает статус:

```text
failed
```

Эта проверка не является полноценной биометрической идентификацией. Она используется как простая техническая защита от очевидных ошибок генерации: сильно изменённого лица, отсутствия лица или некорректного результата.

Возможное улучшение проекта — заменить эту эвристику на embedding-based проверку через InsightFace / ArcFace или аналогичную open-source модель.

---

## Стили генерации

Основной стиль:

```text
ai_business
```

Он включает:

- корпоративный светло-голубой градиентный фон;
- AI-дорисовку деловой одежды;
- защиту и восстановление лица;
- проверку похожести.

Также в коде предусмотрены базовые стили без AI-дорисовки:

```text
default_business
blue_business
warm_office
gray_minimal
```

---

## Хранение и очистка файлов

Для каждого задания создаётся локальный каталог:

```text
app/storage/jobs/<job_id>/
```

Сгенерированные изображения не должны попадать в репозиторий.

Очистить job-файлы:

```bash
rm -rf app/storage/jobs/*
touch app/storage/jobs/.gitkeep
```

Очистить Docker build cache:

```bash
docker builder prune
```

Остановить сервисы:

```bash
docker compose --profile ai down
```

Остановить сервисы и удалить volumes PostgreSQL и Redis:

```bash
docker compose --profile ai down -v
```

Для промышленного распределённого развёртывания локальные job-каталоги следует заменить на S3-совместимое объектное хранилище.

---

## Возможные проблемы

### `Invalid base64 image`

Проверьте, что base64-строка не содержит:

```text
CR
LF
табы
пробелы
```

перед отправкой.

---

### Задание остаётся в `queued`

Проверьте Redis:

```bash
docker compose exec redis redis-cli ping
docker compose exec redis redis-cli LLEN avatar_jobs
```

Проверьте worker:

```bash
docker compose --profile ai ps -a
docker compose logs worker
```

В логе worker должна быть зарегистрирована задача:

```text
app.tasks.avatar_jobs.process_avatar_job
```

Проверьте активные и зарезервированные задания:

```bash
docker compose exec worker \
  celery -A app.celery_app.celery_app inspect active

docker compose exec worker \
  celery -A app.celery_app.celery_app inspect reserved
```

---

### `avatar_worker` завершается во время BiRefNet

Посмотрите состояние контейнеров и потребление памяти:

```bash
docker compose --profile ai ps -a
watch -n 1 'docker stats --no-stream avatar_api avatar_worker avatar_ai_inpaint avatar_redis avatar_db'
```

Тяжёлая сегментация выполняется в `avatar_worker`, а не в `avatar_api`.

Проверьте параметры:

```env
REMBG_MAX_INPUT_SIZE=1280
REMBG_ALPHA_MATTING=false
REMBG_OMP_NUM_THREADS=2
```

В `docker-compose.yml` должны оставаться:

```text
--concurrency=1
--max-tasks-per-child=1
```

---

### `avatar_api` недоступен или возвращает 503 при создании задания

Проверьте Redis и API:

```bash
docker compose logs api
docker compose logs redis
docker compose exec redis redis-cli ping
```

HTTP `503 Service Unavailable` при `POST /api/v1/avatar-jobs` означает, что API не смог опубликовать задание в очередь.

---

### AI-сервис не найден

Проверьте, что запущен профиль `ai`:

```bash
docker compose --profile ai up -d
```

В списке должен быть контейнер:

```text
avatar_ai_inpaint
```

Проверьте healthcheck и логи:

```bash
docker compose --profile ai ps -a
docker compose logs ai_inpaint
```

---

### Ошибка CUDA или нехватка VRAM

Проверьте фактические настройки API:

```bash
docker compose exec api sh -lc \
  'env | grep -E "AI_DTYPE|AI_LOW_VRAM|AI_MODEL_ID"'
```

Для текущей GTX 1660 ожидается:

```text
AI_DTYPE=float32
AI_LOW_VRAM=true
AI_MODEL_ID=Lykon/dreamshaper-8-inpainting
```

Контроль GPU:

```bash
watch -n 1 nvidia-smi
```

AI-сервис содержит внутреннюю блокировку, поэтому несколько одновременных запросов не должны выполнять несколько GPU-инференсов параллельно.

---

### Миграции не применились

Проверьте одноразовый контейнер:

```bash
docker compose --profile ai ps -a
docker compose logs migrate
```

Успешный статус:

```text
avatar_migrate    Exited (0)
```

Проверка ревизии:

```bash
docker compose exec api alembic current
```

Если база тестовая и может быть удалена, её можно создать заново:

```bash
docker compose --profile ai down -v
docker compose --profile ai up -d
```

Эта команда также удалит очередь Redis.

---

### Недостаточно места на диске

Модели и Docker-образы занимают много места.

Основные места, где расходуется диск:

```text
models/rembg
models/huggingface
Docker Desktop WSL data
Ubuntu WSL ext4.vhdx
```

Проверка размера внутри проекта:

```bash
du -sh models/*
du -sh app/storage
docker system df
```

Очистка Docker build cache:

```bash
docker builder prune
```

Более агрессивная очистка неиспользуемых Docker-данных:

```bash
docker system prune -a
```

Осторожно: команда удаляет неиспользуемые образы. После неё Docker может заново скачивать зависимости и пересобирать образы.

Если нужно полностью удалить ещё и неиспользуемые volumes:

```bash
docker system prune -a --volumes
```

Осторожно: команда может удалить volumes PostgreSQL и Redis.

Если во время экспорта крупного AI-образа Docker Desktop завершается с `EOF`, `SIGBUS`, `containerd failed` или `Input/output error`, сначала проверьте свободное место на системном диске. При повреждении внутреннего Docker-хранилища используйте:

```text
Docker Desktop → Troubleshoot → Clean / Purge data
```

Для WSL-проекта следует очищать набор данных `WSL 2`. Это удаляет Docker-образы, контейнеры и volumes, но не удаляет пользовательский Ubuntu-дистрибутив и проект в `/home/<user>/Projects`.

---

#### Почему место может не вернуться сразу

В WSL2 и Docker Desktop данные хранятся внутри виртуальных дисков `.vhdx`.

Даже если удалить файлы внутри Ubuntu или Docker, Windows может не сразу вернуть свободное место на диске `C:`. Файлы внутри виртуального диска удалены, но сам `.vhdx` остаётся прежнего размера, пока его не сжать.

#### Найти большие `.vhdx` файлы

Откройте PowerShell и выполните:

```powershell
Get-ChildItem -Path "$env:LOCALAPPDATA" -Recurse -Filter *.vhdx -ErrorAction SilentlyContinue |
Select-Object FullName, @{Name="SizeGB";Expression={[math]::Round($_.Length / 1GB, 2)}} |
Sort-Object SizeGB -Descending
```

Если самым большим оказался файл Docker, например:

```text
C:\Users\<USER>\AppData\Local\Docker\wsl\disk\docker_data.vhdx
```

значит место занято Docker Desktop.

Если самым большим оказался файл Ubuntu, например:

```text
C:\Users\<USER>\AppData\Local\Packages\CanonicalGroupLimited...\LocalState\ext4.vhdx
```

значит место занято WSL Ubuntu.

---

#### Подготовка к сжатию `.vhdx`

Перед сжатием виртуальных дисков нужно освободить место внутри Linux-файловой системы.

В WSL Ubuntu выполните:

```bash
sudo fstrim -av
```

Потом закройте все WSL-терминалы.

Откройте PowerShell от имени администратора и остановите WSL:

```powershell
wsl --shutdown
```

Также закройте Docker Desktop через меню в системном трее:

```text
Docker Desktop → Quit Docker Desktop
```

---

#### Сжатие `.vhdx` через diskpart

Откройте PowerShell от имени администратора.

Для Docker Desktop VHDX:

```powershell
$vhd = "C:\Users\<USER>\AppData\Local\Docker\wsl\disk\docker_data.vhdx"

@"
select vdisk file="$vhd"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@ | diskpart
```

Для Ubuntu WSL VHDX:

```powershell
$vhd = "C:\Users\<USER>\AppData\Local\Packages\<UbuntuPackage>\LocalState\ext4.vhdx"

@"
select vdisk file="$vhd"
attach vdisk readonly
compact vdisk
detach vdisk
exit
"@ | diskpart
```

В путях нужно заменить:

```text
<USER>
<UbuntuPackage>
```

на реальные значения из вывода команды поиска `.vhdx`.

---

#### Если diskpart пишет, что файл используется

Если появляется ошибка вида:

```text
The process cannot access the file because it is being used by another process
```

значит WSL или Docker Desktop всё ещё используют виртуальный диск.

Повторите:

```powershell
wsl --shutdown
```

Проверьте, что Docker Desktop полностью закрыт, и снова выполните `diskpart`.

---

#### Проверка после сжатия

После сжатия снова проверьте свободное место на системном диске.

Затем можно открыть Docker Desktop, WSL и проверить:

```bash
docker system df
df -h
du -sh models/*
```

Если Docker-образы были удалены, потребуется повторная сборка:

```bash
docker compose build --provenance=false api
docker compose build --provenance=false ai_inpaint
docker compose --profile ai up -d
```

---

## Лицензии и ограничения

Проект использует open-source библиотеки и модели.

Для промышленного и коммерческого использования необходимо отдельно проверить:

- лицензии выбранных моделей;
- лицензии Python-библиотек;
- правила обработки персональных данных сотрудников;
- правила хранения и передачи фотографий сотрудников.
