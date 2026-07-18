# Corporate Avatar Service

Сервис генерации корпоративных аватаров сотрудников по исходной фотографии.

Проект принимает изображение через HTTP API или расширение 1С:УНФ, ставит задачу в очередь, генерирует портрет с сохранением личности, заменяет фон на корпоративный и сохраняет входные и итоговые изображения в S3-совместимом объектном хранилище.

## Архитектура

```text
1С:УНФ / HTTP-клиент
        │
        ▼
FastAPI
  ├─ проверка API-ключа и входного изображения
  ├─ запись задания в PostgreSQL
  ├─ сохранение source.png локально
  └─ публикация source.png в S3
        │
        ▼
Redis → Celery worker, concurrency=1
        │
        ▼
Проверка и подготовка лица
        │
        ▼
ConsistentID + Realistic Vision V6 + OpenPose ControlNet
        │
        ▼
Проверка сходства antelopev2 + buffalo_l
        │
        ▼
InSwapper при улучшении результата
        │
        ▼
BiRefNet Portrait → корпоративный фон
        │
        ▼
S3: кандидаты, swapped-варианты и result.png
PostgreSQL: статус, метрики и метаданные артефактов
```

Основные контейнеры:

| Контейнер | Назначение |
|---|---|
| `avatar_api` | FastAPI и работа с заданиями |
| `avatar_worker` | последовательная генерация изображений |
| `avatar_db` | PostgreSQL |
| `avatar_redis` | очередь Celery |
| `avatar_migrate` | миграции Alembic |
| `avatar_models_init` | автоматическая загрузка моделей |
| `avatar_minio` | локальное S3-совместимое хранилище |
| `avatar_minio_init` | создание приватного бакета |

Тяжёлые модели не загружаются в API-процесс. Генерация выполняется отдельным worker с `concurrency=1`, `prefetch-multiplier=1` и `max-tasks-per-child=1`. После этапов обработки освобождаются PyTorch-, CUDA- и ONNX-ресурсы.

## Генеративный пайплайн

1. Исходник преобразуется в RGB PNG и при необходимости уменьшается до допустимого размера.
2. InsightFace `antelopev2` обнаруживает лицо, определяет landmarks и формирует embedding.
3. `buffalo_l` формирует независимый embedding для дополнительной проверки личности.
4. Из исходника создаётся face-centric reference для ConsistentID.
5. ConsistentID выполняет identity-preserving генерацию на базе Realistic Vision V6.
6. ControlNet v1.1 OpenPose задаёт центрированную верхнюю позу сотрудника.
7. Сиды проверяются последовательно: `44 - 144 - 244`. Следующий сид запускается только при отклонении предыдущего результата.
8. Каждый кандидат проверяется двумя независимыми InsightFace-моделями.
9. InSwapper применяется к подходящим кандидатам; ухудшающие swap-варианты отбрасываются.
10. BiRefNet Portrait отделяет человека от фона и заменяет фон на `#D5E0E8`.
11. Финальный результат и созданные изображения публикуются в S3.

## Используемые модели

- `SG161222/Realistic_Vision_V6.0_B1_noVAE` - базовая Stable Diffusion 1.5 модель;
- `stabilityai/sd-vae-ft-mse` - VAE;
- `JackAILab/ConsistentID`, `ConsistentID-v1.bin` - сохранение личности;
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` - визуальное кодирование;
- `lllyasviel/control_v11p_sd15_openpose` - контроль позы и композиции;
- InsightFace `antelopev2` - детекция, landmarks и проверка сходства;
- InsightFace `buffalo_l` - независимый face embedding;
- `inswapper_128.onnx` - дополнительное восстановление личности;
- `BiRefNet-portrait` - выделение человека и замена фона.

Модели загружаются контейнером `models-init` в локальный каталог `models/` и затем используются worker в offline-режиме.

## Проверка входного изображения

Перед генерацией сервис:

- проверяет корректность base64 и формата изображения;
- ограничивает размер входного файла;
- исправляет EXIF-ориентацию;
- отклоняет изображение без обнаруженного лица;
- отклоняет слишком маленькое лицо;
- отклоняет несколько значимых лиц;
- повторяет детекцию после локального улучшения контраста, если лицо не найдено сразу;
- сохраняет предупреждения о размытии, слишком тёмном или пересвеченном кадре и лице у границы изображения.

После генерации результат принимается только при прохождении порогов сходства `antelopev2`, `buffalo_l` и их среднего значения.

## Хранение файлов

Локальный каталог:

```text
app/storage/jobs/<job_id>/
```

используется worker как рабочее пространство и диагностический кэш.

В S3 сохраняются только изображения, имеющие прикладную ценность:

```text
corporate-avatar-service/jobs/<job_id>/
├── input/
│   └── source.png
├── candidates/
│   ├── candidate_01_seed_44.png
│   ├── candidate_01_seed_44_swapped.png
│   ├── candidate_02_seed_144.png
│   └── ...
└── output/
    └── result.png
```

Технические логи, embeddings, JSON-файлы, маски и `control_pose.png` в S3 не публикуются.

Метаданные объектов хранятся в PostgreSQL в таблице `avatar_job_artifacts`: тип артефакта, имя файла, seed, номер попытки, S3 key, размер и SHA-256.

Если worker работает на другой машине и локального `source.png` нет, он скачивает исходник из S3. API также может вернуть `result.png` непосредственно из S3. Для просмотра объектов локально используется MinIO Console.

## Структура проекта

```text
alembic/versions/        миграции PostgreSQL
app/api/                 HTTP-маршруты
app/core/                настройки и API-key security
app/db/                  SQLAlchemy-модели и сессии
app/schemas/             Pydantic-схемы
app/services/            обработка изображений, модели и S3
app/tasks/               Celery-задачи
app/storage/             локальное рабочее хранилище
scripts/                 AI subprocess, matting и загрузка моделей
onec_config/             расширение 1С:УНФ
Dockerfile               API и миграции
Dockerfile.ai            AI worker
Dockerfile.models        загрузчик моделей
docker-compose.yml       инфраструктура проекта
```

## Требования

- Docker Desktop с WSL2 или Linux с Docker Engine;
- Docker Compose;
- NVIDIA GPU и поддержка GPU-контейнеров;
- свободное место для моделей и Docker volumes;
- доступ в интернет при первой загрузке моделей.

Текущая конфигурация выполняет генерацию в FP32 с последовательной загрузкой компонентов и рассчитана на один GPU-worker. 

Тестирование проекта проводилось на машине с GTX 1660 + Intel Xeon E5-2689 + 32 GB RAM DDR3

## Настройка

Создайте рабочий `.env` из примера:

```bash
cp .env.example .env
```

Перед запуском измените как минимум:

```dotenv
API_KEY=change_me
POSTGRES_PASSWORD=avatar_pass
MINIO_ROOT_USER=avatar_minio
MINIO_ROOT_PASSWORD=avatar_minio_secret
S3_ACCESS_KEY_ID=avatar_minio
S3_SECRET_ACCESS_KEY=avatar_minio_secret
```

`.env` содержит секреты и не должен попадать в Git.

Для встроенного MinIO используются:

```text
S3 API:        http://localhost:9000
MinIO Console: http://localhost:9001
```

Бакет `corporate-avatars` создаётся автоматически и остаётся приватным.

## Первый запуск

Из корня проекта:

```bash
docker compose --profile s3 --profile ai build
docker compose --profile s3 --profile ai up -d
```

При первом запуске `models-init` скачает модели в `./models`. Это может занять продолжительное время.

Проверить контейнеры:

```bash
docker compose --profile s3 --profile ai ps -a
```

Ожидаемое состояние:

- `avatar_db`, `avatar_redis`, `avatar_api`, `avatar_worker`, `avatar_minio` — запущены;
- `avatar_migrate`, `avatar_models_init`, `avatar_minio_init` — завершены с кодом `0`.

## Последующие запуски

```bash
docker compose --profile s3 --profile ai up -d
```

После изменения Python-кода или зависимостей:

```bash
docker compose --profile s3 --profile ai up -d --build
```

## Остановка

Остановить контейнеры без удаления данных:

```bash
docker compose --profile s3 --profile ai stop
```

Остановить и удалить контейнеры и сеть, сохранив volumes:

```bash
docker compose --profile s3 --profile ai down
```

Полностью удалить PostgreSQL, Redis и MinIO volumes:

```bash
docker compose --profile s3 --profile ai down -v
```

Каталог `models/` является bind mount и командой `down -v` не удаляется.

## Логи

Worker:

```bash
docker compose --profile s3 --profile ai logs --tail=250 -f worker
```

API:

```bash
docker compose --profile s3 --profile ai logs --tail=200 -f api
```

MinIO:

```bash
docker compose --profile s3 logs --tail=200 -f minio
```

## Очистка заданий

Перед полной очисткой остановите worker:

```bash
docker compose --profile s3 --profile ai stop worker
```

Очистить очередь Redis:

```bash
docker compose exec redis redis-cli -n 0 FLUSHDB
```

Удалить задания и метаданные артефактов из PostgreSQL:

```bash
docker compose exec db psql -U avatar_user -d avatar_db \
  -c "TRUNCATE TABLE avatar_job_artifacts, avatar_jobs RESTART IDENTITY CASCADE;"
```

Удалить локальные job-файлы:

```bash
rm -rf app/storage/jobs/*
```

Удалить все объекты из проектного S3-бакета:

```bash
docker compose --profile s3 run --rm --entrypoint /bin/sh minio-init -c '
mc alias set local http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" &&
mc rm --recursive --force "local/$S3_BUCKET_NAME"
'
```

Снова запустить worker:

```bash
docker compose --profile s3 --profile ai up -d worker
```

## Очистка моделей

Удалить скачанные модели:

```bash
rm -rf models/*
```

Загрузить их заново:

```bash
docker compose --profile ai run --rm models-init
```

## Основные интерфейсы

- `GET /health` — состояние API;
- `POST /api/v1/avatar-jobs` — создание задания;
- `GET /api/v1/avatar-jobs/{job_id}` — статус задания;
- `GET /api/v1/avatar-jobs/{job_id}/result` — финальное PNG в base64;
- `GET /api/v1/avatar-jobs/{job_id}/artifacts` — список S3-артефактов и временные ссылки.

Защищённые маршруты требуют заголовок `X-API-Key`.

## Интеграция с 1С:УНФ

Расширение находится в каталоге:

```text
onec_config/
```

Оно отправляет фотографию в API, получает `job_id`, отслеживает статус и загружает итоговый аватар. Длительная генерация выполняется асинхронно и не блокирует HTTP-запрос создания задания.
