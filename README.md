# corporate-avatar-service

Сервис генерации корпоративных аватаров сотрудников на основе портретной фотографии.

Проект принимает исходное портретное изображение, формирует аватар в корпоративном стиле и возвращает PNG-результат:

- заменяет фон на единый светлый корпоративный градиент;
- дорисовывает одежду на деловую;
- сохраняет лицо и проверяет похожесть результата на исходное фото;
- предоставляет HTTP API для интеграции с 1С:УНФ;
- содержит расширение 1С:УНФ для вызова сервиса из интерфейса 1С.

## Стек

- Python 3.11
- FastAPI
- PostgreSQL
- SQLAlchemy
- Docker / Docker Compose
- rembg + `birefnet-portrait` для удаления фона
- OpenCV Haar Cascade для детекции лица
- DreamShaper inpainting (`Lykon/dreamshaper-8-inpainting`) для AI-дорисовки деловой одежды
- 1С:УНФ, расширение `.cfe`

## Архитектура

```text
1С:УНФ / curl / Swagger
        |
        | HTTP API
        v
FastAPI service
        |
        | сохраняет задачу
        v
PostgreSQL
        |
        | обработка изображения
        v
rembg / birefnet-portrait
        |
        | маски лица, одежды и фона
        v
AI inpainting service / DreamShaper
        |
        | проверка похожести лица
        v
PNG avatar result
```

Основной API работает в контейнере `avatar_api`.

AI inpainting вынесен в отдельный контейнер `avatar_ai_inpaint`, чтобы тяжёлые зависимости PyTorch / diffusers не смешивались с основным API-контейнером.

PostgreSQL работает в контейнере `avatar_db`.

## Структура проекта

```text
app/
  api/                 FastAPI routes
  core/                настройки и безопасность
  db/                  SQLAlchemy models/session
  schemas/             Pydantic-схемы API
  services/            CV/AI/image-processing логика
  storage/             локальное хранилище job-файлов, не коммитится

scripts/
  ai_service.py        отдельный FastAPI-сервис для Stable Diffusion inpainting
  create_avatar_request_from_image.py
                       генерация JSON-запроса с base64-изображением
  make_test_base64.py  тестовая генерация base64
  run_inpainting_experiment.py
                       экспериментальный запуск AI inpainting

onec_config/
  КорпоративныеАватарыИИ.cfe
                       расширение 1С:УНФ

Dockerfile             основной API-контейнер
Dockerfile.ai          AI-контейнер
docker-compose.yml     запуск API + PostgreSQL + AI profile
docker-compose.ai.yml  отдельный запуск AI-сервиса для экспериментов
requirements.txt       зависимости основного API
requirements-ai.txt    зависимости AI-сервиса
.env.example           пример переменных окружения
```

## Требования к окружению

Рекомендуемая среда:

- Windows 10/11;
- WSL2 Ubuntu;
- Docker Desktop с включённой интеграцией WSL2;
- NVIDIA GPU с поддержкой CUDA для режима `ai_business`;
- установленный NVIDIA-драйвер;
- установленная 1С:Предприятие и учебная база 1С:УНФ для проверки расширения.

Проверка GPU в Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если GPU не доступен, сервис можно теоретически переключить на CPU через `.env`, но генерация AI-одежды будет очень медленной.

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

### Описание переменных `.env`

| Переменная | Назначение |
|---|---|
| `APP_NAME` | Название приложения FastAPI |
| `APP_ENV` | Окружение: `dev`, `prod` и т.п. |
| `API_KEY` | API-ключ, который передаётся в HTTP-заголовке `x-api-key` |
| `POSTGRES_DB` | Имя базы PostgreSQL |
| `POSTGRES_USER` | Пользователь PostgreSQL |
| `POSTGRES_PASSWORD` | Пароль PostgreSQL |
| `POSTGRES_HOST` | Host PostgreSQL внутри Docker Compose, обычно `db` |
| `POSTGRES_PORT` | Порт PostgreSQL |
| `DATABASE_URL` | SQLAlchemy DSN для подключения к PostgreSQL |
| `STORAGE_DIR` | Папка хранения изображений внутри контейнера |
| `MAX_IMAGE_MB` | Максимальный размер входного изображения |
| `U2NET_HOME` | Папка хранения моделей rembg внутри API-контейнера |
| `REMBG_MODEL_NAME` | Модель удаления фона. Текущий рекомендуемый вариант: `birefnet-portrait` |
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
| `AI_DTYPE` | Тип вычислений: `float32` выбран для стабильности на GTX 1660 |
| `AI_LOW_VRAM` | Режим экономии видеопамяти |
| `AI_OUTPUT_NAME` | Имя итогового AI-файла в папке job |
| `AI_INPAINT_TIMEOUT_SECONDS` | Таймаут запроса к AI-сервису |
| `AI_DEFAULT_STEPS` | Количество diffusion-шагов по умолчанию |
| `AI_DEFAULT_GUIDANCE_SCALE` | Guidance scale по умолчанию |
| `AI_DEFAULT_STRENGTH` | Strength по умолчанию |
| `AI_RESTORE_FACE_AFTER_INPAINT` | Возвращать ли защищённую область лица после генерации |

Секретные значения, например `API_KEY`, должны храниться в `.env`.

## Запуск через Docker Compose

Первый запуск может занять много времени: Docker скачает базовые образы и зависимости, а сервисы скачивают модели в локальные папки `models/rembg` и `models/huggingface`.

Полный запуск API + PostgreSQL + AI:

```bash
docker compose --profile ai up --build
```

Запуск в фоне:

```bash
docker compose --profile ai up -d --build
```

Проверка контейнеров:

```bash
docker ps
```

Ожидаемые контейнеры:

```text
avatar_api
avatar_ai_inpaint
avatar_db
```

Проверка API:

```bash
curl http://localhost:8000/health
```

Проверка AI-сервиса из API-контейнера:

```bash
docker exec -it avatar_api python -c "import httpx; print(httpx.get('http://ai_inpaint:8010/health', timeout=10).json())"
```

Swagger UI доступен по адресу:

```text
http://localhost:8000/docs
```

## API

Все API-методы защищены заголовком:

```text
x-api-key: change_me
```

Значение должно совпадать с `API_KEY` из `.env`.

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

Сервис создаёт задачу и сразу возвращает `job_id`. Обработка выполняется в фоне.

Пример ответа:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "status": "queued",
  "face_similarity_score": null
}
```

### Получить статус задачи

```http
GET /api/v1/avatar-jobs/{job_id}
```

Возможные статусы:

```text
queued
processing
done
failed
```

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

## Интеграция с 1С:УНФ

В проекте находится расширение:

```text
onec_config/КорпоративныеАватарыИИ.cfe
```

Расширение содержит обработку:

```text
Генерация корпоративного аватара
```

Обработка позволяет:

- выбрать портретное фото сотрудника;
- отправить фото в FastAPI-сервис;
- получить `job_id`;
- периодически проверять статус задачи;

После выполнения обработки автоматически откроется результат стандартным просмотрщиком Windows с возможностью скачать готовый PNG.

### Подключение расширения в 1С:УНФ

1. Откройте информационную базу УНФ в режиме **Конфигуратор**.
2. Откройте меню **Конфигурация → Расширения конфигурации**.
3. Добавьте расширение из файла `.cfe`.
4. Убедитесь, что расширение активно.
5. Обновите конфигурацию базы данных.
6. Запустите 1С в режиме предприятия.

### Использование обработки

В учебной / демо УНФ обработку можно открыть через подсистему "Корпоративные аватары ИИ", а также через:

```text
Функции для технического специалиста → Обработки → Генерация корпоративного аватара
```

На форме обработки:

1. Проверьте параметры подключения:
   - сервер: `localhost`;
   - порт: `8000`;
   - API-ключ: `change_me`;
   - стиль: `ai_business`.
2. Нажмите **Выбрать фото**.
3. Выберите портретное изображение.
4. Нажмите **Сгенерировать аватар**.
5. Дождитесь статуса завершения.
6. Готовый PNG откроется автоматически.

Перед запуском генерации из 1С сервис должен быть запущен:

```bash
docker compose --profile ai up -d
```

## Используемые модели

### Удаление фона

Используется `rembg` с моделью:

```text
birefnet-portrait
```

Эта модель выбрана после сравнения с `u2net`, `isnet-general-use` и `u2net_human_seg`, так как дала лучший результат на портретных фото, особенно на волосах.

### Генерация деловой одежды

Используется inpainting-модель:

```text
Lykon/dreamshaper-8-inpainting
```

AI-сервис получает:

- `result.png` — изображение после замены фона;
- `clothes_mask.png` — маска области одежды;
- `face_restore_mask.png` — маска для восстановления лица после генерации.

После inpainting лицо частично возвращается из исходного результата, чтобы сохранить индивидуальность.

### Проверка сохранности лица

После генерации выполняется face similarity check. Если score ниже порога `FACE_SIMILARITY_THRESHOLD`, задача получает статус `failed`.

## Стили

Основной стиль для выполнения ТЗ:

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

## Очистка временных файлов

Очистить job-файлы:

```bash
rm -rf app/storage/jobs/*
```

Очистить Docker build cache:

```bash
docker builder prune
```

Остановить сервисы:

```bash
docker compose --profile ai down
```

Остановить и удалить volume PostgreSQL, если нужно полностью сбросить базу:

```bash
docker compose --profile ai down -v
```

## Что не коммитить

В репозиторий не должны попадать:

```text
.env
models/
app/storage/jobs/
build.log
*.onnx
*.safetensors
*.pt
*.pth
```

Файл `.gitignore` уже содержит соответствующие правила.

## Возможные проблемы

### AI-сервис не найден

Проверьте, что запущен профиль `ai`:

```bash
docker compose --profile ai up -d
```

### Недостаточно места на диске

Модели и Docker-образы занимают много места. Основные папки:

```text
models/rembg
models/huggingface
Docker Desktop WSL data
```

Можно посмотреть размер:

```bash
du -sh models/*
docker system df
```
