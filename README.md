# corporate-avatar-service

Сервис генерации корпоративных аватаров сотрудников на основе портретной фотографии.

Проект принимает исходное фото сотрудника, формирует корпоративный аватар и возвращает PNG-результат:

- заменяет фон на единый светлый корпоративный градиент;
- дорисовывает / заменяет одежду на деловую;
- сохраняет лицо и проверяет похожесть результата на исходное фото;
- предоставляет HTTP API;
- содержит расширение 1С:УНФ для вызова сервиса из интерфейса 1С.

---

## Содержание

- [Стек](#стек)
- [Архитектура](#архитектура)
- [Структура проекта](#структура-проекта)
- [Требования к окружению](#требования-к-окружению)
- [Первый запуск проекта с нуля](#первый-запуск-проекта-с-нуля)
- [Быстрый запуск после первой сборки](#быстрый-запуск-после-первой-сборки)
- [Когда нужна пересборка](#когда-нужна-пересборка)
- [Остановка проекта](#остановка-проекта)
- [Настройка и описание .env](#настройка-env)
- [API](#api)
- [Тестирование API из WSL](#тестирование-api-из-wsl)
- [Интеграция с 1С:УНФ](#интеграция-с-1сунф)
- [Используемые модели](#используемые-модели)
- [Стили генерации](#стили-генерации)
- [Очистка временных файлов](#очистка-временных-файлов)
- [Возможные проблемы](#возможные-проблемы)

---

## Стек

- Python 3.11
- FastAPI
- PostgreSQL
- SQLAlchemy
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
        | восстановление лица
        | проверка похожести лица
        v
PNG avatar result
```

Основной API работает в контейнере:

```text
avatar_api
```

AI inpainting вынесен в отдельный контейнер:

```text
avatar_ai_inpaint
```

База PostgreSQL работает в контейнере:

```text
avatar_db
```

Разделение API и AI-контейнера сделано специально: PyTorch, diffusers и модели Stable Diffusion занимают много места и требуют GPU, поэтому они не смешиваются с основным API-контейнером.

---

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
  ai_service.py
                       отдельный FastAPI-сервис для Stable Diffusion inpainting

  create_avatar_request_from_image.py
                       генерация JSON-запроса с base64-изображением

  make_test_base64.py
                       тестовая генерация base64

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

Если GPU не доступен, можно попробовать переключить AI на CPU через `.env`:

```env
AI_DEVICE=cpu
AI_DTYPE=float32
```

Но генерация AI-одежды на CPU будет очень медленной.

---

## Первый запуск проекта с нуля

### 1. Открыть Docker Desktop

Перед запуском проекта на Windows необходимо открыть Docker Desktop и дождаться, пока Docker Engine полностью запустится.

Проверьте Docker из WSL:

```bash
docker version
docker compose version
```

Если команды не работают, проверьте, что в Docker Desktop включена WSL Integration для вашей Ubuntu.

Обычно это находится здесь:

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
REMBG_MODEL_NAME=birefnet-portrait
AI_MODEL_ID=Lykon/dreamshaper-8-inpainting
AI_DEVICE=cuda
AI_DTYPE=float32
AI_LOW_VRAM=true
```

---

### 4. Проверить GPU в Docker

Для режима `ai_business` желательно наличие NVIDIA GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если команда показывает видеокарту, GPU-доступ из Docker работает.

---

### 5. Собрать основной API-контейнер

```bash
docker compose build api
```

Этот контейнер содержит:

- FastAPI;
- SQLAlchemy;
- rembg;
- OpenCV;
- основную бизнес-логику сервиса.

---

### 6. Собрать AI-контейнер

```bash
docker compose build ai_inpaint
```

Этот шаг может выполняться долго, потому что устанавливаются:

- PyTorch;
- diffusers;
- transformers;
- accelerate;
- другие AI-зависимости.

---

### 7. Запустить проект

Полный запуск API + PostgreSQL + AI-сервиса:

```bash
docker compose --profile ai up
```

Запуск в фоне:

```bash
docker compose --profile ai up -d
```

При первом запуске модели будут скачаны автоматически:

```text
models/rembg
models/huggingface
```

---

### 8. Проверить контейнеры

```bash
docker ps
```

Ожидаемые контейнеры:

```text
avatar_api
avatar_ai_inpaint
avatar_db
```

---

### 9. Проверить API

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

### 10. Проверить AI-сервис

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
docker ps
curl http://localhost:8000/health
```

---

## Запуск со сборкой одной командой

Можно использовать короткий вариант:

```bash
docker compose --profile ai up --build
```

Эта команда одновременно пересобирает изменённые образы и запускает контейнеры.

Но для первого запуска рекомендуется выполнять шаги отдельно:

```bash
docker compose build api
docker compose build ai_inpaint
docker compose --profile ai up
```

---

## Когда нужна пересборка

### Изменился код в `app/`

Например, изменены FastAPI routes, сервисы обработки изображений, настройки:

```bash
docker compose build api
docker compose --profile ai up -d
```

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

Остановить контейнеры и удалить volume PostgreSQL:

```bash
docker compose --profile ai down -v
```

Команду `down -v` используйте осторожно: она удалит данные PostgreSQL.

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

---

## Описание переменных `.env`

| Переменная | Назначение |
|---|---|
| `APP_NAME` | Название приложения FastAPI |
| `APP_ENV` | Окружение приложения, например `dev` |
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
| `REMBG_MODEL_NAME` | Модель удаления фона. Рекомендуемый вариант: `birefnet-portrait` |
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
| `AI_DEFAULT_STEPS` | Значение по умолчанию для количества diffusion-шагов |
| `AI_DEFAULT_GUIDANCE_SCALE` | Значение по умолчанию для guidance scale |
| `AI_DEFAULT_STRENGTH` | Значение по умолчанию для strength |
| `AI_RESTORE_FACE_AFTER_INPAINT` | Возвращать ли защищённую область лица после генерации |

Важно: в текущей версии для основного стиля `ai_business` итоговые параметры AI-генерации зафиксированы в коде `app/api/routes_avatar_jobs.py`:

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

Сервис создаёт задачу и сразу возвращает `job_id`. Обработка выполняется в фоне.

Пример ответа:

```json
{
  "job_id": "00000000-0000-0000-0000-000000000000",
  "status": "queued",
  "face_similarity_score": null
}
```

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

---

### Подключение расширения в 1С:УНФ

1. Откройте информационную базу УНФ в режиме **Конфигуратор**.
2. Откройте меню:

   ```text
   Конфигурация → Расширения конфигурации
   ```

3. Добавьте расширение из файла:

   ```text
   onec_config/КорпоративныеАватарыИИ.cfe
   ```

4. Убедитесь, что расширение активно.
5. Обновите конфигурацию базы данных.
6. Запустите 1С в режиме предприятия.

---

### Использование обработки

В учебной / демо УНФ обработку можно открыть через подсистему Корпоративные аватары ИИ, а также через:

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

2. Нажмите **Выбрать фото**.

3. Выберите портретное изображение.

4. Нажмите **Сгенерировать аватар**.

5. Обработка создаст задачу и будет периодически проверять статус.

6. После завершения готовый PNG откроется автоматически.

Перед запуском генерации из 1С сервис должен быть запущен:

```bash
docker compose --profile ai up -d
```

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

## Очистка временных файлов

Сгенерированные изображения не должны попадать в репозиторий.

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

Остановить сервисы и удалить volume PostgreSQL:

```bash
docker compose --profile ai down -v
```

Команда `down -v` удалит данные PostgreSQL.

---

## Возможные проблемы

### `Invalid base64 image`

Проверьте, что ваша base64 строка не содержит следующих символов:

```text
CR
LF
табы
пробелы
```

перед отправкой.

---

### `avatar_api` пропадает из `docker ps`

Возможна нехватка RAM при работе.

В коде предусмотрена очистка rembg-сессии перед AI inpainting.

При повторении проблемы увеличьте память Docker Desktop:

```text
Docker Desktop → Settings → Resources → Memory
```

---

### AI-сервис не найден

Проверьте, что запущен профиль `ai`:

```bash
docker compose --profile ai up -d
```

В `docker ps` должен быть контейнер:

```text
avatar_ai_inpaint
```

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

Осторожно: команда удаляет неиспользуемые образы. После неё Docker может заново скачивать и пересобирать образы.

Если нужно полностью удалить ещё и неиспользуемые volumes:

```bash
docker system prune -a --volumes
```

Осторожно: эта команда может удалить volume PostgreSQL с тестовыми данными.

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

Проверьте, что Docker Desktop полностью закрыт.

После этого снова выполните команду `diskpart`.

---

#### Проверка после сжатия

После сжатия снова проверьте свободное место на системном диске.

Затем можно открыть Docker Desktop, WSL и проверить:

```bash
docker system df
df -h
du -sh models/*
```

Если Docker-образы были удалены через `docker system prune -a`, при следующем запуске проекта потребуется повторная сборка:

```bash
docker compose build api
docker compose build ai_inpaint
docker compose --profile ai up
```

---

## Лицензии и ограничения

Проект использует open-source библиотеки и модели.

Для промышленного и коммерческого использования необходимо отдельно проверить:

- лицензии выбранных моделей;
- лицензии Python-библиотек;
- правила обработки персональных данных сотрудников;
- правила хранения и передачи фотографий сотрудников.
