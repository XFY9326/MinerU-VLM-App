FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --no-cache && \
    rm -rf ${UV_CACHE_DIR}

COPY . .

RUN addgroup --system appuser && \
    adduser --system --ingroup appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

ENV VLM_SERVER_URL="http://mineru-vlm:30000"

VOLUME [ "/app/files" ]

ENTRYPOINT ["uv", "run", "--", "uvicorn", "main:api_app", "--host=0.0.0.0", "--port=7860"]