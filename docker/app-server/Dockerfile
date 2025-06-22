FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN python3 -m pip install --upgrade pip && \
    pip install 'uv' -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

WORKDIR /app
COPY . .

RUN python3 -m uv sync

ENV VLM_SERVER_URL="http://mineru-vlm:30000"

VOLUME [ "/app/files" ]

ENTRYPOINT ["python3", "-m", "uv", "run", "--", "uvicorn", "main:api_app", "--host=0.0.0.0", "--port=7860"]

