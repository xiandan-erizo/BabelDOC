FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

WORKDIR /app

# Optional system libs for common wheels; keep minimal
# 使用阿里云 Debian 镜像源加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml README.md /app/
COPY babeldoc /app/babeldoc
COPY web /app/web

# Install server deps and project
RUN pip install --upgrade pip \
    && pip install fastapi uvicorn[standard] python-multipart \
    && pip install -e .

EXPOSE 8000

CMD ["uvicorn", "web.server.main:app", "--host", "0.0.0.0", "--port", "8000"]