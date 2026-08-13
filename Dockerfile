FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    XDG_CACHE_HOME=/home/appuser/.cache \
    TORCH_HOME=/home/appuser/.cache/torch \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir \
        "numpy>=1.26,<3" \
        "Pillow>=10,<12" \
        "torch>=2.3,<3" \
        "torchvision>=0.18,<1" \
        "gradio>=6.22,<7"

RUN python -m pip install --no-cache-dir --no-deps .

RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home \
        --home-dir /home/appuser --shell /usr/sbin/nologin appuser \
    && mkdir -p "${TORCH_HOME}" "${HF_HOME}" \
    && chown -R appuser:appuser /home/appuser

COPY app.py ./
COPY examples ./examples
COPY docs/assets ./docs/assets

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/gradio_api/info', timeout=4).close()"]

CMD ["python", "/app/app.py"]
