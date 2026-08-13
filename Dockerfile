FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app.py ./

RUN python -m pip install --no-cache-dir ".[demo]"

EXPOSE 7860

CMD ["python", "app.py"]
