FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY . ./
COPY docker/entrypoint.py /entrypoint.py

RUN pip install --no-cache-dir .

EXPOSE 8000

# Render injects $PORT. The entrypoint binds uvicorn to 0.0.0.0:$PORT and also
# starts the Celery worker (all 5 queues) and Beat so web + workers share the
# same persistent disk inside this one container.
CMD ["python", "/entrypoint.py"]
