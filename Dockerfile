FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app

WORKDIR /app

# Run as a non-root user (uid 1000 matches the usual first Linux host user, which keeps
# the ./docs bind mount writable).
RUN groupadd --system --gid 1000 app && useradd --system --uid 1000 --gid app --home-dir /app app

COPY requirements.txt .
RUN pip install -r requirements.txt

# All top-level modules (app, config, auth, vector_store, ...) - no list to keep in sync.
COPY --chown=app:app *.py ./
COPY --chown=app:app templates ./templates

# docs = PDF/DOCX files, qdrant_db = vector index, data = users.json
RUN mkdir -p /app/docs /app/qdrant_db /app/data && chown -R app:app /app
USER app
VOLUME ["/app/docs", "/app/qdrant_db", "/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Bind 0.0.0.0 inside the container. Keep ONE worker: local-mode Qdrant allows a single process,
# and the rate limiter state is in-process.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-timeout", "900"]
