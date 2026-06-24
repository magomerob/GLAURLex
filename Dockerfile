# ---- builder ----
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-ansi --no-root

COPY src ./src

RUN poetry install --only main --no-ansi

# ---- runtime ----
FROM python:3.11-slim-bookworm

ARG PUID=1000
ARG PGID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    GLAURLEX_DATA_DIR=/data

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

COPY src ./src

# Crear un usuario no-root con UID/GID parametrizables para que coincidan
# con el propietario del volumen en el host. UID 0 (root) está prohibido.
RUN set -eux; \
    if [ "${PUID}" = "0" ] || [ "${PGID}" = "0" ]; then \
        echo "PUID/PGID must not be 0 (root)" >&2; exit 1; \
    fi; \
    groupadd --system --gid "${PGID}" appuser; \
    useradd --system --uid "${PUID}" --gid "${PGID}" \
        --home /home/appuser --create-home --shell /usr/sbin/nologin appuser; \
    mkdir -p /data; \
    chown -R appuser:appuser /app /data

USER appuser

VOLUME ["/data"]
EXPOSE 8501

CMD ["streamlit", "run", "src/glaurlex/ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
