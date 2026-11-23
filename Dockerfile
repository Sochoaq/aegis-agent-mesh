# syntax=docker/dockerfile:1
# ==============================================================================
# Multi-Stage Dockerfile - Aegis Agent Mesh (API, Web Console & LLMOps CLI)
# ==============================================================================

# --- Stage 1: Builder (Installs dependencies into an isolated virtual env) ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime (Minimal hardened non-root production image) ---
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Aegis Agent Mesh" \
      org.opencontainers.image.description="Autonomous multi-agent incident triage, hybrid RRF-RAG, and LLMOps guardrail engine" \
      org.opencontainers.image.authors="Santiago Alejandro Ochoa Quesada" \
      org.opencontainers.image.source="https://github.com/Sochoaq/aegis-agent-mesh" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="/opt/venv/bin:$PATH" \
    AEGIS_DB_PATH="/app/data/aegis_audit.db"

RUN groupadd --gid 10001 aegis && \
    useradd --uid 10001 --gid aegis --no-create-home --shell /usr/sbin/nologin aegis && \
    mkdir -p /app/data && \
    chown -R aegis:aegis /app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=aegis:aegis src ./src
COPY --chown=aegis:aegis benchmarks ./benchmarks

USER aegis

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2)"]

CMD ["python", "-m", "aegis.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
