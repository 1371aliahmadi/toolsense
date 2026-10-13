# ── ToolSense API — Dockerfile ────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean
#
# Build:   docker build -t toolsense-api .
# Run:     docker run -p 8000:8000 -v $(pwd)/data:/app/data toolsense-api
# Compose: docker compose up

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and config
COPY src/      ./src/
COPY config/   ./config/
COPY app/      ./app/

# Data is mounted at runtime (see docker-compose.yml)
# This avoids baking large binary files into the image
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd -m toolsense && chown -R toolsense /app
USER toolsense

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
