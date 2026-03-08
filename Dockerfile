# ─────────────────────────────────────────────────────────────────────────────
# 5G-DDoS MCP Server — Dockerfile
#
# Multi-stage build: slim production image with all ML + LLM dependencies.
#
# Build:
#   docker build -t 5g-ddos-mcp .
#
# Run with Claude backend:
#   docker run --env-file .env -p 8000:8000 5g-ddos-mcp
#
# Run with Ollama backend (Ollama must be running on host):
#   docker run --env-file .env -p 8000:8000 \
#     -e LLM_BACKEND=ollama \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     5g-ddos-mcp
#
# Mount pre-trained models:
#   docker run --env-file .env -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/data:/app/data:ro \
#     5g-ddos-mcp
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uv


# ── Stage 2: Production ───────────────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL org.opencontainers.image.title="5G-DDoS MCP Server"
LABEL org.opencontainers.image.description="MCP server for 5G DDoS detection using NCSRD-DS-5GDDoS dataset"
LABEL org.opencontainers.image.source="https://github.com/ncsrd/5g-ddos-mcp"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime libs for XGBoost / NumPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/        ./src/
COPY scripts/    ./scripts/

# Create data and models directories (mount in docker-compose / k8s)
RUN mkdir -p /app/data /app/models

# Non-root user for security
RUN useradd -r -s /bin/false -u 1000 mcpserver && \
    chown -R mcpserver:mcpserver /app
USER mcpserver

# MCP server uses stdio transport by default
# Override with environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
ENV LOG_LEVEL=INFO

EXPOSE 8000

# Health check (MCP server may not expose HTTP; adapt as needed)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "from src.config import server_config; print('OK')" || exit 1

CMD ["python", "-m", "src.server"]
