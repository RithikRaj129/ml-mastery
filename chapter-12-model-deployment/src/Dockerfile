# Dockerfile - Multi-stage build for optimised ML service image
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ── Final image (slim, no build tools) ───────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code and model
COPY main.py .
COPY model_pipeline.joblib .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV MODEL_PATH=/app/model_pipeline.joblib \
    PORT=8000 \
    WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

# Start with gunicorn (WSGI) or uvicorn (ASGI)
CMD ["uvicorn", "main:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "4",
     "--log-level", "info"]
