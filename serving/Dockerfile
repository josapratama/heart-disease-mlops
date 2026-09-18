# ────────────────────────────────────────────────────────────
# Dockerfile untuk Heart Disease Flask App
# Load model TensorFlow langsung (standalone, tanpa TF Serving)
# ────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="josa_pratama"
LABEL description="Heart Disease Classification Model Serving"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code dan model
COPY app.py .
COPY model/ ./model/

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Jalankan dengan Gunicorn
CMD python app.py
