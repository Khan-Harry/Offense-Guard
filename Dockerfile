FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY . .

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Hugging Face standard non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
