# Multi-stage Docker build for Enterprise Data Modeling Application
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/reports /app/models

# Set permissions
RUN chmod +x /app/entrypoint.sh

# Expose ports
EXPOSE 8501 8000 5432

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Production stage
FROM base as production

# Install production dependencies
RUN pip install --no-cache-dir gunicorn uvicorn

# Copy production configuration (skip for now)
# COPY docker/production/ /app/

# Run as non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command
CMD ["/app/entrypoint.sh"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Copy development configuration
COPY docker/development/ /app/

# Expose additional ports for development
EXPOSE 8888 8080

CMD ["/app/entrypoint-dev.sh"]
