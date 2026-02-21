# OomLlama - Efficient LLM inference with .oom format
# 2x smaller than GGUF using Q2 quantization
#
# Usage: docker run jtmeent/oomllama --help
#        docker run jtmeent/oomllama "Hello world!"

FROM python:3.11-slim

LABEL maintainer="Humotica AI Lab <ai@humotica.nl>"
LABEL description="OomLlama - Efficient LLM inference with .oom format (2x smaller than GGUF)"
LABEL version="0.6.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python package directly (no build needed)
COPY python/oomllama /app/oomllama

# Install dependencies
RUN pip install --no-cache-dir requests

# Create models directory
RUN mkdir -p /root/.cache/oomllama

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from oomllama import OomLlama; print('OK')" || exit 1

# Default entrypoint - CLI
ENTRYPOINT ["python", "-m", "oomllama"]
CMD ["--list"]
