# CPU image by default so it builds and runs on a laptop.
# For GPU: swap the base for nvidia/cuda:12.4.1-runtime-ubuntu22.04, install a
# CUDA torch build, and run with `--gpus all`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU wheels keep the image ~1GB instead of ~6GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY bench/ bench/
COPY tests/ tests/
COPY Makefile .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "src.server.app"]
CMD ["--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--host", "0.0.0.0", "--port", "8000"]
