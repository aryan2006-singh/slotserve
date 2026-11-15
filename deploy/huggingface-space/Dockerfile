# Hugging Face Space (free "CPU basic" hardware: 2 vCPU, 16 GB RAM).
# Builds straight from the GitHub repo, so the Space only needs this file and
# README.md. Set REPO / REF as Space variables to point at a fork or branch.
FROM python:3.11-slim

ARG REPO=https://github.com/aryan2006-singh/slotserve.git
ARG REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; give it a writable home and model cache.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

RUN git clone --depth 1 --branch "$REF" "$REPO" . \
    && pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Download the weights at build time so the Space starts fast after a sleep.
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer as T; \
m='HuggingFaceTB/SmolLM2-135M-Instruct'; T.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"

EXPOSE 7860
CMD ["python", "-m", "src.server.app", \
     "--model", "HuggingFaceTB/SmolLM2-135M-Instruct", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--max-slots", "4", "--max-seq-len", "1024"]
