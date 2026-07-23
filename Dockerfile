FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch first. amd64 gets the CUDA 12.8 build (required for RTX 50-series /
# Blackwell sm_120 support; torch>=2.7.0 is the first release with Blackwell kernels).
# arm64 gets the CUDA 13.0 build (PyTorch's cu128 index has no arm64 wheels; cu130 does).
# NOTE: cu130 wheels need a driver whose CUDA ceiling (`nvidia-smi`) is >= 13.0 — see
# docs/torch-cuda-version.md. If your arm64 target's driver only covers 12.x, this
# build won't run on GPU; fall back to a plain `pip install torch` (CPU-only) instead.
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu130; \
    else \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY highlight.py .
COPY model/ ./model/

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "highlight.py"]