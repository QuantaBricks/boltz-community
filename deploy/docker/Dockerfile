# Build from the repo root:  docker build -f deploy/docker/Dockerfile -t boltz2-server .
# The base only needs the CUDA driver shim; the PyTorch cu128 wheels bundle
# their own CUDA/cuDNN libraries.
FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12.8 wheels (needed for RTX 50xx / sm_120 as well as older cards)
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --default-timeout=300 --retries 10 torch --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY server.py ./
RUN pip install --default-timeout=300 --retries 10 ".[server]"

# Weights (~5 GB) and CCD data are NOT baked into the image: `boltz predict`
# downloads them into BOLTZ_CACHE on the first job after a replica starts.
# (No persistent volume, so each fresh replica downloads once.)
ENV BOLTZ_CACHE=/opt/boltz-cache
RUN mkdir -p /opt/boltz-cache

EXPOSE 8080
# The gateway reaches the container over IPv6, so bind to "::".
CMD ["python", "server.py", "--host", "::", "--port", "8080"]
