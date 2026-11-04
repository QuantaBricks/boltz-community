# Thin image: Python + repo source + launcher. PyTorch/CUDA wheels, boltz deps and
# the model weights are installed/downloaded when the container starts (see launcher.py).
# Build from the repo root:  docker build -f deploy/docker/Dockerfile -t boltz2-server:thin .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    BOLTZ_CACHE=/opt/boltz-cache \
    BOLTZ_VENV=/opt/venv \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY server.py ./
COPY deploy/docker/launcher.py ./launcher.py

EXPOSE 8080
CMD ["python", "-u", "launcher.py"]
