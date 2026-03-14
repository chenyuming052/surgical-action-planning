FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
COPY surgcast/ surgcast/
RUN pip install --no-cache-dir -e .

# Optional: W&B for experiment tracking
RUN pip install --no-cache-dir wandb

# Copy configs and scripts
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

ENTRYPOINT ["python"]
