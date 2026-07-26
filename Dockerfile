# UK GDP regime forecasting:  pipeline image.
# Build:  docker compose build pipeline
# Run:    docker compose run --rm pipeline     (executes `make data`)

FROM python:3.11-slim

WORKDIR /app

# Build deps in case any wheel needs compiling. The slim base ships without
# gcc; pandas/numpy/pyarrow wheels are usually prebuilt, but keep this for
# resilience across architectures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so the layer caches across rebuilds.
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -e .

# Application source.
COPY src/ ./src/
COPY data/__init__.py ./data/
COPY data/schemas/ ./data/schemas/
COPY config/ ./config/
COPY tests/ ./tests/
COPY Makefile ./

# Runtime directories: gitignored locally; created empty here so the volume
# mounts (or first-run write) have a target.
RUN mkdir -p data/raw data/interim data/processed data/lineage logs

ENV PYTHONPATH=/app

# Default: build the final dataset end-to-end.
CMD ["make", "data"]
