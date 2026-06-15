# Makefile for UK GDP Regime Forecasting
# Run `make help` to see available targets.

.PHONY: help install download process data all test lint format format-check clean

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies into the current venv"
	@echo "  download      Download raw data from all sources (ONS, BoE, FRED, BoE_YC)"
	@echo "  process       Build final dataset from cached raw CSVs (skips downloads)"
	@echo "  data          End-to-end: download + process; writes data/processed/final_dataset.parquet"
	@echo "  all           Build dataset, run tests, run lint, check formatting"
	@echo "  test          Run the pytest suite"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run black formatter"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  clean         Remove build artefacts and cache"

install:
	pip install -e ".[dev,notebooks]"

download:
	python -m src.data.build_dataset --download-only

process:
	python -m src.data.build_dataset --process-only

data:
	python -m src.data.build_dataset

all: data test lint format-check

test:
	pytest

lint:
	ruff check src/ tests/

format:
	black src/ tests/

format-check:
	black --check src/ tests/

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
