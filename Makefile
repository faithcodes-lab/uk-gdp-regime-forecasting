# Makefile for UK GDP Regime Forecasting
# Run `make help` to see available targets.

.PHONY: help install test lint format clean

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies into the current venv"
	@echo "  test          Run the pytest suite"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run black formatter"
	@echo "  format-check  Check formatting without modifying files"
	@echo "  clean         Remove build artefacts and cache"

install:
	pip install -e ".[dev,notebooks]"

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