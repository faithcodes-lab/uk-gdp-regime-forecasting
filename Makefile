# Makefile for UK GDP Regime Forecasting
# Run `make help` to see available targets.

.PHONY: help install download process data eda regimes break-tests figure-regimes figure-cv-splits tune train evaluate shap all test lint format format-check clean

help:
	@echo "Available targets:"
	@echo "  install       Install dependencies into the current venv"
	@echo "  download      Download raw data from all sources (ONS, BoE, FRED, BoE_YC)"
	@echo "  process       Build final dataset from cached raw CSVs (skips downloads)"
	@echo "  data          End-to-end: download + process; writes data/processed/final_dataset.parquet"
	@echo "  eda           Run exploratory data analysis; writes results/figures/eda/ and results/eda-summary.md"
	@echo "  regimes       Add regime column to final dataset (run after 'make data'); writes data/processed/final_dataset.parquet"
	@echo "  break-tests   Run Chow tests + Bai-Perron sweep on gdp_growth; writes results/regimes/"
	@echo "  figure-regimes Render publication-quality regime figure; writes results/figures/regime_visualisation.{png,pdf}"
	@echo "  figure-cv-splits Render the two CV-scheme figures; writes results/figures/cv_splits_{expanding,regime_aligned}.{png,pdf}"
	@echo "  tune          Force re-tuning of all models and train; writes results/tuning/ and results/models/"
	@echo "  train         Train all four models using cached tuning where available; writes results/models/"
	@echo "  evaluate      Run the Sprint 4 evaluation end-to-end; writes results/{predictions,metrics,tables,figures}/"
	@echo "  shap          Run the regime-aware SHAP analysis end-to-end; writes results/shap/ and results/figures/shap_*"
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

eda:
	PYTHONPATH=. python scripts/eda.py

regimes:
	PYTHONPATH=. python scripts/add_regime_column.py

break-tests:
	PYTHONPATH=. python -m src.regimes.run_analysis

figure-regimes:
	PYTHONPATH=. python -m src.regimes.visualise

figure-cv-splits:
	python -m src.models.visualise_cv

tune:
	python -m src.models.train_all --retune

train:
	python -m src.models.train_all

evaluate:
	PYTHONPATH=. python scripts/run_evaluation.py

shap:
	PYTHONPATH=. python -m src.run_shap_analysis

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
