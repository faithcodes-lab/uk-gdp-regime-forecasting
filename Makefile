.PHONY: data train evaluate shap dashboard clean all test lint

data:
	python data/scripts/download_ons_gdp.py
	python data/scripts/download_boe_rates.py
	python data/scripts/download_oecd.py
	python data/scripts/build_dataset.py

train:
	python src/models/arima_baseline.py
	python src/models/ridge_baseline.py
	python src/models/xgboost_model.py
	python src/models/lightgbm_model.py

evaluate:
	python src/evaluation/metrics.py
	python src/evaluation/regime_evaluation.py
	python src/evaluation/statistical_tests.py

shap:
	python src/explainability/shap_analysis.py
	python src/explainability/regime_shap.py
	python src/explainability/stability_metrics.py

dashboard:
	python dashboard/app.py

all: data train evaluate shap

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv
	rm -f data/processed/*.parquet
	rm -f results/figures/*.png
	rm -f results/tables/*.csv
