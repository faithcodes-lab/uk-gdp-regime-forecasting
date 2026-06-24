# src/regimes

Detection, assignment, and visualisation of the six UK economic regimes used in this project.

This package owns the structural-breaks methodology that supports the regime-aware SHAP analysis at the heart of the dissertation. Three break tests, one assignment routine, one quality figure, and one orchestrator that ties them together.

## What is in here

| Module | Purpose | Key public symbols |
|---|---|---|
| chow.py | Chow (1960) F-test for a single known structural breakpoint | `chow_test()`, `ChowTestResult`, `InsufficientObservationsError` |
| bai_perron.py | Multi-break detection via the `ruptures` library (PELT, BinSeg, sensitivity sweep) | `detect_breaks_pelt()`, `detect_breaks_binseg()`, `tune_penalty()` |
| volatility.py | ICSS variance change-point test (Inclan-Tiao 1994) for variance shifts | `icss_test()`, `ICSSResult` |
| assign.py | Assigns each row of the final dataset to one of the six regimes | `assign_regimes()` |
| visualise.py | Figure of GDP growth with the six regime bands | `plot_gdp_with_regimes()` |
| run_analysis.py | Orchestrator: runs Chow, Bai-Perron, and ICSS on the live dataset and writes the `results/regimes/` artefacts | `main()` |

Each `*.py` module is exercised by a matching test file under `tests/`.

## How to re-run the analysis

After a data refresh:

```
make data             # rebuild data/processed/final_dataset.parquet from raw downloads
make regimes          # add the regime column to the parquet
make break-tests      # Chow + Bai-Perron + ICSS; writes results/regimes/
make figure-regimes   # publication PNG + PDF in results/figures/
```

Each target is independent and idempotent. `make regimes` reads from `config/regimes.yaml`; `make break-tests` and `make figure-regimes` operate on the parquet that `make data` (and then `make regimes`) produced.

## If the regime boundaries change

1. Edit `config/regimes.yaml`.
2. Re-run `make regimes` to rewrite the regime column on the parquet.
3. Re-run `make break-tests`: only the Chow test changes meaningfully, because Bai-Perron and ICSS are series-only and regenerate identical output.
4. Re-run `make figure-regimes`. The figure auto-loads from `regimes.yaml`.
5. Append a new entry to `report/decision-log.md` explaining why the boundaries moved.

## Date-basis convention

The dataset stores quarter-end dates (for example 2008-03-31 for Q1 2008); `config/regimes.yaml` stores quarter-start dates (for example 2008-01-01 for the GFC). Every module in this package normalises both sides to quarter-start before comparing, so the boundary quarter (Q1 2008 in the example) routes to the new regime regardless of which date basis the caller supplies.

## Break-index to date convention

A break at array index `i` maps to `series.index[i]`, the first observation of the new regime. This convention is the same in `chow.py`, `bai_perron.py`, `volatility.py`, and `assign.py`. Each test file includes a dedicated check.

## Running the test suite for this package

```
pytest tests/test_chow.py
pytest tests/test_bai_perron.py
pytest tests/test_assign.py
pytest tests/test_visualise.py
pytest tests/test_volatility.py
```

Each test file is self-contained and uses synthetic data; only `tests/test_visualise.py` writes temporary PNG / PDF files (via pytest's `tmp_path` fixture).

## Reading the results

| Path | Source | What it shows |
|---|---|---|
| `results/regimes/chow_test_results.json` | `run_analysis.py` | F statistic and p-value at each of the five hypothesised boundaries |
| `results/regimes/bai_perron_sweep.json` | `run_analysis.py` | All breakpoints detected at each penalty in the grid |
| `results/regimes/bai_perron_sensitivity.csv` | `run_analysis.py` | Same content, flat CSV for quick eyeballing |
| `results/regimes/icss_results.json` | `run_analysis.py` | ICSS variance breaks plus the GFC-validation block |
| `results/figures/regime_visualisation.png` / `.pdf` | `visualise.py` | The methodology-chapter figure |

## Notes on volatility.py and the GFC boundary dates

`volatility.py` implements the ICSS variance break test as a third instrument alongside Chow and Bai-Perron. The orchestrator writes its result to `results/regimes/icss_results.json` with a GFC-validation block that lists every detected variance break falling inside the GFC regime window and reports its distance, in quarters, from the current GFC start date.

The test itself is a standard part of the methodology. The exact GFC and Post-GFC Recovery boundary dates are an open question under supervisor discussion (see the relevant entry in `report/decision-log.md`); the dates may or may not move in a future revision of `config/regimes.yaml`.

## Why this package exists

The dissertation's novel contribution is regime-aware SHAP analysis. That analysis only makes sense if the regime definitions are defensible. This package produces the evidence base for that defence: literature-motivated boundaries from `config/regimes.yaml`, statistical validation from the three break tests (Chow, Bai-Perron, ICSS), and a clean figure for the methodology chapter. Methodological decisions made here are recorded in `report/decision-log.md`.
