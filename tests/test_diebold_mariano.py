"""Tests for src/evaluation/diebold_mariano.py.

The HLN correction test (test_dm_hln_correction_shrinks_statistic) is the
methodological heart, and the two hand-computed loss tests
(test_dm_power_2_squared_loss_hand_computed and the absolute-loss twin)
verify the arithmetic against values computed independently in the test,
not just that the code runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.diebold_mariano import DMTestResult, diebold_mariano_test


def test_dm_identical_forecasts_returns_p_value_one():
    """When errors_a equals errors_b, the test returns statistic 0 and p_value 1."""
    errors = np.array([0.1, 0.2, -0.1, 0.3, -0.2, 0.0, 0.4, -0.3])
    result = diebold_mariano_test(errors, errors)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.raw_dm_statistic == 0.0


def test_dm_clearly_better_forecast_returns_significant_p_value():
    """When model A's errors are an order of magnitude smaller, p_value is very small."""
    rng = np.random.default_rng(42)
    errors_a = rng.normal(scale=0.1, size=50)
    errors_b = rng.normal(scale=1.0, size=50)
    result = diebold_mariano_test(errors_a, errors_b, power=2)
    assert result.p_value < 0.001
    # statistic should be negative because A's squared losses are smaller, so d < 0
    assert result.statistic < 0


def test_dm_hln_correction_shrinks_statistic():
    """For h=1, the HLN-corrected statistic equals raw_dm_statistic * sqrt((n-1)/n)."""
    rng = np.random.default_rng(42)
    errors_a = rng.normal(scale=0.5, size=32)
    errors_b = rng.normal(scale=1.0, size=32)
    result = diebold_mariano_test(errors_a, errors_b, h=1, power=2)
    expected_factor = float(np.sqrt((32 - 1) / 32))
    assert result.statistic == pytest.approx(result.raw_dm_statistic * expected_factor)


def test_dm_bonferroni_correction_multiplies_raw_p_value():
    """With n_comparisons=6, p_value_bonferroni equals min(p_value * 6, 1.0)."""
    rng = np.random.default_rng(42)
    errors_a = rng.normal(scale=0.5, size=50)
    errors_b = rng.normal(scale=1.0, size=50)
    result = diebold_mariano_test(errors_a, errors_b, power=2, n_comparisons=6)
    expected = min(result.p_value * 6, 1.0)
    assert result.p_value_bonferroni == pytest.approx(expected)


def test_dm_bonferroni_capped_at_one():
    """When raw_p * n_comparisons exceeds 1, the corrected p_value is capped at 1.0."""
    rng = np.random.default_rng(42)
    errors_a = rng.normal(scale=1.0, size=50)
    errors_b = errors_a + rng.normal(scale=0.001, size=50)  # nearly identical
    result = diebold_mariano_test(errors_a, errors_b, power=2, n_comparisons=10)
    assert result.p_value_bonferroni == 1.0


def test_dm_power_2_squared_loss_hand_computed():
    """Squared-loss DM statistic matches values computed independently from the formulas.

    errors_a = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] (four 1s, three 2s, three 3s)
    errors_b = [2] * 10
    d = e_a^2 - e_b^2 = [-3, 0, 5, -3, 0, 5, -3, 0, 5, -3]
    d_bar = (4*(-3) + 3*0 + 3*5) / 10 = 3 / 10 = 0.3
    gamma_0 = (4*(-3.3)^2 + 3*(-0.3)^2 + 3*(4.7)^2) / 10 = 110.10 / 10 = 11.01
    raw_dm = 0.3 / sqrt(11.01 / 10)
    HLN = raw_dm * sqrt(9 / 10)
    """
    errors_a = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
    errors_b = np.array([2.0] * 10)

    expected_raw_dm = 0.3 / (11.01 / 10) ** 0.5
    expected_hln = expected_raw_dm * (9 / 10) ** 0.5

    result = diebold_mariano_test(errors_a, errors_b, power=2)
    assert result.raw_dm_statistic == pytest.approx(expected_raw_dm)
    assert result.statistic == pytest.approx(expected_hln)


def test_dm_power_1_absolute_loss_hand_computed():
    """Absolute-loss DM statistic matches values computed independently from the formulas.

    errors_a = [1, -2, 3, -1, 2, -3, 1, -2, 3, -1]
    errors_b = [2, -1, 1, -2, 1, -2, 2, -1, 1, -2]
    |e_a| = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    |e_b| = [2, 1, 1, 2, 1, 2, 2, 1, 1, 2]
    d = |e_a| - |e_b| = [-1, 1, 2, -1, 1, 1, -1, 1, 2, -1]
        (four -1, four 1, two 2)
    d_bar = (4*(-1) + 4*1 + 2*2) / 10 = 4 / 10 = 0.4
    gamma_0 = (4*(-1.4)^2 + 4*(0.6)^2 + 2*(1.6)^2) / 10 = 14.40 / 10 = 1.44
    raw_dm = 0.4 / sqrt(1.44 / 10)
    HLN = raw_dm * sqrt(9 / 10); algebraically HLN^2 = 0.16 * 0.9 / 0.144 = 1.0
    """
    errors_a = np.array([1.0, -2.0, 3.0, -1.0, 2.0, -3.0, 1.0, -2.0, 3.0, -1.0])
    errors_b = np.array([2.0, -1.0, 1.0, -2.0, 1.0, -2.0, 2.0, -1.0, 1.0, -2.0])

    expected_raw_dm = 0.4 / (1.44 / 10) ** 0.5
    expected_hln = expected_raw_dm * (9 / 10) ** 0.5  # algebraically 1.0

    result = diebold_mariano_test(errors_a, errors_b, power=1)
    assert result.raw_dm_statistic == pytest.approx(expected_raw_dm)
    assert result.statistic == pytest.approx(expected_hln)
    assert result.statistic == pytest.approx(1.0)


def test_dm_handles_small_sample_n_32():
    """Realistic project scale: n=32 (one expanding-window scheme's predictions per model)."""
    rng = np.random.default_rng(42)
    errors_a = rng.normal(scale=0.5, size=32)
    errors_b = rng.normal(scale=1.0, size=32)
    result = diebold_mariano_test(errors_a, errors_b, power=2)
    assert result.n_observations == 32
    assert isinstance(result.statistic, float)
    assert 0.0 <= result.p_value <= 1.0


def test_dm_raises_on_mismatched_lengths():
    """Length mismatch between errors_a and errors_b raises ValueError."""
    with pytest.raises(ValueError, match="length mismatch"):
        diebold_mariano_test(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_dm_raises_on_empty_arrays():
    """Empty errors_a or errors_b raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        diebold_mariano_test(np.array([]), np.array([]))


def test_dm_raises_on_invalid_power():
    """power values other than 1 or 2 raise ValueError."""
    with pytest.raises(ValueError, match="power must be"):
        diebold_mariano_test(np.array([1.0, 2.0]), np.array([1.0, 2.0]), power=3)


def test_dm_raises_on_nan_in_errors():
    """NaN values in either errors array raise ValueError."""
    with pytest.raises(ValueError, match="NaN"):
        diebold_mariano_test(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))


def test_dm_result_dataclass_has_all_documented_fields():
    """DMTestResult exposes all eight fields with the documented types."""
    result = diebold_mariano_test(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8]))
    assert isinstance(result, DMTestResult)
    for field_name in (
        "statistic",
        "p_value",
        "p_value_bonferroni",
        "n_observations",
        "n_comparisons",
        "loss_power",
        "horizon",
        "raw_dm_statistic",
    ):
        assert hasattr(result, field_name), f"missing field {field_name}"
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.p_value_bonferroni, float)
    assert isinstance(result.n_observations, int)
    assert isinstance(result.n_comparisons, int)
    assert isinstance(result.loss_power, int)
    assert isinstance(result.horizon, int)
    assert isinstance(result.raw_dm_statistic, float)
