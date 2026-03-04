"""Tests for baseline packings."""

from semicircle_packing.baselines import circular_baseline, grid_baseline
from semicircle_packing.scoring import validate_and_score
from semicircle_packing.config import N


def test_circular_baseline_valid():
    scs = circular_baseline()
    assert len(scs) == N
    result = validate_and_score(scs)
    assert result.valid, f"Baseline invalid: {result.errors}"


def test_circular_baseline_score_range():
    scs = circular_baseline()
    result = validate_and_score(scs)
    assert result.valid
    assert 4.0 < result.score < 7.0, f"Unexpected score: {result.score}"


def test_grid_baseline_valid():
    scs = grid_baseline()
    assert len(scs) == N
    result = validate_and_score(scs)
    assert result.valid, f"Grid baseline invalid: {result.errors}"


def test_grid_baseline_score_range():
    scs = grid_baseline()
    result = validate_and_score(scs)
    assert result.valid
    assert 3.0 < result.score < 4.0, f"Unexpected score: {result.score}"


def test_grid_beats_circular():
    circ = validate_and_score(circular_baseline())
    grid = validate_and_score(grid_baseline())
    assert grid.score < circ.score
