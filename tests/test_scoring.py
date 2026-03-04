"""Tests for scoring and validation."""

import math
import pytest
from semicircle_packing.geometry import Semicircle
from semicircle_packing.scoring import (
    minimum_enclosing_circle,
    validate_and_score,
)
import numpy as np


def test_mec_single_point():
    pts = np.array([[1.0, 2.0]])
    cx, cy, r = minimum_enclosing_circle(pts)
    assert abs(cx - 1.0) < 1e-6
    assert abs(cy - 2.0) < 1e-6
    assert r < 1e-6


def test_mec_two_points():
    pts = np.array([[0.0, 0.0], [2.0, 0.0]])
    cx, cy, r = minimum_enclosing_circle(pts)
    assert abs(cx - 1.0) < 1e-6
    assert abs(cy) < 1e-6
    assert abs(r - 1.0) < 1e-6


def test_mec_square():
    """MEC of a unit square centered at origin."""
    pts = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    cx, cy, r = minimum_enclosing_circle(pts)
    assert abs(cx) < 1e-6
    assert abs(cy) < 1e-6
    assert abs(r - math.sqrt(2)) < 1e-6


def test_validate_wrong_count():
    result = validate_and_score([Semicircle(0, 0, 0)])
    assert not result.valid
    assert "Expected 15" in result.errors[0]


def test_validate_overlapping():
    # 15 semicircles all at origin -> overlapping
    scs = [Semicircle(0, 0, i * 0.1) for i in range(15)]
    result = validate_and_score(scs)
    assert not result.valid
    assert any("overlap" in e for e in result.errors)


def test_validate_valid_spread():
    """15 semicircles spread far apart should be valid."""
    scs = [Semicircle(i * 10, 0, 0) for i in range(15)]
    result = validate_and_score(scs)
    assert result.valid
    assert result.score is not None
    assert result.score > 0
