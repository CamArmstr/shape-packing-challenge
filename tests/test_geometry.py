"""Tests for semicircle geometry."""

import math
import pytest
from semicircle_packing.geometry import (
    Semicircle,
    semicircle_polygon,
    semicircles_overlap,
    semicircle_contained_in_circle,
    semicircle_boundary_points,
)


def test_polygon_area():
    """A unit semicircle polygon should have area close to pi/2."""
    sc = Semicircle(0, 0, 0)
    poly = semicircle_polygon(sc)
    expected = math.pi / 2
    assert abs(poly.area - expected) < 0.01


def test_no_overlap_far_apart():
    a = Semicircle(0, 0, 0)
    b = Semicircle(10, 0, math.pi)
    assert not semicircles_overlap(a, b)


def test_overlap_same_position():
    a = Semicircle(0, 0, 0)
    b = Semicircle(0, 0, math.pi / 4)
    assert semicircles_overlap(a, b)


def test_no_overlap_back_to_back():
    """Two semicircles back-to-back sharing only the flat edge should not overlap."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(0, 0, math.pi)
    assert not semicircles_overlap(a, b)


def test_overlap_slight_intrusion():
    """Semicircles barely overlapping."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(1.5, 0, math.pi)
    assert semicircles_overlap(a, b)


def test_containment_in_large_circle():
    sc = Semicircle(0, 0, 0)
    assert semicircle_contained_in_circle(sc, 0, 0, 10)


def test_not_contained_in_small_circle():
    sc = Semicircle(0, 0, 0)
    assert not semicircle_contained_in_circle(sc, 0, 0, 0.5)


def test_boundary_points_shape():
    sc = Semicircle(0, 0, 0)
    pts = semicircle_boundary_points(sc, 100)
    assert pts.shape == (100, 2)
