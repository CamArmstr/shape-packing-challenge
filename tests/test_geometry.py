"""Tests for semicircle geometry."""

import math
from semicircle_packing.geometry import (
    Semicircle,
    semicircle_polygon,
    semicircles_overlap,
    semicircle_contained_in_circle,
    semicircle_boundary_points,
    farthest_boundary_point_from,
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


# -- Overlap regression tests ------------------------------------------------
# These cover bugs found during the competition.


def test_no_overlap_back_to_back_6dp_pi():
    """Back-to-back with 6dp pi (the server's rounding) should not overlap."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(0, 0, 3.141593)
    assert not semicircles_overlap(a, b)


def test_no_overlap_up_vs_down():
    """Semicircles at same center facing opposite vertical directions."""
    a = Semicircle(0, 0, math.pi / 2)
    b = Semicircle(0, 0, -math.pi / 2)
    assert not semicircles_overlap(a, b)


def test_no_overlap_tangent():
    """Two semicircles touching at exactly one point (tangent)."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(2, 0, math.pi)
    assert not semicircles_overlap(a, b)


def test_overlap_tiny_intrusion():
    """Arcs overlapping by 0.001 in distance — must be caught."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(1.999, 0, math.pi)
    assert semicircles_overlap(a, b)


def test_no_overlap_tiny_gap():
    """Arcs separated by 0.001 — must not be flagged."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(2.001, 0, math.pi)
    assert not semicircles_overlap(a, b)


def test_overlap_one_inside_other():
    """One semicircle fully inside another."""
    a = Semicircle(0, 0, 0)
    b = Semicircle(0.1, 0.1, 0)
    assert semicircles_overlap(a, b)


def test_overlap_diagonal_orientation():
    """Overlapping semicircles at diagonal angles."""
    a = Semicircle(0, 0, math.pi / 4)
    b = Semicircle(0.5, 0.5, math.pi / 4 + math.pi)
    assert semicircles_overlap(a, b)


def test_overlap_flat_edge_crosses_arc():
    """Flat edge of one crosses arc of other — the wedge-shaped overlap case."""
    a = Semicircle(-0.2224, -0.4517, 3.0167)
    b = Semicircle(-0.7929, 0.0007, 2.2313)
    assert semicircles_overlap(a, b)


def test_overlap_degrees_not_radians():
    """Theta in degrees produces garbage orientations that should overlap."""
    # These two are close enough in position that random orientations overlap
    a = Semicircle(-0.758, 5.119, 88.624)
    b = Semicircle(-0.439, 4.320, 24.102)
    assert semicircles_overlap(a, b)


# -- Farthest boundary point tests -------------------------------------------


def test_farthest_point_arc_midpoint():
    """Farthest point from a query behind the semicircle is the arc midpoint."""
    sc = Semicircle(0, 0, 0)
    # Query far to the left — farthest point should be the arc tip at (1, 0)
    fx, fy = farthest_boundary_point_from(sc, -10, 0)
    assert abs(fx - 1.0) < 1e-10
    assert abs(fy - 0.0) < 1e-10


def test_farthest_point_endpoint():
    """Farthest point from a query on the arc side is a flat-edge endpoint."""
    sc = Semicircle(0, 0, 0)  # arc faces right, flat edge is vertical
    # Query far to the right — farthest should be an endpoint at (0, ±1)
    fx, fy = farthest_boundary_point_from(sc, 10, 0)
    assert abs(fx) < 1e-10
    assert abs(abs(fy) - 1.0) < 1e-10


def test_farthest_point_diagonal():
    """Farthest point in a diagonal direction stays on the arc."""
    sc = Semicircle(0, 0, 0)
    fx, fy = farthest_boundary_point_from(sc, -5, -5)
    # Should be on the arc at angle pi/4
    assert abs(fx - math.cos(math.pi / 4)) < 1e-10
    assert abs(fy - math.sin(math.pi / 4)) < 1e-10
