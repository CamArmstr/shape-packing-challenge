"""Semicircle geometry: representation, Shapely polygon, overlap & containment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from .config import POLYGON_ARC_POINTS, MEC_BOUNDARY_POINTS


@dataclass(frozen=True)
class Semicircle:
    """A unit semicircle.

    (x, y) is the center of the full disk.
    theta is the angle (radians) the curved part extends toward.
    The flat edge passes through (x, y) perpendicular to theta.
    """
    x: float
    y: float
    theta: float


def semicircle_polygon(sc: Semicircle, n_arc: int = POLYGON_ARC_POINTS) -> Polygon:
    """Build a Shapely Polygon for the given semicircle."""
    from .config import RADIUS

    # Arc from theta - pi/2 to theta + pi/2
    angles = np.linspace(sc.theta - math.pi / 2, sc.theta + math.pi / 2, n_arc)
    arc_x = sc.x + RADIUS * np.cos(angles)
    arc_y = sc.y + RADIUS * np.sin(angles)

    # Close with the flat edge (first and last arc points connect through center line)
    coords = list(zip(arc_x, arc_y))
    return Polygon(coords)


def semicircle_boundary_points(sc: Semicircle, n: int = MEC_BOUNDARY_POINTS) -> np.ndarray:
    """Sample n points along the boundary of a semicircle (arc + flat edge).

    Returns an (n, 2) array.
    """
    from .config import RADIUS

    # Half the points on the arc, half on the flat edge
    n_arc = n // 2
    n_flat = n - n_arc

    # Arc points
    angles = np.linspace(sc.theta - math.pi / 2, sc.theta + math.pi / 2, n_arc)
    arc = np.column_stack([sc.x + RADIUS * np.cos(angles),
                           sc.y + RADIUS * np.sin(angles)])

    # Flat edge endpoints
    perp = sc.theta + math.pi / 2
    ex1 = sc.x + RADIUS * math.cos(perp), sc.y + RADIUS * math.sin(perp)
    ex2 = sc.x - RADIUS * math.cos(perp), sc.y - RADIUS * math.sin(perp)

    t = np.linspace(0, 1, n_flat).reshape(-1, 1)
    flat = np.array(ex1) * (1 - t) + np.array(ex2) * t

    return np.vstack([arc, flat])


def _overlap_polygon_arc_points(a: Semicircle, b: Semicircle) -> int:
    """Choose enough arc samples to resolve near-tangent overlaps."""
    from .config import RADIUS

    center_distance = math.hypot(a.x - b.x, a.y - b.y)
    overlap_depth = (2 * RADIUS) - center_distance
    if overlap_depth <= 0:
        return POLYGON_ARC_POINTS

    # Keep the inscribed polygon close enough to the true arc that tiny but
    # valid overlaps near tangency are still represented.
    target_sagitta = max(overlap_depth / 4, 1e-12)
    max_segment_angle = 2 * math.acos(max(-1.0, min(1.0, 1.0 - target_sagitta / RADIUS)))
    if max_segment_angle <= 0:
        return max(POLYGON_ARC_POINTS, 32768)

    segment_count = math.ceil(math.pi / max_segment_angle)
    return max(POLYGON_ARC_POINTS, min(segment_count + 1, 32768))


def semicircles_overlap(a: Semicircle, b: Semicircle) -> bool:
    """Return True when two semicircles share interior area."""
    from .config import RADIUS

    if math.hypot(a.x - b.x, a.y - b.y) >= 2 * RADIUS:
        return False

    n_arc = _overlap_polygon_arc_points(a, b)
    pa = semicircle_polygon(a, n_arc=n_arc)
    pb = semicircle_polygon(b, n_arc=n_arc)

    # Interior intersections are overlaps; pure boundary contact is allowed.
    return pa.intersects(pb) and not pa.touches(pb)


def semicircle_contained_in_circle(sc: Semicircle, cx: float, cy: float, cr: float) -> bool:
    """Check whether a semicircle is fully contained in a circle (cx, cy, cr)."""
    from .config import CONTAINMENT_TOL
    pts = semicircle_boundary_points(sc)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return bool(np.all(dists <= cr + CONTAINMENT_TOL))
