"""Semicircle geometry: representation, Shapely polygon, overlap & containment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, Point

from .config import POLYGON_ARC_POINTS, OVERLAP_TOL, MEC_BOUNDARY_POINTS


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


def semicircles_overlap(a: Semicircle, b: Semicircle) -> bool:
    """Return True if two semicircles have overlapping interior area."""
    pa = semicircle_polygon(a)
    pb = semicircle_polygon(b)
    return pa.intersection(pb).area > OVERLAP_TOL


def semicircle_contained_in_circle(sc: Semicircle, cx: float, cy: float, cr: float) -> bool:
    """Check whether a semicircle is fully contained in a circle (cx, cy, cr)."""
    from .config import CONTAINMENT_TOL
    pts = semicircle_boundary_points(sc)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return bool(np.all(dists <= cr + CONTAINMENT_TOL))
