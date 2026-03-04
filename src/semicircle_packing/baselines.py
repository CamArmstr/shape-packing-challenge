"""Baseline packing strategies."""

from __future__ import annotations

import math

from .config import N, RADIUS
from .geometry import Semicircle


def circular_baseline() -> list[Semicircle]:
    """Place semicircles evenly around a ring, flat edges facing inward.

    Each semicircle points outward (theta = angle from center).
    Score ~5.9. Very naive.
    """
    d_angle = 2 * math.pi / N
    ring_radius = RADIUS / math.sin(math.pi / N) + RADIUS * 0.1

    semicircles = []
    for i in range(N):
        angle = i * d_angle
        x = ring_radius * math.cos(angle)
        y = ring_radius * math.sin(angle)
        theta = angle
        semicircles.append(Semicircle(x=x, y=y, theta=theta))

    return semicircles


def grid_baseline() -> list[Semicircle]:
    """Pack semicircles as paired full disks in a 3x2 grid + 3 extras on top.

    6 full disks (12 semicircles) arranged in a 3x2 grid, plus 3 semicircles
    on top pointing upward. Score ~3.35.
    """
    semicircles = []

    # 3x2 grid of full disks (each = 2 back-to-back semicircles)
    for row in range(2):
        for col in range(3):
            cx = (col - 1) * 2  # x = -2, 0, 2
            cy = row * 2        # y = 0, 2
            # One semicircle pointing right, one pointing left
            semicircles.append(Semicircle(x=cx, y=cy, theta=0))
            semicircles.append(Semicircle(x=cx, y=cy, theta=math.pi))

    # 3 extra semicircles on top, pointing up
    # Row 1 disks extend to y=3, so place flat edges at y=3
    for col in range(3):
        cx = (col - 1) * 2  # x = -2, 0, 2
        semicircles.append(Semicircle(x=cx, y=3, theta=math.pi / 2))

    return semicircles
