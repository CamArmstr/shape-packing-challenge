"""Your packing solution. Modify this file to beat the baseline."""

from semicircle_packing import Semicircle
from semicircle_packing.baselines import grid_baseline


def solve() -> list[Semicircle]:
    """Return a list of 15 unit semicircles packed into the smallest circle.

    Each semicircle is Semicircle(x, y, theta) where:
      - (x, y) is the center of the full disk
      - theta (radians) is the direction the curved part extends
      - The flat edge passes through (x, y) perpendicular to theta

    Score = radius of the minimum enclosing circle (lower is better).
    """
    return grid_baseline()
