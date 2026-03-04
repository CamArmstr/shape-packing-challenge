"""Matplotlib visualization of a semicircle packing."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from .geometry import Semicircle, semicircle_polygon


def plot_packing(
    semicircles: list[Semicircle],
    mec: tuple[float, float, float] | None = None,
    save_path: str | None = None,
) -> None:
    """Plot the semicircle packing and optional enclosing circle."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fill_color = "#B0B0B0"
    edge_color = "#666666"
    label_color = "black"

    for i, sc in enumerate(semicircles):
        poly = semicircle_polygon(sc)
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, fc=fill_color, ec=edge_color,
                linewidth=1.4, zorder=2, alpha=0.85)
        centroid = poly.centroid
        ax.text(centroid.x, centroid.y, str(i),
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=label_color, zorder=3)

    if mec is not None:
        cx, cy, cr = mec

        outline = Circle(
            (cx, cy), cr, fill=False, ec="#CC0000", linewidth=2.5,
            linestyle="solid", zorder=4,
        )
        ax.add_patch(outline)

    # Title
    score_str = f"{mec[2]:.4f}" if mec else "?"
    ax.set_title(
        f"{len(semicircles)} semicircles   |   score = {score_str}",
        fontsize=15, fontweight="bold", color="#333333", pad=18,
    )

    ax.autoscale_view()
    pad = (mec[2] if mec else 5) * 0.15
    ax.margins(pad / 10)

    # Clean axes
    ax.grid(False)
    ax.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()
