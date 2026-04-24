"""Render solution figures for the research paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import os


def make_semicircle_polygon(x, y, theta, n_points=128):
    """Create a semicircle polygon matching the challenge scorer.

    theta is the direction the curved part extends toward.
    Arc sweeps from theta - pi/2 to theta + pi/2.
    """
    angles = np.linspace(theta - np.pi / 2, theta + np.pi / 2, n_points)
    coords = list(zip(x + np.cos(angles), y + np.sin(angles)))
    return Polygon(coords)


def render_solution(solution, R, title, filename, figsize=(5, 5)):
    """Render a solution as a clean figure."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect('equal')

    colors_inner = '#4477AA'  # blue
    colors_mid = '#EE7733'    # orange
    colors_outer = '#228833'  # green

    radii = [np.sqrt(s['x']**2 + s['y']**2) for s in solution]

    # Enclosing circle
    enc_circle = plt.Circle((0, 0), R, fill=False, edgecolor='#CC3311',
                            linewidth=2.0, linestyle='-', zorder=10)
    ax.add_patch(enc_circle)

    for i, s in enumerate(solution):
        semi = make_semicircle_polygon(s['x'], s['y'], s['theta'])
        if semi.is_empty:
            continue

        r = radii[i]
        if r < 1.0:
            color = colors_inner
        elif r > R - 0.5:
            color = colors_outer
        else:
            color = colors_mid

        if semi.geom_type == 'Polygon':
            xs, ys = semi.exterior.xy
            ax.fill(xs, ys, alpha=0.4, color=color, zorder=2)
            ax.plot(xs, ys, color='black', linewidth=0.8, zorder=3)
        elif semi.geom_type == 'MultiPolygon':
            for geom in semi.geoms:
                xs, ys = geom.exterior.xy
                ax.fill(xs, ys, alpha=0.4, color=color, zorder=2)
                ax.plot(xs, ys, color='black', linewidth=0.8, zorder=3)

    ax.plot(0, 0, 'k.', markersize=3, zorder=5)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    margin = R * 0.15
    ax.set_xlim(-R - margin, R + margin)
    ax.set_ylim(-R - margin, R + margin)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'  Saved: {filename}')


def main():
    os.makedirs('figures', exist_ok=True)

    milestones = [
        ('solutions/R2.970026_mar28.json', 2.970, 'Phase 1: SA + Hill-Climber\n$R = 2.970$'),
        ('solutions/R2.961486.json', 2.961, 'Phase 2: LNS Basin Jump\n$R = 2.961$'),
        ('solutions/R2.960373.json', 2.960, 'Phase 2: Exact Polishing\n$R = 2.960$'),
        ('solutions/R2.948997.json', 2.949, 'Phase 3: Fast MCMC Explorer\n$R = 2.949$'),
        ('best_solution.json', 2.9486, 'Phase 4: Nano Polish\n$R = 2.948572$'),
    ]

    print('Rendering milestone figures...')
    for filepath, approx_r, title in milestones:
        if not os.path.exists(filepath):
            print(f'  SKIP (not found): {filepath}')
            continue
        with open(filepath) as f:
            sol = json.load(f)

        basename = os.path.basename(filepath).replace('.json', '')
        if basename.startswith('R'):
            actual_r = float(basename.replace('R', '').replace('_mar28', '').replace('_current', ''))
        else:
            actual_r = approx_r

        safe_name = basename.replace('.', '_').replace('_mar28', '')
        render_solution(sol, actual_r, title, f'figures/{safe_name}.pdf')
        render_solution(sol, actual_r, title, f'figures/{safe_name}.png')

    print('\nRendering main figure (best solution)...')
    with open('best_solution.json') as f:
        sol = json.load(f)
    render_solution(sol, 2.948572,
                    'Best Solution: $R = 2.948572$\n3-11-1 Topology',
                    'figures/best_solution.pdf', figsize=(6, 6))
    render_solution(sol, 2.948572,
                    'Best Solution: $R = 2.948572$\n3-11-1 Topology',
                    'figures/best_solution.png', figsize=(6, 6))

    print('\nDone!')


if __name__ == '__main__':
    main()
