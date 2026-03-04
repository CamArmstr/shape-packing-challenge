"""CLI entry point: python run.py"""

import argparse
import json
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Semicircle Packing Challenge")
    parser.add_argument(
        "--visualize", action="store_true",
        help="show a matplotlib plot of the packing",
    )
    parser.add_argument(
        "--save-plot", type=str, metavar="FILE",
        help="save plot to file (png, svg, pdf)",
    )
    parser.add_argument(
        "--export", type=str, metavar="FILE",
        help="export results to JSON file (standardized format)",
    )
    args = parser.parse_args()

    from solve import solve
    from semicircle_packing.scoring import validate_and_score, print_report

    start = time.time()
    semicircles = solve()
    solve_elapsed = time.time() - start

    val_start = time.time()
    result = validate_and_score(semicircles)
    val_elapsed = time.time() - val_start

    print_report(result)
    print(f"  Solve time:      {solve_elapsed:.3f}s")
    print(f"  Validation time: {val_elapsed:.3f}s")
    print()

    if args.export:
        _export_results(args.export, semicircles, result, solve_elapsed, val_elapsed)

    if args.visualize or args.save_plot:
        from semicircle_packing.visualization import plot_packing
        plot_packing(semicircles, result.mec, save_path=args.save_plot)


def _export_results(path, semicircles, result, solve_time, val_time):
    """Export results in standardized JSON format."""
    import datetime
    from semicircle_packing.config import N, RADIUS

    data = {
        "challenge": "semicircle_packing",
        "version": "0.1.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "n": N,
            "radius": RADIUS,
        },
        "solution": [
            {"x": sc.x, "y": sc.y, "theta": sc.theta}
            for sc in semicircles
        ],
        "result": {
            "valid": result.valid,
            "score": result.score,
            "mec": {
                "cx": result.mec[0],
                "cy": result.mec[1],
                "radius": result.mec[2],
            } if result.mec else None,
            "errors": result.errors,
        },
        "timing": {
            "solve_seconds": round(solve_time, 6),
            "validation_seconds": round(val_time, 6),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results exported to {path}")
    print()


if __name__ == "__main__":
    main()
