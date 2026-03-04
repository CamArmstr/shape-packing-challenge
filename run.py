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
        help="export solution coordinates to JSON file",
    )
    parser.add_argument(
        "--from-json", type=str, metavar="FILE",
        help="load and score a JSON solution file instead of running solve()",
    )
    args = parser.parse_args()

    from semicircle_packing.scoring import validate_and_score, print_report
    from semicircle_packing.geometry import Semicircle

    if args.from_json:
        semicircles = _load_solution(args.from_json)
        solve_elapsed = 0.0
    else:
        from solve import solve

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
        data = [{"x": sc.x, "y": sc.y, "theta": sc.theta} for sc in semicircles]
        with open(args.export, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Solution exported to {args.export}")
        print()

    if args.visualize or args.save_plot:
        from semicircle_packing.visualization import plot_packing
        plot_packing(semicircles, result.mec, save_path=args.save_plot)


def _load_solution(path: str) -> list:
    """Load a solution from a JSON file: [{"x": ..., "y": ..., "theta": ...}, ...]"""
    from semicircle_packing.geometry import Semicircle

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of {x, y, theta} objects")

    semicircles = []
    for i, item in enumerate(data):
        for key in ("x", "y", "theta"):
            if key not in item:
                raise ValueError(f"Item {i} missing required field '{key}'")
        semicircles.append(Semicircle(x=float(item["x"]), y=float(item["y"]), theta=float(item["theta"])))

    return semicircles


if __name__ == "__main__":
    main()
