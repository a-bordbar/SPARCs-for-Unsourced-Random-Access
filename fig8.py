#!/usr/bin/env python3
"""Generate both panels of Figure 8 with one numerical solve."""

from __future__ import annotations

import argparse
from pathlib import Path

from fig8_common import get_solution, print_summary, save_curve_csv
from fig8a import plot_fig8a
from fig8b import plot_fig8b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 8(a) and Figure 8(b)."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/fig8/fig8_solution.npz"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("data/fig8/fig8_curves.csv"),
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=Path("plots/fig8a.png"),
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=Path("plots/fig8b.png"),
    )
    parser.add_argument(
        "--pdf-output-a",
        type=Path,
        default=Path("plots/fig8a.pdf"),
    )
    parser.add_argument(
        "--pdf-output-b",
        type=Path,
        default=Path("plots/fig8b.pdf"),
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    solution = get_solution(
        cache_path=args.cache,
        recompute=args.recompute,
    )

    print_summary(solution)
    save_curve_csv(solution, args.csv_output)

    plot_fig8a(
        solution,
        args.output_a,
        None if args.no_pdf else args.pdf_output_a,
        args.show,
    )
    plot_fig8b(
        solution,
        args.output_b,
        None if args.no_pdf else args.pdf_output_b,
        args.show,
    )

    print(f"Saved CSV: {args.csv_output}")
    print(f"Saved PNG: {args.output_a}")
    print(f"Saved PNG: {args.output_b}")
    if not args.no_pdf:
        print(f"Saved PDF: {args.pdf_output_a}")
        print(f"Saved PDF: {args.pdf_output_b}")


if __name__ == "__main__":
    main()
