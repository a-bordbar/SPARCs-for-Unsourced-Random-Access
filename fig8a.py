#!/usr/bin/env python3
"""Plot Figure 8(a): fixed-point residual g(eta)."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fig8_common import (
    Fig8Solution,
    get_solution,
    print_summary,
    save_curve_csv,
)


def value_at(solution: Fig8Solution, curve, eta_value: float) -> float:
    import numpy as np
    return float(np.interp(eta_value, solution.eta, curve))


def plot_fig8a(
    solution: Fig8Solution,
    output: Path,
    pdf_output: Path | None,
    show: bool,
) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "fig8_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    if not show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    line_star, = ax.plot(
        solution.eta,
        solution.g_star,
        "-.",
        linewidth=1.8,
    )
    line_alg, = ax.plot(
        solution.eta,
        solution.g_alg,
        ":",
        linewidth=2.0,
    )
    line_opt, = ax.plot(
        solution.eta,
        solution.g_opt,
        "-",
        linewidth=1.8,
    )
    line_mix, = ax.plot(
        solution.eta,
        solution.g_mix,
        "--",
        linewidth=1.8,
    )

    ax.axhline(0.0, linewidth=0.8)

    ax.annotate(
        r"$P_{\rm opt}$",
        xy=(0.30, value_at(solution, solution.g_opt, 0.30)),
        xytext=(0.38, 0.27),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$\alpha_1P_{\rm opt}+\alpha_2P_\ast$",
        xy=(0.42, value_at(solution, solution.g_mix, 0.42)),
        xytext=(0.47, 0.08),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$P_{\rm alg}$",
        xy=(0.26, value_at(solution, solution.g_alg, 0.26)),
        xytext=(0.30, -0.30),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$P_\ast$",
        xy=(0.55, value_at(solution, solution.g_star, 0.55)),
        xytext=(0.61, -0.65),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$g(\eta)$")
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-1.0, 0.4)
    ax.grid(True, alpha=0.30, linewidth=0.7)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)

    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output)

    if show:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 8(a) from the SPARC-URA paper."
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
        "--output",
        type=Path,
        default=Path("plots/fig8a.png"),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("plots/fig8a.pdf"),
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
        args.output,
        None if args.no_pdf else args.pdf_output,
        args.show,
    )

    print(f"Saved CSV: {args.csv_output}")
    print(f"Saved PNG: {args.output}")
    if not args.no_pdf:
        print(f"Saved PDF: {args.pdf_output}")


if __name__ == "__main__":
    main()
