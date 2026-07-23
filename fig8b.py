#!/usr/bin/env python3
"""Plot Figure 8(b): normalized integral G(eta) of g(eta)."""

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


def plot_fig8b(
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

    ax_curves = [
        (solution.G_star, "-."),
        (solution.G_alg, ":"),
        (solution.G_mix, "--"),
        (solution.G_opt, "-"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for curve, style in ax_curves:
        ax.plot(
            solution.eta,
            curve,
            style,
            linewidth=1.8,
        )

    ax.annotate(
        r"$P_\ast$",
        xy=(0.20, value_at(solution, solution.G_star, 0.20)),
        xytext=(0.27, value_at(solution, solution.G_star, 0.20) + 0.17),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$P_{\rm alg}$",
        xy=(0.24, value_at(solution, solution.G_alg, 0.24)),
        xytext=(0.34, value_at(solution, solution.G_alg, 0.24) + 0.19),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$\alpha_1P_{\rm opt}+\alpha_2P_\ast$",
        xy=(0.30, value_at(solution, solution.G_mix, 0.30)),
        xytext=(0.41, value_at(solution, solution.G_mix, 0.30) + 0.12),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        r"$P_{\rm opt}$",
        xy=(0.48, value_at(solution, solution.G_opt, 0.48)),
        xytext=(0.56, value_at(solution, solution.G_opt, 0.48) + 0.12),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$G(\eta)$")
    ax.set_xlim(0.0, 0.9)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels([])
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
        description="Reproduce Figure 8(b) from the SPARC-URA paper."
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
        default=Path("plots/fig8b.png"),
    )
    parser.add_argument(
        "--pdf-output",
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

    plot_fig8b(
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
