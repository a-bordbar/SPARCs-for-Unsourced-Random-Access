#!/usr/bin/env python3
"""Plot Figure 6 theoretical curves and empirical AMP markers together."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_THEORY_DIR = Path("data/fig6_data")
DEFAULT_OUTPUT = Path("plots/fig6_all.png")
DEFAULT_PDF_OUTPUT = Path("plots/fig6_all.pdf")


def parse_j_values(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one J value is required.")
    return values


def configure_matplotlib(show: bool) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "fig6_plot_all_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    if not show:
        matplotlib.use("Agg")


def load_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {type(exc).__name__}: {exc}")
        return None
    if data.size == 0 or data.dtype.names is None:
        print(f"WARNING: empty or malformed CSV skipped: {path}")
        return None
    return np.atleast_1d(data)


def discover_amp_csvs(data_root: Path) -> list[Path]:
    candidates = sorted(data_root.glob("fig6_amp_empirical_roc*/fig6_amp_results.csv"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def finite_xy(data: np.ndarray, x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data[x_name], dtype=float)
    y = np.asarray(data[y_name], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def plot_theory(ax, theory_dir: Path, j_values: Iterable[int]) -> None:
    colors = {15: "C0", 20: "C1"}
    for J in j_values:
        path = theory_dir / f"fig6_J{int(J)}.csv"
        data = load_csv(path)
        if data is None:
            print(f"WARNING: theoretical curve missing: {path}")
            continue
        required = {"S_in", "E_required_dB"}
        if not required.issubset(data.dtype.names or ()):
            print(f"WARNING: theoretical CSV missing columns {required}: {path}")
            continue
        x, y = finite_xy(data, "E_required_dB", "S_in")
        order = np.argsort(y)
        ax.plot(
            x[order],
            y[order],
            "-",
            color=colors.get(int(J), None),
            linewidth=1.7,
            label=f"Theory J={int(J)}",
        )

    asym_path = theory_dir / "fig6_asymptotic.csv"
    asym = load_csv(asym_path)
    if asym is not None and asym.dtype.names is not None:
        if {"S_in", "E_optimal_asymptotic_dB"}.issubset(asym.dtype.names):
            x, y = finite_xy(asym, "E_optimal_asymptotic_dB", "S_in")
            order = np.argsort(y)
            ax.plot(x[order], y[order], "k-", linewidth=1.7, label="Optimal asymptotic")
        if {"S_in", "E_amp_asymptotic_dB"}.issubset(asym.dtype.names):
            x, y = finite_xy(asym, "E_amp_asymptotic_dB", "S_in")
            order = np.argsort(y)
            ax.plot(x[order], y[order], "k--", linewidth=1.7, label="AMP asymptotic")
    else:
        print(f"WARNING: asymptotic theoretical CSV missing: {asym_path}")


def marker_for_j(J: int) -> str:
    return {15: "o", 20: "s", 30: "^", 40: "D", 50: "v", 60: "P"}.get(int(J), "o")


def color_for_j(J: int) -> str | None:
    return {15: "C0", 20: "C1", 30: "C2", 40: "C3", 50: "C4", 60: "C5"}.get(int(J), None)


def source_label(path: Path) -> str:
    parent = path.parent.name
    if parent.startswith("fig6_amp_empirical_roc_"):
        return parent.replace("fig6_amp_empirical_roc_", "")
    if parent == "fig6_amp_empirical_roc":
        return "empirical ROC"
    return parent


def plot_empirical(ax, amp_csvs: list[Path], j_values: tuple[int, ...], include_ceiling: bool) -> None:
    required = {"J", "S_in", "E_required_dB", "status"}
    plotted = 0
    for path in amp_csvs:
        data = load_csv(path)
        if data is None:
            continue
        if not required.issubset(data.dtype.names or ()):
            print(f"WARNING: AMP CSV missing columns {required}: {path}")
            continue
        source = source_label(path)
        for J in j_values:
            j_mask = np.asarray(data["J"], dtype=int) == int(J)
            if not np.any(j_mask):
                continue
            rows = data[j_mask]
            statuses = np.asarray(rows["status"]).astype(str)
            ok_mask = statuses == "ok"
            if include_ceiling:
                ok_mask |= statuses == "ceiling_reached"
            x = np.asarray(rows["E_required_dB"], dtype=float)
            y = np.asarray(rows["S_in"], dtype=float)
            mask = ok_mask & np.isfinite(x) & np.isfinite(y)
            if not np.any(mask):
                omitted = int(np.count_nonzero(statuses == "ceiling_reached"))
                if omitted:
                    print(f"WARNING: {omitted} ceiling-reached J={J} point(s) omitted from {path}")
                continue
            order = np.argsort(y[mask])
            ax.plot(
                x[mask][order],
                y[mask][order],
                linestyle="None",
                marker=marker_for_j(int(J)),
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.4,
                color=color_for_j(int(J)),
                label=f"AMP {source} J={int(J)}",
            )
            plotted += int(np.count_nonzero(mask))
    if plotted == 0:
        print("WARNING: no empirical AMP markers were plotted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Fig. 6 theoretical curves and empirical AMP calibrated/ROC markers."
    )
    parser.add_argument(
        "--j-values",
        type=parse_j_values,
        default=None,
        help="Comma-separated J values to plot. Default: all fig6_J*.csv files in --theory-dir.",
    )
    parser.add_argument("--theory-dir", type=Path, default=DEFAULT_THEORY_DIR)
    parser.add_argument(
        "--amp-csv",
        type=Path,
        action="append",
        default=None,
        help="AMP results CSV. May be supplied multiple times. Defaults to data/fig6_amp_empirical_roc*/fig6_amp_results.csv.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Root used for automatic AMP CSV discovery.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--include-ceiling", action="store_true")
    parser.add_argument("--xlim", type=parse_j_values_or_float_pair, default=(-5.0, 6.0))
    parser.add_argument("--ylim", type=parse_j_values_or_float_pair, default=(0.0, 2.5))
    return parser.parse_args()


def parse_j_values_or_float_pair(text: str) -> tuple[float, float]:
    parts = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values, e.g. -5,6.")
    return (parts[0], parts[1])


def discover_theory_j_values(theory_dir: Path) -> tuple[int, ...]:
    values: list[int] = []
    for path in sorted(theory_dir.glob("fig6_J*.csv")):
        stem = path.stem
        try:
            values.append(int(stem.replace("fig6_J", "")))
        except ValueError:
            continue
    return tuple(sorted(set(values)))


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.show)
    import matplotlib.pyplot as plt

    j_values = args.j_values if args.j_values is not None else discover_theory_j_values(args.theory_dir)
    if not j_values:
        raise SystemExit(f"No theoretical J CSV files found in {args.theory_dir}")

    amp_csvs = args.amp_csv if args.amp_csv else discover_amp_csvs(args.data_root)
    if not amp_csvs:
        print(f"WARNING: no AMP result CSVs found under {args.data_root}")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_theory(ax, args.theory_dir, j_values)
    plot_empirical(ax, amp_csvs, j_values, args.include_ceiling)

    ax.set_xlabel(r"$\mathcal{E}_{\mathrm{in}}\,[\mathrm{dB}]$")
    ax.set_ylabel(r"$S_{\mathrm{in}}$")
    ax.set_xlim(*args.xlim)
    ax.set_ylim(*args.ylim)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(f"Saved PNG: {args.output}")
    if not args.no_pdf:
        args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf_output)
        print(f"Saved PDF: {args.pdf_output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
