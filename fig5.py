"""Reproduce Figure 5 with parallel RS-potential evaluations.

The script searches for the inner energy required to meet the per-section
missed-detection/false-alarm target for each pair of inner sum-rate S_in and
section size J.  Each grid point is independent, so the search is parallelized
with a process pool. 
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.stats import norm
from rs_potential import minimize_rs_potential


ALPHA = 2.0
L = 8
J_VEC = np.array([15, 20, 30, 40, 50, 60], dtype=np.int64)
S_IN_VEC = np.linspace(0.05, 2.5, 250)
E_IN_DB_VEC = np.arange(-6.0, 3.01, 0.01)
PMD_TARGET = 0.05 / L
PFA_SCALE = 0.01
DEFAULT_DATA_DIR = Path("data/fig5_data")
DEFAULT_PNG_OUTPUT = Path("plots/fig5.png")
DEFAULT_PDF_OUTPUT = Path("plots/fig5.pdf")


@dataclass(frozen=True)
class GridTask:
    s_idx: int
    j_idx: int
    S_in: float
    J: int
    Ka: int
    gamma_target: float


@dataclass(frozen=True)
class GridResult:
    s_idx: int
    j_idx: int
    S_in: float
    J: int
    Ka: int
    beta: float
    q_active: float
    gamma_target: float
    E_required_dB: float
    eta_required: float
    gamma_effective_required: float


def compute_ka_vec(J_vec: np.ndarray, alpha: float) -> np.ndarray:
    return np.rint(2.0 ** (J_vec / alpha)).astype(np.int64)


def compute_gamma_targets(J_vec: np.ndarray, Ka_vec: np.ndarray) -> np.ndarray:
    gamma_target_vec = np.zeros(len(J_vec), dtype=float)
    for j_idx, J in enumerate(J_vec):
        Ka = int(Ka_vec[j_idx])
        pfa_target = PFA_SCALE * Ka / (2.0 ** int(J))
        gamma_target_vec[j_idx] = (norm.isf(PMD_TARGET) + norm.isf(pfa_target)) ** 2
    return gamma_target_vec


def make_tasks(
    J_vec: np.ndarray,
    Ka_vec: np.ndarray,
    S_in_vec: np.ndarray,
    gamma_target_vec: np.ndarray,
) -> list[GridTask]:
    return [
        GridTask(
            s_idx=s_idx,
            j_idx=j_idx,
            S_in=float(S_in),
            J=int(J),
            Ka=int(Ka_vec[j_idx]),
            gamma_target=float(gamma_target_vec[j_idx]),
        )
        for s_idx, S_in in enumerate(S_in_vec)
        for j_idx, J in enumerate(J_vec)
    ]


def solve_grid_point(task: GridTask) -> GridResult:
    R_in = task.S_in / task.Ka
    beta = (2.0 ** task.J) * R_in / task.J
    q_active = -np.expm1(task.Ka * np.log1p(-(2.0 ** (-task.J))))

    E_required_dB = float("nan")
    eta_required = float("nan")
    gamma_effective_required = float("nan")

    for E_in_dB in E_IN_DB_VEC:
        E_in = 10.0 ** (E_in_dB / 10.0)
        P_hat = 2.0 * task.J * E_in

        result = minimize_rs_potential(
            P_hat=P_hat,
            beta=beta,
            q_active=q_active,
        )

        eta_opt = result.eta
        gamma_effective = eta_opt * P_hat

        if gamma_effective >= task.gamma_target:
            E_required_dB = float(E_in_dB)
            eta_required = float(eta_opt)
            gamma_effective_required = float(gamma_effective)
            break

    return GridResult(
        s_idx=task.s_idx,
        j_idx=task.j_idx,
        S_in=task.S_in,
        J=task.J,
        Ka=task.Ka,
        beta=float(beta),
        q_active=float(q_active),
        gamma_target=task.gamma_target,
        E_required_dB=E_required_dB,
        eta_required=eta_required,
        gamma_effective_required=gamma_effective_required,
    )


def run_grid(tasks: list[GridTask], workers: int) -> list[GridResult]:
    if workers == 1:
        return [solve_grid_point(task) for task in tasks]

    results: list[GridResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(solve_grid_point, task): task
            for task in tasks
        }
        completed = 0
        total = len(tasks)
        for future in concurrent.futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            print(
                f"[{completed:4d}/{total}] "
                f"S_in={result.S_in:.3f}, "
                f"J={result.J:2d}, "
                f"Ka={result.Ka:10d}, "
                f"beta={result.beta:.6e}, "
                f"E_required={result.E_required_dB:.3f} dB, "
                f"eta={result.eta_required:.6e}",
                flush=True,
            )
    return results


def assemble_arrays(
    results: list[GridResult],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E_required_dB = np.full(shape, np.nan)
    eta_required = np.full(shape, np.nan)
    gamma_effective_required = np.full(shape, np.nan)
    beta_mat = np.full(shape, np.nan)
    q_active_mat = np.full(shape, np.nan)

    for result in results:
        E_required_dB[result.j_idx, result.s_idx] = result.E_required_dB
        eta_required[result.j_idx, result.s_idx] = result.eta_required
        gamma_effective_required[result.j_idx, result.s_idx] = result.gamma_effective_required
        beta_mat[result.j_idx, result.s_idx] = result.beta
        q_active_mat[result.j_idx, result.s_idx] = result.q_active

    return (
        E_required_dB,
        eta_required,
        gamma_effective_required,
        beta_mat,
        q_active_mat,
    )


def asymptotic_energy(S_in_vec: np.ndarray, alpha: float) -> np.ndarray:
    return (2.0 ** (2.0 * S_in_vec * (1.0 - 1.0 / alpha)) - 1.0) / (2.0 * S_in_vec)


def save_csv_files(
    data_dir: Path,
    J_vec: np.ndarray,
    Ka_vec: np.ndarray,
    S_in_vec: np.ndarray,
    gamma_target_vec: np.ndarray,
    E_required_dB: np.ndarray,
    eta_required: np.ndarray,
    gamma_effective_required: np.ndarray,
    beta_mat: np.ndarray,
    q_active_mat: np.ndarray,
    E_asymptotic_dB: np.ndarray,
) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for j_idx, J in enumerate(J_vec):
        data = np.column_stack(
            (
                S_in_vec,
                E_required_dB[j_idx, :],
                eta_required[j_idx, :],
                gamma_effective_required[j_idx, :],
                beta_mat[j_idx, :],
                q_active_mat[j_idx, :],
                np.full_like(S_in_vec, gamma_target_vec[j_idx], dtype=float),
                np.full_like(S_in_vec, Ka_vec[j_idx], dtype=float),
            )
        )
        path = data_dir / f"fig5_J{int(J)}.csv"
        np.savetxt(
            path,
            data,
            delimiter=",",
            header=(
                "S_in,E_required_dB,eta_required,gamma_effective_required,"
                "beta,q_active,gamma_target,Ka"
            ),
            comments="",
            fmt="%.18e",
        )
        saved_paths.append(path)

    asymptotic_path = data_dir / "fig5_asymptotic.csv"
    np.savetxt(
        asymptotic_path,
        np.column_stack((S_in_vec, E_asymptotic_dB)),
        delimiter=",",
        header="S_in,E_asymptotic_dB",
        comments="",
        fmt="%.18e",
    )
    saved_paths.append(asymptotic_path)

    return saved_paths


def configure_matplotlib(show: bool) -> None:
    plot_cache_dir = Path(tempfile.gettempdir()) / "fig5_plot_cache"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    (plot_cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir))

    if not show:
        import matplotlib

        matplotlib.use("Agg")


def plot_figure(
    J_vec: np.ndarray,
    S_in_vec: np.ndarray,
    E_required_dB: np.ndarray,
    E_asymptotic_dB: np.ndarray,
    png_output: Path,
    pdf_output: Path | None,
    show: bool,
) -> None:
    configure_matplotlib(show)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for j_idx, J in enumerate(J_vec):
        ax.plot(
            E_required_dB[j_idx, :],
            S_in_vec,
            linewidth=1.5,
            label=fr"$J={int(J)}$",
        )

    ax.plot(
        E_asymptotic_dB,
        S_in_vec,
        "k--",
        linewidth=1.8,
        label="Asymptotic limit",
    )

    ax.set_xlabel(r"$\mathcal{E}_{\mathrm{in}}\,[\mathrm{dB}]$")
    ax.set_ylabel(r"$S_{\mathrm{in}}$")
    ax.set_xlim(-5.0, 2.0)
    ax.set_ylim(0.0, 2.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=300)
    if pdf_output is not None:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_workers = min(os.cpu_count() or 1, len(J_VEC) * len(S_IN_VEC))
    parser = argparse.ArgumentParser(
        description="Parallel reproduction of Figure 5 and CSV data export."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of worker processes. Default: {default_workers}.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for CSV outputs. Default: data/fig5_data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PNG_OUTPUT,
        help="Output PNG path. Default: plots/fig5.png.",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=DEFAULT_PDF_OUTPUT,
        help="Output PDF path. Default: plots/fig5.pdf.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip saving the PDF.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    J_vec = J_VEC.copy()
    S_in_vec = S_IN_VEC.copy()
    Ka_vec = compute_ka_vec(J_vec, ALPHA)
    gamma_target_vec = compute_gamma_targets(J_vec, Ka_vec)
    tasks = make_tasks(J_vec, Ka_vec, S_in_vec, gamma_target_vec)

    print(
        f"Evaluating {len(tasks)} grid points with {args.workers} worker(s).",
        flush=True,
    )
    results = run_grid(tasks, workers=args.workers)

    shape = (len(J_vec), len(S_in_vec))
    (
        E_required_dB,
        eta_required,
        gamma_effective_required,
        beta_mat,
        q_active_mat,
    ) = assemble_arrays(results, shape)

    E_asymptotic = asymptotic_energy(S_in_vec, ALPHA)
    E_asymptotic_dB = 10.0 * np.log10(E_asymptotic)

    csv_paths = save_csv_files(
        data_dir=args.data_dir,
        J_vec=J_vec,
        Ka_vec=Ka_vec,
        S_in_vec=S_in_vec,
        gamma_target_vec=gamma_target_vec,
        E_required_dB=E_required_dB,
        eta_required=eta_required,
        gamma_effective_required=gamma_effective_required,
        beta_mat=beta_mat,
        q_active_mat=q_active_mat,
        E_asymptotic_dB=E_asymptotic_dB,
    )

    pdf_output = None if args.no_pdf else args.pdf_output
    plot_figure(
        J_vec=J_vec,
        S_in_vec=S_in_vec,
        E_required_dB=E_required_dB,
        E_asymptotic_dB=E_asymptotic_dB,
        png_output=args.output,
        pdf_output=pdf_output,
        show=args.show,
    )

    for path in csv_paths:
        print(f"Saved CSV: {path}")
    print(f"Saved PNG: {args.output}")
    if pdf_output is not None:
        print(f"Saved PDF: {pdf_output}")


if __name__ == "__main__":
    main()
