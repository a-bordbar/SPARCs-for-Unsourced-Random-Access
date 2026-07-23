#!/usr/bin/env python3
"""Reproduce Figure 7 of "SPARCs for Unsourced Random Access".

Figure 7 fixes alpha = 2 and S_in = 2, varies J, and plots the
required inner energy for:

1. optimal decoding: global minimum of the RS potential;
2. AMP decoding: smallest stationary point of the RS potential.

This script uses the same `rs_potential.py` helper used by the Figure 5
and Figure 6 theoretical scripts. Put this file in the same directory as
`rs_potential.py`.
"""

from __future__ import annotations

import os

# Avoid nested BLAS/OpenMP parallelism when using ProcessPoolExecutor.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def update(self, n=1):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False


try:
    from rs_potential import minimize_rs_potential
except ImportError as exc:
    raise SystemExit(
        "Could not import rs_potential.py. Put fig7.py in the same directory "
        "as rs_potential.py, then run it from that directory."
    ) from exc


ALPHA_DEFAULT = 2.0
S_IN_DEFAULT = 2.0
L_DEFAULT = 8
J_MIN_DEFAULT = 5
J_MAX_DEFAULT = 60
J_STEP_DEFAULT = 1

ENERGY_MIN_DB_DEFAULT = -2.0
ENERGY_MAX_DB_DEFAULT = 6.0
ENERGY_STEP_DB_DEFAULT = 0.10
REFINE_TOLERANCE_DB_DEFAULT = 0.002

PMD_SCALE = 0.05
PFA_SCALE = 0.01

DEFAULT_DATA_DIR = Path("data/fig7_data")
DEFAULT_CSV_OUTPUT = DEFAULT_DATA_DIR / "fig7_results.csv"
DEFAULT_PNG_OUTPUT = Path("plots/fig7.png")
DEFAULT_PDF_OUTPUT = Path("plots/fig7.pdf")


@dataclass(frozen=True)
class Fig7Config:
    alpha: float
    S_in: float
    L: int
    energy_min_db: float
    energy_max_db: float
    energy_step_db: float
    refine_tolerance_db: float


@dataclass(frozen=True)
class JTask:
    J: int
    config: Fig7Config


@dataclass(frozen=True)
class BranchResult:
    energy_db: float
    eta: float
    gamma_effective: float
    status: str


@dataclass(frozen=True)
class JResult:
    J: int
    Ka: int
    R_in: float
    beta: float
    q_active: float
    pmd_target: float
    pfa_target: float
    gamma_target: float
    optimal: BranchResult
    algorithmic: BranchResult


def compute_ka(J: int, alpha: float) -> int:
    """Use the same finite-J convention as the Figure 5/6 scripts."""
    return int(np.rint(2.0 ** (J / alpha)))


def compute_q_active(J: int, Ka: int) -> float:
    """q = 1 - (1 - 2^{-J})^{Ka}, evaluated stably."""
    x = 2.0 ** (-J)
    return float(-np.expm1(Ka * np.log1p(-x)))


def compute_targets(J: int, Ka: int, L: int) -> tuple[float, float, float]:
    pmd_target = PMD_SCALE / L
    pfa_target = PFA_SCALE * Ka / (2.0 ** J)

    if not (0.0 < pmd_target < 1.0):
        raise ValueError(f"Invalid pmd target: {pmd_target}")
    if not (0.0 < pfa_target < 1.0):
        raise ValueError(f"Invalid pfa target for J={J}: {pfa_target}")

    gamma_target = (
        norm.isf(pmd_target) + norm.isf(pfa_target)
    ) ** 2

    return float(pmd_target), float(pfa_target), float(gamma_target)


def evaluate_branch(
    *,
    J: int,
    beta: float,
    q_active: float,
    energy_db: float,
    branch: str,
) -> tuple[float, float]:
    """Return (eta, eta * P_hat) for one energy and one potential selection."""
    E_in = 10.0 ** (energy_db / 10.0)
    P_hat = 2.0 * J * E_in

    if branch == "optimal":
        # The helper's default is the global minimum.
        result = minimize_rs_potential(
            P_hat=P_hat,
            beta=beta,
            q_active=q_active,
        )
    elif branch == "algorithmic":
        # AMP is described by the smallest stationary point.
        result = minimize_rs_potential(
            P_hat=P_hat,
            beta=beta,
            q_active=q_active,
            selection="first",
        )
    else:
        raise ValueError(f"Unknown branch: {branch}")

    eta = float(result.eta)
    gamma_effective = eta * P_hat

    if not np.isfinite(eta) or eta < 0.0:
        raise RuntimeError(
            f"Invalid eta={eta} for J={J}, E={energy_db:.6f} dB, "
            f"branch={branch}."
        )

    return eta, float(gamma_effective)


def refine_crossing(
    *,
    J: int,
    beta: float,
    q_active: float,
    gamma_target: float,
    branch: str,
    low_db: float,
    high_db: float,
    tolerance_db: float,
) -> BranchResult:
    """Refine the first fail/pass bracket by bisection."""
    eta_high, gamma_high = evaluate_branch(
        J=J,
        beta=beta,
        q_active=q_active,
        energy_db=high_db,
        branch=branch,
    )

    if gamma_high < gamma_target:
        return BranchResult(
            energy_db=float("nan"),
            eta=float("nan"),
            gamma_effective=float("nan"),
            status="invalid_bracket",
        )

    while high_db - low_db > tolerance_db:
        mid_db = 0.5 * (low_db + high_db)
        eta_mid, gamma_mid = evaluate_branch(
            J=J,
            beta=beta,
            q_active=q_active,
            energy_db=mid_db,
            branch=branch,
        )

        if gamma_mid >= gamma_target:
            high_db = mid_db
            eta_high = eta_mid
            gamma_high = gamma_mid
        else:
            low_db = mid_db

    return BranchResult(
        energy_db=float(high_db),
        eta=float(eta_high),
        gamma_effective=float(gamma_high),
        status="ok",
    )


def solve_j(task: JTask) -> JResult:
    J = task.J
    cfg = task.config

    Ka = compute_ka(J, cfg.alpha)
    R_in = cfg.S_in / Ka
    beta = (2.0 ** J) * R_in / J
    q_active = compute_q_active(J, Ka)
    pmd_target, pfa_target, gamma_target = compute_targets(J, Ka, cfg.L)

    energy_grid = np.arange(
        cfg.energy_min_db,
        cfg.energy_max_db + 0.5 * cfg.energy_step_db,
        cfg.energy_step_db,
        dtype=float,
    )

    brackets: dict[str, tuple[float, float] | None] = {
        "optimal": None,
        "algorithmic": None,
    }

    previous_energy = float(energy_grid[0])

    for index, energy_db in enumerate(energy_grid):
        for branch in ("optimal", "algorithmic"):
            if brackets[branch] is not None:
                continue

            _, gamma_effective = evaluate_branch(
                J=J,
                beta=beta,
                q_active=q_active,
                energy_db=float(energy_db),
                branch=branch,
            )

            if gamma_effective >= gamma_target:
                if index == 0:
                    low_db = float(energy_grid[0] - cfg.energy_step_db)
                else:
                    low_db = previous_energy

                brackets[branch] = (low_db, float(energy_db))

        if all(value is not None for value in brackets.values()):
            break

        previous_energy = float(energy_db)

    branch_results: dict[str, BranchResult] = {}

    for branch in ("optimal", "algorithmic"):
        bracket = brackets[branch]

        if bracket is None:
            branch_results[branch] = BranchResult(
                energy_db=float("nan"),
                eta=float("nan"),
                gamma_effective=float("nan"),
                status="ceiling_reached",
            )
            continue

        branch_results[branch] = refine_crossing(
            J=J,
            beta=beta,
            q_active=q_active,
            gamma_target=gamma_target,
            branch=branch,
            low_db=bracket[0],
            high_db=bracket[1],
            tolerance_db=cfg.refine_tolerance_db,
        )

    return JResult(
        J=J,
        Ka=Ka,
        R_in=float(R_in),
        beta=float(beta),
        q_active=float(q_active),
        pmd_target=pmd_target,
        pfa_target=pfa_target,
        gamma_target=gamma_target,
        optimal=branch_results["optimal"],
        algorithmic=branch_results["algorithmic"],
    )


def run_tasks(tasks: list[JTask], workers: int) -> list[JResult]:
    if workers == 1:
        results = []
        for task in tqdm(tasks, desc="J values", unit="J"):
            result = solve_j(task)
            print_result(result)
            results.append(result)
        return results

    results: list[JResult] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(tasks))
    ) as executor:
        futures = {
            executor.submit(solve_j, task): task.J
            for task in tasks
        }

        with tqdm(total=len(futures), desc="J values", unit="J") as progress:
            for future in concurrent.futures.as_completed(futures):
                J = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Figure 7 worker failed for J={J}") from exc

                results.append(result)
                print_result(result)
                progress.update(1)

    return sorted(results, key=lambda item: item.J)


def format_energy(branch: BranchResult) -> str:
    if branch.status != "ok" or not np.isfinite(branch.energy_db):
        return branch.status
    return f"{branch.energy_db:.4f} dB"


def print_result(result: JResult) -> None:
    print(
        f"J={result.J:2d} "
        f"Ka={result.Ka:10d} "
        f"optimal={format_energy(result.optimal):>14s} "
        f"algorithmic={format_energy(result.algorithmic):>14s}",
        flush=True,
    )


def save_csv(results: list[JResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = np.array(
        [
            [
                result.J,
                result.Ka,
                result.R_in,
                result.beta,
                result.q_active,
                result.pmd_target,
                result.pfa_target,
                result.gamma_target,
                result.optimal.energy_db,
                result.optimal.eta,
                result.optimal.gamma_effective,
                result.algorithmic.energy_db,
                result.algorithmic.eta,
                result.algorithmic.gamma_effective,
            ]
            for result in results
        ],
        dtype=float,
    )

    header = (
        "J,Ka,R_in,beta,q_active,pmd_target,pfa_target,gamma_target,"
        "E_optimal_dB,eta_optimal,gamma_optimal,"
        "E_algorithmic_dB,eta_algorithmic,gamma_algorithmic"
    )

    np.savetxt(
        path,
        rows,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.18e",
    )


def configure_matplotlib(show: bool) -> None:
    cache_dir = Path(tempfile.gettempdir()) / "fig7_plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    if not show:
        import matplotlib
        matplotlib.use("Agg")


def plot_results(
    *,
    results: list[JResult],
    config: Fig7Config,
    png_output: Path,
    pdf_output: Path | None,
    show: bool,
) -> None:
    configure_matplotlib(show)
    import matplotlib.pyplot as plt

    J_values = np.array([result.J for result in results], dtype=float)
    optimal_db = np.array(
        [result.optimal.energy_db for result in results],
        dtype=float,
    )
    algorithmic_db = np.array(
        [result.algorithmic.energy_db for result in results],
        dtype=float,
    )

    E_optimal_asymptotic = (
        2.0 ** (
            2.0
            * config.S_in
            * (1.0 - 1.0 / config.alpha)
        )
        - 1.0
    ) / (2.0 * config.S_in)

    E_optimal_asymptotic_db = 10.0 * np.log10(E_optimal_asymptotic)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.plot(
        J_values,
        algorithmic_db,
        "--",
        linewidth=1.8,
        label="algorithmic",
    )
    ax.plot(
        J_values,
        optimal_db,
        "-",
        linewidth=1.8,
        label="optimal",
    )
    ax.axhline(
        E_optimal_asymptotic_db,
        linestyle="-.",
        linewidth=1.4,
        label="optimal asymptotic",
    )

    finite_algorithmic = np.isfinite(algorithmic_db)
    if np.any(finite_algorithmic):
        finite_indices = np.flatnonzero(finite_algorithmic)
        local_index = int(np.nanargmin(algorithmic_db[finite_algorithmic]))
        minimum_index = int(finite_indices[local_index])
        J_star = int(J_values[minimum_index])

        ax.plot(
            [J_star],
            [algorithmic_db[minimum_index]],
            marker="o",
            linestyle="None",
        )
        ax.annotate(
            rf"$J^\star\approx {J_star}$",
            xy=(J_star, algorithmic_db[minimum_index]),
            xytext=(J_star + 4, algorithmic_db[minimum_index] + 0.45),
            arrowprops={"arrowstyle": "->"},
        )

    ax.set_xlabel(r"$J$")
    ax.set_ylabel(r"$\mathcal{E}_{\mathrm{in}}/N_0\,[\mathrm{dB}]$")
    ax.set_xlim(0.0, max(60.0, float(np.max(J_values))))
    ax.set_ylim(-2.0, 6.0)
    ax.grid(True, alpha=0.28, linewidth=0.7)
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


def validate_configuration(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.j_min < 1:
        raise SystemExit("--j-min must be positive.")
    if args.j_max < args.j_min:
        raise SystemExit("--j-max must be at least --j-min.")
    if args.j_step < 1:
        raise SystemExit("--j-step must be positive.")
    if args.alpha <= 1.0:
        raise SystemExit("--alpha must exceed 1.")
    if args.s_in <= 0.0:
        raise SystemExit("--s-in must be positive.")
    if args.L < 1:
        raise SystemExit("--L must be positive.")
    if args.energy_max_db <= args.energy_min_db:
        raise SystemExit("--energy-max-db must exceed --energy-min-db.")
    if args.energy_step_db <= 0.0:
        raise SystemExit("--energy-step-db must be positive.")
    if args.refine_tolerance_db <= 0.0:
        raise SystemExit("--refine-tolerance-db must be positive.")


def parse_args() -> argparse.Namespace:
    default_workers = max(1, min(8, os.cpu_count() or 1))

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figure 7: required inner energy versus J for "
            "optimal and AMP decoding."
        )
    )
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--s-in", type=float, default=S_IN_DEFAULT)
    parser.add_argument("--L", type=int, default=L_DEFAULT)
    parser.add_argument("--j-min", type=int, default=J_MIN_DEFAULT)
    parser.add_argument("--j-max", type=int, default=J_MAX_DEFAULT)
    parser.add_argument("--j-step", type=int, default=J_STEP_DEFAULT)
    parser.add_argument(
        "--energy-min-db",
        type=float,
        default=ENERGY_MIN_DB_DEFAULT,
    )
    parser.add_argument(
        "--energy-max-db",
        type=float,
        default=ENERGY_MAX_DB_DEFAULT,
    )
    parser.add_argument(
        "--energy-step-db",
        type=float,
        default=ENERGY_STEP_DB_DEFAULT,
    )
    parser.add_argument(
        "--refine-tolerance-db",
        type=float,
        default=REFINE_TOLERANCE_DB_DEFAULT,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PNG_OUTPUT,
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=DEFAULT_PDF_OUTPUT,
    )
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_configuration(args)

    config = Fig7Config(
        alpha=float(args.alpha),
        S_in=float(args.s_in),
        L=int(args.L),
        energy_min_db=float(args.energy_min_db),
        energy_max_db=float(args.energy_max_db),
        energy_step_db=float(args.energy_step_db),
        refine_tolerance_db=float(args.refine_tolerance_db),
    )

    J_values = list(
        range(
            int(args.j_min),
            int(args.j_max) + 1,
            int(args.j_step),
        )
    )

    tasks = [JTask(J=J, config=config) for J in J_values]

    print("Figure 7 theoretical reproduction")
    print("---------------------------------")
    print(f"alpha={config.alpha}")
    print(f"S_in={config.S_in}")
    print(f"L={config.L}")
    print(f"J={J_values[0]}:{args.j_step}:{J_values[-1]}")
    print(f"workers={args.workers}")
    print(
        "Optimal curve: global minimum of the RS potential.\n"
        "Algorithmic curve: smallest stationary point of the RS potential."
    )

    results = run_tasks(tasks, workers=int(args.workers))

    csv_output = args.csv_output
    if csv_output == DEFAULT_CSV_OUTPUT and args.data_dir != DEFAULT_DATA_DIR:
        csv_output = args.data_dir / "fig7_results.csv"

    save_csv(results, csv_output)

    pdf_output = None if args.no_pdf else args.pdf_output
    plot_results(
        results=results,
        config=config,
        png_output=args.output,
        pdf_output=pdf_output,
        show=bool(args.show),
    )

    finite_algorithmic = [
        result
        for result in results
        if result.algorithmic.status == "ok"
        and np.isfinite(result.algorithmic.energy_db)
    ]
    if finite_algorithmic:
        best = min(
            finite_algorithmic,
            key=lambda item: item.algorithmic.energy_db,
        )
        print(
            f"Algorithmic minimum: J*={best.J}, "
            f"E_in={best.algorithmic.energy_db:.4f} dB"
        )

    print(f"Saved CSV: {csv_output}")
    print(f"Saved PNG: {args.output}")
    if pdf_output is not None:
        print(f"Saved PDF: {pdf_output}")


if __name__ == "__main__":
    main()
