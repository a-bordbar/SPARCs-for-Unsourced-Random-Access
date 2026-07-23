#!/usr/bin/env python3
"""
At each energy, the threshold is selected from the same Monte Carlo AMP score
population used to measure pmd and pfa. The selected threshold is the lowest
tie-safe threshold satisfying the empirical false-alarm constraint, which
minimizes empirical missed detections subject to that constraint.
"""

from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import argparse
import concurrent.futures
import csv
import hashlib
import math
import multiprocessing as mp
import queue as queue_module
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import fig6_AMP_optimized as opt


THRESHOLD_METHOD = "same_sample_empirical_roc_order_statistic"
SCHEMA_VERSION = 3

PROGRESS_COLUMNS = [
    "J",
    "Ka",
    "S_in",
    "n",
    "M",
    "operator",
    "denoiser",
    "trials",
    "E_required_dB",
    "threshold_z",
    "pmd",
    "pfa",
    "pmd_standard_error",
    "pfa_standard_error",
    "true_active",
    "true_inactive",
    "missed_detections",
    "false_alarms",
    "max_missed_detections",
    "max_false_alarms",
    "mean_iterations",
    "converged_fraction",
    "elapsed_seconds",
    "seed",
    "status",
    "config_key",
    "threshold_method",
    "schema_version",
]


@dataclass(frozen=True)
class RocConfig:
    base: opt.AmpConfig
    threshold_method: str = THRESHOLD_METHOD
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class RocEvaluationResult:
    E_in_dB: float
    threshold_z: float
    pmd: float
    pfa: float
    pmd_standard_error: float
    pfa_standard_error: float
    true_active: int
    true_inactive: int
    missed_detections: int
    false_alarms: int
    max_missed_detections: int
    max_false_alarms: int
    mean_iterations: float
    converged_fraction: float
    elapsed_seconds: float
    passed: bool
    failed: bool
    error_message: str
    peak_inactive_tail_size: int


@dataclass(frozen=True)
class RocSearchResult:
    J: int
    Ka: int
    S_in: float
    n: int
    M: int
    operator: str
    denoiser: str
    trials: int
    E_required_dB: float
    threshold_z: float
    pmd: float
    pfa: float
    pmd_standard_error: float
    pfa_standard_error: float
    true_active: int
    true_inactive: int
    missed_detections: int
    false_alarms: int
    max_missed_detections: int
    max_false_alarms: int
    mean_iterations: float
    converged_fraction: float
    elapsed_seconds: float
    seed: int
    status: str
    config_key: str
    threshold_method: str
    schema_version: int


def progress_log(cfg: RocConfig, message: str) -> None:
    opt.progress_log(cfg.base, message)


def parse_j_values(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one J value is required.")
    if any(j <= 0 for j in values):
        raise argparse.ArgumentTypeError("J values must be positive integers.")
    return values


def strict_max_errors(target: float, opportunities: int) -> int:
    if opportunities < 0:
        raise ValueError("opportunities must be nonnegative.")
    if opportunities == 0:
        return -1
    return max(-1, min(opportunities, int(math.ceil(float(target) * opportunities) - 1)))


def update_top_tail(tail: np.ndarray, values: np.ndarray, k_keep: int) -> np.ndarray:
    if k_keep <= 0 or values.size == 0:
        return tail
    vals = values.astype(np.float64, copy=False)
    combined = vals if tail.size == 0 else np.concatenate((tail, vals))
    if combined.size > k_keep:
        idx = combined.size - k_keep
        combined = np.partition(combined, idx)[idx:]
    return combined


def empirical_roc_from_outputs(
    outputs: list[opt.AmpOutput],
    op: opt.OperatingPoint,
    chunk_size: int,
) -> tuple[float, int, int, int, int, int, int, float, float, int]:
    valid_outputs = [out for out in outputs if not out.failed]
    n_active = int(sum(np.count_nonzero(out.support_true) for out in valid_outputs))
    n_inactive = int(sum(out.support_true.size - np.count_nonzero(out.support_true) for out in valid_outputs))
    max_missed = strict_max_errors(op.pmd_target, n_active)
    max_false = strict_max_errors(op.pfa_target, n_inactive)
    k_keep = max_false + 1

    active_chunks: list[np.ndarray] = []
    inactive_tail = np.empty(0, dtype=np.float64)
    peak_tail = 0
    for out in valid_outputs:
        scale = math.sqrt(out.tau_sq_final)
        for start in range(0, out.pseudo_final.size, chunk_size):
            stop = min(start + chunk_size, out.pseudo_final.size)
            truth = out.support_true[start:stop]
            scores = out.pseudo_final[start:stop].astype(np.float64) / scale
            if np.any(truth):
                active_chunks.append(scores[truth].copy())
            if np.any(~truth):
                inactive_tail = update_top_tail(inactive_tail, scores[~truth], k_keep)
                peak_tail = max(peak_tail, int(inactive_tail.size))

    if n_inactive == 0:
        threshold_z = -math.inf
    elif max_false < 0:
        threshold_z = math.inf
    elif max_false >= n_inactive:
        threshold_z = -math.inf
    else:
        if inactive_tail.size < k_keep:
            raise AssertionError("inactive top-tail buffer is smaller than required.")
        tail_desc = np.sort(inactive_tail)[::-1]
        boundary = float(tail_desc[max_false])
        threshold_z = float(np.nextafter(boundary, np.inf))

    false_alarms = int(np.count_nonzero(inactive_tail >= threshold_z)) if np.isfinite(threshold_z) else (0 if threshold_z > 0 else n_inactive)
    if active_chunks:
        active_scores = np.concatenate(active_chunks)
        missed = int(np.count_nonzero(active_scores < threshold_z))
    else:
        missed = 0

    pmd = missed / n_active if n_active else float("nan")
    pfa = false_alarms / n_inactive if n_inactive else float("nan")
    pmd_se = math.sqrt(pmd * (1.0 - pmd) / n_active) if n_active and np.isfinite(pmd) else float("nan")
    pfa_se = math.sqrt(pfa * (1.0 - pfa) / n_inactive) if n_inactive and np.isfinite(pfa) else float("nan")

    if n_active:
        assert (missed <= max_missed) == (pmd < op.pmd_target)
    if n_inactive:
        assert (false_alarms <= max_false) == (pfa < op.pfa_target)
    return threshold_z, n_active, n_inactive, missed, false_alarms, max_missed, max_false, pmd_se, pfa_se, peak_tail


def make_trial_contexts(cfg: RocConfig, op: opt.OperatingPoint) -> list[opt.TrialContext]:
    return opt.make_trial_contexts(cfg.base, op)


def evaluate_energy(
    cfg: RocConfig,
    op: opt.OperatingPoint,
    contexts: list[opt.TrialContext],
    E_in_dB: float,
    progress_queue: Any | None = None,
    phase: str = "coarse",
) -> RocEvaluationResult:
    start = time.time()
    outputs: list[opt.AmpOutput] = []
    failed = False
    messages: list[str] = []
    for ctx in contexts:
        out = opt.run_amp_to_pseudo(cfg.base, op, ctx, E_in_dB)
        outputs.append(out)
        if out.failed:
            failed = True
            messages.append(f"trial {ctx.trial_idx}: {out.error_message}")

    if outputs and not failed:
        (
            threshold_z,
            n_active,
            n_inactive,
            missed,
            false_alarms,
            max_missed,
            max_false,
            pmd_se,
            pfa_se,
            peak_tail,
        ) = empirical_roc_from_outputs(outputs, op, cfg.base.denoiser_chunk_size)
    else:
        threshold_z = float("nan")
        n_active = n_inactive = missed = false_alarms = max_missed = max_false = peak_tail = 0
        pmd_se = pfa_se = float("nan")

    pmd = missed / n_active if n_active else float("nan")
    pfa = false_alarms / n_inactive if n_inactive else float("nan")
    mean_iter = float(np.mean([out.iterations for out in outputs])) if outputs else float("nan")
    conv_frac = float(np.mean([out.converged for out in outputs])) if outputs else float("nan")
    passed = (not failed) and false_alarms <= max_false and missed <= max_missed
    elapsed = time.time() - start

    result = RocEvaluationResult(
        float(E_in_dB),
        threshold_z,
        pmd,
        pfa,
        pmd_se,
        pfa_se,
        n_active,
        n_inactive,
        missed,
        false_alarms,
        max_missed,
        max_false,
        mean_iter,
        conv_frac,
        elapsed,
        passed,
        failed,
        "; ".join(messages),
        peak_tail,
    )
    log_energy_result(cfg, op, result, progress_queue, phase)
    return result


def log_energy_result(
    cfg: RocConfig,
    op: opt.OperatingPoint,
    ev: RocEvaluationResult,
    progress_queue: Any | None,
    phase: str,
) -> None:
    message = (
        f"J={op.J} S_in={op.S_in:.3f} E={ev.E_in_dB:.3f} dB\n"
        f"threshold_z={ev.threshold_z:.6g}\n"
        f"missed={ev.missed_detections}/{ev.true_active} max={ev.max_missed_detections} "
        f"pmd={ev.pmd:.4e} target={op.pmd_target:.4e}\n"
        f"false={ev.false_alarms}/{ev.true_inactive} max={ev.max_false_alarms} "
        f"pfa={ev.pfa:.4e} target={op.pfa_target:.4e}\n"
        f"mean_iter={ev.mean_iterations:.2f} conv_frac={ev.converged_fraction:.2f} "
        f"{'PASS' if ev.passed else 'fail'}"
    )
    if progress_queue is not None:
        progress_queue.put(
            {
                "phase": phase,
                "J": op.J,
                "S_in": op.S_in,
                "E_in_dB": ev.E_in_dB,
                "pmd": ev.pmd,
                "pfa": ev.pfa,
                "passed": ev.passed,
                "elapsed": ev.elapsed_seconds,
                "message": message,
            }
        )
    else:
        progress_log(cfg, message)


def config_key(cfg: RocConfig, op: opt.OperatingPoint) -> str:
    fields = (
        cfg.base.alpha,
        cfg.base.L,
        op.J,
        op.S_in,
        op.Ka,
        op.n,
        cfg.base.trials,
        cfg.base.seed,
        cfg.base.operator,
        cfg.base.denoiser,
        cfg.base.energy_min_db,
        cfg.base.energy_max_db,
        cfg.base.energy_step_db,
        cfg.base.refine_tolerance_db,
        cfg.base.required_consecutive_passes,
        cfg.base.max_iterations,
        cfg.base.min_iterations,
        cfg.base.tolerance,
        cfg.base.damping,
        cfg.base.tau_floor,
        cfg.base.denoiser_chunk_size,
        cfg.threshold_method,
        "strict_integer_targets",
        cfg.schema_version,
    )
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]


def make_search_result(
    cfg: RocConfig,
    op: opt.OperatingPoint,
    E_req: float,
    ev: RocEvaluationResult,
    elapsed: float,
    status: str,
) -> RocSearchResult:
    return RocSearchResult(
        op.J,
        op.Ka,
        op.S_in,
        op.n,
        op.M,
        cfg.base.operator,
        cfg.base.denoiser,
        cfg.base.trials,
        E_req,
        ev.threshold_z,
        ev.pmd,
        ev.pfa,
        ev.pmd_standard_error,
        ev.pfa_standard_error,
        ev.true_active,
        ev.true_inactive,
        ev.missed_detections,
        ev.false_alarms,
        ev.max_missed_detections,
        ev.max_false_alarms,
        ev.mean_iterations,
        ev.converged_fraction,
        elapsed,
        cfg.base.seed,
        status,
        config_key(cfg, op),
        cfg.threshold_method,
        cfg.schema_version,
    )


def search_operating_point(args: tuple[Any, ...]) -> RocSearchResult:
    if len(args) == 2:
        cfg, op = args
        progress_queue = None
    else:
        cfg, op, progress_queue = args
    try:
        total_start = time.time()
        contexts = make_trial_contexts(cfg, op)
        energies = np.arange(
            cfg.base.energy_min_db,
            cfg.base.energy_max_db + 0.5 * cfg.base.energy_step_db,
            cfg.base.energy_step_db,
        )
        evals: dict[float, RocEvaluationResult] = {}
        consecutive = 0
        bracket_low: float | None = None
        bracket_high: float | None = None
        show_nested_progress = cfg.base.workers == 1
        energy_desc = f"J={op.J} S_in={op.S_in:.3f}"
        for e in opt.tqdm(energies, desc=energy_desc, unit="energy", leave=False, disable=not show_nested_progress):
            ev = evaluate_energy(cfg, op, contexts, float(e), progress_queue, "coarse")
            evals[float(e)] = ev
            if ev.passed:
                consecutive += 1
                if consecutive >= cfg.base.required_consecutive_passes:
                    hi_idx = int(np.where(np.isclose(energies, e))[0][0]) - cfg.base.required_consecutive_passes + 1
                    bracket_high = float(energies[hi_idx])
                    bracket_low = float(energies[hi_idx - 1]) if hi_idx > 0 else float(cfg.base.energy_min_db)
                    break
            else:
                consecutive = 0

        if bracket_high is None:
            last = evals[float(energies[-1])]
            if progress_queue is None:
                progress_log(cfg, f"WARNING: ceiling reached for J={op.J} S_in={op.S_in:.3f}; no finite plot marker will be shown.")
            return make_search_result(cfg, op, float("nan"), last, time.time() - total_start, "ceiling_reached")

        lo = bracket_low
        hi = bracket_high
        hi_eval = evals.get(hi) or evaluate_energy(cfg, op, contexts, hi, progress_queue, "refine")
        refine_total = max(1, int(math.ceil(math.log2(max((hi - lo) / cfg.base.refine_tolerance_db, 1.0)))))
        with opt.tqdm(
            total=refine_total,
            desc=f"refine J={op.J} S_in={op.S_in:.3f}",
            unit="step",
            leave=False,
            disable=not show_nested_progress,
        ) as refine_bar:
            while hi - lo > cfg.base.refine_tolerance_db:
                mid = 0.5 * (lo + hi)
                ev = evaluate_energy(cfg, op, contexts, mid, progress_queue, "refine")
                if ev.passed:
                    hi = mid
                    hi_eval = ev
                else:
                    lo = mid
                refine_bar.set_postfix(E=f"{mid:.3f}", pmd=f"{ev.pmd:.2e}", pfa=f"{ev.pfa:.2e}", pass_=ev.passed)
                refine_bar.update(1)

        return make_search_result(cfg, op, hi, hi_eval, time.time() - total_start, "ok")
    except Exception as exc:
        raise RuntimeError(f"{type(exc).__name__}: {exc}; operating point J={op.J} S_in={op.S_in}") from exc


def make_operating_points(cfg: RocConfig) -> list[opt.OperatingPoint]:
    return opt.make_operating_points(cfg.base)


def estimate_worker_memory_gb(cfg: RocConfig, op: opt.OperatingPoint) -> float:
    return opt.estimate_worker_memory_gb(cfg.base, op)


def parse_progress_value(key: str, value: str) -> Any:
    if key in {
        "J",
        "Ka",
        "n",
        "M",
        "trials",
        "true_active",
        "true_inactive",
        "missed_detections",
        "false_alarms",
        "max_missed_detections",
        "max_false_alarms",
        "seed",
        "schema_version",
    }:
        return int(value)
    if key in {
        "S_in",
        "E_required_dB",
        "threshold_z",
        "pmd",
        "pfa",
        "pmd_standard_error",
        "pfa_standard_error",
        "mean_iterations",
        "converged_fraction",
        "elapsed_seconds",
    }:
        return float(value)
    return value


def read_progress(path: Path) -> dict[str, RocSearchResult]:
    if not path.exists():
        return {}
    rows: dict[str, RocSearchResult] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(PROGRESS_COLUMNS):
            print(f"WARNING: incompatible progress schema in {path}; ignoring old rows.", flush=True)
            return {}
        for row in reader:
            try:
                if int(row.get("schema_version", "0")) != SCHEMA_VERSION:
                    print(f"WARNING: incompatible schema_version in {path}; skipping row.", flush=True)
                    continue
                result = RocSearchResult(**{k: parse_progress_value(k, row[k]) for k in PROGRESS_COLUMNS})
                rows[result.config_key] = result
            except Exception as exc:
                print(f"WARNING: skipping malformed progress row in {path}: {type(exc).__name__}: {exc}", flush=True)
    return rows


def atomic_write_results(path: Path, results: list[RocSearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROGRESS_COLUMNS)
            writer.writeheader()
            for result in sorted(results, key=lambda x: (x.J, x.S_in)):
                writer.writerow({key: getattr(result, key) for key in PROGRESS_COLUMNS})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def drain_progress_queue(progress_queue: Any, progress_bar: Any) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue_module.Empty:
            break
        if event.get("phase") == "refine" and progress_bar.total is not None and progress_bar.n >= progress_bar.total:
            progress_bar.total += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            J=event.get("J"),
            S_in=f"{event.get('S_in'):.3f}",
            E=f"{event.get('E_in_dB'):.3f}",
            status="PASS" if event.get("passed") else "fail",
        )


def startup_summary(cfg: RocConfig, points: list[opt.OperatingPoint]) -> None:
    print(f"Empirical threshold method: {cfg.threshold_method}")
    print(f"Trials: {cfg.base.trials}")
    print("Targets: pmd < 0.05/L, pfa < 0.01*Ka/2**J")
    print(f"Operator: {cfg.base.operator}")
    print(f"J values: {','.join(str(j) for j in cfg.base.J_values)}")
    print(
        "At each energy, the threshold is selected from the same Monte Carlo AMP\n"
        "score population used to measure pmd and pfa. The selected threshold is the\n"
        "lowest tie-safe threshold satisfying the empirical false-alarm constraint,\n"
        "which minimizes empirical missed detections subject to that constraint."
    )
    if cfg.base.operator == "hadamard":
        print(
            "WARNING: structured Hadamard backend is a computationally tractable "
            "approximation, not the exact i.i.d.-Gaussian ensemble assumed by theory."
        )
    max_mem = max(estimate_worker_memory_gb(cfg, point) for point in points)
    for point in points:
        print(
            f"Operating point: J={point.J} Ka={point.Ka} S_in={point.S_in:.3f} "
            f"n={point.n} M={point.M} pmd_target={point.pmd_target:.4e} "
            f"pfa_target={point.pfa_target:.4e}"
        )
    print(f"Estimated per-worker memory: {max_mem:.2f} GB")
    if cfg.base.workers * max_mem > cfg.base.max_total_memory_gb:
        print(
            f"WARNING: workers * estimated memory = {cfg.base.workers * max_mem:.2f} GB "
            f"exceeds {cfg.base.max_total_memory_gb:.2f} GB"
        )


def run_search(cfg: RocConfig) -> list[RocSearchResult]:
    points = make_operating_points(cfg)
    startup_summary(cfg, points)
    progress_path = cfg.base.data_dir / "fig6_amp_progress.csv"
    existing = read_progress(progress_path) if cfg.base.resume else {}
    results = list(existing.values())
    pending = [point for point in points if config_key(cfg, point) not in existing]
    if pending:
        worker_count = max(1, min(cfg.base.workers, len(pending)))
        if worker_count == 1:
            for point in opt.tqdm(pending, desc="operating points", unit="point"):
                result = search_operating_point((cfg, point))
                results = [old for old in results if old.config_key != result.config_key] + [result]
                atomic_write_results(progress_path, results)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {executor.submit(search_operating_point, (cfg, point)): point for point in pending}
                with opt.tqdm(total=len(futures), desc="operating points", unit="point") as point_bar:
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        results = [old for old in results if old.config_key != result.config_key] + [result]
                        atomic_write_results(progress_path, results)
                        point_bar.set_postfix(J=result.J, S_in=f"{result.S_in:.3f}", status=result.status)
                        point_bar.update(1)
    atomic_write_results(cfg.base.data_dir / "fig6_amp_results.csv", results)
    atomic_write_results(progress_path, results)
    return sorted(results, key=lambda x: (x.J, x.S_in))


def maybe_load_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_results(cfg: RocConfig, results: list[RocSearchResult]) -> None:
    import matplotlib

    if not cfg.base.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {15: ("o", "Empirical ROC J=15"), 20: ("s", "Empirical ROC J=20")}

    for J in cfg.base.J_values:
        theory = maybe_load_csv(Path("data/fig6_data") / f"fig6_J{J}.csv")
        if theory is not None and theory.dtype.names and {"S_in", "E_required_dB"}.issubset(theory.dtype.names):
            ax.plot(theory["E_required_dB"], theory["S_in"], "-", label=f"Theory J={J}")

    asym = maybe_load_csv(Path("data/fig6_data") / "fig6_asymptotic.csv")
    if asym is not None and asym.dtype.names:
        if {"S_in", "E_optimal_asymptotic_dB"}.issubset(asym.dtype.names):
            ax.plot(asym["E_optimal_asymptotic_dB"], asym["S_in"], "k-", label="Optimal asymptotic")
        if {"S_in", "E_amp_asymptotic_dB"}.issubset(asym.dtype.names):
            ax.plot(asym["E_amp_asymptotic_dB"], asym["S_in"], "k--", label="AMP asymptotic")

    for J in cfg.base.J_values:
        ok_rows = [row for row in results if row.J == J and row.status == "ok" and np.isfinite(row.E_required_dB)]
        ceiling_rows = [row for row in results if row.J == J and row.status == "ceiling_reached"]
        marker, label = styles.get(J, ("o", f"Empirical ROC J={J}"))
        ax.plot(
            [row.E_required_dB for row in ok_rows],
            [row.S_in for row in ok_rows],
            linestyle="None",
            marker=marker,
            label=label,
        )
        if ceiling_rows:
            print(f"WARNING: {len(ceiling_rows)} J={J} ceiling-reached point(s) omitted from the plot.")

    ax.set_xlabel(r"$\mathcal{E}_{\mathrm{in}}\,[\mathrm{dB}]$")
    ax.set_ylabel(r"$S_{\mathrm{in}}$")
    ax.set_xlim(-5, 6)
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    cfg.base.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.base.output, dpi=300)
    if not cfg.base.no_pdf:
        cfg.base.pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.base.pdf_output)
    if cfg.base.show:
        plt.show()
    plt.close(fig)


def exhaustive_roc(
    active_scores: np.ndarray,
    inactive_scores: np.ndarray,
    pmd_target: float,
    pfa_target: float,
) -> tuple[bool, float, int, int, int, int]:
    n_active = int(active_scores.size)
    n_inactive = int(inactive_scores.size)
    max_missed = strict_max_errors(pmd_target, n_active)
    max_false = strict_max_errors(pfa_target, n_inactive)
    candidates = np.unique(np.concatenate((active_scores, inactive_scores)))
    thresholds = [float(np.nextafter(np.max(candidates), np.inf))]
    thresholds.extend(float(x) for x in candidates)
    thresholds.append(float(np.nextafter(np.min(candidates), -np.inf)))
    best: tuple[bool, float, int, int, int, int] | None = None
    for threshold in thresholds:
        false = int(np.count_nonzero(inactive_scores >= threshold))
        if false > max_false:
            continue
        missed = int(np.count_nonzero(active_scores < threshold))
        feasible = missed <= max_missed
        item = (feasible, threshold, missed, false, max_missed, max_false)
        if best is None or missed < best[2] or (missed == best[2] and threshold < best[1]):
            best = item
    if best is None:
        threshold = float(np.nextafter(np.max(inactive_scores), np.inf))
        false = int(np.count_nonzero(inactive_scores >= threshold))
        missed = int(np.count_nonzero(active_scores < threshold))
        best = (False, threshold, missed, false, max_missed, max_false)
    return best


def synthetic_outputs(active_scores: np.ndarray, inactive_scores: np.ndarray) -> list[opt.AmpOutput]:
    pseudo = np.concatenate((active_scores, inactive_scores)).astype(np.float32)
    support = np.concatenate((np.ones(active_scores.size, dtype=bool), np.zeros(inactive_scores.size, dtype=bool)))
    return [opt.AmpOutput(pseudo, 1.0, support, 1, True, False, None, (1.0,), (0.0,))]


def validate_only() -> None:
    for target, n in [(0.1, 10), (0.1, 11), (0.00625, 14436), (0.0000552, 2607004)]:
        max_errors = strict_max_errors(target, n)
        for err in {0, max_errors, max_errors + 1, max(0, max_errors - 1)}:
            if 0 <= err <= n:
                assert (err <= max_errors) == (err / n < target)

    rng = np.random.default_rng(123)
    active = rng.normal(2.0, 1.0, size=50)
    inactive = rng.normal(0.0, 1.0, size=500)
    pmd_target = 0.2
    pfa_target = 0.05
    outputs = synthetic_outputs(active, inactive)
    fake_op = opt.OperatingPoint(5, 6, 1.0, 0, 10, 32, active.size + inactive.size, pmd_target, pfa_target)
    roc = empirical_roc_from_outputs(outputs, fake_op, 64)
    exhaustive = exhaustive_roc(active, inactive, pmd_target, pfa_target)
    assert (roc[3] <= roc[5] and roc[4] <= roc[6]) == exhaustive[0]
    assert roc[3] == exhaustive[2]

    inactive_ties = np.array([4.0, 4.0, 3.0, 2.0, 1.0])
    active_ties = np.array([5.0, 3.5, 0.0])
    tie_outputs = synthetic_outputs(active_ties, inactive_ties)
    tie_op = opt.OperatingPoint(5, 6, 1.0, 0, 10, 32, 8, 1.0, 0.4)
    tie = empirical_roc_from_outputs(tie_outputs, tie_op, 4)
    assert tie[4] <= tie[6]

    tail = np.empty(0, dtype=np.float64)
    values = rng.normal(size=1000)
    for chunk in np.array_split(values, 17):
        tail = update_top_tail(tail, chunk, 23)
    assert np.allclose(np.sort(tail), np.sort(values)[-23:])

    active_outputs = synthetic_outputs(np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0, 5.0]))
    active_op = opt.OperatingPoint(5, 6, 1.0, 0, 10, 32, 6, 0.5, 0.34)
    active_roc = empirical_roc_from_outputs(active_outputs, active_op, 2)
    threshold = active_roc[0]
    assert active_roc[3] == int(np.count_nonzero(np.array([0.0, 1.0, 2.0]) < threshold))

    cfg = RocConfig(
        opt.AmpConfig(
            J_values=(8,),
            s_in_values=(0.5,),
            L=2,
            operator="dense-gaussian",
            trials=2,
            max_iterations=5,
            energy_min_db=2.0,
            energy_max_db=2.0,
            required_consecutive_passes=1,
            no_pdf=True,
            resume=False,
        )
    )
    op = make_operating_points(cfg)[0]
    contexts1 = make_trial_contexts(cfg, op)
    contexts2 = make_trial_contexts(cfg, op)
    ev1 = evaluate_energy(cfg, op, contexts1, 2.0)
    ev2 = evaluate_energy(cfg, op, contexts2, 2.0)
    assert ev1.threshold_z == ev2.threshold_z
    assert ev1.missed_detections == ev2.missed_detections
    assert ev1.false_alarms == ev2.false_alarms
    assert ev1.pmd == ev2.pmd
    assert ev1.pfa == ev2.pfa
    assert ev1.passed == ev2.passed

    ctx = contexts1[0]
    assert ctx.operator is ctx.operator
    assert np.array_equal(ctx.support_true, contexts2[0].support_true)
    assert np.array_equal(ctx.noise, contexts2[0].noise)
    assert np.array_equal(ctx.signal_base, contexts2[0].signal_base)
    e1 = 0.0
    e2 = 3.0
    y1 = np.float32(math.sqrt(2.0 * op.J * 10.0 ** (e1 / 10.0))) * ctx.signal_base + ctx.noise
    y2 = np.float32(math.sqrt(2.0 * op.J * 10.0 ** (e2 / 10.0))) * ctx.signal_base + ctx.noise
    assert not np.array_equal(y1, y2)

    assert cfg.base.trials == 2
    assert not hasattr(cfg, "calibration_trials")
    assert not hasattr(cfg, "evaluation_trials")

    out_a = opt.run_amp_to_pseudo(cfg.base, op, ctx, 2.0)
    out_b = opt.run_amp_to_pseudo(cfg.base, op, ctx, 2.0)
    assert np.allclose(out_a.pseudo_final, out_b.pseudo_final, rtol=1e-6, atol=1e-6)
    assert math.isclose(out_a.tau_sq_final, out_b.tau_sq_final, rel_tol=1e-12, abs_tol=1e-12)

    had_cfg = RocConfig(
        opt.AmpConfig(
            J_values=(5,),
            s_in_values=(1.0,),
            L=2,
            operator="hadamard",
            trials=1,
            max_iterations=2,
            energy_min_db=1.0,
            energy_max_db=1.0,
            required_consecutive_passes=1,
            no_pdf=True,
            resume=False,
        )
    )
    had_op = make_operating_points(had_cfg)[0]
    had_ctx = make_trial_contexts(had_cfg, had_op)[0]
    assert not hasattr(had_ctx.operator, "A")
    had_ev = evaluate_energy(had_cfg, had_op, [had_ctx], 1.0)
    assert np.isfinite(had_ev.pmd) or had_ev.failed

    mp_cfg = RocConfig(
        opt.AmpConfig(
            J_values=(5,),
            s_in_values=(0.5, 1.0),
            L=2,
            operator="dense-gaussian",
            trials=1,
            max_iterations=2,
            energy_min_db=1.0,
            energy_max_db=1.0,
            required_consecutive_passes=1,
            workers=2,
            data_dir=Path(tempfile.mkdtemp(prefix="fig6_roc_validate_")),
            no_pdf=True,
            resume=False,
        )
    )
    try:
        mp_results = run_search(mp_cfg)
        assert len(mp_results) == 2
    except PermissionError as exc:
        print(f"WARNING: multiprocessing smoke test skipped by environment: {exc}")
        mp_results = [search_operating_point((RocConfig(opt.AmpConfig(**{**mp_cfg.base.__dict__, 'workers': 1})), p)) for p in make_operating_points(mp_cfg)]

    old_path = mp_cfg.base.data_dir / "old.csv"
    old_path.write_text("J,Ka,schema_version,pfa_calibration_factor\n1,1,2,0.8\n")
    assert read_progress(old_path) == {}
    progress_path = mp_cfg.base.data_dir / "new.csv"
    atomic_write_results(progress_path, mp_results)
    loaded = read_progress(progress_path)
    assert len(loaded) == len(mp_results)
    print("Validation passed.")


def benchmark_only() -> None:
    cfg = RocConfig(
        opt.AmpConfig(
            J_values=(8,),
            s_in_values=(0.5,),
            L=2,
            operator="dense-gaussian",
            trials=2,
            max_iterations=5,
            energy_min_db=2.0,
            energy_max_db=2.0,
            required_consecutive_passes=1,
            no_pdf=True,
            resume=False,
        )
    )
    op = make_operating_points(cfg)[0]
    start = time.perf_counter()
    contexts = make_trial_contexts(cfg, op)
    context_time = time.perf_counter() - start

    start = time.perf_counter()
    outputs = [opt.run_amp_to_pseudo(cfg.base, op, ctx, 2.0) for ctx in contexts]
    amp_time = time.perf_counter() - start

    start = time.perf_counter()
    result = empirical_roc_from_outputs(outputs, op, cfg.base.denoiser_chunk_size)
    roc_time = time.perf_counter() - start

    print(f"TrialContext construction time: {context_time:.4f}s")
    print(f"AMP time: {amp_time:.4f}s")
    print("inactive top-tail update time: included in empirical ROC pass")
    print("active-score collection time: included in empirical ROC pass")
    print("threshold extraction time: included in empirical ROC pass")
    print("error-counting time: included in empirical ROC pass")
    print(f"total energy-evaluation time: {context_time + amp_time + roc_time:.4f}s")
    print(f"peak retained inactive-tail size: {result[-1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-sample empirical ROC AMP simulation for Figure 6 markers.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--j-values", type=parse_j_values, default=(15,))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--operator", choices=["hadamard", "dense-gaussian"], default="hadamard")
    parser.add_argument("--denoiser", choices=["or", "binomial"], default="or")
    parser.add_argument("--s-in-values", type=opt.parse_s_values, default=tuple(float(x) for x in opt.DEFAULT_S_IN_VEC))
    parser.add_argument("--energy-min-db", type=float, default=-5.0)
    parser.add_argument("--energy-max-db", type=float, default=6.0)
    parser.add_argument("--energy-step-db", type=float, default=0.25)
    parser.add_argument("--refine-tolerance-db", type=float, default=0.02)
    parser.add_argument("--required-consecutive-passes", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--tau-floor", type=float, default=1e-12)
    parser.add_argument("--denoiser-chunk-size", type=int, default=8192)
    parser.add_argument("--max-matrix-gb", type=float, default=2.0)
    parser.add_argument("--max-total-memory-gb", type=float, default=64.0)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--parallelize-trials", action="store_true")
    parser.add_argument("--disable-numba", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fig6_amp_empirical_roc"))
    parser.add_argument("--output", type=Path, default=Path("plots/fig6_AMP_empirical_roc.png"))
    parser.add_argument("--pdf-output", type=Path, default=Path("plots/fig6_AMP_empirical_roc.pdf"))
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RocConfig:
    if args.debug and not args.full:
        j_values = args.j_values if args.j_values != (20,) else (8,)
        L = 2
        s_values = (0.25, 0.5)
        operator = "dense-gaussian" if args.operator == "hadamard" else args.operator
        max_iterations = 10
        energy_step_db = 1.0
    else:
        j_values = args.j_values
        L = 8
        s_values = args.s_in_values
        operator = args.operator
        max_iterations = args.max_iterations
        energy_step_db = args.energy_step_db
    points_count = len(j_values) * len(s_values)
    workers = args.workers if args.workers is not None else max(1, min(8, points_count, os.cpu_count() or 1))
    base = opt.AmpConfig(
        L=L,
        J_values=tuple(int(j) for j in j_values),
        s_in_values=tuple(float(x) for x in s_values),
        trials=args.trials,
        seed=args.seed,
        operator=operator,
        denoiser=args.denoiser,
        energy_min_db=args.energy_min_db,
        energy_max_db=args.energy_max_db,
        energy_step_db=energy_step_db,
        refine_tolerance_db=args.refine_tolerance_db,
        required_consecutive_passes=args.required_consecutive_passes,
        max_iterations=max_iterations,
        min_iterations=min(args.min_iterations, max_iterations),
        tolerance=args.tolerance,
        damping=args.damping,
        tau_floor=args.tau_floor,
        denoiser_chunk_size=args.denoiser_chunk_size,
        max_matrix_gb=args.max_matrix_gb,
        max_total_memory_gb=args.max_total_memory_gb,
        workers=workers,
        parallelize_trials=False,
        data_dir=args.data_dir,
        output=args.output,
        pdf_output=args.pdf_output,
        no_pdf=args.no_pdf,
        show=args.show,
        resume=args.resume,
        disable_numba=args.disable_numba,
    )
    if base.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if base.trials < 1:
        raise SystemExit("--trials must be at least 1.")
    if base.required_consecutive_passes < 1:
        raise SystemExit("--required-consecutive-passes must be at least 1.")
    if not 0.0 < base.damping <= 1.0:
        raise SystemExit("--damping must lie in (0, 1].")
    if args.parallelize_trials:
        print("Note: --parallelize-trials is accepted for CLI compatibility; this driver parallelizes operating points only.")
    return RocConfig(base)


def main() -> None:
    args = parse_args()
    opt.USE_NUMBA = (opt.njit is not None) and (not args.disable_numba)
    if args.validate_only:
        validate_only()
        return
    if args.benchmark_only:
        benchmark_only()
        return
    cfg = build_config(args)
    results = run_search(cfg)
    plot_results(cfg, results)
    print(f"Saved results: {cfg.base.data_dir / 'fig6_amp_results.csv'}")
    print(f"Saved progress: {cfg.base.data_dir / 'fig6_amp_progress.csv'}")
    print(f"Saved PNG: {cfg.base.output}")
    if not cfg.base.no_pdf:
        print(f"Saved PDF: {cfg.base.pdf_output}")


if __name__ == "__main__":
    main()
