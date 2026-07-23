#!/usr/bin/env python3
"""Finite-dimensional AMP simulation for empirical Figure 6 markers.

This script runs actual AMP iterations for the J=15 and J=20 empirical markers
corresponding to Figure 6 of "SPARCs for Unsourced Random Access." The default
large backend is a structured randomized Hadamard approximation, not the exact
i.i.d.-Gaussian ensemble used by the theory. This is because for J=20 the exact dense Gaussian backend would require 2^20 * 2^20 * 4 bytes = 4 TB of memory, which is not feasible. 
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from scipy.special import expit, gammaln, logsumexp
from scipy.stats import binom, norm

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback keeps the script runnable.
    class tqdm:  # type: ignore[no-redef]
        def __init__(
            self,
            iterable=None,
            total=None,
            desc=None,
            unit=None,
            position=None,
            leave=True,
            disable=False,
            **kwargs,
        ):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item

        def update(self, n=1):
            self.n += n

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        @staticmethod
        def write(message: str):
            print(message, flush=True)


def progress_log(cfg: Any, message: str) -> None:
    if getattr(cfg, "workers", 1) == 1:
        tqdm.write(message)
    else:
        print(message, flush=True)


DEFAULT_S_IN_VEC = np.array(
    [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
    dtype=float,
)

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
    "pmd",
    "pfa",
    "pmd_standard_error",
    "pfa_standard_error",
    "mean_iterations",
    "converged_fraction",
    "elapsed_seconds",
    "seed",
    "status",
    "config_key",
]


@dataclass(frozen=True)
class AmpConfig:
    alpha: float = 2.0
    L: int = 8
    J_values: tuple[int, ...] = (15, 20)
    s_in_values: tuple[float, ...] = tuple(float(x) for x in DEFAULT_S_IN_VEC)
    trials: int = 10
    seed: int = 12345
    operator: str = "hadamard"
    denoiser: str = "or"
    energy_min_db: float = -5.0
    energy_max_db: float = 6.0
    energy_step_db: float = 0.25
    refine_tolerance_db: float = 0.02
    required_consecutive_passes: int = 2
    max_iterations: int = 50
    min_iterations: int = 5
    tolerance: float = 1e-6
    damping: float = 1.0
    tau_floor: float = 1e-12
    denoiser_chunk_size: int = 8192
    binomial_tail_tolerance: float = 1e-14
    max_matrix_gb: float = 2.0
    max_total_memory_gb: float = 64.0
    workers: int = 1
    parallelize_trials: bool = False
    data_dir: Path = Path("data/fig6_amp")
    output: Path = Path("fig6_AMP.png")
    pdf_output: Path = Path("fig6_AMP.pdf")
    no_pdf: bool = False
    show: bool = False
    resume: bool = True


@dataclass(frozen=True)
class OperatingPoint:
    J: int
    Ka: int
    S_in: float
    s_idx: int
    n: int
    M_section: int
    M: int
    pmd_target: float
    pfa_target: float


@dataclass(frozen=True)
class TrialResult:
    true_active: int
    true_inactive: int
    missed_active: int
    false_active: int
    iterations: int
    converged: bool
    failed: bool
    tau_sq_trajectory: tuple[float, ...]
    relative_change_trajectory: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationResult:
    E_in_dB: float
    pmd: float
    pfa: float
    pmd_standard_error: float
    pfa_standard_error: float
    mean_iterations: float
    converged_fraction: float
    elapsed_seconds: float
    passed: bool
    failed: bool


@dataclass(frozen=True)
class SearchResult:
    J: int
    Ka: int
    S_in: float
    n: int
    M: int
    operator: str
    denoiser: str
    trials: int
    E_required_dB: float
    pmd: float
    pfa: float
    pmd_standard_error: float
    pfa_standard_error: float
    mean_iterations: float
    converged_fraction: float
    elapsed_seconds: float
    seed: int
    status: str
    config_key: str


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fwht_inplace(x: np.ndarray) -> None:
    n = x.shape[0]
    if not is_power_of_two(n):
        raise ValueError("Walsh-Hadamard input length must be a power of two.")
    h = 1
    while h < n:
        y = x.reshape(-1, h * 2)
        a = y[:, :h].copy()
        b = y[:, h : 2 * h]
        y[:, :h] = a + b
        y[:, h : 2 * h] = a - b
        h *= 2
    x /= math.sqrt(n)


def stable_seed(*items: int) -> int:
    payload = ",".join(str(int(x)) for x in items).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def rng_from_items(*items: int) -> np.random.Generator:
    return np.random.default_rng(stable_seed(*items))


class DenseGaussianOperator:
    def __init__(self, n: int, M: int, seed: int, max_matrix_gb: float):
        bytes_required = n * M * np.dtype(np.float32).itemsize
        gb_required = bytes_required / 1024**3
        if gb_required > max_matrix_gb:
            raise MemoryError(
                f"Dense Gaussian matrix would require {gb_required:.2f} GB, "
                f"exceeding --max-matrix-gb={max_matrix_gb:.2f}."
            )
        rng = rng_from_items(seed, 991, n, M)
        self.A = rng.normal(0.0, 1.0 / math.sqrt(n), size=(n, M)).astype(np.float32)
        self.shape = (n, M)
        self.dtype = np.float32

    def matvec(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x.astype(np.float32, copy=False)

    def rmatvec(self, z: np.ndarray) -> np.ndarray:
        return self.A.T @ z.astype(np.float32, copy=False)


class HadamardOperator:
    def __init__(self, n: int, J: int, L: int, seed: int, s_idx: int, trial_idx: int):
        self.n = int(n)
        self.J = int(J)
        self.L = int(L)
        self.m = 1 << int(J)
        self.M = self.L * self.m
        self.seed = int(seed)
        self.s_idx = int(s_idx)
        self.trial_idx = int(trial_idx)
        self.blocks = int(math.ceil(self.n / self.m))
        self.scale = math.sqrt(self.m / self.n)
        self.shape = (self.n, self.M)
        self.dtype = np.float32
        if not is_power_of_two(self.m):
            raise ValueError("Hadamard section length must be a power of two.")

    def _block_rng(self, section_idx: int, block_idx: int) -> np.random.Generator:
        return rng_from_items(self.seed, self.J, self.s_idx, self.trial_idx, section_idx, block_idx)

    def _signs_perm(self, section_idx: int, block_idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = self._block_rng(section_idx, block_idx)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.m)
        perm = rng.permutation(self.m)
        return signs, perm

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        y = np.zeros(self.n, dtype=np.float32)
        for ell in range(self.L):
            x_sec = x[ell * self.m : (ell + 1) * self.m]
            for b in range(self.blocks):
                start = b * self.m
                stop = min((b + 1) * self.m, self.n)
                if start >= stop:
                    continue
                r = stop - start
                signs, perm = self._signs_perm(ell, b)
                tmp = (x_sec * signs).astype(np.float32, copy=True)
                fwht_inplace(tmp)
                y[start:stop] += self.scale * tmp[perm[:r]]
        return y

    def rmatvec(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float32, copy=False)
        x = np.zeros(self.M, dtype=np.float32)
        for ell in range(self.L):
            x_sec = x[ell * self.m : (ell + 1) * self.m]
            for b in range(self.blocks):
                start = b * self.m
                stop = min((b + 1) * self.m, self.n)
                if start >= stop:
                    continue
                r = stop - start
                signs, perm = self._signs_perm(ell, b)
                tmp = np.zeros(self.m, dtype=np.float32)
                tmp[perm[:r]] = z[start:stop]
                fwht_inplace(tmp)
                x_sec += self.scale * signs * tmp
        return x


def make_operator(cfg: AmpConfig, op: OperatingPoint, trial_idx: int):
    if cfg.operator == "dense-gaussian":
        return DenseGaussianOperator(op.n, op.M, cfg.seed + 17 * trial_idx + 1009 * op.J, cfg.max_matrix_gb)
    if cfg.operator == "hadamard":
        return HadamardOperator(op.n, op.J, cfg.L, cfg.seed, op.s_idx, trial_idx)
    raise ValueError(f"Unknown operator: {cfg.operator}")


def validate_adjoint(A: Any, rng: np.random.Generator, tol: float = 1e-5) -> float:
    n, M = A.shape
    x = rng.normal(size=M).astype(np.float32)
    z = rng.normal(size=n).astype(np.float32)
    Ax = A.matvec(x)
    ATz = A.rmatvec(z)
    lhs = np.vdot(Ax, z)
    rhs = np.vdot(x, ATz)
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)
    if rel > tol:
        raise AssertionError(f"Adjoint validation failed: relative error {rel:.3e} > {tol:.3e}.")
    return float(rel)


def generate_signal(op: OperatingPoint, P_hat: float, seed: int, trial_idx: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    rng = rng_from_items(seed, op.J, op.s_idx, trial_idx, 777)
    s = np.zeros(op.M, dtype=np.float32)
    for ell in range(L):
        idx = rng.integers(0, op.M_section, size=op.Ka)
        s[ell * op.M_section : (ell + 1) * op.M_section] = np.bincount(
            idx,
            minlength=op.M_section,
        ).astype(np.float32)
    theta_true = np.float32(math.sqrt(P_hat)) * s
    support_true = s > 0
    return theta_true, support_true


def denoiser_or(
    u: np.ndarray,
    tau_sq: float,
    Ka: int,
    J: int,
    P_hat: float,
    tau_floor: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    tau_sq = max(float(tau_sq), float(tau_floor))
    q_active = -math.expm1(Ka * math.log1p(-(2.0 ** (-J))))
    log_prior_odds = math.log(q_active) - math.log1p(-q_active)
    sqrtP = math.sqrt(P_hat)
    theta = np.empty_like(u, dtype=np.float32)
    deriv = np.empty_like(u, dtype=np.float32)
    for start in range(0, u.size, chunk_size):
        stop = min(start + chunk_size, u.size)
        uu = u[start:stop].astype(np.float64)
        logits = log_prior_odds + sqrtP * uu / tau_sq - P_hat / (2.0 * tau_sq)
        p = expit(logits)
        theta[start:stop] = (sqrtP * p).astype(np.float32)
        deriv[start:stop] = ((P_hat / tau_sq) * p * (1.0 - p)).astype(np.float32)
    return theta, deriv


def binomial_k_range(Ka: int, q: float, tol: float) -> np.ndarray:
    lo = int(max(0, math.floor(binom.ppf(tol / 2.0, Ka, q)) - 2))
    hi = int(min(Ka, math.ceil(binom.isf(tol / 2.0, Ka, q)) + 2))
    return np.arange(lo, hi + 1, dtype=np.int64)


def denoiser_binomial(
    u: np.ndarray,
    tau_sq: float,
    Ka: int,
    J: int,
    P_hat: float,
    tau_floor: float,
    chunk_size: int,
    tail_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau_sq = max(float(tau_sq), float(tau_floor))
    q = 2.0 ** (-J)
    k = binomial_k_range(Ka, q, tail_tol)
    theta_k = math.sqrt(P_hat) * k.astype(np.float64)
    log_prior = (
        gammaln(Ka + 1)
        - gammaln(k + 1)
        - gammaln(Ka - k + 1)
        + k * math.log(q)
        + (Ka - k) * math.log1p(-q)
    )
    theta = np.empty_like(u, dtype=np.float32)
    deriv = np.empty_like(u, dtype=np.float32)
    for start in range(0, u.size, chunk_size):
        stop = min(start + chunk_size, u.size)
        uu = u[start:stop].astype(np.float64)
        log_like = -0.5 * (uu[:, None] - theta_k[None, :]) ** 2 / tau_sq
        log_post = log_like + log_prior[None, :]
        log_post -= logsumexp(log_post, axis=1)[:, None]
        weights = np.exp(log_post)
        mean = weights @ theta_k
        second = weights @ (theta_k**2)
        var = np.maximum(second - mean**2, 0.0)
        theta[start:stop] = mean.astype(np.float32)
        deriv[start:stop] = (var / tau_sq).astype(np.float32)
    return theta, deriv


def denoise(cfg: AmpConfig, u: np.ndarray, tau_sq: float, op: OperatingPoint, P_hat: float) -> tuple[np.ndarray, np.ndarray]:
    if cfg.denoiser == "or":
        return denoiser_or(u, tau_sq, op.Ka, op.J, P_hat, cfg.tau_floor, cfg.denoiser_chunk_size)
    if cfg.denoiser == "binomial":
        return denoiser_binomial(
            u,
            tau_sq,
            op.Ka,
            op.J,
            P_hat,
            cfg.tau_floor,
            cfg.denoiser_chunk_size,
            cfg.binomial_tail_tolerance,
        )
    raise ValueError(f"Unknown denoiser: {cfg.denoiser}")


def amp_trial(cfg: AmpConfig, op: OperatingPoint, E_in_dB: float, trial_idx: int) -> TrialResult:
    E_in = 10.0 ** (E_in_dB / 10.0)
    P_hat = 2.0 * op.J * E_in
    A = make_operator(cfg, op, trial_idx)
    theta_true, support_true = generate_signal(op, P_hat, cfg.seed, trial_idx, cfg.L)
    rng = rng_from_items(cfg.seed, op.J, op.s_idx, trial_idx, 999)
    noise = rng.normal(size=op.n).astype(np.float32)
    y = A.matvec(theta_true) + noise

    theta_hat = np.zeros(op.M, dtype=np.float32)
    residual = y.copy().astype(np.float32)
    tau_hist: list[float] = []
    rel_hist: list[float] = []
    converged = False
    failed = False

    try:
        for iteration in range(1, cfg.max_iterations + 1):
            tau_sq = float(np.dot(residual, residual) / op.n)
            tau_hist.append(tau_sq)
            pseudo = A.rmatvec(residual) + theta_hat
            theta_next, derivative = denoise(cfg, pseudo, tau_sq, op, P_hat)
            onsager = float(np.sum(derivative, dtype=np.float64) / op.n)
            residual_next = y - A.matvec(theta_next) + np.float32(onsager) * residual

            if cfg.damping != 1.0:
                theta_next = cfg.damping * theta_next + (1.0 - cfg.damping) * theta_hat
                residual_next = cfg.damping * residual_next + (1.0 - cfg.damping) * residual

            if not (np.all(np.isfinite(theta_next)) and np.all(np.isfinite(residual_next))):
                failed = True
                break

            denom = max(float(np.linalg.norm(theta_hat)), float(np.linalg.norm(theta_next)), 1.0)
            rel_change = float(np.linalg.norm(theta_next - theta_hat) / denom)
            rel_hist.append(rel_change)
            theta_hat = theta_next.astype(np.float32, copy=False)
            residual = residual_next.astype(np.float32, copy=False)

            if iteration >= cfg.min_iterations and rel_change <= cfg.tolerance:
                converged = True
                break
    except Exception:
        failed = True

    if failed:
        return TrialResult(0, 0, 0, 0, len(rel_hist), False, True, tuple(tau_hist), tuple(rel_hist))

    tau_sq_final = max(float(np.dot(residual, residual) / op.n), cfg.tau_floor)
    pseudo_final = A.rmatvec(residual) + theta_hat

    q_active = -math.expm1(op.Ka * math.log1p(-(2.0 ** (-op.J))))
    prior = math.log(q_active) - math.log1p(-q_active)
    sqrtP = math.sqrt(P_hat)
    threshold_score = prior + math.sqrt(P_hat / tau_sq_final) * norm.isf(op.pfa_target) - P_hat / (2.0 * tau_sq_final)
    threshold_u = math.sqrt(tau_sq_final) * norm.isf(op.pfa_target)
    threshold_score_from_u = prior + sqrtP * threshold_u / tau_sq_final - P_hat / (2.0 * tau_sq_final)
    if not math.isclose(threshold_score, threshold_score_from_u, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError("Score and pseudo-data thresholds disagree.")

    score = prior + sqrtP * pseudo_final.astype(np.float64) / tau_sq_final - P_hat / (2.0 * tau_sq_final)
    support_hat = score >= threshold_score

    true_active = int(np.count_nonzero(support_true))
    true_inactive = int(support_true.size - true_active)
    missed_active = int(np.count_nonzero(support_true & ~support_hat))
    false_active = int(np.count_nonzero(~support_true & support_hat))

    return TrialResult(
        true_active,
        true_inactive,
        missed_active,
        false_active,
        len(rel_hist),
        converged,
        False,
        tuple(tau_hist),
        tuple(rel_hist),
    )


def pooled_eval(
    cfg: AmpConfig,
    op: OperatingPoint,
    E_in_dB: float,
    progress_queue: Any | None = None,
    phase: str = "coarse",
) -> EvaluationResult:
    start = time.time()
    trial_results = [amp_trial(cfg, op, E_in_dB, t) for t in range(cfg.trials)]
    failed = any(r.failed for r in trial_results)
    true_active = sum(r.true_active for r in trial_results)
    true_inactive = sum(r.true_inactive for r in trial_results)
    missed = sum(r.missed_active for r in trial_results)
    false = sum(r.false_active for r in trial_results)
    pmd = missed / true_active if true_active else float("nan")
    pfa = false / true_inactive if true_inactive else float("nan")
    pmd_se = math.sqrt(pmd * (1.0 - pmd) / true_active) if true_active and np.isfinite(pmd) else float("nan")
    pfa_se = math.sqrt(pfa * (1.0 - pfa) / true_inactive) if true_inactive and np.isfinite(pfa) else float("nan")
    mean_iter = float(np.mean([r.iterations for r in trial_results]))
    conv_frac = float(np.mean([r.converged for r in trial_results]))
    elapsed = time.time() - start
    passed = (not failed) and pmd <= op.pmd_target and pfa <= op.pfa_target

    message = (
        f"J={op.J} Ka={op.Ka} S_in={op.S_in:.3f} n={op.n} "
        f"E={E_in_dB:.3f} dB trials={cfg.trials} pmd={pmd:.4e} pfa={pfa:.4e} "
        f"mean_iter={mean_iter:.2f} conv_frac={conv_frac:.2f} elapsed={elapsed:.1f}s"
    )
    if progress_queue is not None:
        progress_queue.put(
            {
                "phase": phase,
                "J": op.J,
                "S_in": op.S_in,
                "E_in_dB": E_in_dB,
                "pmd": pmd,
                "pfa": pfa,
                "passed": passed,
                "elapsed": elapsed,
                "message": message,
            }
        )
    else:
        progress_log(cfg, message)

    return EvaluationResult(E_in_dB, pmd, pfa, pmd_se, pfa_se, mean_iter, conv_frac, elapsed, passed, failed)


def config_key(cfg: AmpConfig, op: OperatingPoint) -> str:
    fields = (
        cfg.alpha,
        cfg.L,
        op.J,
        op.S_in,
        op.Ka,
        op.n,
        cfg.trials,
        cfg.seed,
        cfg.operator,
        cfg.denoiser,
        cfg.energy_min_db,
        cfg.energy_max_db,
        cfg.energy_step_db,
        cfg.refine_tolerance_db,
        cfg.required_consecutive_passes,
        cfg.max_iterations,
        cfg.min_iterations,
        cfg.tolerance,
        cfg.damping,
        cfg.tau_floor,
        cfg.denoiser_chunk_size,
    )
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()[:16]


def make_search_result(cfg: AmpConfig, op: OperatingPoint, E_req: float, ev: EvaluationResult, elapsed: float, status: str) -> SearchResult:
    return SearchResult(
        op.J,
        op.Ka,
        op.S_in,
        op.n,
        op.M,
        cfg.operator,
        cfg.denoiser,
        cfg.trials,
        E_req,
        ev.pmd,
        ev.pfa,
        ev.pmd_standard_error,
        ev.pfa_standard_error,
        ev.mean_iterations,
        ev.converged_fraction,
        elapsed,
        cfg.seed,
        status,
        config_key(cfg, op),
    )


def search_operating_point(args: tuple[AmpConfig, OperatingPoint]) -> SearchResult:
    if len(args) == 2:
        cfg, op = args
        progress_queue = None
    else:
        cfg, op, progress_queue = args
    total_start = time.time()
    energies = np.arange(cfg.energy_min_db, cfg.energy_max_db + 0.5 * cfg.energy_step_db, cfg.energy_step_db)
    evals: dict[float, EvaluationResult] = {}
    consecutive = 0
    bracket_low: float | None = None
    bracket_high: float | None = None

    show_nested_progress = cfg.workers == 1
    energy_desc = f"J={op.J} S_in={op.S_in:.3f}"
    for e in tqdm(energies, desc=energy_desc, unit="energy", leave=False, disable=not show_nested_progress):
        ev = pooled_eval(cfg, op, float(e), progress_queue, "coarse")
        evals[float(e)] = ev
        progress_log(
            cfg,
            f"{energy_desc}: E={float(e):.3f} dB "
            f"{'PASS' if ev.passed else 'fail'} "
            f"pmd={ev.pmd:.4e} pfa={ev.pfa:.4e}"
        )
        if ev.passed:
            consecutive += 1
            if consecutive >= cfg.required_consecutive_passes:
                hi_idx = int(np.where(np.isclose(energies, e))[0][0]) - cfg.required_consecutive_passes + 1
                bracket_high = float(energies[hi_idx])
                bracket_low = float(energies[hi_idx - 1]) if hi_idx > 0 else float(cfg.energy_min_db)
                break
        else:
            consecutive = 0

    if bracket_high is None:
        last = evals[float(energies[-1])]
        return make_search_result(cfg, op, float("nan"), last, time.time() - total_start, "ceiling_reached")

    lo = bracket_low
    hi = bracket_high
    hi_eval = evals.get(hi) or pooled_eval(cfg, op, hi)
    refine_total = max(1, int(math.ceil(math.log2(max((hi - lo) / cfg.refine_tolerance_db, 1.0)))))
    with tqdm(
        total=refine_total,
        desc=f"refine J={op.J} S_in={op.S_in:.3f}",
        unit="step",
        leave=False,
        disable=not show_nested_progress,
    ) as refine_bar:
        while hi - lo > cfg.refine_tolerance_db:
            mid = 0.5 * (lo + hi)
            ev = pooled_eval(cfg, op, mid, progress_queue, "refine")
            if ev.passed:
                hi = mid
                hi_eval = ev
            else:
                lo = mid
            refine_bar.set_postfix(E=f"{mid:.3f}", pmd=f"{ev.pmd:.2e}", pfa=f"{ev.pfa:.2e}", pass_=ev.passed)
            refine_bar.update(1)

    return make_search_result(cfg, op, hi, hi_eval, time.time() - total_start, "ok")


def make_operating_points(cfg: AmpConfig) -> list[OperatingPoint]:
    points = []
    for s_idx, S_in in enumerate(cfg.s_in_values):
        for J in cfg.J_values:
            Ka = int(round(2.0 ** (J / cfg.alpha)))
            R_in = S_in / Ka
            n = int(round(cfg.L * J / R_in))
            m_sec = 1 << J
            M = cfg.L * m_sec
            pmd_target = 0.05 / cfg.L
            pfa_target = 0.01 * Ka / (2.0**J)
            points.append(OperatingPoint(J, Ka, float(S_in), s_idx, n, m_sec, M, pmd_target, pfa_target))
    return points


def estimate_worker_memory_gb(cfg: AmpConfig, op: OperatingPoint) -> float:
    float_arrays = 7 * op.M + 3 * op.n + op.M_section
    bool_arrays = 2 * op.M
    bytes_est = 4 * float_arrays + bool_arrays
    if cfg.operator == "dense-gaussian":
        bytes_est += 4 * op.n * op.M
    return bytes_est / 1024**3


def parse_progress_value(k: str, v: str) -> Any:
    if k in {"J", "Ka", "n", "M", "trials", "seed"}:
        return int(v)
    if k in {
        "S_in",
        "E_required_dB",
        "pmd",
        "pfa",
        "pmd_standard_error",
        "pfa_standard_error",
        "mean_iterations",
        "converged_fraction",
        "elapsed_seconds",
    }:
        return float(v)
    return v


def read_progress(path: Path) -> dict[str, SearchResult]:
    if not path.exists():
        return {}
    rows: dict[str, SearchResult] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                result = SearchResult(**{k: parse_progress_value(k, row[k]) for k in PROGRESS_COLUMNS})
                rows[result.config_key] = result
            except Exception:
                continue
    return rows


def atomic_write_results(path: Path, results: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROGRESS_COLUMNS)
            writer.writeheader()
            for r in sorted(results, key=lambda x: (x.J, x.S_in)):
                writer.writerow({k: getattr(r, k) for k in PROGRESS_COLUMNS})
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
            phase=event.get("phase"),
            status="PASS" if event.get("passed") else "fail",
        )


def run_search(cfg: AmpConfig) -> list[SearchResult]:
    points = make_operating_points(cfg)
    max_mem = max(estimate_worker_memory_gb(cfg, p) for p in points)

    print(f"Operator: {'exact dense Gaussian' if cfg.operator == 'dense-gaussian' else 'structured randomized Hadamard'}")
    print(f"Denoiser: {'OR' if cfg.denoiser == 'or' else 'full binomial'}")
    if cfg.operator == "hadamard":
        print(
            "WARNING: structured Hadamard backend is a computationally tractable "
            "approximation, not the exact i.i.d.-Gaussian ensemble assumed by theory."
        )
    for p in points:
        print(f"Operating point: J={p.J} Ka={p.Ka} S_in={p.S_in:.3f} n={p.n} M={p.M}")
    print(f"Estimated per-worker memory: {max_mem:.2f} GB")
    if cfg.workers * max_mem > cfg.max_total_memory_gb:
        print(
            f"WARNING: workers * estimated memory = {cfg.workers * max_mem:.2f} GB "
            f"exceeds {cfg.max_total_memory_gb:.2f} GB"
        )

    progress_path = cfg.data_dir / "fig6_amp_progress.csv"
    existing = read_progress(progress_path) if cfg.resume else {}
    results = list(existing.values())
    pending = [p for p in points if config_key(cfg, p) not in existing]

    if pending:
        worker_count = max(1, min(cfg.workers, len(pending)))
        if worker_count == 1:
            for p in tqdm(pending, desc="operating points", unit="point"):
                r = search_operating_point((cfg, p))
                results = [x for x in results if x.config_key != r.config_key] + [r]
                atomic_write_results(progress_path, results)
        else:
            candidate_count = len(
                np.arange(cfg.energy_min_db, cfg.energy_max_db + 0.5 * cfg.energy_step_db, cfg.energy_step_db)
            )
            manager = mp.Manager()
            progress_queue = manager.Queue()
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
                    futures = {ex.submit(search_operating_point, (cfg, p, progress_queue)): p for p in pending}
                    remaining = set(futures)
                    with tqdm(total=len(futures), desc="operating points", unit="point") as point_bar, tqdm(
                        total=len(futures) * candidate_count,
                        desc="energy evaluations",
                        unit="eval",
                    ) as eval_bar:
                        while remaining:
                            done, remaining = concurrent.futures.wait(
                                remaining,
                                timeout=0.5,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            drain_progress_queue(progress_queue, eval_bar)
                            for future in done:
                                r = future.result()
                                results = [x for x in results if x.config_key != r.config_key] + [r]
                                atomic_write_results(progress_path, results)
                                point_bar.set_postfix(J=r.J, S_in=f"{r.S_in:.3f}", status=r.status)
                                point_bar.update(1)
                        drain_progress_queue(progress_queue, eval_bar)
            finally:
                manager.shutdown()

    atomic_write_results(cfg.data_dir / "fig6_amp_results.csv", results)
    atomic_write_results(progress_path, results)
    return sorted(results, key=lambda x: (x.J, x.S_in))


def maybe_load_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_results(cfg: AmpConfig, results: list[SearchResult]) -> None:
    import matplotlib

    if not cfg.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {15: ("o", "Empirical J=15"), 20: ("s", "Empirical J=20")}

    for J in cfg.J_values:
        theory = maybe_load_csv(Path("data/fig6_data") / f"fig6_J{J}.csv")
        if theory is not None and theory.dtype.names and {"S_in", "E_required_dB"}.issubset(theory.dtype.names):
            ax.plot(theory["E_required_dB"], theory["S_in"], "-", label=f"Theory J={J}")

    asym = maybe_load_csv(Path("data/fig6_data") / "fig6_asymptotic.csv")
    if asym is not None and asym.dtype.names:
        if {"S_in", "E_optimal_asymptotic_dB"}.issubset(asym.dtype.names):
            ax.plot(asym["E_optimal_asymptotic_dB"], asym["S_in"], "k-", label="Optimal asymptotic")
        if {"S_in", "E_amp_asymptotic_dB"}.issubset(asym.dtype.names):
            ax.plot(asym["E_amp_asymptotic_dB"], asym["S_in"], "k--", label="AMP asymptotic")

    for J in cfg.J_values:
        rows = [r for r in results if r.J == J and r.status == "ok" and np.isfinite(r.E_required_dB)]
        marker, label = styles.get(J, ("o", f"Empirical J={J}"))
        ax.plot(
            [r.E_required_dB for r in rows],
            [r.S_in for r in rows],
            linestyle="None",
            marker=marker,
            label=label,
        )

    ax.set_xlabel(r"$\mathcal{E}_{\mathrm{in}}\,[\mathrm{dB}]$")
    ax.set_ylabel(r"$S_{\mathrm{in}}$")
    ax.set_xlim(-5, 6)
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.output, dpi=300)
    if not cfg.no_pdf:
        fig.savefig(cfg.pdf_output)
    if cfg.show:
        plt.show()
    plt.close(fig)


def validate_only() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=256).astype(np.float32)
    y = x.copy()
    fwht_inplace(y)
    fwht_inplace(y)
    assert np.allclose(x, y, atol=1e-5)

    cfg = AmpConfig(J_values=(8,), s_in_values=(0.5,), L=2, operator="hadamard", trials=1, max_iterations=10)
    op = make_operating_points(cfg)[0]
    A = HadamardOperator(op.n, op.J, cfg.L, cfg.seed, 0, 0)
    validate_adjoint(A, rng)
    Ad = DenseGaussianOperator(32, 64, 1, 1.0)
    validate_adjoint(Ad, rng)

    u = rng.normal(size=64).astype(np.float32)
    th, der = denoiser_or(u, 1.2, 5, 8, 10.0, 1e-12, 16)
    eps = 1e-3
    thp, _ = denoiser_or(u + eps, 1.2, 5, 8, 10.0, 1e-12, 16)
    thm, _ = denoiser_or(u - eps, 1.2, 5, 8, 10.0, 1e-12, 16)
    assert np.allclose((thp - thm) / (2 * eps), der, rtol=2e-2, atol=2e-2)
    assert np.all((th / math.sqrt(10.0) >= 0) & (th / math.sqrt(10.0) <= 1))
    assert np.all(der >= 0)

    thb, derb = denoiser_binomial(u, 1.2, 5, 8, 10.0, 1e-12, 16, 1e-14)
    thbp, _ = denoiser_binomial(u + eps, 1.2, 5, 8, 10.0, 1e-12, 16, 1e-14)
    thbm, _ = denoiser_binomial(u - eps, 1.2, 5, 8, 10.0, 1e-12, 16, 1e-14)
    assert np.allclose((thbp - thbm) / (2 * eps), derb, rtol=3e-2, atol=3e-2)
    assert np.all(derb >= 0)

    tiny = AmpConfig(
        J_values=(8,),
        s_in_values=(0.5,),
        L=2,
        operator="dense-gaussian",
        trials=1,
        max_iterations=5,
        energy_min_db=5,
        energy_max_db=5,
    )
    res1 = amp_trial(tiny, op, 5.0, 0)
    res2 = amp_trial(tiny, op, 5.0, 0)
    assert res1 == res2
    assert res1.true_active >= 0
    assert config_key(tiny, op) == config_key(tiny, op)
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3)
    print("Validation passed.")


def parse_s_values(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one S_in value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-dimensional AMP simulation for Figure 6 empirical markers.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--operator", choices=["hadamard", "dense-gaussian"], default="hadamard")
    parser.add_argument("--denoiser", choices=["or", "binomial"], default="or")
    parser.add_argument("--s-in-values", type=parse_s_values, default=tuple(float(x) for x in DEFAULT_S_IN_VEC))
    parser.add_argument("--energy-min-db", type=float, default=-5.0)
    parser.add_argument("--energy-max-db", type=float, default=6.0)
    parser.add_argument("--energy-step-db", type=float, default=0.25)
    parser.add_argument("--refine-tolerance-db", type=float, default=0.02)
    parser.add_argument("--required-consecutive-passes", type=int, default=2)
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
    parser.add_argument("--data-dir", type=Path, default=Path("data/fig6_amp"))
    parser.add_argument("--output", type=Path, default=Path("fig6_AMP.png"))
    parser.add_argument("--pdf-output", type=Path, default=Path("fig6_AMP.pdf"))
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_only()
        return

    J_values = (8,) if args.debug else (15, 20)
    L = 2 if args.debug else 8
    s_values = (0.25, 0.5) if args.debug else args.s_in_values
    operator = "dense-gaussian" if args.debug and args.operator == "hadamard" else args.operator
    trials = 2 if args.debug else args.trials
    max_iterations = 10 if args.debug else args.max_iterations
    energy_step_db = 1.0 if args.debug else args.energy_step_db
    points_count = len(J_values) * len(s_values)
    default_workers = max(1, min(8, points_count, os.cpu_count() or 1))
    workers = args.workers if args.workers is not None else default_workers

    cfg = AmpConfig(
        L=L,
        J_values=J_values,
        s_in_values=tuple(float(x) for x in s_values),
        trials=trials,
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
        parallelize_trials=args.parallelize_trials,
        data_dir=args.data_dir,
        output=args.output,
        pdf_output=args.pdf_output,
        no_pdf=args.no_pdf,
        show=args.show,
        resume=args.resume,
    )

    if cfg.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if not 0.0 < cfg.damping <= 1.0:
        raise SystemExit("--damping must lie in (0, 1].")
    if cfg.parallelize_trials:
        print("Note: --parallelize-trials is accepted, but nested process pools are intentionally avoided.")

    results = run_search(cfg)
    plot_results(cfg, results)
    print(f"Saved results: {cfg.data_dir / 'fig6_amp_results.csv'}")
    print(f"Saved progress: {cfg.data_dir / 'fig6_amp_progress.csv'}")
    print(f"Saved PNG: {cfg.output}")
    if not cfg.no_pdf:
        print(f"Saved PDF: {cfg.pdf_output}")


if __name__ == "__main__":
    main()
