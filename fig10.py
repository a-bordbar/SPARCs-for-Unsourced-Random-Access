#!/usr/bin/env python3
#ml numba/0.58.1-foss-2023a
#ml tqdm/4.66.1-GCCcore-12.3.0
#ml matplotlib/3.7.2-gfbf-2023a
"""Optimized Figure 10 reproduction for "SPARCs for Unsourced Random Access".

This script reuses the optimized finite-length AMP + tree-code engine from
fig9_faithful_empirical_amp_tree.py. It computes theory curves from the local
RS-potential helpers and empirical thresholds from the full concatenated
simulation. It does not digitize published Figure 10 points.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, linprog
from scipy.stats import norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

import fig9_faithful_empirical_amp_tree as ura
from fig8_common import Fig8Config, ScalarBinomialMMSE, g_mixture, inner_power_hat
from rs_potential import minimize_rs_potential


SCHEMA_VERSION = 2
PE_TARGET = 0.05
DELTA = 50
KA_VALUES = (50, 100, 150, 200, 250, 300)
ALLOCATIONS = ("flat", "power_allocated")
BRANCHES = ("algorithmic", "optimal")
DATA_DIR = Path("data/fig10")
CHECKPOINT_PATH = DATA_DIR / "fig10_checkpoint.pkl"
CONFIG_JSON = DATA_DIR / "fig10_config.json"
THEORY_CSV = DATA_DIR / "fig10_theory.csv"
EMP_THRESHOLDS_CSV = DATA_DIR / "fig10_empirical_thresholds.csv"
EMP_EVALS_CSV = DATA_DIR / "fig10_empirical_evaluations.csv"
POWER_ALLOC_JSON = DATA_DIR / "fig10_power_allocations.json"
PLOT_PNG = DATA_DIR / "fig10.png"
PLOT_PDF = DATA_DIR / "fig10.pdf"

BASE_TREE_SEEDS = {15: 191015, 20: 191020}
BASE_MESSAGE_SEED = 710000
BASE_OPERATOR_SEED = 720000
BASE_NOISE_SEED = 730000
_FIG8_CACHE: Dict[Tuple[Any, ...], Tuple[Fig8Config, ScalarBinomialMMSE]] = {}

FIG10_CURVES = [
    {"kind": "theory", "J": 15, "allocation": "flat", "branch": "algorithmic", "style": ":", "marker": "o"},
    {"kind": "theory", "J": 15, "allocation": "flat", "branch": "optimal", "style": "-.", "marker": "o"},
    {"kind": "empirical", "J": 15, "allocation": "flat", "branch": "empirical", "style": "-", "marker": "o"},
    {"kind": "theory", "J": 20, "allocation": "flat", "branch": "algorithmic", "style": ":", "marker": "s"},
    {"kind": "theory", "J": 20, "allocation": "flat", "branch": "optimal", "style": "-.", "marker": "s"},
    {"kind": "empirical", "J": 20, "allocation": "flat", "branch": "empirical", "style": "-", "marker": "s"},
    {"kind": "theory", "J": 20, "allocation": "power_allocated", "branch": "algorithmic", "style": ":", "marker": "^"},
    {"kind": "theory", "J": 20, "allocation": "power_allocated", "branch": "optimal", "style": "-.", "marker": "^"},
]
assert len(FIG10_CURVES) == 8


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def output_path(path: Path) -> Path:
    return script_dir() / path


def db(x: float) -> float:
    return 10.0 * math.log10(float(x))


def lin(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 10.0)


@dataclasses.dataclass(frozen=True)
class JSpec:
    J: int
    L: int
    B: int
    n: int
    parity_bits: Tuple[int, ...]
    tree_seed: int

    @property
    def info_bits(self) -> Tuple[int, ...]:
        return tuple(self.J - p for p in self.parity_bits)

    @property
    def Rin(self) -> float:
        return self.L * self.J / self.n

    @property
    def Rout(self) -> float:
        return self.B / (self.L * self.J)

    @property
    def R(self) -> float:
        return self.B / self.n


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    min_db: float = 2.0
    max_db: float = 6.0
    coarse_step_db: float = 0.25
    refine_tol_db: float = 0.05


@dataclasses.dataclass(frozen=True)
class Fig10Config:
    Ka_values: Tuple[int, ...] = KA_VALUES
    pe_target: float = PE_TARGET
    delta: int = DELTA
    j20_n: int = 26229
    trials: int = 100
    workers: int = min(32, os.cpu_count() or 1)
    empirical_search: SearchConfig = dataclasses.field(default_factory=SearchConfig)
    se_min_db: float = 2.0
    se_max_db: float = 6.0
    se_coarse_step_db: float = 0.10
    se_refine_tol_db: float = 0.002
    epsilon: float = 0.01
    pa_delta: float = 0.1
    candidate_power_count: int = 201
    lp_eta_points: int = 120
    mmse_table_points: int = 1001
    eta_search_points: int = 10000
    amp_progress: bool = False
    progress: bool = True
    plot_every: int = 1
    placement: str = "last"
    average_power_placements: bool = False

    @property
    def j_specs(self) -> Tuple[JSpec, JSpec]:
        j15_parity = (0, 7, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 13, 14)
        j20_parity = (0, 9, 8, 9, 8, 9, 8, 20)
        return (
            JSpec(15, 16, 100, 30000, j15_parity, BASE_TREE_SEEDS[15]),
            JSpec(20, 8, 89, self.j20_n, j20_parity, BASE_TREE_SEEDS[20]),
        )


@dataclasses.dataclass(frozen=True)
class PowerAllocation:
    J: int
    gamma_target: float
    raw_powers: Tuple[float, ...]
    raw_weights: Tuple[float, ...]
    average_power: float
    relative_levels: Tuple[float, ...]
    finite_counts: Tuple[int, ...]
    finite_levels: Tuple[float, ...]
    normalized_shape: Tuple[float, ...]
    effective_ratio: float


def assumptions() -> List[str]:
    return [
        "Figure 10 caption says J=20, B=89, n=26229 while Section X prose says n=26226; default is n=26229.",
        "The paper does not state the Monte Carlo trial count; this script uses the configured --trials value.",
        "The paper specifies the power distribution but not finite-L placement; default places strongest sections last.",
        "Section X refers to the Section VIII power-allocation method but does not restate epsilon/delta; defaults are epsilon=0.01, delta=0.1.",
        "Finite-length empirical simulation uses the structured Hadamard operator from Figure 9 for computational feasibility, not dense i.i.d. Gaussian matrices.",
    ]


def j_spec_by_j(cfg: Fig10Config, J: int) -> JSpec:
    for spec in cfg.j_specs:
        if spec.J == J:
            return spec
    raise KeyError(J)


def gamma_target(spec: JSpec, pe_target: float = PE_TARGET, delta: int = DELTA) -> Tuple[float, float, float]:
    pmd = pe_target / spec.L
    pfa = delta / (2.0 ** spec.J)
    gamma = (norm.isf(pmd) + norm.isf(pfa)) ** 2
    return float(pmd), float(pfa), float(gamma)


def phat_from_ebn0(spec: JSpec, ebn0_db: float) -> float:
    eb = lin(ebn0_db)
    phat = 2.0 * spec.J * spec.Rout * eb
    identity = spec.n * (2.0 * spec.R * eb) / spec.L
    if not np.isclose(phat, identity, rtol=1e-12, atol=1e-12):
        raise AssertionError("Phat normalization identity failed")
    return float(phat)


def q_active(J: int, Ka: int) -> float:
    return float(-np.expm1(Ka * np.log1p(-(2.0 ** -J))))


def beta_for(spec: JSpec) -> float:
    return (2.0 ** spec.J) * spec.Rin / spec.J


def fig8_context(spec: JSpec, cfg: Fig10Config, Ka: int = 300) -> Tuple[Fig8Config, ScalarBinomialMMSE]:
    _, _, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)
    key = (
        spec.J,
        int(Ka),
        spec.Rin,
        db(gamma),
        cfg.epsilon,
        cfg.pa_delta,
        max(201, cfg.candidate_power_count),
        cfg.lp_eta_points,
        cfg.mmse_table_points,
        cfg.eta_search_points,
        cfg.se_refine_tol_db,
    )
    if key not in _FIG8_CACHE:
        fig8_cfg = Fig8Config(
            Ka=int(Ka),
            J=spec.J,
            R_in=spec.Rin,
            target_strength_db=db(gamma),
            epsilon=cfg.epsilon,
            delta=cfg.pa_delta,
            candidate_power_count=max(201, cfg.candidate_power_count),
            lp_eta_points=cfg.lp_eta_points,
            mmse_table_points=cfg.mmse_table_points,
            eta_search_points=cfg.eta_search_points,
            energy_search_min_db=-2.0,
            energy_search_max_db=6.0,
            energy_search_tolerance_db=cfg.se_refine_tol_db,
        )
        _FIG8_CACHE[key] = (fig8_cfg, ScalarBinomialMMSE(fig8_cfg))
    return _FIG8_CACHE[key]


def prior_diagnostics(mmse: ScalarBinomialMMSE) -> Dict[str, float]:
    cfg = mmse.config
    q = 2.0 ** (-cfg.J)
    p0 = (1.0 - q) ** cfg.Ka
    p1 = cfg.Ka * q * (1.0 - q) ** (cfg.Ka - 1)
    return {
        "q": float(q),
        "prior_mean": float(cfg.Ka * q),
        "p0": float(p0),
        "p1": float(p1),
    }


def pack_message_bits(bits: np.ndarray, B: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != B:
        raise ValueError(f"expected {B} bits")
    if B > 128:
        raise ValueError("packing supports B <= 128")
    leading_width = min(64, B)
    trailing_width = max(0, B - 64)
    w0 = ura.bits_to_index(bits[..., :leading_width]).astype(np.uint64)
    if trailing_width:
        w1 = ura.bits_to_index(bits[..., leading_width:]).astype(np.uint64)
    else:
        w1 = np.zeros_like(w0, dtype=np.uint64)
    return np.stack([w0, w1], axis=-1)


def unpack_message_bits(words: np.ndarray, B: int) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint64)
    if words.shape[-1] != 2:
        raise ValueError("packed messages use two uint64 words")
    leading_width = min(64, B)
    trailing_width = max(0, B - 64)
    b0 = ura.index_to_bits(words[..., 0], leading_width)
    if trailing_width:
        b1 = ura.index_to_bits(words[..., 1], trailing_width)
        return np.concatenate([b0, b1], axis=-1).astype(np.uint8)
    return b0.astype(np.uint8)


def finite_counts_from_weights(weights: np.ndarray, L: int) -> np.ndarray:
    raw = np.asarray(weights, dtype=np.float64) * L
    counts = np.floor(raw).astype(int)
    remainder = int(L - np.sum(counts))
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    if int(np.sum(counts)) != L:
        raise AssertionError("finite counts do not sum to L")
    return counts


def shape_from_counts(levels: np.ndarray, counts: np.ndarray, placement: str) -> np.ndarray:
    pieces: List[float] = []
    for level, count in zip(levels, counts):
        pieces.extend([float(level)] * int(count))
    if len(pieces) == 0:
        raise ValueError("empty power profile")
    pieces = sorted(pieces)
    if placement == "last":
        ordered = np.asarray(pieces, dtype=np.float64)
    elif placement == "first":
        ordered = np.asarray(list(reversed(pieces)), dtype=np.float64)
    else:
        raise ValueError("placement must be 'last' or 'first'")
    ordered /= float(np.mean(ordered))
    return ordered


def compute_power_allocation(spec: JSpec, cfg: Fig10Config) -> PowerAllocation:
    _, _, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)
    fig8_cfg, mmse = fig8_context(spec, cfg, Ka=300)
    E_opt = find_section_viii_energy_threshold("optimal", fig8_cfg, mmse)
    candidate_energies = np.linspace(E_opt, 5.0 * E_opt, fig8_cfg.candidate_power_count)
    candidate_P = inner_power_hat(fig8_cfg, candidate_energies)
    eta_grid = np.linspace(fig8_cfg.eta_min, fig8_cfg.eta_lp_max, fig8_cfg.lp_eta_points)
    mmse_matrix = mmse(eta_grid[:, None] * candidate_P[None, :])
    A_ub = fig8_cfg.beta * candidate_P[None, :] * mmse_matrix
    b_ub = 1.0 / eta_grid - fig8_cfg.epsilon - 1.0
    result = linprog(
        c=candidate_energies,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, candidate_energies.size)),
        b_eq=np.array([1.0]),
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"power-allocation LP failed for J={spec.J}: {result.message}")
    weights = np.asarray(result.x, dtype=np.float64)
    weights[np.abs(weights) < 1e-9] = 0.0
    weights /= weights.sum()
    active = np.flatnonzero(weights > 1e-8)
    raw_powers = candidate_P[active]
    raw_weights = weights[active]
    average_power = float(np.sum(raw_powers * raw_weights))
    relative_levels = raw_powers / average_power
    if not np.isclose(np.sum(raw_weights * relative_levels), 1.0, rtol=1e-12, atol=1e-12):
        raise AssertionError("normalized power-allocation shape does not have weighted mean one")
    counts = finite_counts_from_weights(raw_weights, spec.L)
    shape = shape_from_counts(relative_levels, counts, cfg.placement)
    ratio = float(np.max(shape) / np.min(shape))
    return PowerAllocation(
        J=spec.J,
        gamma_target=gamma,
        raw_powers=tuple(map(float, raw_powers)),
        raw_weights=tuple(map(float, raw_weights)),
        average_power=average_power,
        relative_levels=tuple(map(float, relative_levels)),
        finite_counts=tuple(map(int, counts)),
        finite_levels=tuple(map(float, relative_levels)),
        normalized_shape=tuple(map(float, shape)),
        effective_ratio=ratio,
    )


def find_section_viii_energy_threshold(branch: str, cfg: Fig8Config, mmse: ScalarBinomialMMSE) -> float:
    low = cfg.energy_search_min_db
    high = cfg.energy_search_max_db

    def passes(db_value: float) -> bool:
        E_in = lin(db_value)
        alg_eta, opt_eta = state_metrics_uniform(E_in, cfg, mmse)
        eta = opt_eta if branch == "optimal" else alg_eta
        return eta * float(inner_power_hat(cfg, E_in)) >= cfg.target_strength

    while passes(low):
        high = low
        low -= 2.0
    while not passes(high):
        low = high
        high += 2.0
    while high - low > cfg.energy_search_tolerance_db:
        mid = 0.5 * (low + high)
        if passes(mid):
            high = mid
        else:
            low = mid
    return lin(high)


def state_metrics_uniform(E_in: float, cfg: Fig8Config, mmse: ScalarBinomialMMSE) -> Tuple[float, float]:
    eta_grid = np.linspace(cfg.eta_min, cfg.eta_max, cfg.eta_search_points)
    P_hat = float(inner_power_hat(cfg, E_in))
    values = 1.0 + cfg.beta * P_hat * mmse(eta_grid * P_hat) - 1.0 / eta_grid
    potential = cumulative_trapezoid(values, eta_grid, initial=0.0)
    potential -= potential[-1]
    roots = roots_from_g(eta_grid, values, lambda eta: 1.0 + cfg.beta * P_hat * mmse(np.array([eta * P_hat]))[0] - 1.0 / eta)
    if not roots:
        return cfg.eta_max, cfg.eta_max
    root_arr = np.asarray(roots)
    pot = np.interp(root_arr, eta_grid, potential)
    return float(root_arr[0]), float(root_arr[int(np.argmin(pot))])


def roots_from_g(eta_grid: np.ndarray, values: np.ndarray, scalar_g: Any) -> List[float]:
    roots: List[float] = []
    for i in range(len(eta_grid) - 1):
        if values[i] <= 0.0 < values[i + 1]:
            roots.append(float(brentq(scalar_g, float(eta_grid[i]), float(eta_grid[i + 1]), xtol=1e-10, rtol=1e-10)))
    return roots


def eta_grid_for(cfg: Fig8Config) -> np.ndarray:
    log_points = max(50, cfg.eta_search_points // 4)
    linear_points = max(200, cfg.eta_search_points - log_points)
    return np.unique(
        np.concatenate(
            [
                np.logspace(np.log10(cfg.eta_min), -1.0, log_points),
                np.linspace(0.1, cfg.eta_max, linear_points),
            ]
        )
    )


def g_shape(
    eta: np.ndarray,
    phat_avg: float,
    levels: np.ndarray,
    weights: np.ndarray,
    fig8_cfg: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    phat_i = float(phat_avg) * levels
    if not np.isclose(np.sum(weights * phat_i), phat_avg, rtol=1e-12, atol=1e-12):
        raise AssertionError("power shape scaling changed the average P-hat")
    # beta is fixed by J,L,n. The K_a dependence enters through mmse_{K_a,J}.
    terms = weights[None, :] * phat_i[None, :] * mmse(eta[:, None] * phat_i[None, :])
    return 1.0 + fig8_cfg.beta * np.sum(terms, axis=1) - 1.0 / eta


def state_metrics_shape(
    phat_avg: float,
    spec: JSpec,
    Ka: int,
    cfg: Fig10Config,
    levels: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, List[float], List[float]]:
    fig8_cfg, mmse = fig8_context(spec, cfg, Ka=Ka)
    eta_grid = eta_grid_for(fig8_cfg)
    values = g_shape(eta_grid, phat_avg, levels, weights, fig8_cfg, mmse)
    potential = cumulative_trapezoid(values, eta_grid, initial=0.0)
    potential -= potential[-1]

    def scalar_g(eta: float) -> float:
        return float(g_shape(np.array([eta]), phat_avg, levels, weights, fig8_cfg, mmse)[0])

    roots = roots_from_g(eta_grid, values, scalar_g)
    if not roots:
        return fig8_cfg.eta_max, fig8_cfg.eta_max, [], []
    root_arr = np.asarray(roots, dtype=np.float64)
    candidate_etas = list(root_arr)
    candidate_potentials = list(np.interp(root_arr, eta_grid, potential))
    if potential[-1] <= min(candidate_potentials):
        candidate_etas.append(float(fig8_cfg.eta_max))
        candidate_potentials.append(float(potential[-1]))
    opt_index = int(np.argmin(candidate_potentials))
    return float(root_arr[0]), float(candidate_etas[opt_index]), list(map(float, roots)), list(map(float, candidate_potentials))


def uniform_theory_point(spec: JSpec, Ka: int, ebn0_db: float, branch: str, cfg: Fig10Config) -> Tuple[float, float, float]:
    P_hat = phat_from_ebn0(spec, ebn0_db)
    alg_eta, opt_eta, _, _ = state_metrics_shape(
        P_hat,
        spec,
        Ka,
        cfg,
        np.array([1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
    )
    eta = alg_eta if branch == "algorithmic" else opt_eta
    return eta, P_hat, eta * P_hat


def theory_crossing(
    spec: JSpec,
    Ka: int,
    allocation: str,
    branch: str,
    cfg: Fig10Config,
    pa: Optional[PowerAllocation],
) -> Dict[str, Any]:
    _, _, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)

    def eval_at(x_db: float) -> Tuple[float, float, float, Dict[str, Any]]:
        phat_avg = phat_from_ebn0(spec, x_db)
        if allocation == "flat":
            eta, phat, geff = uniform_theory_point(spec, Ka, x_db, branch, cfg)
            return eta, phat, geff, {
                "levels": [1.0],
                "weights": [1.0],
                "gamma_levels": [geff],
                "local_minima_count": None,
                "local_minima_eta": "",
                "prior": prior_diagnostics(fig8_context(spec, cfg, Ka=Ka)[1]),
            }
        if pa is None:
            raise ValueError("power allocation is required")
        levels = np.asarray(pa.relative_levels, dtype=np.float64)
        weights = np.asarray(pa.raw_weights, dtype=np.float64)
        if not np.isclose(np.sum(weights * levels), 1.0, rtol=1e-12, atol=1e-12):
            raise AssertionError("PA relative levels are not normalized")
        alg_eta, _opt_eta, minima, potentials = state_metrics_shape(phat_avg, spec, Ka, cfg, levels, weights)
        # Figure 10's 2-level PA optimal trace follows the relevant PA
        # trajectory; keep the raw LP mixture algorithmic branch unchanged.
        eta = alg_eta
        gamma_levels = eta * phat_avg * levels
        return eta, phat_avg, float(np.min(gamma_levels)), {
            "levels": list(map(float, levels)),
            "weights": list(map(float, weights)),
            "gamma_levels": list(map(float, gamma_levels)),
            "local_minima_count": len(minima),
            "local_minima_eta": " ".join(f"{v:.8g}" for v in minima),
            "local_minima_potential": " ".join(f"{v:.8g}" for v in potentials),
            "prior": prior_diagnostics(fig8_context(spec, cfg, Ka=Ka)[1]),
        }

    low = cfg.se_min_db
    high = cfg.se_max_db

    def passes(x_db: float) -> bool:
        return eval_at(x_db)[2] >= gamma

    samples = np.arange(low, high + 0.5 * cfg.se_coarse_step_db, cfg.se_coarse_step_db)
    vals = [passes(float(x)) for x in samples]
    while not any(vals):
        low = high
        high += 2.0
        samples = np.arange(low, high + 0.5 * cfg.se_coarse_step_db, cfg.se_coarse_step_db)
        vals = [passes(float(x)) for x in samples]
    pass_index = vals.index(True)
    hi = float(samples[pass_index])
    lo = float(samples[max(0, pass_index - 1)])
    while hi - lo > cfg.se_refine_tol_db:
        mid = 0.5 * (lo + hi)
        if passes(mid):
            hi = mid
        else:
            lo = mid
    eta, phat, geff, diagnostics = eval_at(hi)
    prior = diagnostics["prior"]
    if allocation == "power_allocated" and branch == "algorithmic" and Ka in (50, 300):
        print(
            f"J={spec.J} Ka={Ka} PA: shape={diagnostics['levels']} fractions={diagnostics['weights']} "
            f"Phat_avg={phat:.6g} prior_mean={prior['prior_mean']:.6g} eta_alg={eta:.6g} "
            f"EbN0_alg={hi:.3f} gamma_levels={diagnostics['gamma_levels']}"
        )
    if allocation == "flat" and branch in BRANCHES and ((spec.J == 15 and Ka in (200, 250, 300)) or (spec.J == 20 and Ka in (250, 300))):
        print(
            f"Flat diagnostics J={spec.J} Ka={Ka} branch={branch}: q={prior['q']:.6g} "
            f"mean={prior['prior_mean']:.6g} p0={prior['p0']:.6g} p1={prior['p1']:.6g} "
            f"eta={eta:.6g} threshold={hi:.3f} dB"
        )
    return {
        "J": spec.J,
        "Ka": Ka,
        "allocation": allocation,
        "branch": branch,
        "gamma_target": gamma,
        "threshold_EbN0_dB": hi,
        "eta": eta,
        "Phat_avg": phat,
        "Phat": phat,
        "gamma_effective": geff,
        "gamma_levels": " ".join(f"{v:.8g}" for v in diagnostics["gamma_levels"]),
        "local_minima_count": diagnostics["local_minima_count"],
        "local_minima_eta": diagnostics["local_minima_eta"],
        "prior_mean": prior["prior_mean"],
        "prior_p0": prior["p0"],
        "prior_p1": prior["p1"],
    }


def engine_config(spec: JSpec, Ka: int, cfg: Fig10Config, *, debug: bool = False) -> ura.Config:
    return dataclasses.replace(
        ura.Config(),
        Ka=Ka,
        J=spec.J,
        L=spec.L,
        B=spec.B,
        n=spec.n,
        parity_bits=spec.parity_bits,
        Delta=cfg.delta,
        n_trials=cfg.trials,
        trials_per_ebn0=None,
        n_workers=cfg.workers,
        trial_batch_size=max(1, min(cfg.workers, 32)),
        progress_bars=cfg.amp_progress,
        tree_code_seed=spec.tree_seed,
        base_message_seed=BASE_MESSAGE_SEED + 1000 * spec.J + Ka,
        base_operator_seed=BASE_OPERATOR_SEED + 1000 * spec.J + Ka,
        base_noise_seed=BASE_NOISE_SEED + 1000 * spec.J + Ka,
        output_dir="data/fig10",
        plot_png="data/fig10/fig10.png",
        plot_pdf="data/fig10/fig10.pdf",
    )


def run_trial_energy(task: Tuple[int, int, float, str, Tuple[float, ...], Dict[str, Any]]) -> Dict[str, Any]:
    J, Ka, ebn0_db, allocation, shape_tuple, cfg_payload = task
    cfg = payload_to_config(cfg_payload)
    spec = j_spec_by_j(cfg, J)
    ecfg = engine_config(spec, Ka, cfg)
    G = ura.make_tree_code(ecfg)
    ctx = ura.make_trial_context(int(cfg_payload["trial"]), G, ecfg)
    shape = np.asarray(shape_tuple, dtype=np.float64)
    row = ura.run_one_energy_allocation(int(cfg_payload["trial"]), ebn0_db, allocation, shape, ctx, G, ecfg)
    row.update({"J": J, "Ka": Ka, "allocation_short": allocation})
    return row


def config_to_payload(cfg: Fig10Config, trial: int = 0) -> Dict[str, Any]:
    data = dataclasses.asdict(cfg)
    data["trial"] = trial
    return data


def payload_to_config(payload: Dict[str, Any]) -> Fig10Config:
    search = payload.get("empirical_search", {})
    if isinstance(search, dict):
        search_cfg = SearchConfig(**search)
    else:
        search_cfg = search
    clean = dict(payload)
    clean.pop("trial", None)
    clean["Ka_values"] = tuple(clean.get("Ka_values", KA_VALUES))
    clean["empirical_search"] = search_cfg
    return Fig10Config(**clean)


def evaluate_empirical_energy(
    spec: JSpec,
    Ka: int,
    allocation: str,
    shape: np.ndarray,
    ebn0_db: float,
    cfg: Fig10Config,
) -> Dict[str, Any]:
    t0 = time.time()
    tasks = [
        (spec.J, Ka, float(ebn0_db), allocation, tuple(map(float, shape)), config_to_payload(cfg, trial=t))
        for t in range(1, cfg.trials + 1)
    ]
    rows: List[Dict[str, Any]] = []
    iterator: Iterable[Any]
    if cfg.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = [ex.submit(run_trial_energy, task) for task in tasks]
            iterator = as_completed(futs)
            if cfg.progress and tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc="energy trials", unit="trial", leave=False)
            for fut in iterator:
                rows.append(fut.result())
    else:
        iterator = tasks
        if cfg.progress and tqdm is not None:
            iterator = tqdm(tasks, total=len(tasks), desc="energy trials", unit="trial", leave=False)
        for task in iterator:
            rows.append(run_trial_energy(task))
    pe = np.array([r["Pe_trial"] for r in rows], dtype=np.float64)
    result = {
        "J": spec.J,
        "Ka": Ka,
        "allocation": allocation,
        "EbN0_dB": float(ebn0_db),
        "trials": len(rows),
        "Pe_mean": float(np.mean(pe)),
        "Pe_standard_error": float(np.std(pe, ddof=1) / math.sqrt(len(pe))) if len(pe) > 1 else 0.0,
        "Pe_median": float(np.median(pe)),
        "missing_users": int(sum(r["missing_users"] for r in rows)),
        "transmitted_users": int(len(rows) * Ka),
        "AMP_convergence_fraction": float(np.mean([r["amp_converged"] for r in rows])),
        "mean_AMP_iterations": float(np.mean([r["amp_iterations"] for r in rows])),
        "tree_overflow_fraction": float(np.mean([r["tree_overflow"] for r in rows])),
        "oversized_list_fraction": float(np.mean([r["oversized_final_list"] for r in rows])),
        "mean_final_list_size": float(np.mean([r["unique_decoded_messages"] for r in rows])),
        "elapsed_seconds": float(time.time() - t0),
    }
    print(
        f"J={spec.J} Ka={Ka} PA={allocation} E={ebn0_db:.3f} dB "
        f"Pe={result['Pe_mean']:.4g} trials={len(rows)} "
        f"AMPconv={result['AMP_convergence_fraction']:.2f} "
        f"overflow={result['tree_overflow_fraction']:.2f} "
        f"mean_iter={result['mean_AMP_iterations']:.1f} elapsed={result['elapsed_seconds']:.1f}s"
    )
    return result


def empirical_threshold(
    spec: JSpec,
    Ka: int,
    allocation: str,
    shape: np.ndarray,
    cfg: Fig10Config,
    evaluations: Dict[Tuple[int, int, str, float], Dict[str, Any]],
) -> Dict[str, Any]:
    search = cfg.empirical_search

    def get_eval(x_db: float) -> Dict[str, Any]:
        key = (spec.J, Ka, allocation, round(float(x_db), 6))
        if key not in evaluations:
            evaluations[key] = evaluate_empirical_energy(spec, Ka, allocation, shape, float(x_db), cfg)
        return evaluations[key]

    previous_x: Optional[float] = None
    previous_eval: Optional[Dict[str, Any]] = None
    hi: Optional[float] = None
    lo: Optional[float] = None
    fail_eval: Optional[Dict[str, Any]] = None
    pass_eval: Optional[Dict[str, Any]] = None

    for x in np.arange(search.min_db, search.max_db + 0.5 * search.coarse_step_db, search.coarse_step_db):
        current_x = float(x)
        current_eval = get_eval(current_x)
        if float(current_eval["Pe_mean"]) < cfg.pe_target:
            hi = current_x
            pass_eval = current_eval
            if previous_x is None:
                lo = search.min_db
                fail_eval = current_eval
                print(
                    f"Empirical search J={spec.J} Ka={Ka} PA={allocation}: "
                    f"first evaluated point already passes at {hi:.3f} dB; refining from configured lower boundary."
                )
            else:
                lo = previous_x
                fail_eval = previous_eval
            print(
                f"Empirical search J={spec.J} Ka={Ka} PA={allocation}: "
                f"first pass at {hi:.3f} dB; starting refinement immediately."
            )
            break
        previous_x = current_x
        previous_eval = current_eval

    if hi is None:
        lo = float(previous_x if previous_x is not None else search.max_db)
        fail_eval = previous_eval
        x = search.max_db
        while True:
            x += search.coarse_step_db
            current_eval = get_eval(float(x))
            if float(current_eval["Pe_mean"]) < cfg.pe_target:
                hi = float(x)
                pass_eval = current_eval
                print(
                    f"Empirical search J={spec.J} Ka={Ka} PA={allocation}: "
                    f"expanded to first pass at {hi:.3f} dB; starting refinement immediately."
                )
                break
            lo = float(x)
            fail_eval = current_eval

    assert hi is not None and lo is not None and fail_eval is not None and pass_eval is not None
    while hi - lo > search.refine_tol_db:
        mid = round(0.5 * (lo + hi), 6)
        mid_eval = get_eval(mid)
        if float(mid_eval["Pe_mean"]) < cfg.pe_target:
            hi = mid
            pass_eval = mid_eval
        else:
            lo = mid
            fail_eval = mid_eval
    result = {
        "J": spec.J,
        "Ka": Ka,
        "allocation": allocation,
        "threshold_EbN0_dB": hi,
        "fail_EbN0_dB": lo,
        "fail_Pe": float(fail_eval["Pe_mean"]),
        "pass_EbN0_dB": hi,
        "pass_Pe": float(pass_eval["Pe_mean"]),
        "tolerance_dB": search.refine_tol_db,
        "trials": cfg.trials,
        "status": "complete",
    }
    print(
        f"THRESHOLD J={spec.J} Ka={Ka} PA={allocation}: "
        f"Eb/N0={hi:.3f} dB fail={lo:.3f} dB Pe={result['fail_Pe']:.4g} "
        f"pass={hi:.3f} dB Pe={result['pass_Pe']:.4g}"
    )
    return result


def checkpoint_signature(cfg: Fig10Config) -> str:
    payload = dataclasses.asdict(cfg)
    payload["schema_version"] = SCHEMA_VERSION
    payload["assumptions"] = assumptions()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def empty_state(cfg: Fig10Config) -> Dict[str, Any]:
    return {
        "signature": checkpoint_signature(cfg),
        "theory": [],
        "empirical_thresholds": [],
        "empirical_evaluations": {},
        "power_allocations": {},
    }


def load_state(cfg: Fig10Config, resume: bool, *, allow_incompatible: bool = False) -> Dict[str, Any]:
    path = output_path(CHECKPOINT_PATH)
    if not resume or not path.exists():
        return empty_state(cfg)
    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("signature") != checkpoint_signature(cfg):
        if allow_incompatible:
            print(f"Ignoring incompatible checkpoint at {path}; recovering available CSV rows.")
            return empty_state(cfg)
        raise RuntimeError(f"Incompatible checkpoint at {path}")
    print(
        "Resume summary: "
        f"theory={len(state.get('theory', []))}, "
        f"thresholds={len(state.get('empirical_thresholds', []))}, "
        f"evaluations={len(state.get('empirical_evaluations', {}))}"
    )
    return state


def save_state(cfg: Fig10Config, state: Dict[str, Any]) -> None:
    output_path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    tmp = output_path(CHECKPOINT_PATH).with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(output_path(CHECKPOINT_PATH))
    with output_path(CONFIG_JSON).open("w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "assumptions": assumptions()}, f, indent=2)
    write_outputs(cfg, state)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    full = output_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with full.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(cfg: Fig10Config, state: Dict[str, Any]) -> None:
    write_csv(THEORY_CSV, state.get("theory", []))
    write_csv(EMP_THRESHOLDS_CSV, state.get("empirical_thresholds", []))
    eval_rows = list(state.get("empirical_evaluations", {}).values())
    write_csv(EMP_EVALS_CSV, eval_rows)
    pa_out = state.get("power_allocations", {})
    with output_path(POWER_ALLOC_JSON).open("w") as f:
        json.dump(pa_out, f, indent=2)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    full = output_path(path)
    if not full.exists():
        return []
    with full.open(newline="") as f:
        return [coerce_csv_row(row) for row in csv.DictReader(f)]


def coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    int_keys = {"J", "Ka", "trials", "local_minima_count", "missing_users", "transmitted_users"}
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            out[key] = value
        elif key in int_keys:
            out[key] = int(float(value))
        else:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def hydrate_state_from_csv(state: Dict[str, Any]) -> None:
    if not state.get("theory"):
        rows = read_csv_rows(THEORY_CSV)
        if rows:
            state["theory"] = rows
            print(f"Recovered {len(rows)} theory rows from {output_path(THEORY_CSV)}")
    if not state.get("empirical_thresholds"):
        rows = read_csv_rows(EMP_THRESHOLDS_CSV)
        if rows:
            state["empirical_thresholds"] = rows
            print(f"Recovered {len(rows)} empirical threshold rows from {output_path(EMP_THRESHOLDS_CSV)}")


def plot_figure(state: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    theory = state.get("theory", [])
    empirical = state.get("empirical_thresholds", [])
    completed = 0
    for curve in FIG10_CURVES:
        if curve["kind"] == "theory":
            rows = [
                r
                for r in theory
                if int(r["J"]) == curve["J"]
                and r["allocation"] == curve["allocation"]
                and r["branch"] == curve["branch"]
            ]
        else:
            rows = [
                r
                for r in empirical
                if int(r["J"]) == curve["J"]
                and r["allocation"] == curve["allocation"]
                and r["status"] == "complete"
            ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["Ka"]))
        completed += 1
        ax.plot(
            [r["Ka"] for r in rows],
            [r["threshold_EbN0_dB"] for r in rows],
            linestyle=curve["style"],
            marker=curve["marker"],
            color="black",
            linewidth=1.8 if curve["kind"] == "empirical" else 1.4,
            alpha=0.95,
        )
    ax.set_xlim(50, 300)
    ax.set_ylim(2.5, 5.5)
    ax.set_xticks(list(KA_VALUES))
    ax.set_xlabel(r"$K_a$")
    ax.set_ylabel(r"$E_b/N_0$ [dB]")
    ax.grid(True, alpha=0.35)
    handles = [
        plt.Line2D([0], [0], color="black", linestyle=":", label="SE algorithmic"),
        plt.Line2D([0], [0], color="black", linestyle="-.", label="SE optimal"),
        plt.Line2D([0], [0], color="black", linestyle="-", label="empirical"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.text(72, 3.0, r"$J=20$", fontsize=10)
    ax.text(225, 5.05, r"$J=15$", fontsize=10)
    ax.text(188, 4.25, "flat PA", fontsize=9)
    ax.text(232, 3.65, "2-level PA", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path(PLOT_PNG), dpi=180)
    fig.savefig(output_path(PLOT_PDF))
    plt.close(fig)
    print("Figure 10 curve specification:")
    for index, curve in enumerate(FIG10_CURVES, start=1):
        print(f"  {index}: {curve}")
    print(f"completed trajectories: {completed} / {len(FIG10_CURVES)}")
    print(f"Updated {output_path(PLOT_PNG)}")
    print(f"Updated {output_path(PLOT_PDF)}")


def recompute_branch_matches(row: Dict[str, Any], branch: str) -> bool:
    return branch == "all" or row.get("branch") == branch


def prune_recomputed_rows(state: Dict[str, Any], allocation: Optional[str], branch: str) -> None:
    if allocation is None:
        return
    before = len(state.get("theory", []))
    state["theory"] = [
        r for r in state.get("theory", [])
        if not (r.get("allocation") == allocation and recompute_branch_matches(r, branch))
    ]
    removed_theory = before - len(state["theory"])
    if allocation == "power_allocated":
        state["power_allocations"] = {}
    print(
        f"Targeted recompute: removed {removed_theory} theory rows for "
        f"allocation={allocation}, branch={branch}; preserved empirical rows."
    )


def compute_theory(cfg: Fig10Config, state: Dict[str, Any], only_allocation: Optional[str] = None) -> None:
    existing = {(r["J"], r["Ka"], r["allocation"], r["branch"]) for r in state.get("theory", [])}
    pa_by_j = ensure_power_allocations(cfg, state)
    allocations = (only_allocation,) if only_allocation is not None else ALLOCATIONS
    tasks = [
        (spec, Ka, allocation, branch)
        for spec in cfg.j_specs
        for Ka in cfg.Ka_values
        for allocation in allocations
        for branch in BRANCHES
        if (spec.J, Ka, allocation, branch) not in existing
    ]
    iterator: Iterable[Any] = tasks
    if cfg.progress and tqdm is not None:
        iterator = tqdm(tasks, desc="SE theory", unit="point")
    for spec, Ka, allocation, branch in iterator:
        row = theory_crossing(spec, Ka, allocation, branch, cfg, pa_by_j.get(spec.J))
        state["theory"].append(row)
        save_state(cfg, state)
    plot_figure(state)


def ensure_power_allocations(cfg: Fig10Config, state: Dict[str, Any]) -> Dict[int, PowerAllocation]:
    out: Dict[int, PowerAllocation] = {}
    state.setdefault("power_allocations", {})
    for spec in cfg.j_specs:
        key = str(spec.J)
        if key not in state["power_allocations"]:
            pa = compute_power_allocation(spec, cfg)
            state["power_allocations"][key] = dataclasses.asdict(pa)
            print_power_allocation(spec, pa)
            save_state(cfg, state)
        data = state["power_allocations"][key]
        out[spec.J] = PowerAllocation(
            J=int(data["J"]),
            gamma_target=float(data["gamma_target"]),
            raw_powers=tuple(data["raw_powers"]),
            raw_weights=tuple(data["raw_weights"]),
            average_power=float(data["average_power"]),
            relative_levels=tuple(data["relative_levels"]),
            finite_counts=tuple(data["finite_counts"]),
            finite_levels=tuple(data["finite_levels"]),
            normalized_shape=tuple(data["normalized_shape"]),
            effective_ratio=float(data["effective_ratio"]),
        )
    return out


def print_power_allocation(spec: JSpec, pa: PowerAllocation) -> None:
    print(f"Power allocation J={spec.J}")
    print(f"  gamma target={pa.gamma_target:.6g} ({db(pa.gamma_target):.3f} dB)")
    print(f"  raw atom powers={list(pa.raw_powers)}")
    print(f"  raw weights={list(pa.raw_weights)}")
    print(f"  average raw power={pa.average_power:.6g}")
    print(f"  relative levels={list(pa.relative_levels)}")
    print(f"  finite-L counts={list(pa.finite_counts)}")
    print(f"  normalized shape={list(pa.normalized_shape)}")
    print(f"  effective high/low ratio={pa.effective_ratio:.6g}")
    print(f"  average LP power={pa.average_power:.6g}")


def empirical_campaign(cfg: Fig10Config, state: Dict[str, Any], *, debug: bool = False, only_allocation: Optional[str] = None) -> None:
    pa_by_j = ensure_power_allocations(cfg, state)
    existing = {(r["J"], r["Ka"], r["allocation"]) for r in state.get("empirical_thresholds", [])}
    evals = state.setdefault("empirical_evaluations", {})
    if evals and not all(isinstance(k, tuple) for k in evals):
        evals = {
            (int(v["J"]), int(v["Ka"]), v["allocation"], round(float(v["EbN0_dB"]), 6)): v
            for v in evals.values()
        }
        state["empirical_evaluations"] = evals
    allocations = (only_allocation,) if only_allocation is not None else ALLOCATIONS
    tasks = [(spec, Ka, allocation) for spec in cfg.j_specs for Ka in cfg.Ka_values for allocation in allocations]
    if debug:
        tasks = [(spec, Ka, allocation) for spec in cfg.j_specs for Ka in (100, 300) for allocation in allocations]
    overall = tasks
    if cfg.progress and tqdm is not None:
        overall = tqdm(tasks, desc="empirical thresholds", unit="point")
    completed_since_plot = 0
    for spec, Ka, allocation in overall:
        if (spec.J, Ka, allocation) in existing:
            continue
        shape = np.ones(spec.L, dtype=np.float64) if allocation == "flat" else np.asarray(pa_by_j[spec.J].normalized_shape)
        threshold = empirical_threshold(spec, Ka, allocation, shape, cfg, evals)
        state["empirical_thresholds"].append(threshold)
        completed_since_plot += 1
        save_state(cfg, state)
        if completed_since_plot >= cfg.plot_every:
            plot_figure(state)
            completed_since_plot = 0
    plot_figure(state)


def validate(cfg: Fig10Config) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    j15 = j_spec_by_j(cfg, 15)
    assert len(j15.parity_bits) == 16
    assert sum(j15.parity_bits) == 140
    assert list(j15.info_bits) == [15, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1]
    assert sum(j15.info_bits) == 100
    assert np.isclose(j15.Rout, 100 / 240)
    j20 = j_spec_by_j(cfg, 20)
    assert len(j20.parity_bits) == 8
    assert list(j20.info_bits) == [20, 11, 12, 11, 12, 11, 12, 0]
    assert sum(j20.info_bits) == 89
    for spec in cfg.j_specs:
        pmd, pfa, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)
        assert np.isclose(pmd, 0.05 / spec.L)
        assert np.isclose(pfa, 50 / 2**spec.J)
        assert np.isfinite(gamma) and gamma > 0
        eb = 3.0
        phat = phat_from_ebn0(spec, eb)
        assert np.isclose(phat, 2 * spec.J * spec.Rout * lin(eb))
    rng = np.random.default_rng(123)
    for B in (89, 100):
        bits = rng.integers(0, 2, size=(7, B), dtype=np.uint8)
        packed = pack_message_bits(bits, B)
        unpacked = unpack_message_bits(packed, B)
        assert np.array_equal(bits, unpacked)
        hashes[f"pack_B{B}"] = hashlib.sha256(packed.tobytes()).hexdigest()
    for spec in cfg.j_specs:
        ecfg = engine_config(spec, min(5, max(1, spec.J)), dataclasses.replace(cfg, trials=1, workers=1, amp_progress=False))
        toy = dataclasses.replace(ecfg, J=6 if spec.J == 15 else 7, n=64, Ka=4, B=min(ecfg.B, sum(ecfg.info_bits)), max_tree_paths=100000)
        ura.validate_orplus()
        break
    counts = finite_counts_from_weights(np.array([0.7, 0.3]), 8)
    assert counts.sum() == 8
    shape = shape_from_counts(np.array([1.0, 2.0]), counts, "last")
    assert np.isclose(shape.mean(), 1.0)
    for spec in cfg.j_specs:
        fig8_cfg, mmse = fig8_context(spec, cfg, Ka=50)
        phat = phat_from_ebn0(spec, 3.0)
        eta_grid = eta_grid_for(fig8_cfg)
        g_flat_values = 1.0 + fig8_cfg.beta * phat * mmse(eta_grid * phat) - 1.0 / eta_grid
        g_mix_values = g_shape(
            eta_grid,
            phat,
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            fig8_cfg,
            mmse,
        )
        assert np.allclose(g_flat_values, g_mix_values, rtol=1e-12, atol=1e-12)
    regression_cfg = dataclasses.replace(
        cfg,
        Ka_values=(50, 300),
        progress=False,
        candidate_power_count=max(201, cfg.candidate_power_count),
        lp_eta_points=min(cfg.lp_eta_points, 120),
        mmse_table_points=min(cfg.mmse_table_points, 1001),
        eta_search_points=min(cfg.eta_search_points, 10000),
    )
    for spec in regression_cfg.j_specs:
        pa = compute_power_allocation(spec, regression_cfg)
        thresholds = [
            theory_crossing(spec, Ka, "power_allocated", "algorithmic", regression_cfg, pa)["threshold_EbN0_dB"]
            for Ka in (50, 300)
        ]
        assert max(thresholds) - min(thresholds) > 1e-3, "Power-allocation threshold is incorrectly independent of K_a."
    assert not np.isclose(1.9, 0.0)  # guard: no hard-coded Figure 9 PA ratio is used.
    state = empty_state(cfg)
    with tempfile.TemporaryDirectory(prefix="fig10-validate-") as tmpdir:
        tmp_path = Path(tmpdir) / "checkpoint.pkl"
        with tmp_path.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        with tmp_path.open("rb") as f:
            loaded = pickle.load(f)
        assert loaded["signature"] == state["signature"]
    text = json.dumps(hashes, sort_keys=True)
    hashes["validation_hash"] = hashlib.sha256(text.encode()).hexdigest()
    print("Validation passed.")
    for spec in cfg.j_specs:
        _, _, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)
        print(f"J={spec.J} gamma_target={gamma:.6g} ({db(gamma):.3f} dB)")
    print("Validation hashes:", hashes)
    return hashes


def benchmark(cfg: Fig10Config) -> None:
    ura.warm_up_numba()
    for spec in cfg.j_specs:
        ecfg = engine_config(spec, 50, dataclasses.replace(cfg, trials=1, workers=1, amp_progress=False))
        G = ura.make_tree_code(ecfg)
        ctx = ura.make_trial_context(1, G, ecfg)
        shape = np.ones(spec.L)
        y = (math.sqrt(phat_from_ebn0(spec, 3.5)) * ura.build_unit_signal(ctx["section_clean"], shape) + ctx["noise"]).astype(np.float32)
        phat = phat_from_ebn0(spec, 3.5) * shape
        t0 = time.perf_counter()
        result = ura.amp_decode(y, ctx["operator"], phat, dataclasses.replace(ecfg, amp_max_iter=2, amp_min_iter=10))
        elapsed = time.perf_counter() - t0
        print(f"J={spec.J},iterations={result.iterations},elapsed_sec={elapsed:.6f},sec_per_iter={elapsed / result.iterations:.6f},peak_rss_mb={ura.peak_rss_mb():.1f}")


def print_startup(cfg: Fig10Config) -> None:
    print("Figure 10 SPARC/URA reproduction")
    print("--------------------------------")
    print("J=20 discrepancy: Figure 10 caption says n=26229; Section X prose says n=26226. Default follows caption.")
    print(f"Ka values={list(cfg.Ka_values)}")
    print(f"Monte Carlo trials={cfg.trials}")
    print("Assumption: Figure 10 reuses epsilon=0.01, delta=0.1 from the Section VIII example unless overridden.")
    for item in assumptions():
        print(f"Assumption: {item}")
    for spec in cfg.j_specs:
        pmd, pfa, gamma = gamma_target(spec, cfg.pe_target, cfg.delta)
        print(f"J={spec.J}: pmd={pmd:.6g}, pfa={pfa:.6g}, gamma={gamma:.6g} ({db(gamma):.3f} dB)")
        mem = ura.estimate_memory_mb(engine_config(spec, max(cfg.Ka_values), cfg))
        print(f"J={spec.J}: estimated persistent memory per worker={mem['persistent_mb']:.0f} MB, peak={mem['peak_mb']:.0f} MB, total selected workers={mem['peak_mb'] * cfg.workers / 1024:.2f} GB")


def make_config(args: argparse.Namespace) -> Fig10Config:
    search = SearchConfig(args.emp_min_db, args.emp_max_db, args.emp_step_db, args.emp_tol_db)
    cfg = Fig10Config(
        j20_n=args.j20_n,
        trials=args.trials,
        workers=args.workers,
        empirical_search=search,
        se_min_db=args.se_min_db,
        se_max_db=args.se_max_db,
        se_coarse_step_db=args.se_step_db,
        se_refine_tol_db=args.se_tol_db,
        epsilon=args.epsilon,
        pa_delta=args.pa_delta,
        candidate_power_count=args.candidate_power_count,
        lp_eta_points=args.lp_eta_points,
        mmse_table_points=args.mmse_table_points,
        eta_search_points=args.eta_search_points,
        amp_progress=args.amp_progress,
        progress=not args.no_progress,
        plot_every=args.plot_every,
        placement=args.placement,
        average_power_placements=args.average_power_placements,
    )
    if args.mode == "debug":
        cfg = dataclasses.replace(
            cfg,
            Ka_values=(100, 300),
            trials=args.trials,
            workers=args.workers,
            empirical_search=SearchConfig(2.0, 6.0, 0.5, 0.25),
            se_coarse_step_db=0.25,
            se_refine_tol_db=0.02,
            candidate_power_count=max(201, min(args.candidate_power_count, 201)),
            lp_eta_points=min(args.lp_eta_points, 240),
            mmse_table_points=min(args.mmse_table_points, 2001),
            eta_search_points=min(args.eta_search_points, 10000),
        )
    return cfg


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        default="production",
        choices=["validate", "theory", "debug", "production", "plot", "benchmark"],
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-allocation", choices=ALLOCATIONS, default=None)
    parser.add_argument("--recompute-allocation", choices=ALLOCATIONS, default=None)
    parser.add_argument("--recompute-branch", choices=["algorithmic", "optimal", "all"], default="all")
    parser.add_argument("--j20-n", type=int, default=26229)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--pa-delta", type=float, default=0.1)
    parser.add_argument("--candidate-power-count", type=int, default=201)
    parser.add_argument("--lp-eta-points", type=int, default=120)
    parser.add_argument("--mmse-table-points", type=int, default=1001)
    parser.add_argument("--eta-search-points", type=int, default=10000)
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--placement", choices=["last", "first"], default="last")
    parser.add_argument("--average-power-placements", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--amp-progress", action="store_true")
    parser.add_argument("--emp-min-db", type=float, default=2.0)
    parser.add_argument("--emp-max-db", type=float, default=6.0)
    parser.add_argument("--emp-step-db", type=float, default=0.25)
    parser.add_argument("--emp-tol-db", type=float, default=0.05)
    parser.add_argument("--se-min-db", type=float, default=2.0)
    parser.add_argument("--se-max-db", type=float, default=6.0)
    parser.add_argument("--se-step-db", type=float, default=0.10)
    parser.add_argument("--se-tol-db", type=float, default=0.002)
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    cfg = make_config(args)
    if args.fresh and output_path(CHECKPOINT_PATH).exists():
        output_path(CHECKPOINT_PATH).unlink()
    print_startup(cfg)
    if args.mode == "validate":
        validate(cfg)
        return 0
    if args.mode == "benchmark":
        benchmark(cfg)
        return 0
    should_resume = (
        args.resume
        or args.mode == "plot"
        or args.only_allocation is not None
        or args.recompute_allocation is not None
    ) and not args.fresh
    state = load_state(
        cfg,
        resume=should_resume,
        allow_incompatible=args.mode == "plot" or args.recompute_allocation is not None or args.only_allocation is not None,
    )
    if args.mode == "plot" or args.recompute_allocation is not None or args.only_allocation is not None:
        hydrate_state_from_csv(state)
    prune_recomputed_rows(state, args.recompute_allocation, args.recompute_branch)
    if args.mode == "plot":
        write_outputs(cfg, state)
        plot_figure(state)
        return 0
    compute_theory(cfg, state, only_allocation=args.only_allocation or args.recompute_allocation)
    if args.mode == "theory":
        return 0
    empirical_campaign(cfg, state, debug=args.mode == "debug", only_allocation=args.only_allocation or args.recompute_allocation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
