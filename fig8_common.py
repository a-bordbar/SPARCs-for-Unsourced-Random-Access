#!/usr/bin/env python3
"""Shared numerical routines for reproducing Figure 8.

The implementation follows Section VIII of
"SPARCs for Unsourced Random Access" for

    K_a = 300, J = 20, R_in = 0.0061,
    target effective inner-channel strength = 15 dB,
    epsilon = 0.01, delta = 0.1.

It computes:
- P_opt from the global minimum of the scalar RS potential;
- P_alg from the smallest stationary point / AMP branch;
- the power-allocation LP over [P_opt, 5 P_opt];
- g(eta) from equation (99);
- G(eta), a normalized integral of g(eta).

The scalar MMSE uses the exact Binomial(K_a, 2^{-J}) prior.
The binomial support is truncated only after its remaining tail
probability is below `prior_tail_tolerance`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, linprog
from scipy.special import logsumexp
from scipy.stats import binom


CACHE_VERSION = 1


@dataclass(frozen=True)
class Fig8Config:
    Ka: int = 300
    J: int = 20
    R_in: float = 0.0061
    target_strength_db: float = 15.0
    epsilon: float = 0.01
    delta: float = 0.1

    gh_order: int = 60
    prior_tail_tolerance: float = 1e-15

    mmse_t_max: float = 320.0
    mmse_table_points: int = 16001

    eta_min: float = 1e-4
    eta_max: float = 0.9999
    eta_search_points: int = 6000
    eta_plot_min: float = 0.03
    eta_plot_max: float = 0.9
    eta_plot_points: int = 2400

    energy_search_min_db: float = -1.0
    energy_search_max_db: float = 4.0
    energy_search_tolerance_db: float = 2e-4

    candidate_power_count: int = 101
    lp_eta_points: int = 900

    @property
    def beta(self) -> float:
        return (2.0**self.J) * self.R_in / self.J

    @property
    def target_strength(self) -> float:
        return 10.0 ** (self.target_strength_db / 10.0)

    @property
    def eta_lp_max(self) -> float:
        return 1.0 - self.delta


@dataclass
class Fig8Solution:
    config: Fig8Config
    eta: np.ndarray
    g_opt: np.ndarray
    g_alg: np.ndarray
    g_star: np.ndarray
    g_mix: np.ndarray
    G_opt: np.ndarray
    G_alg: np.ndarray
    G_star: np.ndarray
    G_mix: np.ndarray

    E_opt: float
    E_alg: float
    E_star: float
    E_mix: float

    alpha_opt: float
    alpha_star: float

    candidate_energies: np.ndarray
    candidate_weights: np.ndarray

    @property
    def E_opt_db(self) -> float:
        return 10.0 * math.log10(self.E_opt)

    @property
    def E_alg_db(self) -> float:
        return 10.0 * math.log10(self.E_alg)

    @property
    def E_star_db(self) -> float:
        return 10.0 * math.log10(self.E_star)

    @property
    def E_mix_db(self) -> float:
        return 10.0 * math.log10(self.E_mix)


class ScalarBinomialMMSE:
    """Accurate scalar-channel MMSE table for a sparse binomial prior."""

    def __init__(self, config: Fig8Config):
        self.config = config

        success_probability = 2.0 ** (-config.J)
        k_max = None
        tail = None
        for candidate in range(config.Ka + 1):
            tail = float(
                binom.sf(
                    candidate,
                    config.Ka,
                    success_probability,
                )
            )
            if tail <= config.prior_tail_tolerance:
                k_max = candidate
                break

        if k_max is None:
            raise RuntimeError("Could not truncate the binomial prior safely.")

        self.symbols = np.arange(k_max + 1, dtype=np.float64)
        probabilities = binom.pmf(
            self.symbols.astype(int),
            config.Ka,
            success_probability,
        ).astype(np.float64)
        probabilities /= probabilities.sum()

        self.probabilities = probabilities
        self.log_probabilities = np.log(probabilities)
        self.omitted_tail_probability = float(tail)

        nodes, weights = hermgauss(config.gh_order)
        self.noise_nodes = np.sqrt(2.0) * nodes
        self.noise_weights = weights / np.sqrt(np.pi)

        self.t_grid = np.linspace(
            0.0,
            config.mmse_t_max,
            config.mmse_table_points,
            dtype=np.float64,
        )
        self.mmse_grid = self._evaluate_exact(self.t_grid)
        self.interpolator = PchipInterpolator(
            self.t_grid,
            self.mmse_grid,
            extrapolate=False,
        )

    def _evaluate_exact(
        self,
        strengths: np.ndarray,
        *,
        chunk_size: int = 256,
    ) -> np.ndarray:
        strengths_array = np.asarray(strengths, dtype=np.float64)
        flat = strengths_array.reshape(-1)
        output = np.empty_like(flat)

        symbols = self.symbols
        probabilities = self.probabilities
        log_probabilities = self.log_probabilities
        noise_nodes = self.noise_nodes
        noise_weights = self.noise_weights

        for start in range(0, flat.size, chunk_size):
            stop = min(start + chunk_size, flat.size)
            t = np.maximum(flat[start:stop], 0.0)
            sqrt_t = np.sqrt(t)

            observations = (
                sqrt_t[:, None, None] * symbols[None, :, None]
                + noise_nodes[None, None, :]
            )

            residuals = (
                observations[:, :, :, None]
                - sqrt_t[:, None, None, None]
                * symbols[None, None, None, :]
            )

            log_weights = (
                log_probabilities[None, None, None, :]
                - 0.5 * residuals * residuals
            )
            log_normalizer = logsumexp(
                log_weights,
                axis=-1,
                keepdims=True,
            )
            posterior = np.exp(log_weights - log_normalizer)
            posterior_mean = np.sum(
                posterior
                * symbols[None, None, None, :],
                axis=-1,
            )

            squared_error = (
                symbols[None, :, None] - posterior_mean
            ) ** 2

            output[start:stop] = np.sum(
                squared_error
                * probabilities[None, :, None]
                * noise_weights[None, None, :],
                axis=(1, 2),
            )

        return output.reshape(strengths_array.shape)

    def __call__(self, strengths: np.ndarray | float) -> np.ndarray:
        strengths_array = np.asarray(strengths, dtype=np.float64)

        if np.any(strengths_array < 0.0):
            raise ValueError("Scalar-channel strength must be nonnegative.")
        if np.any(strengths_array > self.config.mmse_t_max):
            maximum = float(np.max(strengths_array))
            raise ValueError(
                f"Requested MMSE strength {maximum:.6g} exceeds "
                f"the table maximum {self.config.mmse_t_max:.6g}. "
                "Increase mmse_t_max."
            )

        return np.asarray(
            self.interpolator(strengths_array),
            dtype=np.float64,
        )


def inner_power_hat(config: Fig8Config, E_in: float | np.ndarray) -> np.ndarray:
    """Convert linear E_in into P-hat = 2 J E_in."""
    return 2.0 * config.J * np.asarray(E_in, dtype=np.float64)


def g_flat(
    eta: np.ndarray,
    E_in: float,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    P_hat = float(inner_power_hat(config, E_in))
    return (
        1.0
        + config.beta * P_hat * mmse(eta * P_hat)
        - 1.0 / eta
    )


def g_mixture(
    eta: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    energies = np.asarray(energies, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if energies.ndim != 1 or weights.ndim != 1:
        raise ValueError("energies and weights must be one-dimensional.")
    if energies.size != weights.size:
        raise ValueError("energies and weights must have equal lengths.")

    P_hats = inner_power_hat(config, energies)
    terms = (
        weights[None, :]
        * P_hats[None, :]
        * mmse(eta[:, None] * P_hats[None, :])
    )

    return (
        1.0
        + config.beta * np.sum(terms, axis=1)
        - 1.0 / eta
    )


def normalized_integral(eta: np.ndarray, g_values: np.ndarray) -> np.ndarray:
    """Integrate g and choose the constant so G(eta_max)=0."""
    integral = cumulative_trapezoid(
        g_values,
        eta,
        initial=0.0,
    )
    return integral - integral[-1]


def local_minimum_roots(
    E_in: float,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> list[float]:
    eta_grid = np.linspace(
        config.eta_min,
        config.eta_max,
        config.eta_search_points,
        dtype=np.float64,
    )
    values = g_flat(eta_grid, E_in, config, mmse)

    P_hat = float(inner_power_hat(config, E_in))

    def scalar_g(value: float) -> float:
        return float(
            1.0
            + config.beta
            * P_hat
            * mmse(np.array([value * P_hat]))[0]
            - 1.0 / value
        )

    roots: list[float] = []

    for index in range(eta_grid.size - 1):
        left_value = values[index]
        right_value = values[index + 1]

        # A negative-to-positive crossing is a local minimum of
        # the potential because its derivative is proportional to g.
        if left_value <= 0.0 < right_value:
            root = brentq(
                scalar_g,
                float(eta_grid[index]),
                float(eta_grid[index + 1]),
                xtol=1e-12,
                rtol=1e-12,
            )
            roots.append(float(root))

    return roots


def state_metrics(
    E_in: float,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> tuple[float, float]:
    eta_grid = np.linspace(
        config.eta_min,
        config.eta_max,
        config.eta_search_points,
        dtype=np.float64,
    )
    values = g_flat(eta_grid, E_in, config, mmse)
    potential = normalized_integral(eta_grid, values)

    minimum_roots = local_minimum_roots(E_in, config, mmse)
    if not minimum_roots:
        # At sufficiently high power the good minimum lies extremely
        # close to eta=1 and may be outside the finite grid.
        algorithmic_eta = config.eta_max
        global_eta = config.eta_max
        return algorithmic_eta, global_eta

    algorithmic_eta = minimum_roots[0]

    candidate_etas = np.array(minimum_roots, dtype=np.float64)
    candidate_potentials = np.interp(
        candidate_etas,
        eta_grid,
        potential,
    )
    global_eta = float(
        candidate_etas[int(np.argmin(candidate_potentials))]
    )

    return float(algorithmic_eta), global_eta


def find_energy_threshold(
    *,
    branch: str,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> float:
    if branch not in {"optimal", "algorithmic"}:
        raise ValueError("branch must be 'optimal' or 'algorithmic'.")

    low_db = config.energy_search_min_db
    high_db = config.energy_search_max_db

    def passes(energy_db: float) -> bool:
        E_in = 10.0 ** (energy_db / 10.0)
        algorithmic_eta, global_eta = state_metrics(
            E_in,
            config,
            mmse,
        )
        eta = global_eta if branch == "optimal" else algorithmic_eta
        effective_strength = eta * float(inner_power_hat(config, E_in))
        return effective_strength >= config.target_strength

    if passes(low_db):
        raise RuntimeError(
            f"{branch} threshold is below the configured search interval."
        )
    if not passes(high_db):
        raise RuntimeError(
            f"{branch} threshold is above the configured search interval."
        )

    while (
        high_db - low_db
        > config.energy_search_tolerance_db
    ):
        middle_db = 0.5 * (low_db + high_db)
        if passes(middle_db):
            high_db = middle_db
        else:
            low_db = middle_db

    return 10.0 ** (high_db / 10.0)


def solve_power_allocation(
    E_opt: float,
    config: Fig8Config,
    mmse: ScalarBinomialMMSE,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_energies = np.linspace(
        E_opt,
        5.0 * E_opt,
        config.candidate_power_count,
        dtype=np.float64,
    )
    candidate_P_hats = inner_power_hat(
        config,
        candidate_energies,
    )

    eta_constraints = np.linspace(
        config.eta_min,
        config.eta_lp_max,
        config.lp_eta_points,
        dtype=np.float64,
    )

    mmse_matrix = mmse(
        eta_constraints[:, None]
        * candidate_P_hats[None, :]
    )

    A_ub = (
        config.beta
        * candidate_P_hats[None, :]
        * mmse_matrix
    )
    b_ub = (
        1.0 / eta_constraints
        - config.epsilon
        - 1.0
    )

    result = linprog(
        c=candidate_energies,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.ones(
            (1, candidate_energies.size),
            dtype=np.float64,
        ),
        b_eq=np.array([1.0], dtype=np.float64),
        bounds=(0.0, None),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "Power-allocation LP failed: "
            f"{result.message}"
        )

    weights = np.asarray(result.x, dtype=np.float64)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()

    return candidate_energies, weights


def reduce_to_two_levels(
    candidate_energies: np.ndarray,
    candidate_weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Represent adjacent high-power LP atoms by one weighted level."""
    active = np.flatnonzero(candidate_weights > 1e-8)
    if active.size < 2:
        raise RuntimeError(
            "The LP did not produce the expected nonuniform allocation."
        )

    lowest_index = int(active[np.argmin(candidate_energies[active])])
    alpha_opt = float(candidate_weights[lowest_index])
    E_opt = float(candidate_energies[lowest_index])

    high_indices = active[active != lowest_index]
    alpha_star = float(np.sum(candidate_weights[high_indices]))
    E_star = float(
        np.sum(
            candidate_weights[high_indices]
            * candidate_energies[high_indices]
        )
        / alpha_star
    )

    normalization = alpha_opt + alpha_star
    alpha_opt /= normalization
    alpha_star /= normalization

    return E_opt, E_star, alpha_opt, alpha_star


def compute_solution(config: Fig8Config | None = None) -> Fig8Solution:
    config = config or Fig8Config()
    mmse = ScalarBinomialMMSE(config)

    E_opt = find_energy_threshold(
        branch="optimal",
        config=config,
        mmse=mmse,
    )
    E_alg = find_energy_threshold(
        branch="algorithmic",
        config=config,
        mmse=mmse,
    )

    candidate_energies, candidate_weights = solve_power_allocation(
        E_opt,
        config,
        mmse,
    )

    E_opt_reduced, E_star, alpha_opt, alpha_star = (
        reduce_to_two_levels(
            candidate_energies,
            candidate_weights,
        )
    )

    if not np.isclose(E_opt_reduced, E_opt):
        raise RuntimeError("Unexpected LP support below P_opt.")

    E_mix = alpha_opt * E_opt + alpha_star * E_star

    eta = np.linspace(
        config.eta_plot_min,
        config.eta_plot_max,
        config.eta_plot_points,
        dtype=np.float64,
    )

    g_opt_values = g_flat(eta, E_opt, config, mmse)
    g_alg_values = g_flat(eta, E_alg, config, mmse)
    g_star_values = g_flat(eta, E_star, config, mmse)
    g_mix_values = g_mixture(
        eta,
        np.array([E_opt, E_star]),
        np.array([alpha_opt, alpha_star]),
        config,
        mmse,
    )

    return Fig8Solution(
        config=config,
        eta=eta,
        g_opt=g_opt_values,
        g_alg=g_alg_values,
        g_star=g_star_values,
        g_mix=g_mix_values,
        G_opt=normalized_integral(eta, g_opt_values),
        G_alg=normalized_integral(eta, g_alg_values),
        G_star=normalized_integral(eta, g_star_values),
        G_mix=normalized_integral(eta, g_mix_values),
        E_opt=E_opt,
        E_alg=E_alg,
        E_star=E_star,
        E_mix=E_mix,
        alpha_opt=alpha_opt,
        alpha_star=alpha_star,
        candidate_energies=candidate_energies,
        candidate_weights=candidate_weights,
    )


def _config_signature(config: Fig8Config) -> dict[str, float | int]:
    return {
        key: value
        for key, value in config.__dict__.items()
    }


def save_solution(solution: Fig8Solution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "cache_version": CACHE_VERSION,
        "config": _config_signature(solution.config),
    }

    np.savez_compressed(
        path,
        metadata=np.array(json.dumps(metadata)),
        eta=solution.eta,
        g_opt=solution.g_opt,
        g_alg=solution.g_alg,
        g_star=solution.g_star,
        g_mix=solution.g_mix,
        G_opt=solution.G_opt,
        G_alg=solution.G_alg,
        G_star=solution.G_star,
        G_mix=solution.G_mix,
        E_opt=np.array(solution.E_opt),
        E_alg=np.array(solution.E_alg),
        E_star=np.array(solution.E_star),
        E_mix=np.array(solution.E_mix),
        alpha_opt=np.array(solution.alpha_opt),
        alpha_star=np.array(solution.alpha_star),
        candidate_energies=solution.candidate_energies,
        candidate_weights=solution.candidate_weights,
    )


def load_solution(
    path: Path,
    config: Fig8Config,
) -> Fig8Solution | None:
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("cache_version") != CACHE_VERSION:
                return None
            if metadata.get("config") != _config_signature(config):
                return None

            return Fig8Solution(
                config=config,
                eta=np.asarray(data["eta"]),
                g_opt=np.asarray(data["g_opt"]),
                g_alg=np.asarray(data["g_alg"]),
                g_star=np.asarray(data["g_star"]),
                g_mix=np.asarray(data["g_mix"]),
                G_opt=np.asarray(data["G_opt"]),
                G_alg=np.asarray(data["G_alg"]),
                G_star=np.asarray(data["G_star"]),
                G_mix=np.asarray(data["G_mix"]),
                E_opt=float(data["E_opt"]),
                E_alg=float(data["E_alg"]),
                E_star=float(data["E_star"]),
                E_mix=float(data["E_mix"]),
                alpha_opt=float(data["alpha_opt"]),
                alpha_star=float(data["alpha_star"]),
                candidate_energies=np.asarray(
                    data["candidate_energies"]
                ),
                candidate_weights=np.asarray(
                    data["candidate_weights"]
                ),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def get_solution(
    *,
    cache_path: Path,
    recompute: bool = False,
    config: Fig8Config | None = None,
) -> Fig8Solution:
    config = config or Fig8Config()

    if not recompute:
        cached = load_solution(cache_path, config)
        if cached is not None:
            return cached

    solution = compute_solution(config)
    save_solution(solution, cache_path)
    return solution


def save_curve_csv(solution: Fig8Solution, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack(
        [
            solution.eta,
            solution.g_opt,
            solution.g_alg,
            solution.g_star,
            solution.g_mix,
            solution.G_opt,
            solution.G_alg,
            solution.G_star,
            solution.G_mix,
        ]
    )
    np.savetxt(
        path,
        matrix,
        delimiter=",",
        header=(
            "eta,g_Popt,g_Palg,g_Pstar,g_mixture,"
            "G_Popt,G_Palg,G_Pstar,G_mixture"
        ),
        comments="",
        fmt="%.18e",
    )


def print_summary(solution: Fig8Solution) -> None:
    config = solution.config

    print("Figure 8 numerical reproduction")
    print("--------------------------------")
    print(
        f"K_a={config.Ka}, J={config.J}, "
        f"R_in={config.R_in:.6f}"
    )
    print(
        f"beta={config.beta:.8f}, "
        f"target={config.target_strength_db:.2f} dB"
    )
    print(
        f"epsilon={config.epsilon:.4f}, "
        f"delta={config.delta:.4f}"
    )
    print(f"E_opt  = {solution.E_opt_db:.4f} dB")
    print(f"E_alg  = {solution.E_alg_db:.4f} dB")
    print(
        f"E_star = {solution.E_star_db:.4f} dB "
        f"= {solution.E_star / solution.E_opt:.4f} E_opt"
    )
    print(
        f"alpha_opt={solution.alpha_opt:.6f}, "
        f"alpha_star={solution.alpha_star:.6f}"
    )
    print(f"E_mix  = {solution.E_mix_db:.4f} dB")
    print(
        f"gain E_alg-E_mix = "
        f"{solution.E_alg_db - solution.E_mix_db:.4f} dB"
    )

