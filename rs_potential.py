from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar
from scipy.special import expit, xlogy


__all__ = [
    "RSMinimum",
    "binary_entropy",
    "mutual_information_or",
    "rs_potential_or",
    "minimize_rs_potential",
]


@dataclass(frozen=True)
class RSMinimum:
    eta: float
    potential: float
    local_minima_eta: np.ndarray
    local_minima_potential: np.ndarray


def binary_entropy(p):
    p = np.asarray(p, dtype=float)

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(
            "Probabilities must lie between zero and one."
        )

    result = -(
        xlogy(p, p)
        + xlogy(1.0 - p, 1.0 - p)
    ) / np.log(2.0)

    if result.ndim == 0:
        return float(result)

    return result


@lru_cache(maxsize=None)
def _gauss_hermite_rule(order):
    if order < 1:
        raise ValueError(
            "The Gauss-Hermite order must be positive."
        )

    nodes, weights = hermgauss(order)

    return (
        np.sqrt(2.0) * nodes,
        weights / np.sqrt(np.pi),
    )


def mutual_information_or(
    channel_strength,
    q_active,
    *,
    gh_order=60,
):
    channel_strength = np.asarray(
        channel_strength,
        dtype=float,
    )

    q_active = float(q_active)

    if np.any(channel_strength < 0.0):
        raise ValueError(
            "channel_strength must be nonnegative."
        )

    if not 0.0 < q_active < 1.0:
        raise ValueError(
            "q_active must lie strictly between zero and one."
        )

    nodes, weights = _gauss_hermite_rule(
        gh_order
    )

    sqrt_strength = np.sqrt(
        channel_strength
    )[..., None]

    strength = channel_strength[..., None]

    log_prior_odds = (
        np.log(q_active)
        - np.log1p(-q_active)
    )

    llr_given_zero = (
        log_prior_odds
        + sqrt_strength * nodes
        - 0.5 * strength
    )

    llr_given_one = (
        log_prior_odds
        + sqrt_strength * nodes
        + 0.5 * strength
    )

    posterior_given_zero = expit(
        llr_given_zero
    )

    posterior_given_one = expit(
        llr_given_one
    )

    entropy_given_zero = np.sum(
        weights
        * binary_entropy(
            posterior_given_zero
        ),
        axis=-1,
    )

    entropy_given_one = np.sum(
        weights
        * binary_entropy(
            posterior_given_one
        ),
        axis=-1,
    )

    input_entropy = binary_entropy(
        q_active
    )

    conditional_entropy = (
        (1.0 - q_active)
        * entropy_given_zero
        + q_active
        * entropy_given_one
    )

    result = np.clip(
        input_entropy
        - conditional_entropy,
        0.0,
        input_entropy,
    )

    if result.ndim == 0:
        return float(result)

    return result


def rs_potential_or(
    eta,
    P_hat,
    beta,
    q_active,
    *,
    gh_order=60,
):
    eta = np.asarray(
        eta,
        dtype=float,
    )

    P_hat = float(P_hat)
    beta = float(beta)
    q_active = float(q_active)

    if P_hat < 0.0:
        raise ValueError(
            "P_hat must be nonnegative."
        )

    if beta <= 0.0:
        raise ValueError(
            "beta must be positive."
        )

    if not 0.0 < q_active < 1.0:
        raise ValueError(
            "q_active must lie strictly between zero and one."
        )

    result = np.full_like(
        eta,
        np.inf,
        dtype=float,
    )

    valid = (
        (eta > 0.0)
        & (eta <= 1.0)
    )

    if np.any(valid):
        eta_valid = eta[valid]

        information_term = (
            mutual_information_or(
                eta_valid * P_hat,
                q_active,
                gh_order=gh_order,
            )
        )

        measurement_term = (
            (
                (eta_valid - 1.0)
                * np.log2(np.e)
                - np.log2(eta_valid)
            )
            / (2.0 * beta)
        )

        result[valid] = (
            information_term
            + measurement_term
        )

    if result.ndim == 0:
        return float(result)

    return result


def _make_eta_grid(
    eta_min,
    log_grid_points,
    linear_grid_points,
):
    if not 0.0 < eta_min < 0.1:
        raise ValueError(
            "eta_min must lie between zero and 0.1."
        )

    if log_grid_points < 3:
        raise ValueError(
            "log_grid_points must be at least three."
        )

    if linear_grid_points < 3:
        raise ValueError(
            "linear_grid_points must be at least three."
        )

    return np.unique(
        np.concatenate(
            (
                np.logspace(
                    np.log10(eta_min),
                    -1.0,
                    log_grid_points,
                ),
                np.linspace(
                    0.1,
                    1.0,
                    linear_grid_points,
                ),
            )
        )
    )


def _deduplicate_minima(
    candidate_etas,
    candidate_potentials,
    tolerance,
):
    order = np.argsort(
        candidate_etas
    )

    candidate_etas = np.asarray(
        candidate_etas,
        dtype=float,
    )[order]

    candidate_potentials = np.asarray(
        candidate_potentials,
        dtype=float,
    )[order]

    unique_etas = []
    unique_potentials = []

    for eta, potential in zip(
        candidate_etas,
        candidate_potentials,
    ):
        if not unique_etas:
            unique_etas.append(eta)
            unique_potentials.append(
                potential
            )
            continue

        if abs(
            eta - unique_etas[-1]
        ) <= tolerance:
            if (
                potential
                < unique_potentials[-1]
            ):
                unique_etas[-1] = eta
                unique_potentials[-1] = (
                    potential
                )
        else:
            unique_etas.append(eta)
            unique_potentials.append(
                potential
            )

    return (
        np.asarray(
            unique_etas,
            dtype=float,
        ),
        np.asarray(
            unique_potentials,
            dtype=float,
        ),
    )


def minimize_rs_potential(
    P_hat,
    beta,
    q_active,
    *,
    selection="global",
    gh_order=60,
    eta_min=1e-8,
    log_grid_points=250,
    linear_grid_points=600,
    optimizer_tolerance=1e-10,
    tie_tolerance=1e-10,
):
    P_hat = float(P_hat)
    beta = float(beta)
    q_active = float(q_active)

    if P_hat < 0.0:
        raise ValueError(
            "P_hat must be nonnegative."
        )

    if beta <= 0.0:
        raise ValueError(
            "beta must be positive."
        )

    if not 0.0 < q_active < 1.0:
        raise ValueError(
            "q_active must lie strictly between zero and one."
        )

    if selection not in {
        "global",
        "first",
    }:
        raise ValueError(
            "selection must be either "
            "'global' or 'first'."
        )

    if optimizer_tolerance <= 0.0:
        raise ValueError(
            "optimizer_tolerance must be positive."
        )

    if tie_tolerance < 0.0:
        raise ValueError(
            "tie_tolerance must be nonnegative."
        )

    eta_grid = _make_eta_grid(
        eta_min,
        log_grid_points,
        linear_grid_points,
    )

    potential_grid = rs_potential_or(
        eta_grid,
        P_hat,
        beta,
        q_active,
        gh_order=gh_order,
    )

    interior_indices = np.flatnonzero(
        (
            potential_grid[1:-1]
            <= potential_grid[:-2]
        )
        & (
            potential_grid[1:-1]
            <= potential_grid[2:]
        )
    ) + 1

    candidate_indices = list(
        interior_indices
    )

    if (
        potential_grid[-1]
        <= potential_grid[-2]
    ):
        candidate_indices.append(
            len(eta_grid) - 1
        )

    if not candidate_indices:
        minimum_grid_index = int(
            np.argmin(potential_grid)
        )

        if minimum_grid_index == 0:
            raise RuntimeError(
                "No local minimum was found "
                "above eta_min. Reduce eta_min "
                "or increase the grid resolution."
            )

        candidate_indices.append(
            minimum_grid_index
        )

    candidate_indices = np.unique(
        np.asarray(
            candidate_indices,
            dtype=int,
        )
    )

    candidate_etas = []
    candidate_potentials = []

    def objective(eta_value):
        return rs_potential_or(
            eta_value,
            P_hat,
            beta,
            q_active,
            gh_order=gh_order,
        )

    for index in candidate_indices:
        if index == len(eta_grid) - 1:
            eta_candidate = float(
                eta_grid[index]
            )

            potential_candidate = float(
                potential_grid[index]
            )
        else:
            lower_index = max(
                index - 1,
                0,
            )

            upper_index = min(
                index + 1,
                len(eta_grid) - 1,
            )

            result = minimize_scalar(
                objective,
                bounds=(
                    float(
                        eta_grid[
                            lower_index
                        ]
                    ),
                    float(
                        eta_grid[
                            upper_index
                        ]
                    ),
                ),
                method="bounded",
                options={
                    "xatol":
                        optimizer_tolerance,
                },
            )

            if result.success:
                eta_candidate = float(
                    result.x
                )

                potential_candidate = float(
                    result.fun
                )
            else:
                eta_candidate = float(
                    eta_grid[index]
                )

                potential_candidate = float(
                    potential_grid[index]
                )

        candidate_etas.append(
            eta_candidate
        )

        candidate_potentials.append(
            potential_candidate
        )

    (
        candidate_etas,
        candidate_potentials,
    ) = _deduplicate_minima(
        candidate_etas,
        candidate_potentials,
        tolerance=max(
            10.0
            * optimizer_tolerance,
            1e-12,
        ),
    )

    if selection == "first":
        best_index = int(
            np.argmin(candidate_etas)
        )

    else:
        minimum_potential = float(
            np.min(
                candidate_potentials
            )
        )

        tied_indices = np.flatnonzero(
            candidate_potentials
            <= (
                minimum_potential
                + tie_tolerance
            )
        )

        best_index = int(
            tied_indices[
                np.argmax(
                    candidate_etas[
                        tied_indices
                    ]
                )
            ]
        )

    return RSMinimum(
        eta=float(
            candidate_etas[
                best_index
            ]
        ),
        potential=float(
            candidate_potentials[
                best_index
            ]
        ),
        local_minima_eta=(
            candidate_etas
        ),
        local_minima_potential=(
            candidate_potentials
        ),
    )