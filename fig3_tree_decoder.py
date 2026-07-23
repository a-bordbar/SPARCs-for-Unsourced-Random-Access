from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import argparse
import concurrent.futures
import math
import os

import numpy as np

import utils


L = 8
J_vec = np.array([12, 15, 20])
Ka_vec = np.arange(25, 301, 25)

DEBUG_MODE = False
if DEBUG_MODE:
    J_vec = np.array([12])
    Ka_vec = np.array([25, 50])

TARGET_PUPE = 0.05
MAX_PATHS = 1_000_000
PROGRESS_FILE = Path("tree_code_progress.npz")
LIST_MODES = ("unique_support", "with_multiplicity")
OUTPUT_SELECTION_MODES = ("random", "expected")
REMAINDER_ALLOCATIONS = ("early", "late")


@dataclass
class TreeTrialResult:
    pupe: float
    number_recovered: float
    number_transmitted: int
    number_decoded_messages: int
    number_of_roots: int
    number_of_cap_hits: int
    max_paths_observed: int
    average_final_paths_per_root: float
    number_of_ambiguous_roots: int
    number_of_zero_path_roots: int
    total_surviving_paths_before_deduplication: int
    number_of_unique_surviving_messages_before_truncation: int
    number_of_messages_after_truncation: int
    truncation_applied: bool
    number_of_unique_roots: int
    number_of_colliding_users: int
    shared_root_user_fraction: float


@dataclass
class SimulationResult:
    mean_pupe: float
    parity_profile: np.ndarray
    info_profile: np.ndarray
    B_eff: int
    total_cap_hits: int
    cap_hit_fraction: float
    maximum_paths_observed: int
    average_decoded_messages: float
    average_roots: float
    average_final_paths_per_root: float
    average_ambiguous_roots: float
    average_zero_path_roots: float
    average_surviving_paths_before_deduplication: float
    average_unique_surviving_messages_before_truncation: float
    average_messages_after_truncation: float
    truncation_fraction: float
    average_unique_roots: float
    average_colliding_users: float
    average_shared_root_user_fraction: float


@dataclass
class SearchResult:
    rate: float
    total_parity: float
    mean_pupe: float
    parity_profile: Optional[np.ndarray]
    B_eff: float
    cap_hit_fraction: float
    maximum_paths_observed: float
    average_ambiguous_roots: float
    average_unique_surviving_messages_before_truncation: float
    average_messages_after_truncation: float
    average_shared_root_user_fraction: float
    evaluated_points: dict
    reason: str


def H2(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    out[mask] = -p[mask] * np.log2(p[mask]) - (1.0 - p[mask]) * np.log2(1.0 - p[mask])
    return out


def make_profiles(L_value, J, total_parity, remainder_allocation="late"):
    parity_bits = utils.make_parity_profile(
        L_value,
        J,
        total_parity,
        remainder_allocation=remainder_allocation,
    )
    info_bits = J - parity_bits
    B_eff = int(np.sum(info_bits))
    assert B_eff == L_value * J - total_parity
    return info_bits, parity_bits, B_eff


def resolve_output_limit(output_list_limit, Ka):
    if output_list_limit is None:
        return None
    if isinstance(output_list_limit, str):
        normalized = output_list_limit.lower()
        if normalized == "none":
            return None
        if normalized == "ka":
            return int(Ka)
        return int(output_list_limit)
    return int(output_list_limit)


def validate_modes(list_mode, output_selection_mode, remainder_allocation="late"):
    if list_mode not in LIST_MODES:
        raise ValueError(f"list_mode must be one of {LIST_MODES}.")
    if output_selection_mode not in OUTPUT_SELECTION_MODES:
        raise ValueError(f"output_selection_mode must be one of {OUTPUT_SELECTION_MODES}.")
    if remainder_allocation not in REMAINDER_ALLOCATIONS:
        raise ValueError(f"remainder_allocation must be one of {REMAINDER_ALLOCATIONS}.")


def build_candidates(section_indices, info_bits, parity_bits, J, list_mode="unique_support"):
    if list_mode not in LIST_MODES:
        raise ValueError(f"list_mode must be one of {LIST_MODES}.")

    candidate_information = []
    candidate_parity = []

    for l, indices in enumerate(section_indices):
        if list_mode == "unique_support":
            S_l = np.unique(section_indices[l])
        elif list_mode == "with_multiplicity":
            S_l = np.asarray(section_indices[l])
        else:
            raise ValueError(f"Unsupported list_mode={list_mode!r}.")

        fragments = utils.int_to_bits(S_l, J)
        candidate_information.append(fragments[:, : info_bits[l]])
        candidate_parity.append(fragments[:, info_bits[l] :])

    return candidate_information, candidate_parity


def transmitted_message_matrix(information_blocks):
    return np.concatenate(information_blocks, axis=1)


def root_collision_diagnostics(first_indices, Ka):
    unique_roots, counts = np.unique(first_indices, return_counts=True)
    number_of_unique_roots = len(unique_roots)
    number_of_colliding_users = Ka - number_of_unique_roots
    shared_root_users = int(np.sum(counts[counts > 1]))
    shared_root_user_fraction = shared_root_users / Ka if Ka > 0 else 0.0
    return number_of_unique_roots, number_of_colliding_users, shared_root_user_fraction


def count_recovered_users(decoded_keys, transmitted_keys):
    return sum(key in decoded_keys for key in transmitted_keys)


def apply_global_output_limit(
    all_decoded_keys,
    transmitted_keys,
    Ka,
    max_output_messages,
    output_selection_mode,
    rng,
):
    if output_selection_mode not in OUTPUT_SELECTION_MODES:
        raise ValueError(f"output_selection_mode must be one of {OUTPUT_SELECTION_MODES}.")

    M = len(all_decoded_keys)
    surviving_transmitted_user_count = count_recovered_users(all_decoded_keys, transmitted_keys)

    if max_output_messages is None:
        return (
            1.0 - surviving_transmitted_user_count / Ka,
            float(surviving_transmitted_user_count),
            M,
            False,
            all_decoded_keys,
        )

    max_output_messages = int(max_output_messages)
    if max_output_messages < 0:
        raise ValueError("max_output_messages must be nonnegative or None.")

    truncation_applied = M > max_output_messages
    number_after_truncation = min(M, max_output_messages)

    if output_selection_mode == "expected":
        if M <= max_output_messages:
            expected_number_recovered = float(surviving_transmitted_user_count)
        elif M == 0:
            expected_number_recovered = 0.0
        else:
            expected_number_recovered = surviving_transmitted_user_count * max_output_messages / M

        return (
            1.0 - expected_number_recovered / Ka,
            expected_number_recovered,
            number_after_truncation,
            truncation_applied,
            None,
        )

    decoded_key_list = sorted(all_decoded_keys)
    if not truncation_applied:
        final_decoded_keys = set(decoded_key_list)
    else:
        selected_indices = rng.choice(
            len(decoded_key_list),
            size=max_output_messages,
            replace=False,
        )
        final_decoded_keys = {
            decoded_key_list[index]
            for index in selected_indices
        }

    number_recovered = count_recovered_users(final_decoded_keys, transmitted_keys)
    return (
        1.0 - number_recovered / Ka,
        float(number_recovered),
        len(final_decoded_keys),
        truncation_applied,
        final_decoded_keys,
    )


def tree_trial(
    Ka,
    J,
    L_value,
    total_parity,
    G,
    H,
    rng,
    max_paths=MAX_PATHS,
    validate_true_paths=False,
    list_mode="unique_support",
    max_output_messages=None,
    output_selection_mode="expected",
    remainder_allocation="late",
):
    validate_modes(list_mode, output_selection_mode, remainder_allocation)
    info_bits, parity_bits, _ = make_profiles(
        L_value,
        J,
        total_parity,
        remainder_allocation=remainder_allocation,
    )
    information_blocks, section_indices = utils.encode_users(
        Ka,
        J,
        info_bits,
        parity_bits,
        G,
        rng,
    )

    candidate_information, candidate_parity = build_candidates(
        section_indices,
        info_bits,
        parity_bits,
        J,
        list_mode=list_mode,
    )

    transmitted_messages = transmitted_message_matrix(information_blocks)
    transmitted_keys = [utils.message_key(row) for row in transmitted_messages]

    all_decoded_keys = set()
    number_of_cap_hits = 0
    max_paths_observed = 0
    final_path_counts = []
    number_of_ambiguous_roots = 0
    number_of_zero_path_roots = 0
    total_surviving_paths_before_deduplication = 0
    number_of_roots = candidate_information[0].shape[0]

    (
        number_of_unique_roots,
        number_of_colliding_users,
        shared_root_user_fraction,
    ) = root_collision_diagnostics(section_indices[0], Ka)

    for root_information in candidate_information[0]:
        decode_result = utils.decode_one_root(
            root_information,
            candidate_information,
            candidate_parity,
            H,
            max_paths=max_paths,
        )

        if decode_result.path_counts:
            max_paths_observed = max(max_paths_observed, max(decode_result.path_counts))

        if decode_result.cap_hit:
            number_of_cap_hits += 1
            final_path_counts.append(np.nan)
            continue

        number_of_paths = decode_result.paths.shape[0]
        final_path_counts.append(number_of_paths)
        total_surviving_paths_before_deduplication += number_of_paths

        if number_of_paths == 0:
            number_of_zero_path_roots += 1
        elif number_of_paths > 1:
            number_of_ambiguous_roots += 1

        for path in decode_result.paths:
            all_decoded_keys.add(utils.message_key(path))

    if validate_true_paths:
        for user_idx in range(Ka):
            root_information = information_blocks[0][user_idx]
            genuine_key = utils.message_key(transmitted_messages[user_idx])
            decode_result = utils.decode_one_root(
                root_information,
                candidate_information,
                candidate_parity,
                H,
                max_paths=max_paths,
            )

            if decode_result.path_counts:
                max_paths_observed = max(max_paths_observed, max(decode_result.path_counts))

            if decode_result.cap_hit:
                continue

            surviving_keys = {
                utils.message_key(path)
                for path in decode_result.paths
            }
            assert genuine_key in surviving_keys, (
                f"Genuine path removed for user_idx={user_idx}, "
                f"Ka={Ka}, J={J}, L={L_value}, total_parity={total_parity}, "
                f"path_counts={decode_result.path_counts}."
            )

    number_of_unique_surviving_messages = len(all_decoded_keys)
    pupe, number_recovered, number_after_truncation, truncation_applied, _ = apply_global_output_limit(
        all_decoded_keys,
        transmitted_keys,
        Ka,
        max_output_messages,
        output_selection_mode,
        rng,
    )

    if final_path_counts:
        average_final_paths_per_root = float(np.nanmean(final_path_counts))
    else:
        average_final_paths_per_root = float("nan")

    return TreeTrialResult(
        pupe=pupe,
        number_recovered=number_recovered,
        number_transmitted=Ka,
        number_decoded_messages=number_after_truncation,
        number_of_roots=number_of_roots,
        number_of_cap_hits=number_of_cap_hits,
        max_paths_observed=max_paths_observed,
        average_final_paths_per_root=average_final_paths_per_root,
        number_of_ambiguous_roots=number_of_ambiguous_roots,
        number_of_zero_path_roots=number_of_zero_path_roots,
        total_surviving_paths_before_deduplication=total_surviving_paths_before_deduplication,
        number_of_unique_surviving_messages_before_truncation=number_of_unique_surviving_messages,
        number_of_messages_after_truncation=number_after_truncation,
        truncation_applied=truncation_applied,
        number_of_unique_roots=number_of_unique_roots,
        number_of_colliding_users=number_of_colliding_users,
        shared_root_user_fraction=shared_root_user_fraction,
    )


def summarize_trial_results(trial_results, parity_bits, info_bits, B_eff):
    total_roots = sum(result.number_of_roots for result in trial_results)
    total_cap_hits = sum(result.number_of_cap_hits for result in trial_results)
    cap_hit_fraction = total_cap_hits / total_roots if total_roots > 0 else 0.0

    return SimulationResult(
        mean_pupe=float(np.mean([result.pupe for result in trial_results])),
        parity_profile=parity_bits,
        info_profile=info_bits,
        B_eff=B_eff,
        total_cap_hits=total_cap_hits,
        cap_hit_fraction=float(cap_hit_fraction),
        maximum_paths_observed=max(result.max_paths_observed for result in trial_results),
        average_decoded_messages=float(np.mean([result.number_decoded_messages for result in trial_results])),
        average_roots=float(np.mean([result.number_of_roots for result in trial_results])),
        average_final_paths_per_root=float(np.nanmean([result.average_final_paths_per_root for result in trial_results])),
        average_ambiguous_roots=float(np.mean([result.number_of_ambiguous_roots for result in trial_results])),
        average_zero_path_roots=float(np.mean([result.number_of_zero_path_roots for result in trial_results])),
        average_surviving_paths_before_deduplication=float(
            np.mean([result.total_surviving_paths_before_deduplication for result in trial_results])
        ),
        average_unique_surviving_messages_before_truncation=float(
            np.mean([result.number_of_unique_surviving_messages_before_truncation for result in trial_results])
        ),
        average_messages_after_truncation=float(
            np.mean([result.number_of_messages_after_truncation for result in trial_results])
        ),
        truncation_fraction=float(np.mean([result.truncation_applied for result in trial_results])),
        average_unique_roots=float(np.mean([result.number_of_unique_roots for result in trial_results])),
        average_colliding_users=float(np.mean([result.number_of_colliding_users for result in trial_results])),
        average_shared_root_user_fraction=float(np.mean([result.shared_root_user_fraction for result in trial_results])),
    )


def run_trial_batch(batch_args):
    (
        Ka,
        J,
        L_value,
        total_parity,
        number_of_trials,
        code_seed,
        trial_seed,
        max_paths,
        validate_true_paths,
        list_mode,
        output_list_limit,
        output_selection_mode,
        remainder_allocation,
    ) = batch_args

    info_bits, parity_bits, B_eff = make_profiles(
        L_value,
        J,
        total_parity,
        remainder_allocation=remainder_allocation,
    )
    code_rng = np.random.default_rng(code_seed)
    G, H = utils.make_parity_matrices(info_bits, parity_bits, code_rng)
    trial_rng = np.random.default_rng(trial_seed)
    max_output_messages = resolve_output_limit(output_list_limit, Ka)

    trial_results = [
        tree_trial(
            Ka,
            J,
            L_value,
            total_parity,
            G,
            H,
            trial_rng,
            max_paths=max_paths,
            validate_true_paths=validate_true_paths,
            list_mode=list_mode,
            max_output_messages=max_output_messages,
            output_selection_mode=output_selection_mode,
            remainder_allocation=remainder_allocation,
        )
        for _ in range(number_of_trials)
    ]

    return summarize_trial_results(trial_results, parity_bits, info_bits, B_eff)


def split_trials(number_of_trials, workers):
    workers = max(1, min(workers, number_of_trials))
    base, remainder = divmod(number_of_trials, workers)
    return [
        base + (worker_idx < remainder)
        for worker_idx in range(workers)
        if base + (worker_idx < remainder) > 0
    ]


def weighted_average(results, weights, field):
    return float(
        sum(getattr(result, field) * weight for result, weight in zip(results, weights))
        / float(np.sum(weights))
    )


def aggregate_weighted_simulation_results(results, weights):
    first = results[0]
    weights = np.asarray(weights, dtype=float)
    total_roots = sum(result.average_roots * weight for result, weight in zip(results, weights))
    total_cap_hits = sum(result.total_cap_hits for result in results)
    cap_hit_fraction = total_cap_hits / total_roots if total_roots > 0 else 0.0

    final_path_values = np.array([result.average_final_paths_per_root for result in results], dtype=float)
    final_path_mask = np.isfinite(final_path_values)
    if np.any(final_path_mask):
        average_final_paths_per_root = float(
            np.average(final_path_values[final_path_mask], weights=weights[final_path_mask])
        )
    else:
        average_final_paths_per_root = float("nan")

    return SimulationResult(
        mean_pupe=weighted_average(results, weights, "mean_pupe"),
        parity_profile=first.parity_profile,
        info_profile=first.info_profile,
        B_eff=first.B_eff,
        total_cap_hits=total_cap_hits,
        cap_hit_fraction=float(cap_hit_fraction),
        maximum_paths_observed=max(result.maximum_paths_observed for result in results),
        average_decoded_messages=weighted_average(results, weights, "average_decoded_messages"),
        average_roots=weighted_average(results, weights, "average_roots"),
        average_final_paths_per_root=average_final_paths_per_root,
        average_ambiguous_roots=weighted_average(results, weights, "average_ambiguous_roots"),
        average_zero_path_roots=weighted_average(results, weights, "average_zero_path_roots"),
        average_surviving_paths_before_deduplication=weighted_average(
            results,
            weights,
            "average_surviving_paths_before_deduplication",
        ),
        average_unique_surviving_messages_before_truncation=weighted_average(
            results,
            weights,
            "average_unique_surviving_messages_before_truncation",
        ),
        average_messages_after_truncation=weighted_average(results, weights, "average_messages_after_truncation"),
        truncation_fraction=weighted_average(results, weights, "truncation_fraction"),
        average_unique_roots=weighted_average(results, weights, "average_unique_roots"),
        average_colliding_users=weighted_average(results, weights, "average_colliding_users"),
        average_shared_root_user_fraction=weighted_average(results, weights, "average_shared_root_user_fraction"),
    )


def simulate_tree_pupe(
    Ka,
    J,
    L_value,
    total_parity,
    number_of_trials,
    code_seed=0,
    trial_seed=1,
    max_paths=MAX_PATHS,
    validate_true_paths=False,
    workers=1,
    list_mode="unique_support",
    output_list_limit="Ka",
    output_selection_mode="expected",
    remainder_allocation="late",
):
    validate_modes(list_mode, output_selection_mode, remainder_allocation)
    info_bits, parity_bits, B_eff = make_profiles(
        L_value,
        J,
        total_parity,
        remainder_allocation=remainder_allocation,
    )
    max_output_messages = resolve_output_limit(output_list_limit, Ka)

    if workers <= 1 or number_of_trials == 1:
        code_rng = np.random.default_rng(code_seed)
        G, H = utils.make_parity_matrices(info_bits, parity_bits, code_rng)
        trial_rng = np.random.default_rng(trial_seed)

        trial_results = [
            tree_trial(
                Ka,
                J,
                L_value,
                total_parity,
                G,
                H,
                trial_rng,
                max_paths=max_paths,
                validate_true_paths=validate_true_paths,
                list_mode=list_mode,
                max_output_messages=max_output_messages,
                output_selection_mode=output_selection_mode,
                remainder_allocation=remainder_allocation,
            )
            for _ in range(number_of_trials)
        ]

        return summarize_trial_results(trial_results, parity_bits, info_bits, B_eff)

    batch_sizes = split_trials(number_of_trials, workers)
    batch_args = [
        (
            Ka,
            J,
            L_value,
            total_parity,
            batch_size,
            code_seed,
            trial_seed + 1_000_003 * batch_idx,
            max_paths,
            validate_true_paths,
            list_mode,
            output_list_limit,
            output_selection_mode,
            remainder_allocation,
        )
        for batch_idx, batch_size in enumerate(batch_sizes)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch_args)) as executor:
        batch_results = list(executor.map(run_trial_batch, batch_args))

    return aggregate_weighted_simulation_results(batch_results, batch_sizes)


def log_evaluation(
    stage,
    Ka,
    J,
    L_value,
    total_parity,
    result,
    list_mode,
    output_list_limit,
    output_selection_mode,
    remainder_allocation,
    target_pupe=TARGET_PUPE,
):
    if result.mean_pupe >= target_pupe:
        return

    rate = 1.0 - total_parity / (L_value * J)
    print(
        f"[{stage}] Ka={Ka:3d}, J={J:2d}, P={total_parity:3d}, "
        f"Rout={rate:.4f}, PUPE={result.mean_pupe:.5f}",
        flush=True,
    )


def make_manual_trial_result(
    Ka,
    J,
    L_value,
    total_parity,
    information_blocks,
    section_indices,
    G,
    H,
    rng,
    list_mode="unique_support",
    max_output_messages=None,
    output_selection_mode="expected",
):
    info_bits, parity_bits, _ = make_profiles(L_value, J, total_parity)
    candidate_information, candidate_parity = build_candidates(section_indices, info_bits, parity_bits, J, list_mode)
    transmitted_messages = transmitted_message_matrix(information_blocks)
    transmitted_keys = [utils.message_key(row) for row in transmitted_messages]
    all_decoded_keys = set()
    ambiguous_roots = 0
    zero_path_roots = 0
    surviving_paths = 0
    max_paths_observed = 0

    for root_information in candidate_information[0]:
        decode_result = utils.decode_one_root(root_information, candidate_information, candidate_parity, H, max_paths=100_000)
        max_paths_observed = max(max_paths_observed, max(decode_result.path_counts))
        assert not decode_result.cap_hit
        n_paths = decode_result.paths.shape[0]
        surviving_paths += n_paths
        ambiguous_roots += int(n_paths > 1)
        zero_path_roots += int(n_paths == 0)
        for path in decode_result.paths:
            all_decoded_keys.add(utils.message_key(path))

    pupe, recovered, after_limit, truncation_applied, _ = apply_global_output_limit(
        all_decoded_keys,
        transmitted_keys,
        Ka,
        max_output_messages,
        output_selection_mode,
        rng,
    )
    unique_roots, colliding_users, shared_fraction = root_collision_diagnostics(section_indices[0], Ka)
    return TreeTrialResult(
        pupe=pupe,
        number_recovered=recovered,
        number_transmitted=Ka,
        number_decoded_messages=after_limit,
        number_of_roots=candidate_information[0].shape[0],
        number_of_cap_hits=0,
        max_paths_observed=max_paths_observed,
        average_final_paths_per_root=surviving_paths / candidate_information[0].shape[0],
        number_of_ambiguous_roots=ambiguous_roots,
        number_of_zero_path_roots=zero_path_roots,
        total_surviving_paths_before_deduplication=surviving_paths,
        number_of_unique_surviving_messages_before_truncation=len(all_decoded_keys),
        number_of_messages_after_truncation=after_limit,
        truncation_applied=truncation_applied,
        number_of_unique_roots=unique_roots,
        number_of_colliding_users=colliding_users,
        shared_root_user_fraction=shared_fraction,
    )


def run_validation_tests():
    print("Running validation tests.", flush=True)

    utils.test_bit_conversions()
    print("Bit conversion test passed.", flush=True)

    assert utils.make_parity_profile(
        8,
        12,
        55,
        remainder_allocation="early",
    ).tolist() == [0, 8, 7, 7, 7, 7, 7, 12]
    assert utils.make_parity_profile(
        8,
        12,
        55,
        remainder_allocation="late",
    ).tolist() == [0, 7, 7, 7, 7, 7, 8, 12]
    assert utils.make_parity_profile(
        8,
        12,
        47,
        remainder_allocation="late",
    ).tolist() == [0, 5, 6, 6, 6, 6, 6, 12]
    assert utils.make_parity_profile(
        8,
        12,
        54,
        remainder_allocation="late",
    ).tolist() == [0, 7, 7, 7, 7, 7, 7, 12]
    assert utils.make_parity_profile(
        8,
        20,
        71,
        remainder_allocation="late",
    ).tolist() == [0, 8, 8, 8, 9, 9, 9, 20]
    print("Parity remainder-allocation tests passed.", flush=True)

    one_user = simulate_tree_pupe(
        Ka=1,
        J=4,
        L_value=3,
        total_parity=6,
        number_of_trials=100,
        code_seed=11,
        trial_seed=12,
        max_paths=100_000,
        validate_true_paths=True,
        workers=1,
        list_mode="unique_support",
        output_list_limit="Ka",
        output_selection_mode="random",
    )
    assert one_user.parity_profile.tolist() == [0, 2, 4]
    assert one_user.info_profile.tolist() == [4, 2, 0]
    assert one_user.mean_pupe == 0.0
    assert one_user.cap_hit_fraction == 0.0
    print("Test A passed: one-user PUPE is zero.", flush=True)

    Ka = 2
    J = 2
    L_value = 3
    total_parity = 2
    info_bits, parity_bits, _ = make_profiles(L_value, J, total_parity)
    G = [[None for _ in range(L_value)] for _ in range(L_value)]
    H = [None for _ in range(L_value)]
    G[0][1] = np.zeros((2, 0), dtype=np.uint8)
    G[0][2] = np.zeros((2, 2), dtype=np.uint8)
    G[1][2] = np.zeros((2, 2), dtype=np.uint8)
    H[1] = np.zeros((2, 0), dtype=np.uint8)
    H[2] = np.zeros((4, 2), dtype=np.uint8)
    information_blocks = [
        np.array([[0, 0], [0, 0]], dtype=np.uint8),
        np.array([[0, 1], [1, 0]], dtype=np.uint8),
        np.empty((2, 0), dtype=np.uint8),
    ]
    section_indices = [
        utils.bits_to_int(information_blocks[0]),
        utils.bits_to_int(information_blocks[1]),
        np.array([0, 0], dtype=np.uint64),
    ]
    candidate_information, candidate_parity = build_candidates(section_indices, info_bits, parity_bits, J)
    decode_result = utils.decode_one_root(candidate_information[0][0], candidate_information, candidate_parity, H)
    assert decode_result.paths.shape[0] >= 2
    manual_result = make_manual_trial_result(
        Ka,
        J,
        L_value,
        total_parity,
        information_blocks,
        section_indices,
        G,
        H,
        np.random.default_rng(7),
        max_output_messages=None,
        output_selection_mode="expected",
    )
    assert manual_result.number_of_unique_surviving_messages_before_truncation == decode_result.paths.shape[0]
    assert manual_result.number_of_ambiguous_roots == 1
    print("Test B passed: multiple surviving paths are retained.", flush=True)

    assert manual_result.pupe == 0.0
    assert manual_result.shared_root_user_fraction == 1.0
    assert manual_result.number_of_colliding_users == 1
    print("Test C passed: shared first root is not an automatic failure.", flush=True)

    all_decoded_keys = {b"a", b"b", b"c", b"d"}
    transmitted_keys = [b"a", b"b"]
    pupe, recovered, after_limit, truncated, _ = apply_global_output_limit(
        all_decoded_keys,
        transmitted_keys,
        Ka=2,
        max_output_messages=2,
        output_selection_mode="expected",
        rng=np.random.default_rng(9),
    )
    assert truncated
    assert after_limit == 2
    assert recovered == 1.0
    assert pupe == 0.5
    print("Test D passed: global expected truncation is analytical.", flush=True)

    original_simulate = globals()["simulate_tree_pupe"]

    def fake_simulate(
        Ka,
        J,
        L_value,
        total_parity,
        number_of_trials,
        code_seed=0,
        trial_seed=1,
        max_paths=MAX_PATHS,
        validate_true_paths=False,
        workers=1,
        list_mode="unique_support",
        output_list_limit="Ka",
        output_selection_mode="expected",
        remainder_allocation="late",
    ):
        info_bits, parity_bits, B_eff = make_profiles(
            L_value,
            J,
            total_parity,
            remainder_allocation=remainder_allocation,
        )
        initial = int(round(J + (L_value - 2) * math.log2(Ka)))
        threshold = initial if code_seed >= 10_000 else initial + 2
        return SimulationResult(
            mean_pupe=0.01 if total_parity >= threshold else 0.2,
            parity_profile=parity_bits,
            info_profile=info_bits,
            B_eff=B_eff,
            total_cap_hits=0,
            cap_hit_fraction=0.0,
            maximum_paths_observed=1,
            average_decoded_messages=Ka,
            average_roots=Ka,
            average_final_paths_per_root=1.0,
            average_ambiguous_roots=0.0,
            average_zero_path_roots=0.0,
            average_surviving_paths_before_deduplication=Ka,
            average_unique_surviving_messages_before_truncation=Ka,
            average_messages_after_truncation=Ka,
            truncation_fraction=0.0,
            average_unique_roots=Ka,
            average_colliding_users=0.0,
            average_shared_root_user_fraction=0.0,
        )

    globals()["simulate_tree_pupe"] = fake_simulate
    try:
        search_result = search_tree_rate(
            Ka=8,
            J=4,
            L_value=4,
            target_pupe=0.05,
            number_of_trials_coarse=1,
            number_of_trials_final=1,
            verification_code_seeds=[1],
            workers=1,
            list_mode="unique_support",
            output_list_limit="Ka",
            output_selection_mode="expected",
            remainder_allocation="late",
        )
        expected_threshold = int(round(4 + (4 - 2) * math.log2(8))) + 2
        assert search_result.total_parity == expected_threshold
    finally:
        globals()["simulate_tree_pupe"] = original_simulate
    print("Test E passed: final search continues upward.", flush=True)

    print("All validation tests passed.", flush=True)


def aggregate_verification_results(results):
    return aggregate_weighted_simulation_results(results, [1] * len(results))


def search_tree_rate(
    Ka,
    J,
    L_value=L,
    target_pupe=TARGET_PUPE,
    number_of_trials_coarse=20,
    number_of_trials_final=500,
    verification_code_seeds=None,
    max_paths=MAX_PATHS,
    require_zero_cap_hits=True,
    workers=1,
    list_mode="unique_support",
    output_list_limit="Ka",
    output_selection_mode="expected",
    remainder_allocation="late",
):
    validate_modes(list_mode, output_selection_mode, remainder_allocation)
    if verification_code_seeds is None:
        verification_code_seeds = [1001, 2002, 3003]

    minimum_parity = J
    maximum_parity = (L_value - 1) * J
    initial = int(round(J + (L_value - 2) * math.log2(Ka)))
    initial = int(np.clip(initial, minimum_parity, maximum_parity))

    coarse_cache = {}

    def is_acceptable(result):
        return (
            result.mean_pupe < target_pupe
            and (not require_zero_cap_hits or result.cap_hit_fraction == 0.0)
        )

    def eval_coarse(total_parity):
        if total_parity not in coarse_cache:
            result = simulate_tree_pupe(
                Ka,
                J,
                L_value,
                total_parity,
                number_of_trials=number_of_trials_coarse,
                code_seed=10_000 + 100 * J + Ka + total_parity,
                trial_seed=20_000 + 100 * J + Ka + total_parity,
                max_paths=max_paths,
                workers=workers,
                list_mode=list_mode,
                output_list_limit=output_list_limit,
                output_selection_mode=output_selection_mode,
                remainder_allocation=remainder_allocation,
            )
            coarse_cache[total_parity] = result
            log_evaluation(
                "coarse",
                Ka,
                J,
                L_value,
                total_parity,
                result,
                list_mode,
                output_list_limit,
                output_selection_mode,
                remainder_allocation,
                target_pupe,
            )
        return coarse_cache[total_parity]

    initial_result = eval_coarse(initial)
    candidate = initial

    if is_acceptable(initial_result):
        p = initial - 1
        while p >= minimum_parity:
            result = eval_coarse(p)
            if not is_acceptable(result):
                break
            candidate = p
            p -= 1
    else:
        p = initial + 1
        while p <= maximum_parity:
            result = eval_coarse(p)
            if is_acceptable(result):
                candidate = p
                break
            p += 1

    final_cache = {}

    def eval_final(total_parity):
        if total_parity not in final_cache:
            code_results = []
            for code_seed in verification_code_seeds:
                result = simulate_tree_pupe(
                    Ka,
                    J,
                    L_value,
                    total_parity,
                    number_of_trials=number_of_trials_final,
                    code_seed=code_seed,
                    trial_seed=30_000 + 100 * J + Ka + total_parity + code_seed,
                    max_paths=max_paths,
                    workers=workers,
                    list_mode=list_mode,
                    output_list_limit=output_list_limit,
                    output_selection_mode=output_selection_mode,
                    remainder_allocation=remainder_allocation,
                )
                code_results.append(result)

            final_result = aggregate_verification_results(code_results)
            final_cache[total_parity] = final_result
            log_evaluation(
                "final verification",
                Ka,
                J,
                L_value,
                total_parity,
                final_result,
                list_mode,
                output_list_limit,
                output_selection_mode,
                remainder_allocation,
                target_pupe,
            )
        return final_cache[total_parity]

    first_passing = None
    start_p = max(minimum_parity, candidate - 1)
    p = start_p
    while p <= maximum_parity:
        result = eval_final(p)
        if is_acceptable(result):
            first_passing = p
            break
        p += 1

    if first_passing is not None:
        for p in range(max(minimum_parity, first_passing - 2), min(maximum_parity, first_passing + 1) + 1):
            eval_final(p)

    selected_parity = None
    selected_result = None
    for total_parity in sorted(final_cache):
        result = final_cache[total_parity]
        if is_acceptable(result):
            selected_parity = total_parity
            selected_result = result
            break

    if selected_result is None:
        return SearchResult(
            rate=float("nan"),
            total_parity=float("nan"),
            mean_pupe=float("nan"),
            parity_profile=None,
            B_eff=float("nan"),
            cap_hit_fraction=float("nan"),
            maximum_paths_observed=float("nan"),
            average_ambiguous_roots=float("nan"),
            average_unique_surviving_messages_before_truncation=float("nan"),
            average_messages_after_truncation=float("nan"),
            average_shared_root_user_fraction=float("nan"),
            evaluated_points={
                "coarse": {p: result.__dict__ for p, result in coarse_cache.items()},
                "final": {p: result.__dict__ for p, result in final_cache.items()},
            },
            reason="No verified parity point met PUPE and cap-hit criteria.",
        )

    rate = 1.0 - selected_parity / (L_value * J)
    log_evaluation(
        "final selected",
        Ka,
        J,
        L_value,
        selected_parity,
        selected_result,
        list_mode,
        output_list_limit,
        output_selection_mode,
        remainder_allocation,
        target_pupe,
    )

    return SearchResult(
        rate=rate,
        total_parity=float(selected_parity),
        mean_pupe=selected_result.mean_pupe,
        parity_profile=selected_result.parity_profile,
        B_eff=float(selected_result.B_eff),
        cap_hit_fraction=selected_result.cap_hit_fraction,
        maximum_paths_observed=float(selected_result.maximum_paths_observed),
        average_ambiguous_roots=selected_result.average_ambiguous_roots,
        average_unique_surviving_messages_before_truncation=(
            selected_result.average_unique_surviving_messages_before_truncation
        ),
        average_messages_after_truncation=selected_result.average_messages_after_truncation,
        average_shared_root_user_fraction=selected_result.average_shared_root_user_fraction,
        evaluated_points={
            "coarse": {p: result.__dict__ for p, result in coarse_cache.items()},
            "final": {p: result.__dict__ for p, result in final_cache.items()},
        },
        reason="ok",
    )


def initialize_result_arrays(shape):
    return {
        "R_tree": np.full(shape, np.nan),
        "P_tree": np.full(shape, np.nan),
        "Pe_tree": np.full(shape, np.nan),
        "B_tree": np.full(shape, np.nan),
        "Cap_tree": np.full(shape, np.nan),
        "MaxPaths_tree": np.full(shape, np.nan),
        "AmbiguousRoots_tree": np.full(shape, np.nan),
        "SharedRootFraction_tree": np.full(shape, np.nan),
        "SurvivingMessages_tree": np.full(shape, np.nan),
        "MessagesAfterLimit_tree": np.full(shape, np.nan),
    }


def progress_config(
    list_mode,
    output_list_limit,
    output_selection_mode,
    remainder_allocation,
    target_pupe,
    max_paths,
):
    return {
        "list_mode": str(list_mode),
        "output_list_limit": str(output_list_limit),
        "output_selection_mode": str(output_selection_mode),
        "remainder_allocation": str(remainder_allocation),
        "target_pupe": float(target_pupe),
        "max_paths": int(max_paths),
    }


def config_matches(data, config):
    for key, value in config.items():
        if key not in data:
            return False
        saved = data[key].item() if np.asarray(data[key]).shape == () else data[key]
        if isinstance(value, float):
            if float(saved) != value:
                return False
        elif isinstance(value, int):
            if int(saved) != value:
                return False
        else:
            if str(saved) != str(value):
                return False
    return True


def load_or_initialize_progress(progress_file, J_values, Ka_values, config):
    shape = (len(J_values), len(Ka_values))
    arrays = initialize_result_arrays(shape)
    start_j_idx = 0
    start_ka_idx = 0

    if not progress_file.exists():
        return arrays, start_j_idx, start_ka_idx

    data = np.load(progress_file, allow_pickle=True)
    grid_matches = (
        "J_vec" in data
        and "Ka_vec" in data
        and np.array_equal(data["J_vec"], J_values)
        and np.array_equal(data["Ka_vec"], Ka_values)
    )
    if grid_matches and config_matches(data, config):
        for key in arrays:
            if key in data and data[key].shape == shape:
                arrays[key] = data[key]
        start_j_idx = int(data["last_j_idx"]) if "last_j_idx" in data else 0
        start_ka_idx = int(data["last_ka_idx"]) if "last_ka_idx" in data else 0
        print(f"Loaded progress from {progress_file}.", flush=True)
    else:
        print(f"Ignoring {progress_file}: saved grid or decoder configuration does not match.", flush=True)

    return arrays, start_j_idx, start_ka_idx


def save_progress(progress_file, J_values, Ka_values, arrays, last_j_idx, last_ka_idx, config):
    np.savez(
        progress_file,
        J_vec=J_values,
        Ka_vec=Ka_values,
        R_tree=arrays["R_tree"],
        P_tree=arrays["P_tree"],
        Pe_tree=arrays["Pe_tree"],
        B_tree=arrays["B_tree"],
        Cap_tree=arrays["Cap_tree"],
        MaxPaths_tree=arrays["MaxPaths_tree"],
        AmbiguousRoots_tree=arrays["AmbiguousRoots_tree"],
        SharedRootFraction_tree=arrays["SharedRootFraction_tree"],
        SurvivingMessages_tree=arrays["SurvivingMessages_tree"],
        MessagesAfterLimit_tree=arrays["MessagesAfterLimit_tree"],
        last_j_idx=last_j_idx,
        last_ka_idx=last_ka_idx,
        **config,
    )


def run_sweep(
    J_values,
    Ka_values,
    progress_file=PROGRESS_FILE,
    resume=True,
    list_mode="unique_support",
    output_list_limit="Ka",
    output_selection_mode="expected",
    remainder_allocation="late",
    target_pupe=TARGET_PUPE,
    max_paths=MAX_PATHS,
    **search_kwargs,
):
    config = progress_config(
        list_mode,
        output_list_limit,
        output_selection_mode,
        remainder_allocation,
        target_pupe,
        max_paths,
    )
    if resume:
        arrays, _, _ = load_or_initialize_progress(progress_file, J_values, Ka_values, config)
    else:
        arrays = initialize_result_arrays((len(J_values), len(Ka_values)))

    for j_idx, J in enumerate(J_values):
        for ka_idx, Ka in enumerate(Ka_values):
            if np.isfinite(arrays["R_tree"][j_idx, ka_idx]):
                print(f"Skipping completed point J={J}, Ka={Ka}.", flush=True)
                continue

            print(f"Starting point J={J}, Ka={Ka}.", flush=True)
            result = search_tree_rate(
                int(Ka),
                int(J),
                target_pupe=target_pupe,
                max_paths=max_paths,
                list_mode=list_mode,
                output_list_limit=output_list_limit,
                output_selection_mode=output_selection_mode,
                remainder_allocation=remainder_allocation,
                **search_kwargs,
            )

            arrays["R_tree"][j_idx, ka_idx] = result.rate
            arrays["P_tree"][j_idx, ka_idx] = result.total_parity
            arrays["Pe_tree"][j_idx, ka_idx] = result.mean_pupe
            arrays["B_tree"][j_idx, ka_idx] = result.B_eff
            arrays["Cap_tree"][j_idx, ka_idx] = result.cap_hit_fraction
            arrays["MaxPaths_tree"][j_idx, ka_idx] = result.maximum_paths_observed
            arrays["AmbiguousRoots_tree"][j_idx, ka_idx] = result.average_ambiguous_roots
            arrays["SharedRootFraction_tree"][j_idx, ka_idx] = result.average_shared_root_user_fraction
            arrays["SurvivingMessages_tree"][j_idx, ka_idx] = (
                result.average_unique_surviving_messages_before_truncation
            )
            arrays["MessagesAfterLimit_tree"][j_idx, ka_idx] = result.average_messages_after_truncation

            save_progress(
                progress_file,
                J_values,
                Ka_values,
                arrays,
                j_idx,
                ka_idx,
                config,
            )

            if result.reason != "ok":
                print(f"No reliable point for J={J}, Ka={Ka}: {result.reason}", flush=True)

    return arrays


def plot_results(J_values, Ka_values, arrays):
    import matplotlib.pyplot as plt

    plt.figure()
    for j_idx, J in enumerate(J_values):
        rates = arrays["R_tree"][j_idx].copy()
        unreliable = arrays["Cap_tree"][j_idx] > 0.0
        reliable_rates = rates.copy()
        reliable_rates[unreliable] = np.nan
        plt.plot(Ka_values, reliable_rates, marker="o", label=f"J={J}")

        if np.any(unreliable & np.isfinite(rates)):
            plt.plot(
                Ka_values[unreliable],
                rates[unreliable],
                linestyle="None",
                marker="x",
                label=f"J={J} cap hit",
            )

    plt.xlabel(r"$K_a$")
    plt.ylabel("Outer tree rate")
    plt.ylim([0.3, 0.9])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_progress_comparison(progress_files):
    import matplotlib.pyplot as plt

    plt.figure()
    for progress_file in progress_files:
        data = np.load(progress_file, allow_pickle=True)
        list_mode = str(data["list_mode"].item()) if "list_mode" in data else "unknown"
        output_limit = str(data["output_list_limit"].item()) if "output_list_limit" in data else "unknown"
        selection = str(data["output_selection_mode"].item()) if "output_selection_mode" in data else "unknown"
        remainder_allocation = (
            str(data["remainder_allocation"].item())
            if "remainder_allocation" in data
            else "unknown"
        )
        J_values = data["J_vec"]
        Ka_values = data["Ka_vec"]
        R_tree = data["R_tree"]
        for j_idx, J in enumerate(J_values):
            plt.plot(
                Ka_values,
                R_tree[j_idx],
                marker="o",
                label=(
                    f"{Path(progress_file).name}: J={J}, {list_mode}, "
                    f"alloc={remainder_allocation}, limit={output_limit}, {selection}"
                ),
            )

    plt.xlabel(r"$K_a$")
    plt.ylabel("Outer tree rate")
    plt.ylim([0.3, 0.9])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Fig. 3 outer tree-code simulation.")
    parser.add_argument("--validate-only", action="store_true", help="Run validation tests and stop.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation tests before the sweep.")
    parser.add_argument("--full", action="store_true", help="Run the full Fig. 3 grid.")
    parser.add_argument("--debug", action="store_true", help="Run the small debug grid.")
    parser.add_argument("--no-resume", action="store_true", help="Do not load existing progress.")
    parser.add_argument("--progress-file", default=str(PROGRESS_FILE), help="Progress .npz file.")
    parser.add_argument("--coarse-trials", type=int, default=20, help="Trials per coarse parity evaluation.")
    parser.add_argument("--final-trials", type=int, default=500, help="Trials per code seed in final verification.")
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS, help="Artificial path cap per rooted tree.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes for Monte Carlo trial batches.",
    )
    parser.add_argument("--list-mode", choices=LIST_MODES, default="unique_support")
    parser.add_argument("--remainder-allocation", choices=REMAINDER_ALLOCATIONS, default="late")
    parser.add_argument("--output-list-limit", choices=("none", "Ka"), default="Ka")
    parser.add_argument("--output-selection-mode", choices=OUTPUT_SELECTION_MODES, default="expected")
    parser.add_argument("--target-pupe", type=float, default=TARGET_PUPE)
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plotting after the sweep.")
    parser.add_argument(
        "--compare-progress",
        nargs="+",
        default=None,
        help="Plot one or more existing progress files together and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare_progress:
        plot_progress_comparison(args.compare_progress)
        return

    if not args.skip_validation:
        run_validation_tests()

    if args.validate_only:
        return

    if args.full:
        J_values = np.array([12, 15, 20])
        Ka_values = np.arange(25, 301, 5)
    elif args.debug or DEBUG_MODE:
        J_values = np.array([12])
        Ka_values = np.array([25, 50])
    else:
        J_values = J_vec
        Ka_values = Ka_vec

    arrays = run_sweep(
        J_values,
        Ka_values,
        progress_file=Path(args.progress_file),
        resume=not args.no_resume,
        number_of_trials_coarse=args.coarse_trials,
        number_of_trials_final=args.final_trials,
        max_paths=args.max_paths,
        workers=args.workers,
        list_mode=args.list_mode,
        remainder_allocation=args.remainder_allocation,
        output_list_limit=args.output_list_limit,
        output_selection_mode=args.output_selection_mode,
        target_pupe=args.target_pupe,
    )

    if not args.no_plot:
        plot_results(J_values, Ka_values, arrays)


if __name__ == "__main__":
    main()
