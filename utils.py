from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RootDecodeResult:
    paths: Optional[np.ndarray]
    cap_hit: bool
    path_counts: list[int]


def bits_to_int(bits):
    """
    Convert rows of binary vectors to unsigned integers.

    bits.shape = (number_of_rows, number_of_bits)
    """
    bits = np.asarray(bits, dtype=np.uint8)

    if bits.ndim == 1:
        bits = bits.reshape(1, -1)

    if bits.shape[1] == 0:
        return np.zeros(bits.shape[0], dtype=np.uint64)

    if bits.shape[1] > 64:
        raise ValueError("bits_to_int supports at most 64 bits per row.")

    powers = (
        np.uint64(1)
        << np.arange(
            bits.shape[1] - 1,
            -1,
            -1,
            dtype=np.uint64,
        )
    )

    return bits.astype(np.uint64) @ powers


def int_to_bits(values, width):
    """
    Convert unsigned integers to binary row vectors.
    """
    if width < 0:
        raise ValueError("width must be nonnegative.")
    if width > 64:
        raise ValueError("int_to_bits supports widths up to 64.")

    values = np.asarray(values, dtype=np.uint64).reshape(-1)

    if width == 0:
        return np.empty((len(values), 0), dtype=np.uint8)

    shifts = np.arange(
        width - 1,
        -1,
        -1,
        dtype=np.uint64,
    )

    return (
        (values[:, None] >> shifts[None, :]) & 1
    ).astype(np.uint8)


def message_key(bits):
    """
    Convert a possibly longer-than-64-bit message to a hashable key.
    """
    return np.packbits(
        np.asarray(bits, dtype=np.uint8),
        bitorder="big",
    ).tobytes()


def make_parity_profile(L, J, total_parity, remainder_allocation="late"):
    """
    Construct the Fig. 3 parity profile:

        pi[0]  = 0
        pi[-1] = J
        remaining parity bits distributed uniformly
        over sections 1,...,L-2 in zero-based indexing.

    total_parity = sum(pi)
    """
    if L < 3:
        raise ValueError("L must be at least 3.")

    if J < 1:
        raise ValueError("J must be positive.")

    if remainder_allocation not in ("early", "late"):
        raise ValueError("remainder_allocation must be 'early' or 'late'.")

    minimum_parity = J
    maximum_parity = (L - 1) * J

    if not minimum_parity <= total_parity <= maximum_parity:
        raise ValueError(
            f"total_parity must lie in "
            f"[{minimum_parity}, {maximum_parity}]"
        )

    pi = np.zeros(L, dtype=int)

    # Last section consists entirely of parity.
    pi[-1] = J

    remaining = total_parity - J
    number_of_middle_sections = L - 2

    base, remainder = divmod(
        remaining,
        number_of_middle_sections,
    )

    if base > J or (base == J and remainder > 0):
        raise ValueError("Invalid parity profile.")

    pi[1:L - 1] = base

    # Make the distribution differ by at most one bit.
    if remainder_allocation == "early":
        pi[1:1 + remainder] += 1
    elif remainder > 0:
        pi[L - 1 - remainder:L - 1] += 1

    assert len(pi) == L
    assert pi[0] == 0
    assert pi[-1] == J
    assert np.sum(pi) == total_parity
    assert np.all(pi >= 0)
    assert np.all(pi <= J)

    return pi


def make_parity_matrices(info_bits, parity_bits, rng):
    """
    G[r][l] maps the information bits of section r
    to the parity bits contained in section l.

    G[r][l].shape = (b[r], pi[l])

    H[l] is the vertical stack [G[0][l]; ...; G[l-1][l]], matching
    decoder prefixes [w(1), ..., w(l-1)].
    """
    info_bits = np.asarray(info_bits, dtype=int)
    parity_bits = np.asarray(parity_bits, dtype=int)

    if len(info_bits) != len(parity_bits):
        raise ValueError("info_bits and parity_bits must have the same length.")
    if not np.all(info_bits >= 0):
        raise ValueError("info_bits must be nonnegative.")
    if not np.all(parity_bits >= 0):
        raise ValueError("parity_bits must be nonnegative.")

    L = len(info_bits)

    G = [[None for _ in range(L)] for _ in range(L)]

    # H[l] is the vertically stacked generator matrix that maps
    # all information bits before section l to its parity bits.
    H = [None for _ in range(L)]

    for l in range(1, L):
        matrices = []

        for r in range(l):
            G[r][l] = rng.integers(
                0,
                2,
                size=(info_bits[r], parity_bits[l]),
                dtype=np.uint8,
            )
            assert G[r][l].shape == (info_bits[r], parity_bits[l])

            matrices.append(G[r][l])

        H[l] = np.vstack(matrices)
        assert H[l].shape == (int(np.sum(info_bits[:l])), parity_bits[l])

    return G, H


def encode_users(
    Ka,
    J,
    info_bits,
    parity_bits,
    G,
    rng,
):
    """
    Generate and encode Ka independent users.

    Returns
    -------
    information_blocks:
        information_blocks[l].shape = (Ka, b[l])

    section_indices:
        section_indices[l][k] is the J-bit section index
        transmitted by user k in section l.
    """
    info_bits = np.asarray(info_bits, dtype=int)
    parity_bits = np.asarray(parity_bits, dtype=int)

    if len(info_bits) != len(parity_bits):
        raise ValueError("info_bits and parity_bits must have the same length.")
    if not np.all(info_bits + parity_bits == J):
        raise ValueError("Each section must contain exactly J coded bits.")
    if info_bits[0] != J:
        raise ValueError("The first section must contain J information bits.")
    if parity_bits[0] != 0:
        raise ValueError("The first section must contain no parity bits.")
    if info_bits[-1] != 0:
        raise ValueError("The Fig. 3 profile requires no information bits in the last section.")
    if parity_bits[-1] != J:
        raise ValueError("The Fig. 3 profile requires J parity bits in the last section.")

    L = len(info_bits)

    information_blocks = [
        rng.integers(
            0,
            2,
            size=(Ka, info_bits[l]),
            dtype=np.uint8,
        )
        for l in range(L)
    ]

    section_indices = []

    for l in range(L):
        parity = np.zeros(
            (Ka, parity_bits[l]),
            dtype=np.uint8,
        )

        for r in range(l):
            if info_bits[r] == 0 or parity_bits[l] == 0:
                continue

            contribution = (
                information_blocks[r].astype(np.uint16)
                @ G[r][l].astype(np.uint16)
            ) & 1

            parity ^= contribution.astype(np.uint8)

        coded_fragment = np.concatenate(
            (
                information_blocks[l],
                parity,
            ),
            axis=1,
        )

        assert coded_fragment.shape == (Ka, J)

        indices = bits_to_int(coded_fragment)
        assert np.all(indices >= 0)
        assert np.all(indices < np.uint64(2) ** np.uint64(J))

        section_indices.append(indices)

    return information_blocks, section_indices


def decode_one_root(
    root_information,
    candidate_information,
    candidate_parity,
    H,
    max_paths=500_000,
):
    """
    Decode the tree associated with one element of S_1.

    Returns a RootDecodeResult. paths is None only when max_paths is exceeded.
    path_counts contains the number of surviving paths after the initial root
    and after every subsequent decoded section.
    """
    L = len(candidate_information)

    prefixes = np.asarray(root_information, dtype=np.uint8).reshape(1, -1).copy()
    path_counts = [int(prefixes.shape[0])]

    for l in range(1, L):
        number_of_paths = prefixes.shape[0]

        # Parity expected from each currently surviving prefix.
        if H[l].shape[1] == 0:
            expected_parity_int = np.zeros(
                number_of_paths,
                dtype=np.uint64,
            )
        else:
            expected_parity_bits = (
                prefixes.astype(np.uint16)
                @ H[l].astype(np.uint16)
            ) & 1

            expected_parity_int = bits_to_int(
                expected_parity_bits.astype(np.uint8)
            )

        # Parity actually carried by candidates in S_l.
        actual_parity_int = bits_to_int(
            candidate_parity[l]
        )

        expected_values = np.unique(expected_parity_int)
        actual_values = np.unique(actual_parity_int)

        matching_values = np.intersect1d(
            expected_values,
            actual_values,
            assume_unique=True,
        )

        new_prefixes = []
        new_path_count = 0

        for parity_value in matching_values:
            path_positions = np.flatnonzero(
                expected_parity_int == parity_value
            )

            candidate_positions = np.flatnonzero(
                actual_parity_int == parity_value
            )

            number_of_new_paths = (
                len(path_positions)
                * len(candidate_positions)
            )

            new_path_count += number_of_new_paths

            if new_path_count > max_paths:
                path_counts.append(int(new_path_count))
                return RootDecodeResult(
                    paths=None,
                    cap_hit=True,
                    path_counts=path_counts,
                )

            old_part = np.repeat(
                prefixes[path_positions],
                len(candidate_positions),
                axis=0,
            )

            new_part = np.tile(
                candidate_information[l][candidate_positions],
                (len(path_positions), 1),
            )

            new_prefixes.append(
                np.concatenate(
                    (old_part, new_part),
                    axis=1,
                )
            )

        if not new_prefixes:
            empty_paths = np.empty(
                (
                    0,
                    prefixes.shape[1]
                    + candidate_information[l].shape[1],
                ),
                dtype=np.uint8,
            )
            path_counts.append(0)
            return RootDecodeResult(
                paths=empty_paths,
                cap_hit=False,
                path_counts=path_counts,
            )

        prefixes = np.vstack(new_prefixes)
        path_counts.append(int(prefixes.shape[0]))

    return RootDecodeResult(
        paths=prefixes,
        cap_hit=False,
        path_counts=path_counts,
    )


def test_bit_conversions(max_width=20):
    """
    Test bits_to_int and int_to_bits.

    Widths up to 20 are tested exhaustively by default. Larger widths use a
    deterministic random sample so the test remains fast.
    """
    rng = np.random.default_rng(12345)
    exhaustive_limit_width = 20

    for width in range(max_width + 1):
        if width <= exhaustive_limit_width:
            values = np.arange(2 ** width, dtype=np.uint64)
        else:
            edge_values = np.array(
                [0, 1, 2 ** width - 1],
                dtype=np.uint64,
            )
            random_values = rng.integers(
                0,
                2 ** width,
                size=10_000,
                dtype=np.uint64,
            )
            values = np.unique(np.concatenate((edge_values, random_values)))

        round_trip = bits_to_int(int_to_bits(values, width))
        if not np.array_equal(round_trip, values):
            raise AssertionError(f"Bit conversion round trip failed for width={width}.")

    return True


def H2(p):
    """
    Binary entropy in bits.
    """
    p = np.asarray(p, dtype=float)

    result = np.zeros_like(p)

    valid = (p > 0.0) & (p < 1.0)

    result[valid] = (
        -p[valid] * np.log2(p[valid])
        -(1.0 - p[valid]) * np.log2(1.0 - p[valid])
    )
    return result