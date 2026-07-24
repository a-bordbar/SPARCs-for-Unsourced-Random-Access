#!/usr/bin/env python3
"""Finite-length empirical AMP + tree-code reproduction for Figure 9.

This script intentionally simulates the full concatenated system:

    messages -> random tree encoder -> SPARC inner channel -> AMP OR+
    denoiser -> top-(Ka+Delta) section lists -> tree decoder -> PUPE.

The full-scale default uses a matrix-free structured Hadamard sensing
operator. This is computationally tractable and column-normalized, but it is
not the exact i.i.d. Gaussian ensemble used in the paper. The Gaussian ensemble is not 
feasible for the full-scale parameters due to memory constraints.
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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - reported at runtime
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    tqdm = None


SCHEMA_VERSION = 1
TREE_CODE_SEED = 19051031
BASE_MESSAGE_SEED = 400000
BASE_OPERATOR_SEED = 500000
BASE_NOISE_SEED = 600000


@dataclasses.dataclass(frozen=True)
class Config:
    Ka: int = 300
    J: int = 20
    L: int = 8
    B: int = 89
    n: int = 26229
    parity_bits: Tuple[int, ...] = (0, 9, 8, 9, 8, 9, 8, 20)
    Delta: int = 50
    EbN0_dB: Tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5)
    operator_mode: str = "structured_hadamard"
    high_section_indices: Tuple[int, ...] = (7, 8)  # one-based
    high_to_low_ratio: float = 1.9
    average_over_high_section_pairs: bool = False
    amp_max_iter: int = 100
    amp_tol: float = 1e-5
    amp_min_iter: int = 3
    amp_required_stable_iterations: int = 2
    amp_damping: float = 1.0
    max_tree_paths: int = 2_000_000
    n_trials: int = 10
    n_workers: int = min(8, os.cpu_count())
    trial_batch_size: int = 5
    show_error_bars: bool = False
    progress_bars: bool = True
    debug: bool = False
    validate_only: bool = False
    output_dir: str = "data/fig9"
    plot_png: str = "fig9_empirical_amp_tree.png"
    plot_pdf: str = "fig9_empirical_amp_tree.pdf"
    tree_code_seed: int = TREE_CODE_SEED
    base_message_seed: int = BASE_MESSAGE_SEED
    base_operator_seed: int = BASE_OPERATOR_SEED
    base_noise_seed: int = BASE_NOISE_SEED

    @property
    def info_bits(self) -> Tuple[int, ...]:
        return tuple(self.J - p for p in self.parity_bits)

    @property
    def list_size(self) -> int:
        return self.Ka + self.Delta

    @property
    def N(self) -> int:
        return 1 << self.J

    @property
    def Rin(self) -> float:
        return self.L * self.J / self.n

    @property
    def Rout(self) -> float:
        return self.B / (self.L * self.J)

    @property
    def R(self) -> float:
        return self.B / self.n

    @property
    def mu(self) -> float:
        return self.Ka * self.R

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.output_dir) / "fig9_checkpoint.pkl"

    @property
    def summary_csv(self) -> Path:
        return Path(self.output_dir) / "fig9_summary.csv"

    @property
    def trials_csv(self) -> Path:
        return Path(self.output_dir) / "fig9_trials.csv"


def debug_config() -> Config:
    return dataclasses.replace(
        Config(),
        n_trials=2,
        EbN0_dB=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5),
        n_workers=1,
        trial_batch_size=1,
        debug=True,
        output_dir="data/fig9_debug",
        plot_png="fig9_empirical_amp_tree_debug.png",
        plot_pdf="fig9_empirical_amp_tree_debug.pdf",
    )


def validation_toy_config(operator_mode: str = "dense_gaussian_debug") -> Config:
    return dataclasses.replace(
        Config(),
        Ka=10,
        J=8,
        L=4,
        B=22,
        n=128 if operator_mode == "structured_hadamard" else 160,
        parity_bits=(0, 4, 4, 2),
        Delta=4,
        EbN0_dB=(5.0,),
        operator_mode=operator_mode,
        high_section_indices=(3, 4),
        amp_max_iter=8,
        amp_tol=1e-4,
        max_tree_paths=50_000,
        n_trials=1,
        n_workers=1,
        progress_bars=False,
        output_dir="data/fig9_validation",
    )


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def trial_message_seed(trial: int, cfg: Config) -> int:
    return cfg.base_message_seed + 1009 * int(trial)


def trial_operator_seed(trial: int, cfg: Config) -> int:
    return cfg.base_operator_seed + 1009 * int(trial)


def trial_noise_seed(trial: int, cfg: Config) -> int:
    return cfg.base_noise_seed + 1009 * int(trial)


def bits_to_index(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim == 1:
        powers = (1 << np.arange(bits.size - 1, -1, -1, dtype=np.uint64))
        return np.uint64(bits.astype(np.uint64) @ powers)
    powers = (1 << np.arange(bits.shape[1] - 1, -1, -1, dtype=np.uint64))
    return bits.astype(np.uint64) @ powers


def index_to_bits(values: Union[np.ndarray, int], width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.uint64)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    out = ((arr[..., None] >> shifts) & np.uint64(1)).astype(np.uint8)
    return out


def pack_message_89(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.shape[-1] != 89:
        raise ValueError("pack_message_89 expects 89-bit rows")
    leading = bits[..., :64]
    trailing = bits[..., 64:]
    w0 = bits_to_index(leading).astype(np.uint64)
    w1 = bits_to_index(trailing).astype(np.uint64)
    return np.stack([w0, w1], axis=-1)


def unpack_message_89(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.uint64)
    if keys.shape[-1] != 2:
        raise ValueError("unpack_message_89 expects two-word keys")
    b0 = index_to_bits(keys[..., 0], 64)
    b1 = index_to_bits(keys[..., 1], 25)
    return np.concatenate([b0, b1], axis=-1).astype(np.uint8)


def pack_variable_message(bits: np.ndarray) -> Tuple[int, int]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size > 128:
        raise ValueError("variable message pack supports at most 128 bits")
    first = min(64, bits.size)
    w0 = int(bits_to_index(bits[:first])) if first else 0
    rest = bits.size - first
    w1 = int(bits_to_index(bits[first:])) if rest else 0
    return w0, w1


def key_tuple_from_words(words: np.ndarray) -> Tuple[int, int]:
    return int(words[0]), int(words[1])


def make_tree_code(cfg: Config) -> List[Optional[np.ndarray]]:
    rng = rng_from_seed(cfg.tree_code_seed)
    G: List[Optional[np.ndarray]] = [None] * cfg.L
    info = cfg.info_bits
    for l in range(cfg.L):
        p = cfg.parity_bits[l]
        if p == 0:
            G[l] = np.zeros((sum(info[:l]), 0), dtype=np.uint8)
        else:
            G[l] = rng.integers(0, 2, size=(sum(info[:l]), p), dtype=np.uint8)
    return G


def tree_code_dimensions(G: Sequence[Optional[np.ndarray]]) -> List[Tuple[int, int]]:
    return [tuple(map(int, g.shape)) if g is not None else (0, 0) for g in G]


def encode_messages(messages: np.ndarray, G: Sequence[np.ndarray], cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    messages = np.asarray(messages, dtype=np.uint8)
    if messages.shape != (cfg.Ka, cfg.B):
        raise ValueError(f"messages must have shape {(cfg.Ka, cfg.B)}")
    indices = np.zeros((cfg.Ka, cfg.L), dtype=np.int64)
    fragments = np.zeros((cfg.Ka, cfg.L, cfg.J), dtype=np.uint8)
    pos = 0
    prev_len = 0
    for l in range(cfg.L):
        ib = cfg.info_bits[l]
        pb = cfg.parity_bits[l]
        current = messages[:, pos : pos + ib]
        previous = messages[:, :prev_len]
        if pb:
            parity = (previous @ G[l] % 2).astype(np.uint8)
        else:
            parity = np.zeros((cfg.Ka, 0), dtype=np.uint8)
        frag = np.concatenate([current, parity], axis=1)
        if frag.shape[1] != cfg.J:
            raise AssertionError("coded fragment length mismatch")
        fragments[:, l, :] = frag
        indices[:, l] = bits_to_index(frag).astype(np.int64)
        pos += ib
        prev_len += ib
    if pos != cfg.B:
        raise AssertionError("message bit accounting mismatch")
    return indices, fragments


def multiplicity_vectors(indices: np.ndarray, cfg: Config) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for l in range(cfg.L):
        counts = np.bincount(indices[:, l], minlength=cfg.N).astype(np.float32)
        if int(counts.sum()) != cfg.Ka:
            raise AssertionError("section multiplicity must sum to Ka")
        out.append(counts)
    return out


def fwht_inplace(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    h = 1
    while h < n:
        a = x.reshape(-1, h * 2)
        left = a[:, :h].copy()
        right = a[:, h : 2 * h].copy()
        a[:, :h] = left + right
        a[:, h : 2 * h] = left - right
        h *= 2
    x *= 1.0 / math.sqrt(n)
    return x


@dataclasses.dataclass
class StructuredHadamardOperator:
    cfg: Config
    rows: List[np.ndarray]
    signs: List[np.ndarray]

    @classmethod
    def create(cls, cfg: Config, seed: int) -> "StructuredHadamardOperator":
        rng = rng_from_seed(seed)
        if cfg.n > cfg.N:
            raise ValueError("structured_hadamard requires n <= 2^J")
        rows: List[np.ndarray] = []
        signs: List[np.ndarray] = []
        for _ in range(cfg.L):
            rows.append(rng.choice(cfg.N, size=cfg.n, replace=False).astype(np.int64))
            signs.append(rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=cfg.N, replace=True))
        return cls(cfg=cfg, rows=rows, signs=signs)

    def forward_section(self, l: int, x: np.ndarray) -> np.ndarray:
        work = np.asarray(x, dtype=np.float32).copy()
        work *= self.signs[l]
        fwht_inplace(work)
        return (math.sqrt(self.cfg.N / self.cfg.n) * work[self.rows[l]]).astype(np.float32)

    def adjoint_section(self, l: int, y: np.ndarray) -> np.ndarray:
        work = np.zeros(self.cfg.N, dtype=np.float32)
        work[self.rows[l]] = np.asarray(y, dtype=np.float32) * math.sqrt(self.cfg.N / self.cfg.n)
        fwht_inplace(work)
        work *= self.signs[l]
        return work

    def forward(self, theta_sections: Sequence[np.ndarray]) -> np.ndarray:
        y = np.zeros(self.cfg.n, dtype=np.float32)
        for l, x in enumerate(theta_sections):
            y += self.forward_section(l, x)
        return y

    def adjoint(self, z: np.ndarray) -> List[np.ndarray]:
        return [self.adjoint_section(l, z) for l in range(self.cfg.L)]

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": "structured_hadamard",
            "row_hashes": [hash_array(r) for r in self.rows],
            "sign_hashes": [hash_array(s) for s in self.signs],
        }


@dataclasses.dataclass
class DenseGaussianOperator:
    cfg: Config
    mats: List[np.ndarray]

    @classmethod
    def create(cls, cfg: Config, seed: int) -> "DenseGaussianOperator":
        if cfg.J > 12:
            raise ValueError("dense_gaussian_debug is only allowed for toy dimensions")
        rng = rng_from_seed(seed)
        mats = [
            rng.normal(0.0, 1.0 / math.sqrt(cfg.n), size=(cfg.n, cfg.N)).astype(np.float32)
            for _ in range(cfg.L)
        ]
        return cls(cfg=cfg, mats=mats)

    def forward_section(self, l: int, x: np.ndarray) -> np.ndarray:
        return (self.mats[l] @ np.asarray(x, dtype=np.float32)).astype(np.float32)

    def adjoint_section(self, l: int, y: np.ndarray) -> np.ndarray:
        return (self.mats[l].T @ np.asarray(y, dtype=np.float32)).astype(np.float32)

    def forward(self, theta_sections: Sequence[np.ndarray]) -> np.ndarray:
        y = np.zeros(self.cfg.n, dtype=np.float32)
        for l, x in enumerate(theta_sections):
            y += self.forward_section(l, x)
        return y

    def adjoint(self, z: np.ndarray) -> List[np.ndarray]:
        return [self.adjoint_section(l, z) for l in range(self.cfg.L)]

    def metadata(self) -> Dict[str, Any]:
        return {"mode": "dense_gaussian_debug", "mat_hashes": [hash_array(m) for m in self.mats]}


Operator = Union[StructuredHadamardOperator, DenseGaussianOperator]


def create_operator(cfg: Config, seed: int) -> Operator:
    if cfg.operator_mode == "structured_hadamard":
        return StructuredHadamardOperator.create(cfg, seed)
    if cfg.operator_mode == "dense_gaussian_debug":
        return DenseGaussianOperator.create(cfg, seed)
    raise ValueError(f"unknown operator mode {cfg.operator_mode}")


def hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()[:16]


def make_power_shapes(cfg: Config, pair: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
    flat = np.ones(cfg.L, dtype=np.float64)
    if pair is None:
        high = tuple(i - 1 for i in cfg.high_section_indices)
    else:
        high = tuple(i - 1 for i in pair)
    if len(high) != 2:
        raise ValueError("two-level profile expects exactly two high sections")
    low_factor = cfg.L / ((cfg.L - 2) + 2 * cfg.high_to_low_ratio)
    high_factor = cfg.high_to_low_ratio * low_factor
    two = low_factor * np.ones(cfg.L, dtype=np.float64)
    two[list(high)] = high_factor
    if not np.isclose(two.mean(), 1.0, atol=1e-12):
        raise AssertionError("two-level shape must have unit mean")
    return {"2-level power allocation": two, "flat power allocation": flat}


def phat_avg(EbN0_dB: float, cfg: Config) -> float:
    eb = 10.0 ** (EbN0_dB / 10.0)
    p_user = 2.0 * cfg.R * eb
    p1 = cfg.n * p_user / cfg.L
    p2 = 2.0 * cfg.J * cfg.Rout * eb
    if not np.isclose(p1, p2, rtol=1e-12, atol=1e-12):
        raise AssertionError("Phat expression mismatch")
    return p1


def orplus_denoise(x: np.ndarray, tau2: float, phat_l: float, Ka: int, J: int) -> Tuple[np.ndarray, float]:
    x64 = np.asarray(x, dtype=np.float64)
    tau2 = float(max(tau2, np.finfo(float).tiny))
    q = 2.0 ** (-J)
    p0 = (1.0 - q) ** Ka
    p1 = Ka * q * (1.0 - q) ** (Ka - 1)
    p2 = max(1.0 - p0 - p1, np.finfo(float).tiny)
    a = math.sqrt(phat_l)
    amps = np.array([0.0, a, 2.0 * a], dtype=np.float64)
    logs = np.empty((3, x64.size), dtype=np.float64)
    logs[0] = math.log(max(p0, np.finfo(float).tiny)) - (x64 - amps[0]) ** 2 / (2.0 * tau2)
    logs[1] = math.log(max(p1, np.finfo(float).tiny)) - (x64 - amps[1]) ** 2 / (2.0 * tau2)
    logs[2] = math.log(p2) - (x64 - amps[2]) ** 2 / (2.0 * tau2)
    m = np.max(logs, axis=0)
    w = np.exp(logs - m)
    denom = np.sum(w, axis=0)
    probs = w / denom
    mean = probs[1] * amps[1] + probs[2] * amps[2]
    second = probs[1] * amps[1] ** 2 + probs[2] * amps[2] ** 2
    var = np.maximum(second - mean * mean, 0.0)
    divergence = float(np.sum(var / tau2, dtype=np.float64))
    return mean.astype(np.float32), divergence


def orplus_scalar(x: float, tau2: float, phat_l: float, Ka: int, J: int) -> Tuple[float, float]:
    mean, div = orplus_denoise(np.array([x], dtype=np.float64), tau2, phat_l, Ka, J)
    return float(mean[0]), float(div)


@dataclasses.dataclass
class AMPResult:
    theta: List[np.ndarray]
    iterations: int
    converged: bool
    final_tau2: float
    rel_change: float
    elapsed: float


def progress_iter(iterable: Iterable[int], cfg: Config, **kwargs: Any) -> Iterable[int]:
    if not cfg.progress_bars or tqdm is None:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def amp_decode(
    y: np.ndarray,
    op: Operator,
    phat: np.ndarray,
    cfg: Config,
    progress_desc: str = "",
    progress_position: int = 0,
) -> AMPResult:
    t0 = time.time()
    theta = [np.zeros(cfg.N, dtype=np.float32) for _ in range(cfg.L)]
    z = np.asarray(y, dtype=np.float32).copy()
    stable = 0
    rel_change = math.inf
    final_tau2 = math.nan
    converged = False
    iterator = progress_iter(
        range(1, cfg.amp_max_iter + 1),
        cfg,
        total=cfg.amp_max_iter,
        desc=progress_desc or "AMP",
        unit="iter",
        leave=False,
        position=progress_position,
    )
    for it in iterator:
        final_tau2 = float(np.dot(z.astype(np.float64), z.astype(np.float64)) / cfg.n)
        az = op.adjoint(z)
        theta_new: List[np.ndarray] = []
        divergence = 0.0
        for l in range(cfg.L):
            pseudo = theta[l] + az[l]
            den, div_l = orplus_denoise(pseudo, final_tau2, float(phat[l]), cfg.Ka, cfg.J)
            if cfg.amp_damping != 1.0:
                den = (cfg.amp_damping * den + (1.0 - cfg.amp_damping) * theta[l]).astype(np.float32)
            theta_new.append(den)
            divergence += div_l
        diff2 = 0.0
        norm2 = 0.0
        for old, new in zip(theta, theta_new):
            d = (new.astype(np.float64) - old.astype(np.float64))
            diff2 += float(np.dot(d, d))
            norm2 += float(np.dot(new.astype(np.float64), new.astype(np.float64)))
        rel_change = math.sqrt(diff2) / max(math.sqrt(norm2), 1.0)
        z_new = y.astype(np.float32) - op.forward(theta_new) + z * np.float32(divergence / cfg.n)
        theta = theta_new
        z = z_new.astype(np.float32)
        if it >= cfg.amp_min_iter and rel_change <= cfg.amp_tol:
            stable += 1
            if stable >= cfg.amp_required_stable_iterations:
                converged = True
                break
        else:
            stable = 0
        if tqdm is not None and cfg.progress_bars and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(tau2=f"{final_tau2:.3g}", rel=f"{rel_change:.2g}", refresh=False)
    return AMPResult(theta=theta, iterations=it, converged=converged, final_tau2=final_tau2, rel_change=rel_change, elapsed=time.time() - t0)


def top_section_lists(theta: Sequence[np.ndarray], cfg: Config) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    idxs: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    k = min(cfg.list_size, cfg.N)
    for arr in theta:
        part = np.argpartition(arr, -k)[-k:]
        # Deterministic: score descending, index ascending for ties.
        order = np.lexsort((part, -arr[part]))
        sel = part[order].astype(np.int64)
        idxs.append(sel)
        scores.append(arr[sel].astype(np.float64))
    return idxs, scores


@dataclasses.dataclass
class TreeDecodeResult:
    decoded_keys: List[Tuple[int, int]]
    final_list_size: int
    pre_cap_size: int
    tree_overflow: bool
    overflow_stage: int
    attempted_paths: int
    oversized_final_list: bool


def parity_int(previous_info: np.ndarray, G_l: np.ndarray) -> int:
    if G_l.shape[1] == 0:
        return 0
    p = (previous_info @ G_l % 2).astype(np.uint8)
    return int(bits_to_index(p))


def tree_decode(
    list_indices: Sequence[np.ndarray],
    list_scores: Sequence[np.ndarray],
    G: Sequence[np.ndarray],
    cfg: Config,
) -> TreeDecodeResult:
    cand_info: List[np.ndarray] = []
    cand_parity: List[np.ndarray] = []
    cand_parity_int: List[np.ndarray] = []
    for l in range(cfg.L):
        bits = index_to_bits(list_indices[l], cfg.J)
        ib = cfg.info_bits[l]
        cand_info.append(bits[:, :ib].astype(np.uint8))
        pbits = bits[:, ib:].astype(np.uint8)
        cand_parity.append(pbits)
        if pbits.shape[1] == 0:
            cand_parity_int.append(np.zeros(bits.shape[0], dtype=np.int64))
        else:
            cand_parity_int.append(bits_to_index(pbits).astype(np.int64))

    paths: List[Tuple[np.ndarray, float]] = [
        (cand_info[0][i].copy(), float(list_scores[0][i])) for i in range(len(list_indices[0]))
    ]
    for l in range(1, cfg.L):
        by_expected: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        for info_so_far, metric in paths:
            key = parity_int(info_so_far, G[l])
            by_expected.setdefault(key, []).append((info_so_far, metric))
        by_observed: Dict[int, List[int]] = {}
        for i, key in enumerate(cand_parity_int[l]):
            by_observed.setdefault(int(key), []).append(i)
        attempted = 0
        new_paths: List[Tuple[np.ndarray, float]] = []
        for key in sorted(set(by_expected).intersection(by_observed)):
            left = by_expected[key]
            right = by_observed[key]
            attempted += len(left) * len(right)
            if attempted > cfg.max_tree_paths:
                return TreeDecodeResult([], 0, 0, True, l + 1, attempted, False)
            for info_so_far, metric in left:
                for idx in right:
                    new_info = np.concatenate([info_so_far, cand_info[l][idx]])
                    new_paths.append((new_info, metric + float(list_scores[l][idx])))
        paths = new_paths
        if not paths:
            break
    best: Dict[Tuple[int, int], float] = {}
    for bits, metric in paths:
        key = pack_variable_message(bits)
        if key not in best or metric > best[key]:
            best[key] = metric
    pre_cap = len(best)
    oversized = pre_cap > cfg.Ka
    items = list(best.items())
    items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    if oversized:
        items = items[: cfg.Ka]
    keys = [k for k, _ in items]
    return TreeDecodeResult(
        decoded_keys=keys,
        final_list_size=len(keys),
        pre_cap_size=pre_cap,
        tree_overflow=False,
        overflow_stage=0,
        attempted_paths=0,
        oversized_final_list=oversized,
    )


def pupe(transmitted_bits: np.ndarray, decoded_keys: Sequence[Tuple[int, int]], cfg: Config) -> Tuple[float, int, int]:
    decoded = set(decoded_keys)
    if cfg.B == 89:
        keys = pack_message_89(transmitted_bits)
    else:
        keys = [pack_variable_message(b) for b in transmitted_bits]
    missing = 0
    for k in range(cfg.Ka):
        key = key_tuple_from_words(keys[k]) if cfg.B == 89 else keys[k]
        if key not in decoded:
            missing += 1
    return missing / cfg.Ka, cfg.Ka - missing, missing


def make_trial_context(trial: int, G: Sequence[np.ndarray], cfg: Config) -> Dict[str, Any]:
    msg_rng = rng_from_seed(trial_message_seed(trial, cfg))
    messages = msg_rng.integers(0, 2, size=(cfg.Ka, cfg.B), dtype=np.uint8)
    indices, fragments = encode_messages(messages, G, cfg)
    multiplicities = multiplicity_vectors(indices, cfg)
    op = create_operator(cfg, trial_operator_seed(trial, cfg))
    noise_rng = rng_from_seed(trial_noise_seed(trial, cfg))
    noise = noise_rng.normal(0.0, 1.0, size=cfg.n).astype(np.float32)
    return {
        "messages": messages,
        "indices": indices,
        "fragments": fragments,
        "multiplicities": multiplicities,
        "operator": op,
        "noise": noise,
    }


def run_one_energy_allocation(
    trial: int,
    eb: float,
    allocation_name: str,
    shape: np.ndarray,
    ctx: Dict[str, Any],
    G: Sequence[np.ndarray],
    cfg: Config,
) -> Dict[str, Any]:
    started = time.time()
    avg = phat_avg(eb, cfg)
    phat = avg * np.asarray(shape, dtype=np.float64)
    theta_true = [(math.sqrt(phat[l]) * ctx["multiplicities"][l]).astype(np.float32) for l in range(cfg.L)]
    y = ctx["operator"].forward(theta_true) + ctx["noise"]
    desc = f"T{trial} Eb/N0={eb:g} {allocation_name.replace(' power allocation', '')}"
    amp = amp_decode(y, ctx["operator"], phat, cfg, progress_desc=desc, progress_position=max(0, (trial - 1) % max(cfg.n_workers, 1)))
    list_idx, list_scores = top_section_lists(amp.theta, cfg)
    tree = tree_decode(list_idx, list_scores, G, cfg)
    if tree.tree_overflow:
        pe, recovered, missing = 1.0, 0, cfg.Ka
    else:
        pe, recovered, missing = pupe(ctx["messages"], tree.decoded_keys, cfg)
    return {
        "trial": trial,
        "EbN0_dB": eb,
        "allocation": allocation_name,
        "Pe_trial": pe,
        "recovered_users": recovered,
        "missing_users": missing,
        "unique_decoded_messages": tree.final_list_size,
        "pre_cap_final_list_size": tree.pre_cap_size,
        "tree_overflow": tree.tree_overflow,
        "overflow_stage": tree.overflow_stage,
        "attempted_paths": tree.attempted_paths,
        "oversized_final_list": tree.oversized_final_list,
        "amp_iterations": amp.iterations,
        "amp_converged": amp.converged,
        "final_tau2": amp.final_tau2,
        "rel_change": amp.rel_change,
        "amp_elapsed_sec": amp.elapsed,
        "elapsed_sec": time.time() - started,
    }


def run_trial(trial: int, cfg: Config) -> List[Dict[str, Any]]:
    G = make_tree_code(cfg)
    ctx = make_trial_context(trial, G, cfg)
    if cfg.average_over_high_section_pairs:
        pairs: Iterable[Optional[Tuple[int, int]]] = itertools.combinations(range(1, cfg.L + 1), 2)
    else:
        pairs = [None]
    results: List[Dict[str, Any]] = []
    for pair in pairs:
        shapes = make_power_shapes(cfg, pair)
        for eb in cfg.EbN0_dB:
            for allocation_name, shape in shapes.items():
                try:
                    row = run_one_energy_allocation(trial, eb, allocation_name, shape, ctx, G, cfg)
                    if pair is not None:
                        row["high_sections_pair"] = f"{pair[0]} {pair[1]}"
                    else:
                        row["high_sections_pair"] = " ".join(map(str, cfg.high_section_indices))
                    results.append(row)
                except Exception as exc:
                    raise RuntimeError(
                        f"trial={trial} EbN0={eb} allocation={allocation_name} "
                        f"exception={type(exc).__name__}: {exc}"
                    ) from exc
    return results


def config_signature(cfg: Config) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "Ka": cfg.Ka,
        "J": cfg.J,
        "L": cfg.L,
        "B": cfg.B,
        "n": cfg.n,
        "parity_bits": cfg.parity_bits,
        "Delta": cfg.Delta,
        "EbN0_dB": cfg.EbN0_dB,
        "operator_mode": cfg.operator_mode,
        "high_to_low_ratio": cfg.high_to_low_ratio,
        "high_section_indices": cfg.high_section_indices,
        "average_over_high_section_pairs": cfg.average_over_high_section_pairs,
        "amp_max_iter": cfg.amp_max_iter,
        "amp_tol": cfg.amp_tol,
        "amp_min_iter": cfg.amp_min_iter,
        "amp_required_stable_iterations": cfg.amp_required_stable_iterations,
        "amp_damping": cfg.amp_damping,
        "max_tree_paths": cfg.max_tree_paths,
        "tree_code_seed": cfg.tree_code_seed,
        "base_message_seed": cfg.base_message_seed,
        "base_operator_seed": cfg.base_operator_seed,
        "base_noise_seed": cfg.base_noise_seed,
        "tree_code_policy": "fixed random parity matrices across campaign",
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_checkpoint(cfg: Config) -> Tuple[set[int], List[Dict[str, Any]]]:
    path = cfg.checkpoint_path
    if not path.exists():
        return set(), []
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("signature") != config_signature(cfg):
        raise RuntimeError(f"Incompatible checkpoint at {path}. Move it aside or use the matching configuration.")
    return set(payload.get("completed_trials", [])), list(payload.get("rows", []))


def atomic_save_checkpoint(cfg: Config, completed: set[int], rows: List[Dict[str, Any]]) -> None:
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signature": config_signature(cfg), "completed_trials": sorted(completed), "rows": rows}
    tmp = cfg.checkpoint_path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cfg.checkpoint_path)


def remove_existing_outputs(cfg: Config) -> None:
    for path in [cfg.checkpoint_path, cfg.summary_csv, cfg.trials_csv, Path(cfg.plot_png), Path(cfg.plot_pdf)]:
        if path.exists():
            path.unlink()


def summarize_rows(rows: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for allocation in ("2-level power allocation", "flat power allocation"):
        for eb in cfg.EbN0_dB:
            sub = [r for r in rows if r["allocation"] == allocation and float(r["EbN0_dB"]) == float(eb)]
            if not sub:
                continue
            pe = np.array([r["Pe_trial"] for r in sub], dtype=np.float64)
            total_missing = int(sum(r["missing_users"] for r in sub))
            total_users = int(sum(cfg.Ka for _ in sub))
            mean_pe = float(np.mean(pe))
            weighted = total_missing / total_users
            equal_constant_ka = bool(np.isclose(mean_pe, weighted, atol=1e-15))
            out.append(
                {
                    "allocation": allocation,
                    "EbN0_dB": eb,
                    "N_trials": len(sub),
                    "Pe_mean": mean_pe,
                    "Pe_standard_error": float(np.std(pe, ddof=1) / math.sqrt(len(pe))) if len(pe) > 1 else 0.0,
                    "Pe_median": float(np.median(pe)),
                    "total_missing_users": total_missing,
                    "total_user_transmissions": total_users,
                    "Pe_total_missing_over_users": weighted,
                    "mean_equals_total_ratio_constant_Ka": equal_constant_ka,
                    "AMP_convergence_fraction": float(np.mean([r["amp_converged"] for r in sub])),
                    "tree_overflow_fraction": float(np.mean([r["tree_overflow"] for r in sub])),
                    "oversized_final_list_fraction": float(np.mean([r["oversized_final_list"] for r in sub])),
                    "mean_AMP_iterations": float(np.mean([r["amp_iterations"] for r in sub])),
                    "mean_final_list_size": float(np.mean([r["unique_decoded_messages"] for r in sub])),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def make_plot(summary: List[Dict[str, Any]], cfg: Config) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib import failed: {_PLOT_IMPORT_ERROR}")
    by = {(r["allocation"], float(r["EbN0_dB"])): r for r in summary}
    xs = np.array(cfg.EbN0_dB, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = {
        "2-level power allocation": dict(marker="o", linestyle="-", linewidth=1.8),
        "flat power allocation": dict(marker="s", linestyle="--", linewidth=1.8),
    }
    for allocation in ("2-level power allocation", "flat power allocation"):
        ys = np.array([by.get((allocation, float(x)), {}).get("Pe_mean", np.nan) for x in xs], dtype=np.float64)
        se = np.array([by.get((allocation, float(x)), {}).get("Pe_standard_error", np.nan) for x in xs], dtype=np.float64)
        if cfg.show_error_bars:
            ax.errorbar(xs, ys, yerr=se, label=allocation, **styles[allocation])
        else:
            ax.semilogy(xs, ys, label=allocation, **styles[allocation])
    ax.set_xlim(3.0, 5.5)
    ax.set_ylim(1e-2, 1.0)
    ax.set_xticks(np.arange(3.0, 5.6, 0.5))
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel(r"$P_e$")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    fig.savefig(cfg.plot_png, dpi=180)
    fig.savefig(cfg.plot_pdf)
    plt.close(fig)
    print(f"Updated plot: {cfg.plot_png}")
    print(f"Updated plot: {cfg.plot_pdf}")


def print_startup(cfg: Config) -> None:
    print("Figure 9 finite-length AMP + tree-code reproduction")
    print("----------------------------------------------------")
    print("Startup note: using n=26229, matching Figure 9 caption and introducing paragraph; Section X later says n=26226.")
    print(f"Ka={cfg.Ka}, J={cfg.J}, L={cfg.L}, B={cfg.B}, n={cfg.n}")
    print(f"Rin={cfg.Rin:.8f}")
    print(f"Rout={cfg.Rout:.8f}")
    print(f"R={cfg.R:.8f}")
    print(f"mu={cfg.mu:.8f}")
    print(f"parity={list(cfg.parity_bits)}")
    print(f"info={list(cfg.info_bits)}")
    print(f"Delta={cfg.Delta}, LIST_SIZE={cfg.list_size}")
    print(f"Eb/N0 grid={list(cfg.EbN0_dB)} dB")
    print(f"operator={cfg.operator_mode}")
    print(f"progress bars={'on' if cfg.progress_bars and tqdm is not None else 'off'}")
    if cfg.progress_bars and tqdm is None:
        print("tqdm is not installed; install tqdm to see live progress bars.")
    print("tree code=fixed across campaign")
    print(f"tree-code seed={cfg.tree_code_seed}")
    print(f"tree-code dimensions={tree_code_dimensions(make_tree_code(cfg))}")
    print(f"two-level ratio={cfg.high_to_low_ratio}")
    print(f"high sections={list(cfg.high_section_indices)}")
    print(f"trials={cfg.n_trials}")
    print(f"workers={cfg.n_workers}")
    print(f"plot PNG={cfg.plot_png}")
    print(f"plot PDF={cfg.plot_pdf}")
    print("Plots are regenerated after every completed batch and at resume completion.")
    print("The full simulation uses a structured Hadamard operator as a computationally tractable approximation to the paper's i.i.d. Gaussian ensemble.")
    print("The paper does not specify which two finite-L sections are strong. This reproduction uses sections [7 8].")


def assert_main_parameters(cfg: Config) -> None:
    assert cfg.Ka == 300 and cfg.J == 20 and cfg.L == 8 and cfg.B == 89 and cfg.n == 26229
    assert list(cfg.parity_bits) == [0, 9, 8, 9, 8, 9, 8, 20]
    assert list(cfg.info_bits) == [20, 11, 12, 11, 12, 11, 12, 0]
    assert sum(cfg.info_bits) == 89
    assert cfg.Delta == 50 and cfg.list_size == 350
    assert np.isclose(cfg.Rin, 0.0061, atol=5e-5)
    assert np.isclose(cfg.Rout, 0.55625, atol=1e-12)
    assert np.isclose(cfg.mu, 1.018, atol=5e-4)
    shapes = make_power_shapes(cfg)
    low_factor = cfg.L / (6 + 2 * cfg.high_to_low_ratio)
    high_factor = cfg.high_to_low_ratio * low_factor
    assert np.isclose(low_factor, 0.8163265306, atol=1e-10)
    assert np.isclose(high_factor, 1.5510204082, atol=1e-10)
    assert np.isclose(shapes["2-level power allocation"].mean(), 1.0, atol=1e-12)


def run_campaign(cfg: Config) -> None:
    assert_main_parameters(Config())
    print_startup(cfg)
    completed, rows = load_checkpoint(cfg)
    pending = [t for t in range(1, cfg.n_trials + 1) if t not in completed]
    print(f"resume status: completed_trials={len(completed)}/{cfg.n_trials}, pending_trials={len(pending)}, checkpoint={cfg.checkpoint_path}")
    if not pending:
        summary = summarize_rows(rows, cfg)
        write_csv(cfg.trials_csv, rows)
        write_csv(cfg.summary_csv, summary)
        if summary:
            make_plot(summary, cfg)
        print("No pending trials for this configuration. Existing checkpoint is complete.")
        print(f"summary CSV: {cfg.summary_csv}")
        print(f"trials CSV: {cfg.trials_csv}")
        print(f"plot PNG: {cfg.plot_png}")
        print(f"plot PDF: {cfg.plot_pdf}")
        return
    for start in range(0, len(pending), cfg.trial_batch_size):
        batch = pending[start : start + cfg.trial_batch_size]
        batch_rows: List[Dict[str, Any]] = []
        if cfg.n_workers > 1 and len(batch) > 1:
            with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
                futs = {ex.submit(run_trial, t, cfg): t for t in batch}
                futures_iter = as_completed(futs)
                futures_iter = progress_iter(
                    futures_iter,
                    cfg,
                    total=len(futs),
                    desc=f"batch trials {batch[0]}-{batch[-1]}",
                    unit="trial",
                    leave=True,
                    position=cfg.n_workers + 1,
                )
                for fut in futures_iter:
                    t = futs[fut]
                    trial_rows = fut.result()
                    batch_rows.extend(trial_rows)
                    print_progress(trial_rows)
                    completed.add(t)
        else:
            for t in batch:
                trial_rows = run_trial(t, cfg)
                batch_rows.extend(trial_rows)
                print_progress(trial_rows)
                completed.add(t)
        rows.extend(sorted(batch_rows, key=lambda r: (r["trial"], r["EbN0_dB"], r["allocation"])))
        summary = summarize_rows(rows, cfg)
        atomic_save_checkpoint(cfg, completed, rows)
        write_csv(cfg.trials_csv, rows)
        write_csv(cfg.summary_csv, summary)
        make_plot(summary, cfg)
        print(f"Completed batch {batch}; checkpoint={cfg.checkpoint_path}")


def print_progress(trial_rows: List[Dict[str, Any]]) -> None:
    for r in sorted(trial_rows, key=lambda x: (x["trial"], x["EbN0_dB"], x["allocation"])):
        print(
            "trial={trial} Eb/N0={EbN0_dB:g} allocation={allocation} "
            "Pe={Pe_trial:.4g} AMP_iter={amp_iterations} conv={amp_converged} "
            "final_list={unique_decoded_messages} overflow={tree_overflow} elapsed={elapsed_sec:.2f}s".format(**r),
            flush=True,
        )


def validate_parameters_and_power() -> None:
    cfg = Config()
    assert_main_parameters(cfg)
    shapes = make_power_shapes(cfg)
    for eb in cfg.EbN0_dB:
        avg = phat_avg(eb, cfg)
        assert np.isclose(np.mean(shapes["flat power allocation"]), 1.0)
        assert np.isclose(np.mean(shapes["2-level power allocation"]), 1.0)
        assert np.isclose(np.sum(avg * shapes["flat power allocation"]), cfg.L * avg)
        assert np.isclose(np.sum(avg * shapes["2-level power allocation"]), cfg.L * avg)


def validate_bit_conversion() -> None:
    rng = rng_from_seed(1)
    frags = rng.integers(0, 2, size=(100, 20), dtype=np.uint8)
    vals = bits_to_index(frags)
    assert np.array_equal(index_to_bits(vals, 20), frags)
    msgs = rng.integers(0, 2, size=(100, 89), dtype=np.uint8)
    assert np.array_equal(unpack_message_89(pack_message_89(msgs)), msgs)


def validate_tree_encoder_parity() -> None:
    cfg = validation_toy_config()
    G = make_tree_code(cfg)
    rng = rng_from_seed(2)
    messages = rng.integers(0, 2, size=(cfg.Ka, cfg.B), dtype=np.uint8)
    _, fragments = encode_messages(messages, G, cfg)
    prev = 0
    pos = 0
    for l in range(cfg.L):
        ib = cfg.info_bits[l]
        pb = cfg.parity_bits[l]
        if pb:
            expected = messages[:, :prev] @ G[l] % 2
            actual = fragments[:, l, ib:]
            assert np.array_equal(expected.astype(np.uint8), actual)
        pos += ib
        prev += ib
    assert pos == cfg.B


def true_lists_with_distractors(indices: np.ndarray, cfg: Config, n_distractors: int, seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = rng_from_seed(seed)
    out_idx: List[np.ndarray] = []
    out_scores: List[np.ndarray] = []
    for l in range(cfg.L):
        true_unique = np.unique(indices[:, l])
        distractors = rng.choice(cfg.N, size=n_distractors, replace=False)
        merged = np.unique(np.concatenate([true_unique, distractors])).astype(np.int64)
        scores = np.zeros(len(merged), dtype=np.float64)
        true_set = set(map(int, true_unique))
        for i, v in enumerate(merged):
            scores[i] = 10.0 if int(v) in true_set else rng.random()
        order = np.lexsort((merged, -scores))
        out_idx.append(merged[order])
        out_scores.append(scores[order])
    return out_idx, out_scores


def validate_tree_decoder() -> None:
    cfg = dataclasses.replace(
        validation_toy_config(),
        B=12,
        parity_bits=(0, 6, 7, 7),
        max_tree_paths=200_000,
    )
    G = make_tree_code(cfg)
    for seed in range(3, 200):
        rng = rng_from_seed(seed)
        messages = rng.integers(0, 2, size=(cfg.Ka, cfg.B), dtype=np.uint8)
        indices, _ = encode_messages(messages, G, cfg)
        li0, ls0 = true_lists_with_distractors(indices, cfg, 0, 5)
        dec0 = tree_decode(li0, ls0, G, cfg)
        pe0, _, _ = pupe(messages, dec0.decoded_keys, cfg)
        if pe0 == 0.0 and dec0.pre_cap_size <= cfg.Ka:
            break
    else:
        raise AssertionError("could not find a non-spurious deterministic toy tree instance")
    li, ls = true_lists_with_distractors(indices, cfg, 8, 4)
    dec = tree_decode(li, ls, G, cfg)
    pe, _, _ = pupe(messages, dec.decoded_keys, cfg)
    assert not dec.tree_overflow
    assert pe == 0.0
    li, ls = true_lists_with_distractors(indices, cfg, 0, 5)
    dec = tree_decode(li, ls, G, cfg)
    pe, _, _ = pupe(messages, dec.decoded_keys, cfg)
    assert pe == 0.0


def validate_orplus() -> None:
    rng = rng_from_seed(6)
    x = rng.normal(size=50)
    tau2 = 0.7
    phat = 4.2
    mean, div = orplus_denoise(x, tau2, phat, 300, 20)
    scalar = np.array([orplus_scalar(float(v), tau2, phat, 300, 20)[0] for v in x])
    deriv = np.array([orplus_scalar(float(v), tau2, phat, 300, 20)[1] for v in x])
    assert np.allclose(mean, scalar, atol=1e-7)
    assert np.isclose(div, np.sum(deriv), atol=1e-7)
    eps = 1e-5
    for v in x[:10]:
        mp = orplus_scalar(float(v + eps), tau2, phat, 300, 20)[0]
        mm = orplus_scalar(float(v - eps), tau2, phat, 300, 20)[0]
        fd = (mp - mm) / (2 * eps)
        an = orplus_scalar(float(v), tau2, phat, 300, 20)[1]
        assert np.isclose(fd, an, rtol=2e-4, atol=2e-5)


def validate_hadamard() -> None:
    cfg = dataclasses.replace(validation_toy_config("structured_hadamard"), J=8, n=96)
    op = create_operator(cfg, 9)
    rng = rng_from_seed(10)
    xs = [rng.normal(size=cfg.N).astype(np.float32) for _ in range(cfg.L)]
    y = rng.normal(size=cfg.n).astype(np.float32)
    ax = op.forward(xs)
    aty = op.adjoint(y)
    lhs = float(np.dot(ax.astype(np.float64), y.astype(np.float64)))
    rhs = sum(float(np.dot(xs[l].astype(np.float64), aty[l].astype(np.float64))) for l in range(cfg.L))
    assert np.isclose(lhs, rhs, rtol=2e-6, atol=2e-5)
    for l in range(cfg.L):
        for col in [0, 1, 7, 33, 127]:
            e = np.zeros(cfg.N, dtype=np.float32)
            e[col] = 1.0
            norm = np.linalg.norm(op.forward_section(l, e).astype(np.float64))
            assert np.isclose(norm, 1.0, atol=2e-6)


def deterministic_toy_run(cfg: Config) -> Tuple[Dict[str, Any], Dict[str, str]]:
    G = make_tree_code(cfg)
    ctx = make_trial_context(1, G, cfg)
    shapes = make_power_shapes(cfg)
    row = run_one_energy_allocation(1, cfg.EbN0_dB[0], "flat power allocation", shapes["flat power allocation"], ctx, G, cfg)
    comparable_row = {k: v for k, v in row.items() if k not in ("elapsed_sec", "amp_elapsed_sec")}
    meta = {
        "messages": hash_array(ctx["messages"]),
        "support": hash_array(ctx["indices"]),
        "noise": hash_array(ctx["noise"]),
        "operator": json.dumps(ctx["operator"].metadata(), sort_keys=True),
        "row": json.dumps(comparable_row, sort_keys=True),
    }
    return comparable_row, meta


def validate_determinism_and_pairing() -> None:
    cfg = validation_toy_config("dense_gaussian_debug")
    row1, meta1 = deterministic_toy_run(cfg)
    row2, meta2 = deterministic_toy_run(cfg)
    assert row1 == row2
    assert meta1 == meta2
    G = make_tree_code(cfg)
    ctx = make_trial_context(1, G, cfg)
    shapes = make_power_shapes(cfg)
    common_hash = (hash_array(ctx["messages"]), hash_array(ctx["indices"]), hash_array(ctx["noise"]), json.dumps(ctx["operator"].metadata(), sort_keys=True))
    _ = run_one_energy_allocation(1, cfg.EbN0_dB[0], "flat power allocation", shapes["flat power allocation"], ctx, G, cfg)
    after_hash = (hash_array(ctx["messages"]), hash_array(ctx["indices"]), hash_array(ctx["noise"]), json.dumps(ctx["operator"].metadata(), sort_keys=True))
    _ = run_one_energy_allocation(1, cfg.EbN0_dB[0], "2-level power allocation", shapes["2-level power allocation"], ctx, G, cfg)
    assert common_hash == after_hash


def validate_collision() -> None:
    cfg = validation_toy_config()
    G = make_tree_code(cfg)
    rng = rng_from_seed(12)
    messages = rng.integers(0, 2, size=(cfg.Ka, cfg.B), dtype=np.uint8)
    messages[1] = messages[0]
    indices, _ = encode_messages(messages, G, cfg)
    mult = multiplicity_vectors(indices, cfg)
    assert np.max(mult[0]) >= 2


def validate_final_cap_and_overflow() -> None:
    cfg = validation_toy_config()
    G = make_tree_code(cfg)
    rng = rng_from_seed(13)
    list_indices = []
    list_scores = []
    for _ in range(cfg.L):
        idx = rng.choice(cfg.N, size=cfg.Ka + 20, replace=False).astype(np.int64)
        scores = rng.random(size=idx.size)
        list_indices.append(idx)
        list_scores.append(scores)
    dec = tree_decode(list_indices, list_scores, G, cfg)
    assert dec.final_list_size <= cfg.Ka
    if dec.pre_cap_size > cfg.Ka:
        assert dec.oversized_final_list
    tiny = dataclasses.replace(cfg, max_tree_paths=1)
    dec2 = tree_decode(list_indices, list_scores, G, tiny)
    assert dec2.tree_overflow


def validate_end_to_end() -> None:
    dense = validation_toy_config("dense_gaussian_debug")
    row, _ = deterministic_toy_run(dense)
    assert np.isfinite(row["Pe_trial"])
    structured = validation_toy_config("structured_hadamard")
    row, _ = deterministic_toy_run(structured)
    assert np.isfinite(row["Pe_trial"])
    assert 0.0 <= row["Pe_trial"] <= 1.0


def run_validation() -> None:
    tests = [
        ("parameter/rate and power normalization", validate_parameters_and_power),
        ("bit conversion", validate_bit_conversion),
        ("tree encoder parity", validate_tree_encoder_parity),
        ("perfect-list/noiseless tree decoder", validate_tree_decoder),
        ("OR+ posterior and derivative", validate_orplus),
        ("Hadamard adjoint and column norm", validate_hadamard),
        ("determinism and paired allocations", validate_determinism_and_pairing),
        ("collision multiplicity", validate_collision),
        ("final-list cap and tree overflow", validate_final_cap_and_overflow),
        ("dense and structured end-to-end smoke", validate_end_to_end),
    ]
    print("Running validation suite...")
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"PASS {name} ({time.time() - t0:.2f}s)")
    print("All validation tests passed.")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="production", choices=["validate", "debug", "production"])
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--average-over-high-section-pairs", action="store_true")
    parser.add_argument("--show-error-bars", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--fresh", action="store_true", help="default behavior; kept for compatibility")
    parser.add_argument("--resume", action="store_true", help="resume from a compatible checkpoint instead of starting fresh")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.mode == "validate":
        run_validation()
        return 0
    cfg = debug_config() if args.mode == "debug" else Config()
    if args.trials is not None:
        cfg = dataclasses.replace(cfg, n_trials=args.trials)
    if args.workers is not None:
        cfg = dataclasses.replace(cfg, n_workers=args.workers)
    if args.average_over_high_section_pairs:
        cfg = dataclasses.replace(cfg, average_over_high_section_pairs=True)
    if args.show_error_bars:
        cfg = dataclasses.replace(cfg, show_error_bars=True)
    if args.no_progress:
        cfg = dataclasses.replace(cfg, progress_bars=False)
    if not args.resume:
        remove_existing_outputs(cfg)
        print("Starting fresh: removed existing checkpoint/CSV/plot outputs for this mode.")
    run_campaign(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
