#!/usr/bin/env python3
#ml numba/0.58.1-foss-2023a
#ml tqdm/4.66.1-GCCcore-12.3.0
#ml matplotlib/3.7.2-gfbf-2023a
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

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

try:
    import numba
except Exception as exc:  # pragma: no cover - import-time dependency check
    raise RuntimeError(
        "Numba is required for the optimized Figure 9 simulation. "
        "Install it in this Python environment, e.g. `conda install numba`."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import-time dependency check
    raise RuntimeError(
        "Matplotlib is required to generate Figure 9 plots. "
        "Load or install matplotlib in this Python environment."
    ) from exc

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception as exc:  # pragma: no cover - import-time dependency check
    raise RuntimeError(
        "tqdm is required for Figure 9 progress reporting. "
        "Load or install tqdm in this Python environment."
    ) from exc


SCHEMA_VERSION = 1
TREE_CODE_SEED = 19051031
BASE_MESSAGE_SEED = 400000
BASE_OPERATOR_SEED = 500000
BASE_NOISE_SEED = 600000


def default_worker_count() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return min(32, os.cpu_count() or 1)


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
    n_trials: int = 200
    trials_per_ebn0: Optional[Tuple[int, ...]] = (10, 10, 50, 1000, 200, 200)
    n_workers: int = dataclasses.field(default_factory=default_worker_count)
    trial_batch_size: int = 32
    show_error_bars: bool = False
    progress_bars: bool = True
    debug: bool = False
    validate_only: bool = False
    output_dir: str = "data/fig9"
    plot_png: str = "data/fig9/fig9_empirical_amp_tree.png"
    plot_pdf: str = "data/fig9/fig9_empirical_amp_tree.pdf"
    dense_gaussian_max_gb: float = 1200.0
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

    @property
    def trial_schedule(self) -> Tuple[int, ...]:
        if self.trials_per_ebn0 is None:
            return tuple(int(self.n_trials) for _ in self.EbN0_dB)
        if len(self.trials_per_ebn0) != len(self.EbN0_dB):
            raise ValueError("trials_per_ebn0 must have one entry per Eb/N0 point")
        return tuple(int(v) for v in self.trials_per_ebn0)

    @property
    def max_trials(self) -> int:
        return max(self.trial_schedule)


def debug_config() -> Config:
    return dataclasses.replace(
        Config(),
        n_trials=2,
        trials_per_ebn0=None,
        EbN0_dB=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5),
        n_workers=1,
        trial_batch_size=1,
        debug=True,
        output_dir="data/fig9_debug",
        plot_png="data/fig9_debug/fig9_empirical_amp_tree_debug.png",
        plot_pdf="data/fig9_debug/fig9_empirical_amp_tree_debug.pdf",
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
        trials_per_ebn0=None,
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


def multiplicity_vectors(indices: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.zeros((cfg.L, cfg.N), dtype=np.float32)
    for l in range(cfg.L):
        counts = np.bincount(indices[:, l], minlength=cfg.N).astype(np.float32)
        if int(counts.sum()) != cfg.Ka:
            raise AssertionError("section multiplicity must sum to Ka")
        out[l, :] = counts
    return out


def fwht_inplace_reference(x: np.ndarray) -> np.ndarray:
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


@numba.njit(cache=True, nogil=True)
def fwht_inplace_numba(x: np.ndarray) -> None:
    n = x.shape[0]
    h = 1
    while h < n:
        step = h * 2
        for block_start in range(0, n, step):
            for j in range(block_start, block_start + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h = step
    scale = 1.0 / math.sqrt(n)
    for i in range(n):
        x[i] = x[i] * scale


def fwht_inplace(x: np.ndarray) -> np.ndarray:
    fwht_inplace_numba(x)
    return x


@numba.njit(cache=True, nogil=True)
def squared_norm_float32(x: np.ndarray) -> float:
    total = 0.0
    flat = x.ravel()
    for i in range(flat.size):
        value = float(flat[i])
        total += value * value
    return total


@numba.njit(cache=True, nogil=True)
def residual_update_numba(y: np.ndarray, forward_output: np.ndarray, residual: np.ndarray, coeff: float, out: np.ndarray) -> None:
    for i in range(y.size):
        out[i] = y[i] - forward_output[i] + residual[i] * coeff


@numba.njit(cache=True, nogil=True)
def add_scaled_sections_numba(section_clean: np.ndarray, sqrt_shape: np.ndarray, out: np.ndarray) -> None:
    for i in range(out.size):
        total = 0.0
        for l in range(section_clean.shape[0]):
            total += float(sqrt_shape[l]) * float(section_clean[l, i])
        out[i] = total


@numba.njit(cache=True, nogil=True)
def orplus_denoise_section_numba(
    pseudo: np.ndarray,
    old_theta: np.ndarray,
    output_theta: np.ndarray,
    tau2: float,
    sqrt_phat: float,
    damping: float,
    logp0: float,
    logp1: float,
    logp2: float,
) -> Tuple[float, float, float]:
    divergence = 0.0
    diff2 = 0.0
    norm2 = 0.0
    a1 = sqrt_phat
    a2 = 2.0 * sqrt_phat
    inv2tau = 1.0 / (2.0 * tau2)
    for i in range(pseudo.size):
        x = float(pseudo[i])
        d0 = x
        d1 = x - a1
        d2 = x - a2
        lw0 = logp0 - d0 * d0 * inv2tau
        lw1 = logp1 - d1 * d1 * inv2tau
        lw2 = logp2 - d2 * d2 * inv2tau
        m = lw0
        if lw1 > m:
            m = lw1
        if lw2 > m:
            m = lw2
        w0 = math.exp(lw0 - m)
        w1 = math.exp(lw1 - m)
        w2 = math.exp(lw2 - m)
        denom = w0 + w1 + w2
        p1 = w1 / denom
        p2 = w2 / denom
        mean = p1 * a1 + p2 * a2
        second = p1 * a1 * a1 + p2 * a2 * a2
        var = second - mean * mean
        if var < 0.0:
            var = 0.0
        new_value = damping * mean + (1.0 - damping) * float(old_theta[i])
        output_theta[i] = new_value
        delta = new_value - float(old_theta[i])
        diff2 += delta * delta
        norm2 += new_value * new_value
        divergence += var / tau2
    return divergence, diff2, norm2


@dataclasses.dataclass
class StructuredHadamardOperator:
    cfg: Config
    rows: np.ndarray
    signs: np.ndarray
    scale: float

    @classmethod
    def create(cls, cfg: Config, seed: int) -> "StructuredHadamardOperator":
        rng = rng_from_seed(seed)
        if cfg.n > cfg.N:
            raise ValueError("structured_hadamard requires n <= 2^J")
        rows = np.empty((cfg.L, cfg.n), dtype=np.int32)
        signs = np.empty((cfg.L, cfg.N), dtype=np.float32)
        sign_choices = np.array([-1.0, 1.0], dtype=np.float32)
        for l in range(cfg.L):
            rows[l, :] = rng.choice(cfg.N, size=cfg.n, replace=False).astype(np.int32)
            signs[l, :] = rng.choice(sign_choices, size=cfg.N, replace=True)
        return cls(cfg=cfg, rows=np.ascontiguousarray(rows), signs=np.ascontiguousarray(signs), scale=math.sqrt(cfg.N / cfg.n))

    def forward_section(self, l: int, x: np.ndarray) -> np.ndarray:
        work = np.empty(self.cfg.N, dtype=np.float32)
        out = np.zeros(self.cfg.n, dtype=np.float32)
        self.forward_section_accumulate(l, x, out, work, 1.0)
        return out

    def adjoint_section(self, l: int, y: np.ndarray) -> np.ndarray:
        work = np.empty(self.cfg.N, dtype=np.float32)
        out = np.empty(self.cfg.N, dtype=np.float32)
        self.adjoint_section_into(l, y, out, work)
        return out

    def forward_section_accumulate(self, l: int, x: np.ndarray, output: np.ndarray, work: np.ndarray, coeff: float) -> None:
        np.multiply(np.asarray(x, dtype=np.float32), self.signs[l], out=work)
        fwht_inplace_numba(work)
        output += np.float32(coeff * self.scale) * work[self.rows[l]]

    def adjoint_section_into(self, l: int, y: np.ndarray, output: np.ndarray, work: np.ndarray) -> None:
        work.fill(0.0)
        work[self.rows[l]] = np.asarray(y, dtype=np.float32) * np.float32(self.scale)
        fwht_inplace_numba(work)
        np.multiply(work, self.signs[l], out=output)

    def forward_into(self, theta_sections: np.ndarray, output: np.ndarray, workspace: np.ndarray) -> None:
        output.fill(0.0)
        for l in range(self.cfg.L):
            self.forward_section_accumulate(l, theta_sections[l], output, workspace[l], 1.0)

    def adjoint_into(self, z: np.ndarray, output: np.ndarray, workspace: np.ndarray) -> None:
        for l in range(self.cfg.L):
            self.adjoint_section_into(l, z, output[l], workspace[l])

    def forward(self, theta_sections: Sequence[np.ndarray]) -> np.ndarray:
        y = np.zeros(self.cfg.n, dtype=np.float32)
        work = np.empty((self.cfg.L, self.cfg.N), dtype=np.float32)
        arr = np.asarray(theta_sections, dtype=np.float32)
        self.forward_into(arr, y, work)
        return y

    def adjoint(self, z: np.ndarray) -> List[np.ndarray]:
        out = np.empty((self.cfg.L, self.cfg.N), dtype=np.float32)
        work = np.empty_like(out)
        self.adjoint_into(z, out, work)
        return [out[l].copy() for l in range(self.cfg.L)]

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": "structured_hadamard",
            "row_hashes": [hash_array(self.rows[l]) for l in range(self.cfg.L)],
            "sign_hashes": [hash_array(self.signs[l]) for l in range(self.cfg.L)],
        }


@dataclasses.dataclass
class DenseGaussianOperator:
    cfg: Config
    mats: List[np.ndarray]

    @classmethod
    def create(cls, cfg: Config, seed: int) -> "DenseGaussianOperator":
        estimated_gb = cfg.L * cfg.n * cfg.N * 4.0 / (1024.0 ** 3)
        if cfg.operator_mode == "dense_gaussian_debug" and cfg.J > 12:
            raise ValueError("dense_gaussian_debug is only allowed for toy dimensions")
        if cfg.operator_mode == "dense_gaussian_full" and estimated_gb > cfg.dense_gaussian_max_gb:
            raise MemoryError(
                f"dense Gaussian codebook would require about {estimated_gb:.1f} GiB "
                f"for A_l matrices, above dense_gaussian_max_gb={cfg.dense_gaussian_max_gb:.1f}. "
                "Increase --gaussian-max-gb only on a node with sufficient memory."
            )
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

    def forward_into(self, theta_sections: np.ndarray, output: np.ndarray, workspace: Optional[np.ndarray] = None) -> None:
        output.fill(0.0)
        for l in range(self.cfg.L):
            output += self.forward_section(l, theta_sections[l])

    def adjoint_into(self, z: np.ndarray, output: np.ndarray, workspace: Optional[np.ndarray] = None) -> None:
        for l in range(self.cfg.L):
            output[l, :] = self.adjoint_section(l, z)

    def forward(self, theta_sections: Sequence[np.ndarray]) -> np.ndarray:
        y = np.zeros(self.cfg.n, dtype=np.float32)
        self.forward_into(np.asarray(theta_sections, dtype=np.float32), y, None)
        return y

    def adjoint(self, z: np.ndarray) -> List[np.ndarray]:
        return [self.adjoint_section(l, z) for l in range(self.cfg.L)]

    def metadata(self) -> Dict[str, Any]:
        return {"mode": self.cfg.operator_mode, "mat_hashes": [hash_array(m) for m in self.mats]}


@dataclasses.dataclass
class StreamingGaussianOperator:
    cfg: Config
    seed: int
    chunk_rows: int = 256

    @classmethod
    def create(cls, cfg: Config, seed: int) -> "StreamingGaussianOperator":
        return cls(cfg=cfg, seed=seed)

    def _section_seed(self, l: int) -> int:
        return int((self.seed + 1000003 * (l + 1)) % (2**63 - 1))

    def _row_blocks(self, l: int) -> Iterable[Tuple[int, int, np.ndarray]]:
        rng = rng_from_seed(self._section_seed(l))
        scale = np.float32(1.0 / math.sqrt(self.cfg.n))
        for start in range(0, self.cfg.n, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.cfg.n)
            block = rng.normal(0.0, scale, size=(stop - start, self.cfg.N)).astype(np.float32)
            yield start, stop, block

    def forward_section(self, l: int, x: np.ndarray) -> np.ndarray:
        out = np.empty(self.cfg.n, dtype=np.float32)
        x32 = np.asarray(x, dtype=np.float32)
        for start, stop, block in self._row_blocks(l):
            out[start:stop] = block @ x32
        return out

    def adjoint_section(self, l: int, y: np.ndarray) -> np.ndarray:
        out = np.zeros(self.cfg.N, dtype=np.float32)
        y32 = np.asarray(y, dtype=np.float32)
        for start, stop, block in self._row_blocks(l):
            out += block.T @ y32[start:stop]
        return out

    def forward_into(self, theta_sections: np.ndarray, output: np.ndarray, workspace: Optional[np.ndarray] = None) -> None:
        output.fill(0.0)
        for l in range(self.cfg.L):
            output += self.forward_section(l, theta_sections[l])

    def adjoint_into(self, z: np.ndarray, output: np.ndarray, workspace: Optional[np.ndarray] = None) -> None:
        for l in range(self.cfg.L):
            output[l, :] = self.adjoint_section(l, z)

    def forward(self, theta_sections: Sequence[np.ndarray]) -> np.ndarray:
        y = np.zeros(self.cfg.n, dtype=np.float32)
        self.forward_into(np.asarray(theta_sections, dtype=np.float32), y, None)
        return y

    def adjoint(self, z: np.ndarray) -> List[np.ndarray]:
        out = np.empty((self.cfg.L, self.cfg.N), dtype=np.float32)
        self.adjoint_into(np.asarray(z, dtype=np.float32), out, None)
        return [out[l].copy() for l in range(self.cfg.L)]

    def metadata(self) -> Dict[str, Any]:
        return {"mode": "streaming_gaussian", "seed": int(self.seed), "chunk_rows": int(self.chunk_rows)}


Operator = Union[StructuredHadamardOperator, DenseGaussianOperator, StreamingGaussianOperator]


def create_operator(cfg: Config, seed: int) -> Operator:
    if cfg.operator_mode == "structured_hadamard":
        return StructuredHadamardOperator.create(cfg, seed)
    if cfg.operator_mode in {"dense_gaussian_debug", "dense_gaussian_full"}:
        return DenseGaussianOperator.create(cfg, seed)
    if cfg.operator_mode == "streaming_gaussian":
        return StreamingGaussianOperator.create(cfg, seed)
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
    theta: np.ndarray
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
    # Persistent large buffers owned by this AMP call:
    # theta/theta_new/pseudo/adjoint/fwht_workspace are (L,N) float32 arrays;
    # residual/residual_new/forward_output are n-length float32 arrays.
    theta = np.zeros((cfg.L, cfg.N), dtype=np.float32)
    theta_new = np.empty_like(theta)
    adjoint = np.empty_like(theta)
    pseudo = np.empty_like(theta)
    fwht_workspace = np.empty_like(theta)
    forward_output = np.empty(cfg.n, dtype=np.float32)
    residual = np.asarray(y, dtype=np.float32).copy()
    residual_new = np.empty_like(residual)
    y32 = np.asarray(y, dtype=np.float32)
    stable = 0
    rel_change = math.inf
    final_tau2 = math.nan
    converged = False
    q = 2.0 ** (-cfg.J)
    p0 = (1.0 - q) ** cfg.Ka
    p1 = cfg.Ka * q * (1.0 - q) ** (cfg.Ka - 1)
    p2 = max(1.0 - p0 - p1, np.finfo(float).tiny)
    logp0 = math.log(max(p0, np.finfo(float).tiny))
    logp1 = math.log(max(p1, np.finfo(float).tiny))
    logp2 = math.log(p2)
    sqrt_phat = np.sqrt(np.asarray(phat, dtype=np.float64))
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
        final_tau2 = squared_norm_float32(residual) / cfg.n
        op.adjoint_into(residual, adjoint, fwht_workspace)
        divergence = 0.0
        diff2 = 0.0
        norm2 = 0.0
        for l in range(cfg.L):
            np.add(theta[l], adjoint[l], out=pseudo[l])
            div_l, diff_l, norm_l = orplus_denoise_section_numba(
                pseudo[l],
                theta[l],
                theta_new[l],
                final_tau2,
                float(sqrt_phat[l]),
                float(cfg.amp_damping),
                logp0,
                logp1,
                logp2,
            )
            divergence += div_l
            diff2 += diff_l
            norm2 += norm_l
        rel_change = math.sqrt(diff2) / max(math.sqrt(norm2), 1.0)
        op.forward_into(theta_new, forward_output, fwht_workspace)
        residual_update_numba(y32, forward_output, residual, divergence / cfg.n, residual_new)
        theta, theta_new = theta_new, theta
        residual, residual_new = residual_new, residual
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
    section_clean = precompute_section_clean(op, multiplicities, cfg)
    noise_rng = rng_from_seed(trial_noise_seed(trial, cfg))
    noise = noise_rng.normal(0.0, 1.0, size=cfg.n).astype(np.float32)
    return {
        "messages": messages,
        "indices": indices,
        "fragments": fragments,
        "multiplicities": multiplicities,
        "operator": op,
        "section_clean": section_clean,
        "noise": noise,
    }


def precompute_section_clean(op: Operator, multiplicities: np.ndarray, cfg: Config) -> np.ndarray:
    section_clean = np.empty((cfg.L, cfg.n), dtype=np.float32)
    if isinstance(op, StructuredHadamardOperator):
        work = np.empty(cfg.N, dtype=np.float32)
        tmp = np.zeros(cfg.n, dtype=np.float32)
        for l in range(cfg.L):
            tmp.fill(0.0)
            op.forward_section_accumulate(l, multiplicities[l], tmp, work, 1.0)
            section_clean[l, :] = tmp
    else:
        for l in range(cfg.L):
            section_clean[l, :] = op.forward_section(l, multiplicities[l])
    return section_clean


def build_unit_signal(section_clean: np.ndarray, shape: np.ndarray) -> np.ndarray:
    out = np.empty(section_clean.shape[1], dtype=np.float32)
    add_scaled_sections_numba(section_clean, np.sqrt(np.asarray(shape, dtype=np.float64)), out)
    return out


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
    if "unit_signals" not in ctx:
        ctx["unit_signals"] = {}
    shape_key = tuple(np.asarray(shape, dtype=np.float64).round(15))
    if shape_key not in ctx["unit_signals"]:
        ctx["unit_signals"][shape_key] = build_unit_signal(ctx["section_clean"], shape)
    y = (math.sqrt(avg) * ctx["unit_signals"][shape_key] + ctx["noise"]).astype(np.float32)
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
        for eb_idx, eb in enumerate(cfg.EbN0_dB):
            if trial > cfg.trial_schedule[eb_idx]:
                continue
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
        "trials_per_ebn0": cfg.trials_per_ebn0,
        "trial_schedule": cfg.trial_schedule,
        "max_trials": cfg.max_trials,
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
        "dense_gaussian_max_gb": cfg.dense_gaussian_max_gb,
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
    ax.set_ylim(5e-3, 1.0)
    ax.set_xticks(np.arange(3.0, 5.6, 0.5))
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel(r"$P_e$")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    script_dir = Path(__file__).resolve().parent
    plot_png = script_dir / cfg.plot_png
    plot_pdf = script_dir / cfg.plot_pdf
    fig.savefig(plot_png, dpi=180)
    fig.savefig(plot_pdf)
    plt.close(fig)
    print(f"Updated plot: {plot_png}")
    print(f"Updated plot: {plot_pdf}")


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
    print(f"trials per Eb/N0={list(cfg.trial_schedule)}")
    print(f"operator={cfg.operator_mode}")
    print(f"progress bars={'on' if cfg.progress_bars and tqdm is not None else 'off'}")
    if cfg.progress_bars and tqdm is None:
        print("tqdm is not installed; install tqdm to see live progress bars.")
    print("tree code=fixed across campaign")
    print(f"tree-code seed={cfg.tree_code_seed}")
    print(f"tree-code dimensions={tree_code_dimensions(make_tree_code(cfg))}")
    print(f"two-level ratio={cfg.high_to_low_ratio}")
    print(f"high sections={list(cfg.high_section_indices)}")
    print(f"max trials={cfg.max_trials}")
    print(f"workers={cfg.n_workers}")
    mem = estimate_memory_mb(cfg)
    print(f"estimated persistent memory per worker={mem['persistent_mb']:.0f} MB")
    print(f"estimated peak memory per worker={mem['peak_mb']:.0f} MB")
    print(f"estimated total memory at selected workers={mem['peak_mb'] * cfg.n_workers / 1024:.2f} GB")
    print(f"plot PNG={cfg.plot_png}")
    print(f"plot PDF={cfg.plot_pdf}")
    print("Plots are regenerated after every completed batch and at resume completion.")
    print("The full simulation uses a structured Hadamard operator as a computationally tractable approximation to the paper's i.i.d. Gaussian ensemble.")
    print("The paper does not specify which two finite-L sections are strong. This reproduction uses sections [7 8].")


def estimate_memory_mb(cfg: Config) -> Dict[str, float]:
    f32 = 4
    i32 = 4
    amp_arrays = 5 * cfg.L * cfg.N * f32
    residual_arrays = 3 * cfg.n * f32
    multiplicities = cfg.L * cfg.N * f32
    signs = cfg.L * cfg.N * f32
    rows = cfg.L * cfg.n * i32
    section_clean = cfg.L * cfg.n * f32
    persistent = amp_arrays + residual_arrays + multiplicities + signs + rows + section_clean
    peak = persistent + 2 * cfg.L * cfg.N * f32
    return {"persistent_mb": persistent / (1024 ** 2), "peak_mb": peak / (1024 ** 2)}


def warm_up_numba() -> None:
    x = np.arange(8, dtype=np.float32)
    fwht_inplace_numba(x)
    y = np.ones(8, dtype=np.float32)
    out = np.empty(8, dtype=np.float32)
    _ = squared_norm_float32(y)
    residual_update_numba(y, y, y, 0.1, out)
    sections = np.ones((2, 8), dtype=np.float32)
    weights = np.ones(2, dtype=np.float64)
    add_scaled_sections_numba(sections, weights, out)
    _ = orplus_denoise_section_numba(y, y, out, 1.0, 1.0, 1.0, -0.1, -8.0, -12.0)
    print("Numba kernels warmed up")
    print("FWHT kernel compiled")
    print("OR+ kernel compiled")


def assert_main_parameters(cfg: Config) -> None:
    assert cfg.Ka == 300 and cfg.J == 20 and cfg.L == 8 and cfg.B == 89 and cfg.n == 26229
    assert list(cfg.parity_bits) == [0, 9, 8, 9, 8, 9, 8, 20]
    assert list(cfg.info_bits) == [20, 11, 12, 11, 12, 11, 12, 0]
    assert sum(cfg.info_bits) == 89
    assert cfg.Delta == 50 and cfg.list_size == 350
    assert list(cfg.trial_schedule) == [10, 10, 50, 1000, 200, 200]
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
    warm_up_numba()
    print_startup(cfg)
    completed, rows = load_checkpoint(cfg)
    pending = [t for t in range(1, cfg.max_trials + 1) if t not in completed]
    print(f"resume status: completed_trials={len(completed)}/{cfg.max_trials}, pending_trials={len(pending)}, checkpoint={cfg.checkpoint_path}")
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
    batch_rows: List[Dict[str, Any]] = []
    batch_trials: List[int] = []

    def flush_checkpoint() -> None:
        nonlocal rows, batch_rows, batch_trials
        if not batch_rows:
            return
        rows.extend(sorted(batch_rows, key=lambda r: (r["trial"], r["EbN0_dB"], r["allocation"])))
        rows.sort(key=lambda r: (r["trial"], r["EbN0_dB"], r["allocation"], r.get("high_sections_pair", "")))
        summary = summarize_rows(rows, cfg)
        atomic_save_checkpoint(cfg, completed, rows)
        write_csv(cfg.trials_csv, rows)
        write_csv(cfg.summary_csv, summary)
        make_plot(summary, cfg)
        print(f"Completed trials {batch_trials}; checkpoint={cfg.checkpoint_path}")
        batch_rows = []
        batch_trials = []

    if cfg.n_workers > 1 and len(pending) > 1:
        max_queued = max(cfg.n_workers, 2 * cfg.n_workers)
        pending_iter = iter(pending)
        futures: Dict[Any, int] = {}
        with ProcessPoolExecutor(max_workers=cfg.n_workers) as ex:
            try:
                for _ in range(min(max_queued, len(pending))):
                    t = next(pending_iter)
                    futures[ex.submit(run_trial, t, cfg)] = t
                pbar = tqdm(total=len(pending), desc="production trials", unit="trial", dynamic_ncols=True) if cfg.progress_bars and tqdm is not None else None
                while futures:
                    for fut in as_completed(list(futures.keys())):
                        t = futures.pop(fut)
                        trial_rows = fut.result()
                        batch_rows.extend(trial_rows)
                        print_progress(trial_rows)
                        completed.add(t)
                        batch_trials.append(t)
                        if pbar is not None:
                            pbar.update(1)
                        try:
                            next_t = next(pending_iter)
                        except StopIteration:
                            pass
                        else:
                            futures[ex.submit(run_trial, next_t, cfg)] = next_t
                        if len(batch_trials) >= cfg.trial_batch_size:
                            flush_checkpoint()
                        break
                if pbar is not None:
                    pbar.close()
            except KeyboardInterrupt:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    else:
        for t in pending:
            trial_rows = run_trial(t, cfg)
            batch_rows.extend(trial_rows)
            print_progress(trial_rows)
            completed.add(t)
            batch_trials.append(t)
            if len(batch_trials) >= cfg.trial_batch_size:
                flush_checkpoint()
    flush_checkpoint()


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
    old = rng.normal(size=x.size).astype(np.float32)
    out = np.empty_like(old)
    damping = 0.73
    q = 2.0 ** (-20)
    p0 = (1.0 - q) ** 300
    p1 = 300 * q * (1.0 - q) ** 299
    p2 = max(1.0 - p0 - p1, np.finfo(float).tiny)
    div_fused, diff2_fused, norm2_fused = orplus_denoise_section_numba(
        x.astype(np.float32),
        old,
        out,
        tau2,
        math.sqrt(phat),
        damping,
        math.log(p0),
        math.log(p1),
        math.log(p2),
    )
    undamped, div_ref = orplus_denoise(x.astype(np.float32), tau2, phat, 300, 20)
    damped_ref = (damping * undamped + (1.0 - damping) * old).astype(np.float32)
    assert np.allclose(out, damped_ref, atol=1e-6)
    assert np.isclose(div_fused, div_ref, rtol=1e-7, atol=1e-7)
    assert np.isclose(diff2_fused, np.sum((damped_ref.astype(np.float64) - old.astype(np.float64)) ** 2), rtol=1e-7)
    assert np.isclose(norm2_fused, np.sum(damped_ref.astype(np.float64) ** 2), rtol=1e-7)


def validate_hadamard() -> None:
    cfg = dataclasses.replace(validation_toy_config("structured_hadamard"), J=8, n=96)
    op = create_operator(cfg, 9)
    rng = rng_from_seed(10)
    for nbits in [8, 16, 256]:
        a = rng.normal(size=nbits).astype(np.float32)
        b = a.copy()
        fwht_inplace_reference(a)
        fwht_inplace_numba(b)
        assert np.allclose(a, b, atol=1e-6)
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


def validate_streaming_gaussian() -> None:
    cfg = validation_toy_config("dense_gaussian_debug")
    stream_cfg = dataclasses.replace(cfg, operator_mode="streaming_gaussian")
    stream = create_operator(stream_cfg, 123)
    stream_again = create_operator(stream_cfg, 123)
    rng = rng_from_seed(124)
    x = rng.normal(size=(cfg.L, cfg.N)).astype(np.float32)
    y = rng.normal(size=cfg.n).astype(np.float32)
    ax = stream.forward(x)
    ax_again = stream_again.forward(x)
    assert np.allclose(ax, ax_again, rtol=0.0, atol=0.0)
    aty = np.asarray(stream.adjoint(y))
    lhs = float(np.dot(ax.astype(np.float64), y.astype(np.float64)))
    rhs = sum(float(np.dot(x[l].astype(np.float64), aty[l].astype(np.float64))) for l in range(cfg.L))
    assert np.isclose(lhs, rhs, rtol=2e-6, atol=2e-5)


def validate_unit_signal_identity() -> None:
    cfg = dataclasses.replace(validation_toy_config("structured_hadamard"), J=8, n=128)
    G = make_tree_code(cfg)
    ctx = make_trial_context(1, G, cfg)
    shape = make_power_shapes(cfg)["2-level power allocation"]
    avg = phat_avg(cfg.EbN0_dB[0], cfg)
    direct_theta = np.empty((cfg.L, cfg.N), dtype=np.float32)
    for l in range(cfg.L):
        direct_theta[l, :] = (math.sqrt(avg * shape[l]) * ctx["multiplicities"][l]).astype(np.float32)
    direct = ctx["operator"].forward(direct_theta)
    unit = build_unit_signal(ctx["section_clean"], shape)
    assert np.allclose(direct, math.sqrt(avg) * unit, atol=1e-5)


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
        ("streaming Gaussian matches dense toy operator", validate_streaming_gaussian),
        ("unit-signal precomputation identity", validate_unit_signal_identity),
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


def peak_rss_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024 ** 2)
        return rss / 1024.0
    except Exception:
        return float("nan")


def benchmark_amp(workers: Sequence[int], iterations: int = 2) -> None:
    warm_up_numba()
    print("workers,iterations,elapsed_sec,sec_per_iter,peak_rss_mb")
    for worker_count in workers:
        cfg = dataclasses.replace(
            Config(),
            amp_max_iter=iterations,
            amp_min_iter=iterations + 10,
            amp_required_stable_iterations=iterations + 10,
            progress_bars=False,
            n_workers=int(worker_count),
        )
        G = make_tree_code(cfg)
        ctx = make_trial_context(1, G, cfg)
        shape = make_power_shapes(cfg)["flat power allocation"]
        avg = phat_avg(cfg.EbN0_dB[0], cfg)
        phat = avg * shape
        y = (math.sqrt(avg) * build_unit_signal(ctx["section_clean"], shape) + ctx["noise"]).astype(np.float32)
        t0 = time.perf_counter()
        result = amp_decode(y, ctx["operator"], phat, cfg)
        elapsed = time.perf_counter() - t0
        print(f"{worker_count},{result.iterations},{elapsed:.6f},{elapsed / result.iterations:.6f},{peak_rss_mb():.1f}")


def gaussian_diagnostic_config(args: argparse.Namespace, operator_mode: str) -> Config:
    return dataclasses.replace(
        Config(),
        EbN0_dB=(5.0,),
        trials_per_ebn0=None,
        n_trials=int(args.trials or 1),
        n_workers=1,
        trial_batch_size=1,
        operator_mode=operator_mode,
        progress_bars=not args.no_progress,
        output_dir="data/fig9_gaussian_diagnostic",
        dense_gaussian_max_gb=float(args.gaussian_max_gb),
    )


def run_gaussian_diagnostic(args: argparse.Namespace) -> None:
    print("Controlled i.i.d.-Gaussian codebook diagnostic")
    print("Purpose: compare Eb/N0=5 dB flat-power PUPE for structured Hadamard versus true dense i.i.d. Gaussian.")
    print("This diagnostic does not replace the structured-Hadamard Figure 9 production simulation.")
    print("WARNING: full J=20 dense Gaussian matrices are extremely memory intensive.")
    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    out_dir = Path("data/fig9_gaussian_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "fig9_gaussian_diagnostic_checkpoint.pkl"
    csv_path = out_dir / "fig9_gaussian_diagnostic_trials.csv"
    summary_path = out_dir / "fig9_gaussian_diagnostic_summary.csv"
    if args.resume and checkpoint.exists():
        with checkpoint.open("rb") as f:
            payload = pickle.load(f)
        rows = list(payload.get("rows", []))
        print(f"Resume summary: loaded {len(rows)} diagnostic rows from {checkpoint}")
    completed = {(r["operator"], int(r["trial"])) for r in rows}
    for operator_mode, label in [
        ("structured_hadamard", "Hadamard"),
        (args.gaussian_operator_mode, "Gaussian"),
    ]:
        cfg = gaussian_diagnostic_config(args, operator_mode)
        estimated_gb = cfg.L * cfg.n * cfg.N * 4.0 / (1024.0 ** 3) if operator_mode == "dense_gaussian_full" else 0.0
        if operator_mode == "dense_gaussian_full":
            print(
                f"Dense Gaussian matrix estimate: {estimated_gb:.1f} GiB "
                f"for L={cfg.L}, n={cfg.n}, N={cfg.N}; cap={cfg.dense_gaussian_max_gb:.1f} GiB."
            )
        if operator_mode == "streaming_gaussian":
            print("Streaming Gaussian mode: regenerates deterministic i.i.d. Gaussian row blocks; avoids dense matrix storage but is much slower than Hadamard.")
        G = make_tree_code(cfg)
        shape = make_power_shapes(cfg)["flat power allocation"]
        for trial in range(1, cfg.n_trials + 1):
            if (label, trial) in completed:
                continue
            t0 = time.time()
            ctx = make_trial_context(trial, G, cfg)
            row = run_one_energy_allocation(trial, 5.0, "flat power allocation", shape, ctx, G, cfg)
            row["operator"] = label
            row["operator_mode"] = operator_mode
            row["elapsed_sec_total"] = time.time() - t0
            rows.append(row)
            write_csv(csv_path, rows)
            with checkpoint.open("wb") as f:
                pickle.dump({"rows": rows, "n_trials": cfg.n_trials}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"{label} trial={trial} Pe={row['Pe_trial']:.4g} "
                f"AMP_iter={row['amp_iterations']} conv={row['amp_converged']} "
                f"final_list={row['unique_decoded_messages']} elapsed={row['elapsed_sec_total']:.2f}s"
            )
    for label in ["Hadamard", "Gaussian"]:
        sub = [r for r in rows if r["operator"] == label]
        if not sub:
            continue
        pe = np.asarray([r["Pe_trial"] for r in sub], dtype=np.float64)
        summary.append(
            {
                "operator": label,
                "EbN0_dB": 5.0,
                "allocation": "flat power allocation",
                "trials": len(sub),
                "Pe_mean": float(np.mean(pe)),
                "Pe_standard_error": float(np.std(pe, ddof=1) / math.sqrt(len(pe))) if len(pe) > 1 else 0.0,
                "Pe_median": float(np.median(pe)),
                "AMP_convergence_fraction": float(np.mean([r["amp_converged"] for r in sub])),
                "mean_AMP_iterations": float(np.mean([r["amp_iterations"] for r in sub])),
                "tree_overflow_fraction": float(np.mean([r["tree_overflow"] for r in sub])),
                "mean_final_list_size": float(np.mean([r["unique_decoded_messages"] for r in sub])),
            }
        )
    write_csv(summary_path, summary)
    print(f"Diagnostic trials CSV: {csv_path}")
    print(f"Diagnostic summary CSV: {summary_path}")
    for row in summary:
        print(f"{row['operator']}: Pe_mean={row['Pe_mean']:.6g} over {row['trials']} trials")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="production", choices=["validate", "debug", "production", "benchmark", "gaussian-diagnostic"])
    parser.add_argument("--trials", type=int, default=None, help="uniform trial-count override for every Eb/N0 point")
    parser.add_argument("--workers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="completed trials per checkpoint/CSV/plot flush")
    parser.add_argument("--benchmark-iterations", type=int, default=2)
    parser.add_argument("--operator-mode", choices=["structured_hadamard", "streaming_gaussian", "dense_gaussian_full"], default=None)
    parser.add_argument("--gaussian-operator-mode", choices=["streaming_gaussian", "dense_gaussian_full"], default="streaming_gaussian")
    parser.add_argument("--gaussian-max-gb", type=float, default=1200.0, help="maximum dense Gaussian matrix memory allowed for gaussian-diagnostic")
    parser.add_argument("--average-over-high-section-pairs", action="store_true")
    parser.add_argument("--show-error-bars", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    parser.add_argument("--fresh", action="store_true", help="default behavior; kept for compatibility")
    parser.add_argument("--resume", action="store_true", help="resume from a compatible checkpoint instead of starting fresh")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.mode == "validate":
        warm_up_numba()
        run_validation()
        return 0
    if args.mode == "benchmark":
        benchmark_amp(args.workers or [1], args.benchmark_iterations)
        return 0
    if args.mode == "gaussian-diagnostic":
        run_gaussian_diagnostic(args)
        return 0
    cfg = debug_config() if args.mode == "debug" else Config()
    if args.trials is not None:
        cfg = dataclasses.replace(cfg, n_trials=args.trials, trials_per_ebn0=None)
    if args.workers is not None:
        cfg = dataclasses.replace(cfg, n_workers=int(args.workers[0]))
    if args.operator_mode is not None:
        cfg = dataclasses.replace(cfg, operator_mode=args.operator_mode)
        if args.operator_mode == "streaming_gaussian" and cfg.n_workers > 1:
            print(
                "WARNING: streaming Gaussian is running with concurrent AMP workers. "
                "Each worker regenerates deterministic Gaussian row blocks independently; "
                "this can be CPU and memory-bandwidth intensive."
            )
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, trial_batch_size=int(args.batch_size))
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
