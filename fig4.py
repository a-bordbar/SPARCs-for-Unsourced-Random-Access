"""Reproduce Figure 4 from the SPARC unsourced-random-access paper.

Figure 4 replots the finite-length outer tree-code rates from the Figure 3
progress files against alpha = J / log2(Ka), compares them with the finite-J
entropy bound, and overlays the asymptotic outer-rate curve
R_out(alpha) = 1 - 1 / alpha.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROGRESS_FILES = (
    Path("data/fig3_data/tree_code_progress.npz"),
    Path("tree_code_progress.npz"),
)
REQUIRED_ARRAYS = ("J_vec", "Ka_vec", "R_tree")
OPTIONAL_GRID_ARRAYS = ("Cap_tree", "Pe_tree", "P_tree", "B_tree")
OPTIONAL_SCALAR_METADATA = (
    "remainder_allocation",
    "list_mode",
    "output_list_limit",
    "output_selection_mode",
)
CSV_COLUMNS = (
    "curve_type",
    "source_file",
    "remainder_allocation",
    "J",
    "Ka",
    "alpha",
    "rate",
    "reliable",
    "cap_hit_fraction",
    "pupe",
    "total_parity",
    "effective_information_bits",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
ALPHA_ANALYTIC = np.linspace(1.001, 4.5, 1500)


@dataclass(frozen=True)
class ProgressData:
    path: Path
    J_vec: np.ndarray
    Ka_vec: np.ndarray
    R_tree: np.ndarray
    optional_arrays: dict[str, np.ndarray]
    metadata: dict[str, str]


def resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path)
    candidates = (
        candidate,
        SCRIPT_DIR / candidate,
        SCRIPT_DIR.parent / candidate,
        SCRIPT_DIR / "data" / "fig3_data" / candidate,
    )
    for resolved in candidates:
        if resolved.exists():
            return resolved
    return candidate


def available_npz_files() -> list[Path]:
    paths: set[Path] = set()
    for root in (Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR / "data" / "fig3_data"):
        if root.exists():
            paths.update(root.glob("*.npz"))
            paths.update(root.glob("*/*.npz"))
    return sorted(paths)


def discover_progress_files(inputs: Iterable[str]) -> list[Path]:
    supplied = list(inputs)
    if supplied:
        files = [resolve_existing_path(path) for path in supplied]
    else:
        files = []
        for default in DEFAULT_PROGRESS_FILES:
            resolved = resolve_existing_path(default)
            if resolved.exists():
                files = [resolved]
                break

    missing = [path for path in files if not path.exists()]
    if missing or not files:
        available = available_npz_files()
        available_text = "\n  ".join(str(path) for path in available) or "(none found)"
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise SystemExit(
                f"Missing progress file(s): {missing_text}\n"
                f"Available .npz files:\n  {available_text}"
            )
        raise SystemExit(
            "No progress files were provided and no default progress file was found.\n"
            f"Available .npz files:\n  {available_text}"
        )

    return files


def scalar_metadata(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    return str(data[key])


def load_progress(path: Path) -> ProgressData:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_ARRAYS if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")

        J_vec = np.asarray(data["J_vec"]).reshape(-1)
        Ka_vec = np.asarray(data["Ka_vec"]).reshape(-1)
        R_tree = np.asarray(data["R_tree"], dtype=float)
        expected_shape = (len(J_vec), len(Ka_vec))

        if R_tree.shape != expected_shape:
            raise ValueError(
                f"{path}: R_tree.shape is {R_tree.shape}, expected {expected_shape} "
                "from len(J_vec) and len(Ka_vec)."
            )
        if np.any(Ka_vec <= 1):
            raise ValueError(f"{path}: all Ka_vec entries must be greater than 1.")

        optional_arrays: dict[str, np.ndarray] = {}
        for key in OPTIONAL_GRID_ARRAYS:
            if key in data:
                array = np.asarray(data[key], dtype=float)
                if array.shape != expected_shape:
                    raise ValueError(
                        f"{path}: optional array {key} has shape {array.shape}, "
                        f"expected {expected_shape}."
                    )
                optional_arrays[key] = array

        metadata = {
            key: scalar_metadata(data, key, default="")
            for key in OPTIONAL_SCALAR_METADATA
        }

    return ProgressData(
        path=path,
        J_vec=J_vec,
        Ka_vec=Ka_vec,
        R_tree=R_tree,
        optional_arrays=optional_arrays,
        metadata=metadata,
    )


def h2(p: Any) -> np.ndarray:
    p_array = np.asarray(p, dtype=float)
    out = np.zeros_like(p_array, dtype=float)
    mask = (p_array > 0.0) & (p_array < 1.0)
    out[mask] = (
        -p_array[mask] * np.log2(p_array[mask])
        - (1.0 - p_array[mask]) * np.log2(1.0 - p_array[mask])
    )
    return out


def entropy_bound_rate(J: float, Ka: Any) -> np.ndarray:
    Ka_array = np.asarray(Ka, dtype=float)
    p_active = -np.expm1(Ka_array * np.log1p(-(2.0 ** (-float(J)))))
    return (2.0 ** float(J) / (float(J) * Ka_array)) * h2(p_active)


def color_map_for_j(progress_files: list[ProgressData]) -> dict[int, str]:
    unique_j = sorted({int(J) for progress in progress_files for J in progress.J_vec})
    return {J: f"C{idx % 10}" for idx, J in enumerate(unique_j)}


def allocation_label(progress: ProgressData) -> str:
    allocation = progress.metadata.get("remainder_allocation", "")
    return allocation or progress.path.stem


def tree_legend_label(progress: ProgressData, J: int) -> str:
    allocation = allocation_label(progress)
    if allocation == progress.path.stem:
        return f"{progress.path.stem}, J={J}"
    return f"{progress.path.stem}: {allocation}, J={J}"


def get_optional_value(progress: ProgressData, key: str, j_idx: int, ka_idx: int) -> float:
    array = progress.optional_arrays.get(key)
    if array is None:
        return float("nan")
    return float(array[j_idx, ka_idx])


def make_tree_rows(progress_files: list[ProgressData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for progress in progress_files:
        cap_tree = progress.optional_arrays.get("Cap_tree")
        for j_idx, J_raw in enumerate(progress.J_vec):
            J = int(J_raw)
            alpha_vec = J / np.log2(progress.Ka_vec.astype(float))
            tree_rate = progress.R_tree[j_idx, :]
            if cap_tree is None:
                reliable_vec = np.isfinite(tree_rate)
            else:
                reliable_vec = cap_tree[j_idx, :] == 0.0
                reliable_vec &= np.isfinite(tree_rate)

            for ka_idx, Ka_raw in enumerate(progress.Ka_vec):
                rate = float(tree_rate[ka_idx])
                if not np.isfinite(rate):
                    continue
                rows.append(
                    {
                        "curve_type": "tree_code",
                        "source_file": progress.path.name,
                        "remainder_allocation": progress.metadata.get("remainder_allocation", ""),
                        "J": J,
                        "Ka": int(Ka_raw),
                        "alpha": float(alpha_vec[ka_idx]),
                        "rate": rate,
                        "reliable": bool(reliable_vec[ka_idx]),
                        "cap_hit_fraction": get_optional_value(progress, "Cap_tree", j_idx, ka_idx),
                        "pupe": get_optional_value(progress, "Pe_tree", j_idx, ka_idx),
                        "total_parity": get_optional_value(progress, "P_tree", j_idx, ka_idx),
                        "effective_information_bits": get_optional_value(progress, "B_tree", j_idx, ka_idx),
                    }
                )
    return rows


def make_entropy_rows(progress_files: list[ProgressData]) -> list[dict[str, Any]]:
    unique_j = sorted({int(J) for progress in progress_files for J in progress.J_vec})
    rows: list[dict[str, Any]] = []
    for J in unique_j:
        Ka_analytic = 2.0 ** (J / ALPHA_ANALYTIC)
        rates = entropy_bound_rate(J, Ka_analytic)
        for alpha, Ka_value, rate in zip(ALPHA_ANALYTIC, Ka_analytic, rates):
            rows.append(
                {
                    "curve_type": "entropy_bound",
                    "source_file": "analytic",
                    "remainder_allocation": "",
                    "J": J,
                    "Ka": float(Ka_value),
                    "alpha": float(alpha),
                    "rate": float(rate),
                    "reliable": True,
                    "cap_hit_fraction": float("nan"),
                    "pupe": float("nan"),
                    "total_parity": float("nan"),
                    "effective_information_bits": float("nan"),
                }
            )
    return rows


def make_asymptotic_rows() -> list[dict[str, Any]]:
    rates = 1.0 - 1.0 / ALPHA_ANALYTIC
    return [
        {
            "curve_type": "asymptotic",
            "source_file": "analytic",
            "remainder_allocation": "",
            "J": float("nan"),
            "Ka": float("nan"),
            "alpha": float(alpha),
            "rate": float(rate),
            "reliable": True,
            "cap_hit_fraction": float("nan"),
            "pupe": float("nan"),
            "total_parity": float("nan"),
            "effective_information_bits": float("nan"),
        }
        for alpha, rate in zip(ALPHA_ANALYTIC, rates)
    ]


def csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in CSV_COLUMNS})


def plot_figure(
    progress_files: list[ProgressData],
    tree_rows: list[dict[str, Any]],
    entropy_rows: list[dict[str, Any]],
    asymptotic_rows: list[dict[str, Any]],
    output_path: Path,
    pdf_output_path: Path | None,
    include_unreliable: bool,
    show: bool,
) -> None:
    plot_cache_dir = Path(tempfile.gettempdir()) / "tree_code_plot_cache"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    (plot_cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir))

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    colors = color_map_for_j(progress_files)
    fig, ax = plt.subplots(figsize=(7, 5))

    entropy_seen: set[int] = set()
    asymptotic_label_used = False
    for J in sorted({int(row["J"]) for row in entropy_rows}):
        j_rows = [row for row in entropy_rows if int(row["J"]) == J]
        j_rows.sort(key=lambda row: row["alpha"])
        ax.plot(
            [row["alpha"] for row in j_rows],
            [row["rate"] for row in j_rows],
            linestyle="-",
            color=colors[J],
            label=f"entropy bound, J={J}" if J not in entropy_seen else None,
        )
        entropy_seen.add(J)

    asymptotic_rows_sorted = sorted(asymptotic_rows, key=lambda row: row["alpha"])
    ax.plot(
        [row["alpha"] for row in asymptotic_rows_sorted],
        [row["rate"] for row in asymptotic_rows_sorted],
        linestyle="--",
        color="red",
        label=r"$R_{\mathrm{out}}(\alpha)=1-1/\alpha$" if not asymptotic_label_used else None,
    )
    asymptotic_label_used = True

    for file_idx, progress in enumerate(progress_files):
        marker = MARKERS[file_idx % len(MARKERS)]
        for J_raw in sorted(progress.J_vec):
            J = int(J_raw)
            rows = [
                row
                for row in tree_rows
                if row["source_file"] == progress.path.name and int(row["J"]) == J
            ]
            reliable_rows = sorted(
                [row for row in rows if row["reliable"]],
                key=lambda row: row["alpha"],
            )
            if reliable_rows:
                ax.plot(
                    [row["alpha"] for row in reliable_rows],
                    [row["rate"] for row in reliable_rows],
                    linestyle="None",
                    marker=marker,
                    color=colors[J],
                    label=tree_legend_label(progress, J),
                )

            if include_unreliable:
                unreliable_rows = sorted(
                    [row for row in rows if not row["reliable"]],
                    key=lambda row: row["alpha"],
                )
                if unreliable_rows:
                    ax.plot(
                        [row["alpha"] for row in unreliable_rows],
                        [row["rate"] for row in unreliable_rows],
                        linestyle="None",
                        marker="x",
                        color=colors[J],
                        label=f"{tree_legend_label(progress, J)} unreliable",
                    )

    ax.set_xlabel(r"$\alpha=J/\log_2 K_a$")
    ax.set_ylabel("Rate per user [J bits/outer-cu]")
    ax.set_xlim([1.0, 4.5])
    ax.set_ylim([0.0, 0.9])
    ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if pdf_output_path is not None:
        pdf_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output_path)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Figure 4 from Figure 3 tree-decoder progress files by "
            "plotting rate versus alpha = J/log2(Ka)."
        )
    )
    parser.add_argument(
        "progress_files",
        nargs="*",
        help="Figure 3 progress .npz files produced by fig3_tree_decoder.py.",
    )
    parser.add_argument(
        "--output",
        default="plots/fig4.png",
        help="Output image path. Default: plots/fig4.png",
    )
    parser.add_argument(
        "--pdf-output",
        default="plots/fig4.pdf",
        help="Optional PDF output path. Default: plots/fig4.pdf",
    )
    parser.add_argument(
        "--csv-output",
        default="data/fig4_data.csv",
        help="Output CSV path. Default: data/fig4_data.csv",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively after saving it.",
    )
    parser.add_argument(
        "--include-unreliable",
        action="store_true",
        help="Include Cap_tree > 0 tree-code points as x markers.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip saving the PDF.",
    )
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "reliable_tree": sum(
            row["curve_type"] == "tree_code" and bool(row["reliable"])
            for row in rows
        ),
        "unreliable_tree": sum(
            row["curve_type"] == "tree_code" and not bool(row["reliable"])
            for row in rows
        ),
        "entropy": sum(row["curve_type"] == "entropy_bound" for row in rows),
        "asymptotic": sum(row["curve_type"] == "asymptotic" for row in rows),
    }


def main() -> None:
    args = parse_args()
    progress_paths = discover_progress_files(args.progress_files)
    progress_files = [load_progress(path) for path in progress_paths]

    tree_rows = make_tree_rows(progress_files)
    entropy_rows = make_entropy_rows(progress_files)
    asymptotic_rows = make_asymptotic_rows()
    all_rows = tree_rows + entropy_rows + asymptotic_rows

    output_path = Path(args.output)
    pdf_output_path = None if args.no_pdf else Path(args.pdf_output)
    csv_output_path = Path(args.csv_output)

    write_csv(csv_output_path, all_rows)
    plot_figure(
        progress_files=progress_files,
        tree_rows=tree_rows,
        entropy_rows=entropy_rows,
        asymptotic_rows=asymptotic_rows,
        output_path=output_path,
        pdf_output_path=pdf_output_path,
        include_unreliable=args.include_unreliable,
        show=args.show,
    )

    counts = summarize(all_rows)
    print(f"Saved PNG: {output_path}")
    if pdf_output_path is not None:
        print(f"Saved PDF: {pdf_output_path}")
    print(f"Saved CSV: {csv_output_path}")
    print(
        "Saved rows: "
        f"{counts['reliable_tree']} reliable tree points, "
        f"{counts['unreliable_tree']} unreliable tree points, "
        f"{counts['entropy']} entropy points, "
        f"{counts['asymptotic']} asymptotic points."
    )


if __name__ == "__main__":
    main()
