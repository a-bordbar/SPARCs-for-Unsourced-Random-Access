import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_existing_path(path):
    candidate = Path(path)
    if candidate.exists():
        return candidate

    script_relative = SCRIPT_DIR / candidate
    if script_relative.exists():
        return script_relative

    root_relative = SCRIPT_DIR.parent / candidate
    if root_relative.exists():
        return root_relative

    data_relative = SCRIPT_DIR / "data" / "fig3_data" / candidate
    if data_relative.exists():
        return data_relative

    return candidate


def load_progress(path):
    path = resolve_existing_path(path)
    data = np.load(path, allow_pickle=True)
    required = ["J_vec", "Ka_vec", "R_tree"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    return data


def scalar_metadata(data, key, default="unknown"):
    if key not in data:
        return default
    value = data[key]
    if np.asarray(value).shape == ():
        return str(value.item())
    return str(value)


def available_npz_files():
    paths = set(Path(".").glob("**/*.npz"))
    paths.update(SCRIPT_DIR.glob("*.npz"))
    paths.update((SCRIPT_DIR.parent).glob("*.npz"))
    return sorted(str(path) for path in paths)


def color_for_j(J, j_idx):
    color_cycle = plt_colors()
    return color_cycle[j_idx % len(color_cycle)]


def plt_colors():
    return [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9",
    ]


def plot_progress_file(ax, path):
    data = load_progress(path)
    J_vec = data["J_vec"]
    Ka_vec = data["Ka_vec"]
    R_tree = data["R_tree"]

    allocation = scalar_metadata(data, "remainder_allocation", Path(path).stem)
    list_mode = scalar_metadata(data, "list_mode", "unknown")
    output_limit = scalar_metadata(data, "output_list_limit", "unknown")
    selection = scalar_metadata(data, "output_selection_mode", "unknown")

    for j_idx, J in enumerate(J_vec):
        label = f"{Path(path).stem}: {allocation}, J={int(J)}"
        if list_mode != "unknown" or output_limit != "unknown":
            label += f" ({list_mode}, limit={output_limit}, {selection})"
        ax.plot(
            Ka_vec,
            R_tree[j_idx],
            marker="o",
            linestyle="None",
            color=color_for_j(J, j_idx),
            label=label,
        )


def H2(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    out[mask] = -p[mask] * np.log2(p[mask]) - (1.0 - p[mask]) * np.log2(1.0 - p[mask])
    return out


def plot_fig3_bound(ax, J_vec, Ka_vec):
    p = 2.0 ** (-J_vec)
    for j_idx, J in enumerate(J_vec):
        p0 = (1.0 - p[j_idx]) ** Ka_vec
        R_bound = (2**J / Ka_vec) * H2(p0) / J
        ax.plot(
            Ka_vec,
            R_bound,
            linestyle="-",
            color=color_for_j(J, j_idx),
            label=f"fig3.py bound, J={int(J)}",
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Fig. 3 tree-code rate curves from one or more progress files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Progress .npz files. Pass two three-J files to get six curves.",
    )
    parser.add_argument(
        "--late",
        default=None,
        help="Progress file generated with remainder_allocation=late.",
    )
    parser.add_argument(
        "--early",
        default=None,
        help="Progress file generated with remainder_allocation=early.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. If omitted, show the plot interactively.",
    )
    parser.add_argument(
        "--no-fig3-bound",
        action="store_true",
        help="Do not add the three entropy-bound curves computed in fig3.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plot_cache_dir = Path(tempfile.gettempdir()) / "tree_code_plot_cache"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    (plot_cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir))
    import matplotlib.pyplot as plt

    files = list(args.files)
    if args.late:
        files.append(args.late)
    if args.early:
        files.append(args.early)
    if not files:
        default_file = resolve_existing_path("data/fig3_data/tree_code_progress.npz")
        if not default_file.exists():
            default_file = resolve_existing_path("tree_code_progress.npz")
        if default_file.exists():
            files = [str(default_file)]
        else:
            available = "\n  ".join(available_npz_files())
            raise SystemExit(
                "No progress files were provided and tree_code_progress.npz was not found.\n"
                f"Available .npz files:\n  {available}"
            )

    files = [str(resolve_existing_path(path)) for path in files]
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        available = "\n  ".join(available_npz_files())
        raise SystemExit(
            f"Missing progress file(s): {missing}\n"
            f"Available .npz files:\n  {available}"
        )

    fig, ax = plt.subplots()
    for path in files:
        plot_progress_file(ax, path)

    if not args.no_fig3_bound:
        first = load_progress(files[0])
        plot_fig3_bound(ax, first["J_vec"], first["Ka_vec"])

    ax.set_xlabel(r"$K_a$")
    ax.set_ylabel("Outer tree rate")
    ax.set_ylim([0.3, 0.9])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
