# SPARCs for Unsourced Random Access

This repository contains Python reproductions and simulation utilities for figures from "SPARCs for Unsourced Random Access" by Alexander Fengler, Peter Jung, and Giuseppe Caire. The code combines state-evolution calculations, replica-symmetric potential evaluations, tree-code simulations, and optimized finite-length AMP experiments for unsourced random access.

## Setup

Create a Python environment and install the required packages:

```bash
python -m pip install -r requirements.txt
```

On Vera, load the matching modules before running the optimized simulations:

```bash
ml numba/0.58.1-foss-2023a
ml tqdm/4.66.1-GCCcore-12.3.0
ml matplotlib/3.7.2-gfbf-2023a
```

## Common Commands

Generate or refresh the optimized Figure 9 simulation:

```bash
python fig9_faithful_empirical_amp_tree.py production --resume
```

Run Figure 10 theory only:

```bash
python fig10.py theory --fresh
```

Resume the full Figure 10 empirical campaign:

```bash
python fig10.py production --trials 100 --workers 32 --resume
```

Regenerate plots from existing saved data:

```bash
python fig10.py plot
```

Run validation checks:

```bash
python fig9_faithful_empirical_amp_tree.py validate
python fig10.py validate
```

## Outputs

Generated plot data and figures are stored with their corresponding figure folders:

- `data/fig9/`: Figure 9 CSVs, checkpoint, PNG, and PDF.
- `data/fig10/`: Figure 10 theory/empirical CSVs, checkpoint, allocation metadata, PNG, and PDF.
- `data/fig9_debug/`: debug-mode Figure 9 outputs.

The optimized finite-length simulations use a structured Hadamard sensing operator for computational feasibility; this is not the exact dense i.i.d. Gaussian ensemble assumed in the paper.
