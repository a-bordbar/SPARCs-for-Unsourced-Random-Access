# SPARCs for Unsourced Random Access

This repository reproduces Figures 2-10 from ["SPARCs for Unsourced Random Access"](https://arxiv.org/abs/1901.06234) by Alexander Fengler, Peter Jung, and Giuseppe Caire.

It contains the numerical code and saved outputs used for state-evolution calculations, replica-symmetric potential evaluations, power-allocation optimization, outer tree-code simulations, and finite-length AMP simulations.

## Reproduced Figures

Figures 2-10 have been reproduced. The main scripts are:

| Figure | Script(s) | Main outputs |
| --- | --- | --- |
| 2 | `fig2.py` | `data/fig2_data/` |
| 3 | `fig3.py`, `fig3_tree_decoder.py` | `data/fig3_data/` |
| 4 | `fig4.py` | `plots/fig4.*`, `data/fig4_data.csv` |
| 5 | `fig5.py` | `plots/fig5.*`, `data/fig5_data/` |
| 6 | `fig6_theoretical.py`, `fig6_AMP_calibrated.py`, `fig6_AMP_optimized.py`, `fig6_plot_all_py.py` | `plots/fig6*`, `data/fig6_*` |
| 7 | `fig7.py` | `plots/fig7.*`, `data/fig7_data/` |
| 8 | `fig8.py`, `fig8a.py`, `fig8b.py`, `fig8_common.py` | `plots/fig8*`, `data/fig8/` |
| 9 | `fig9_faithful_empirical_amp_tree.py` | `data/fig9/` |
| 10 | `fig10.py` | `data/fig10/` |

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

## Reproducing the Results

Completed simulation outputs are included in the repository. The commands below are representative entry points for validating, rerunning, resuming, or replotting the heavier computations.

Validate the optimized finite-length simulations:

```bash
python fig9_faithful_empirical_amp_tree.py validate
python fig10.py validate
```

Resume the optimized Figure 9 finite-length AMP/tree-code reproduction:

```bash
python fig9_faithful_empirical_amp_tree.py production --resume
```

Regenerate Figure 10 theory and plot outputs from the saved data:

```bash
python fig10.py theory --fresh
python fig10.py plot
```

Resume the full Figure 10 empirical campaign:

```bash
python fig10.py production --trials 100 --workers 32 --resume
```

## Repository Layout

- `fig*.py`: figure-specific reproduction scripts.
- `rs_potential.py`, `utils.py`: shared numerical helpers.
- `data/`: saved numerical outputs, checkpoints, and figure-specific data.
- `plots/`: generated plots for the earlier figures.

Figure 9 and Figure 10 store their final PNG/PDF outputs inside their corresponding `data/fig9/` and `data/fig10/` folders.

## Faithfulness and Assumptions

The objective is to reproduce the published numerical results as closely as possible from the information provided in the paper. Where an implementation detail is not specified, the corresponding assumption is documented in the code.

For the large finite-length AMP simulations, a structured randomized Hadamard sensing operator is used for computational feasibility. The paper's analysis assumes i.i.d. Gaussian sensing matrices with entries `N(0, 1/n)`, so these finite-length simulations should not be interpreted as an exact realization of the paper's Gaussian matrix ensemble.

## Reference

Alexander Fengler, Peter Jung, and Giuseppe Caire, "SPARCs for Unsourced Random Access," arXiv:1901.06234.
