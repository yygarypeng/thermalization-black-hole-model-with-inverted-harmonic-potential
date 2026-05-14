# Thermalization in a Black Hole Model with an Inverted Harmonic Potential

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)](#license)
[![Jupyter](https://img.shields.io/badge/notebook-Jupyter-orange?logo=jupyter)](greens_func.ipynb)

Research code accompanying the thesis of Amigo on thermalization in a black hole model governed by an inverted harmonic potential.

---

## Table of Contents

- [Thermalization in a Black Hole Model with an Inverted Harmonic Potential](#thermalization-in-a-black-hole-model-with-an-inverted-harmonic-potential)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
    - [`greens_func.ipynb`](#greens_funcipynb)
    - [`core.py`](#corepy)
  - [Method](#method)
    - [Recurrence iteration](#recurrence-iteration)
    - [Pole detection](#pole-detection)
    - [Decay-rate estimation](#decay-rate-estimation)
    - [Power-law fit](#power-law-fit)
  - [Requirements \& Installation](#requirements--installation)
  - [Usage](#usage)
  - [Configuration Guide](#configuration-guide)
  - [Generated Outputs](#generated-outputs)
    - [Figure Descriptions](#figure-descriptions)
  - [Verification](#verification)
  - [License](#license)

---

> **🚧 Work in Progress** — This repository accompanies an ongoing thesis. Details on the overview and physics background will be added as the work progresses.

---

## Repository Structure

```
.
├── greens_func.ipynb   # Main analysis notebook and orchestration
├── core.py             # Numerical kernels, fitting, plotting, and worker functions
├── LICENSE
└── README.md
```

### `greens_func.ipynb`

A Jupyter notebook that configures and orchestrates the analysis:

1. **Configures** the numerical grid and model parameters.
2. **Computes** the Green's-function recurrence over the complex *ω*-plane using `core.py` process workers.
3. **Detects** pole candidates as prominent local maxima in log₁₀|g|.
4. **Estimates** the decay rate *γ* from the poles with non-positive imaginary part.
5. **Sweeps** the coupling *a* and fits the power law *γ = C · a^x*.

The heavy selected-plot and sweep cells call importable `core.py` worker functions in separate processes. This keeps Numba/OpenMP thread pools isolated and avoids the common notebook failure mode where only one CPU core is busy.

### `core.py`

Importable support module used by the notebook for:

- Numba-compiled Green's-function iteration.
- Pole detection and decay-rate fitting.
- Coupling-sweep worker functions.
- Figure generation and sweep-scan serialization.

---

## Method

### Recurrence iteration

Starting from initial complex frequency lines

```python
w0_start_lines = w0_real_arr[None, :] + 1j * w0_imag_values[:, None]
w0_arr = w0_start_lines.ravel()
```

the notebook iterates the two-step map

```
g0 ← i / (ω + i·eps)
g  ← 1 / (i·a·g − i·ω)
ω  ←  ω − i·m
```

for `n_step` steps. Each trajectory is independent, enabling Numba `prange` parallelism over the flattened `w0_arr` starting values.

### Pole detection

The absolute value |g| is evaluated on the (Re ω, Im ω) grid, converted to log₁₀ scale, and scanned for local maxima that stand out above a rolling background. Pole sensitivity is piecewise in `a`: `a <= 0.5` uses `0.6`, `0.5 < a < 3` uses `0.2`, and `a >= 3` uses `0.05`.

### Decay-rate estimation

Poles with Im ω below the default cutoff `-1e-2` contribute exponentially decaying modes. Their imaginary parts are pooled into a summed decay curve and fitted to an exponential via `scipy.optimize.curve_fit` to extract *Γ*.

### Power-law fit

Over a sweep of *a* values the resulting *Γ(a)* data are fitted in log-log space to

```
Γ = C · a^x
```

using `scipy.optimize.curve_fit`.

---

## Requirements & Installation

The code requires Python ≥ 3.11 and the following packages:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations and grid construction |
| `matplotlib` | Visualization |
| `numba` | JIT compilation and multi-threaded iteration |
| `scipy` | Peak detection and curve fitting |
| `jupyter` | Notebook execution |

Install with pip:

```bash
pip install numpy matplotlib numba scipy jupyter
```

or with conda:

```bash
conda install numpy matplotlib numba scipy jupyter
```

> **Note:** The notebook kernel metadata may use the display name `torch`, but no PyTorch dependency is required.

---

## Usage

1. **Clone** the repository and install dependencies (see above).
2. **Launch** Jupyter:

   ```bash
   jupyter notebook greens_func.ipynb
   # or
   jupyter lab greens_func.ipynb
   ```

3. **Run all cells** from top to bottom (`Kernel → Restart & Run All`).

   > The cells are stateful. Later analysis cells depend on variables produced by earlier configuration and computation cells, so running them out of order will raise `NameError`.

For quick exploratory runs, reduce the grid sizes in the configuration cell before running the expensive selected-plot and sweep cells.

---

## Configuration Guide

All tuneable parameters live in the **configuration cell** near the top of the notebook.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_threads` | `os.cpu_count() or 32` | Base Numba thread count |
| `n_w0` | `5_000` | Grid density along Re ω; major cost driver |
| `w0_min / w0_max` | `-5 / 5` | Real-axis scan range |
| `w0_imag_values` | `linspace(10.0, 10.1, 50)` | Imaginary starting lines |
| `a_values` | `linspace(0.01, 10.0, 50)` plus `selected_a_values` | Coupling values for the parameter sweep |
| `m` | `0.05` | Imaginary step per iteration |
| `iter_range` | `20` | Total imaginary range traversed (`n_step = iter_range/m + 1`) |
| `scan_stride` | `1` | Sweep subsampling factor; `1` is full accuracy |
| `selected_n_jobs` | `min(len(selected_a_values), 4)` | Parallel selected-plot worker count |
| `sweep_n_jobs` | `min(len(a_values), 8)` | Parallel sweep worker count |
| `plot_log_vmin / plot_log_vmax` | `-1.3 / 0.3` | Fixed color range for log-scale pole heatmaps |
| `save_sweep_scan_outputs` | `True` | Save heatmap-ready sweep arrays and pole data for later replotting |
| `save_sweep_scan_a_values` | `selected_a_values` | Which sweep scans are saved as `.npz`; use `None` to save every scanned `a` |

**For a quick exploratory run**, reduce `n_w0` to ~500, reduce `w0_imag_values`, and set `a_values = np.linspace(0.01, 10.0, 5)` before running the full production sweep.

---

## Generated Outputs

The notebook writes generated artifacts to ignored local directories:

| Path | Contents |
|------|----------|
| `.worker_results/` | Worker scratch arrays, saved sweep grids, and heatmap-ready `.npz` files |
| `figures/` | Generated PDF figures for selected heatmaps, pole candidates, decay fits, and sweep fits |

Saved full-accuracy sweep scans can be large. Use `save_sweep_scan_a_values = selected_a_values` to keep only the representative scans, or `None` to save every scanned `a` value.

### Figure Descriptions

All figures are written as PDF files to the `figures/` directory. The four representative coupling values used in per-*a* figures are *a* = 0.01, 0.1, 1, and 10.

#### `w0_starting_lines.pdf`

Scatter plot of the initial conditions in the complex *ω*-plane. Each horizontal line corresponds to one of the 50 imaginary starting values (Im(*ω*₀) ∈ [10.0, 10.1]), drawn across the full real-axis scan range. This figure serves as a visual check that the starting grid is populated as intended before the recurrence is run.

#### `g_heatmap_a_<value>.pdf`

Four 2D heatmaps (one per representative *a* value) showing Re(*G*) in the complex *ω*-plane after the full recurrence iteration. The *x*-axis is Re(*ω*) and the *y*-axis is Im(*ω*). Colors use a diverging `coolwarm` palette symmetric around zero and clipped at the 98th percentile of |Re(*G*)|. Narrow streaks of large positive or negative Re(*G*) indicate the locations of quasi-normal-mode poles. Axis limits shrink with *a* to keep the dominant pole structure in view.

#### `pole_candidates_a_<value>.pdf`

Four log-scale heatmaps (one per representative *a* value) of log₁₀|*G*| in the complex *ω*-plane using the `magma` colormap. Detected pole candidates — prominent local maxima that stand out above a rolling background — are overlaid as cyan open circles. These figures are produced at full grid resolution from the selected runs and confirm that the automatic peak-finding algorithm captures the visually apparent poles.

#### `sweep_pole_heatmap_a_<value>.pdf`

Four log-scale heatmaps (one per representative *a* value) reconstructed from the saved `.npz` sweep-scan files in `.worker_results/`. The layout is the same as the `pole_candidates` figures (log₁₀|*G*| background with cyan pole markers), but the color range for each panel is fixed independently to best expose the pole structure at that coupling. These figures are suitable for publication-quality inspection without re-running the expensive sweep.

#### `decay_fit_a_1.pdf`

Time-domain decay curve for the representative case *a* = 1. The solid line is the summed exponential-decay signal constructed from all detected poles whose Im(*ω*) lies below the decay-rate cutoff (default −0.01). The dashed line is the best-fit exponential *A* exp(−*Γ t*) from `scipy.optimize.curve_fit`, with the fitted value of *Γ* shown in the legend. This figure provides a direct sanity check of the decay-rate extraction procedure.

#### `gamma_vs_a_power_law_fit.pdf`

Log-log scatter plot of the fitted decay rate *Γ* as a function of coupling *a* over the full parameter sweep. Error bars reflect the uncertainty returned by `curve_fit` for each individual decay fit. The dashed curve shows the best-fit power law *Γ* = *C* · *a*^*x* obtained by fitting in log-log space, with the exponent *x* and prefactor *C* displayed in the legend. This is the primary result figure of the analysis.

---

## Verification

Cheap syntax check:

```bash
python -m py_compile core.py
```

For runtime smoke tests, prefer a reduced grid in a scratch notebook run. Do not run the full notebook as a routine check unless final figures or production results are needed.

---

## License

Copyright © 2026 yygarypeng and Amigo. All rights reserved.

See [`LICENSE`](LICENSE) for the full terms.
