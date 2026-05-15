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
    - [Saved Sweep Scan Format](#saved-sweep-scan-format)
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
├── figures/            # Tracked generated PDF figures
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

The heavy sweep cell calls importable `core.py` worker functions in separate processes. This keeps Numba/OpenMP thread pools isolated and avoids the common notebook failure mode where only one CPU core is busy.

### `core.py`

Importable support module used by the notebook for:

- Numba-compiled Green's-function iteration.
- Pole detection and decay-rate fitting.
- Coupling-sweep worker execution and sweep-scan serialization.
- Shared plotting helpers for the starting grid, decay fit, and power-law fit.

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

The absolute value |g| is evaluated on the (Re ω, Im ω) grid, converted to log₁₀ scale, and scanned for local maxima that stand out above a rolling background. In the current notebook configuration, pole sensitivity is piecewise in `a`: `a <= 0.5` uses `0.6`, `0.5 < a < 3` uses `0.1`, and `a >= 3` uses `0.001`.

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
| `mplhep` | Optional ATLAS-style plot formatting in the final notebook cell |
| `numba` | JIT compilation and multi-threaded iteration |
| `scipy` | Peak detection and curve fitting |
| `jupyter` | Notebook execution |

Install with pip:

```bash
pip install numpy matplotlib mplhep numba scipy jupyter
```

or with conda:

```bash
conda install numpy matplotlib mplhep numba scipy jupyter
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

For quick exploratory runs, reduce the grid sizes in the configuration cell before running the expensive sweep cell.

---

## Configuration Guide

All tuneable parameters live in the **configuration cell** near the top of the notebook.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `total_cpus` | `32` | Total CPU threads used by notebook and workers |
| `n_w0` | `5_000` | Grid density along Re ω; major cost driver |
| `w0_real_arr_by_a` | from `output_w_range(a_values)` | Coupling-dependent real-axis scan range |
| `w0_imag_values` | `linspace(10.0, 10.05, 50)` | Imaginary starting lines |
| `a_values` | `linspace(0.01, 10.0, 50)` plus `selected_a_values` | Coupling values for the parameter sweep |
| `m` | `0.05` | Imaginary step per iteration |
| `iter_range` | `20` | Total imaginary range traversed (`n_step = iter_range/m + 1`) |
| `sweep_n_jobs` | `min(len(a_values), 8)` | Parallel sweep worker count |
| `sweep_num_threads_per_job` | `4` | Numba threads assigned to each sweep worker |
| `plot_log_vmin / plot_log_vmax` | `-1.3 / 0.3` | Fixed color range for log-scale pole heatmaps |

**For a quick exploratory run**, reduce `n_w0` to ~500, reduce `w0_imag_values`, and set `a_values = np.linspace(0.01, 10.0, 5)` before running the full production sweep.

---

## Generated Outputs

The notebook writes generated artifacts to local output directories:

| Path | Contents |
|------|----------|
| `.worker_results/` | Ignored worker scratch arrays and compressed full-grid sweep `.npz` files |
| `figures/` | Tracked generated PDF figures for heatmaps, pole diagnostics, decay fits, and sweep fits |

The sweep always uses the full grid and saves a compressed raw `.npz` file for every scanned `a` value. These files can be large for production grids.

### Saved Sweep Scan Format

Each `.worker_results/sweep_scan_a_<value>.npz` file stores the raw full-grid data needed to replot a sweep result without rerunning the recurrence:

| Key | Meaning |
|-----|---------|
| `a_value` | Coupling value for this sweep scan |
| `scan_w0_arr` | Flattened complex starting grid `omega_0` |
| `scan_log_abs_g` | `log10(abs(G))` values with shape `(len(scan_w0_arr), n_step)` |
| `pole_w` | Detected pole-candidate locations in the complex `omega` plane |
| `gamma`, `gamma_err` | Decay-rate estimate and fit uncertainty for this `a` |
| `n_poles` | Number of detected pole candidates |
| `m`, `n_step` | Imaginary-axis step size and number of recurrence steps |

The notebook helper `sweep_scan_plot_points(data, xlim=None, ylim=None)` reconstructs plotting coordinates from this format. It infers the original `(imaginary line, real point)` grid from `scan_w0_arr`, computes `x = Re(omega)`, computes `y = Im(omega_0) - m * step`, selects optional `xlim`/`ylim` windows, and returns finite flattened `x`, `y`, `z` arrays for scatter-style heatmap plotting.

### Figure Descriptions

All figures are written as PDF files to the `figures/` directory. The four representative coupling values used in per-*a* figures are *a* = 0.01, 0.1, 1, and 10.

The current notebook regenerates `w0_starting_lines.pdf`, `decay_fit_a_1.pdf`, `gamma_vs_a_power_law_fit.pdf`, and `sweep_pole_heatmap_a_<value>.pdf`. The tracked `g_heatmap_a_<value>.pdf` and `pole_candidates_a_<value>.pdf` files are retained diagnostic figures from the same analysis workflow.

#### `w0_starting_lines.pdf`

Scatter plot of the initial conditions in the complex *ω*-plane. Each horizontal line corresponds to one of the 50 imaginary starting values (Im(*ω*₀) ∈ [10.0, 10.05]), drawn across the representative real-axis scan range. This figure serves as a visual check that the starting grid is populated as intended before the recurrence is run.

#### `g_heatmap_a_<value>.pdf`

Diagnostic heatmaps showing Re(*G*) in the complex *ω*-plane for representative coupling values. These tracked figures are useful for checking the raw real part of the iterated Green's function before reducing the data to log-scale pole candidates.

#### `pole_candidates_a_<value>.pdf`

Diagnostic log-scale heatmaps of log₁₀|*G*| with detected pole candidates overlaid as cyan open circles. These figures show the peak-finding output directly on the computed Green's-function landscape.

#### `sweep_pole_heatmap_a_<value>.pdf`

Log-scale heatmaps reconstructed from the saved `.npz` sweep-scan files in `.worker_results/`. The plotted grid is rebuilt directly from the saved full-grid iteration data, and detected pole candidates are overlaid as cyan open circles. The current notebook applies per-coupling plot windows and color limits for the representative values.

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
