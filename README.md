# Thermalization in a Black Hole Model with an Inverted Harmonic Potential

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/notebook-Jupyter-orange?logo=jupyter)](greens_func.ipynb)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)](LICENSE)

Research code for studying thermalization in a black-hole-inspired model with an inverted harmonic potential. The workflow computes an iterated Green's function over a complex-frequency grid, detects pole candidates, estimates a decay rate, and fits the decay-rate dependence on the coupling parameter `a`.

This is a work-in-progress thesis repository. The notebook is the source of truth for the current analysis run.

## Repository Layout

```text
.
├── greens_func.ipynb   # Main notebook: configuration, orchestration, final plots
├── core.py             # Shared numerical kernels, pole detection, fitting, plotting
├── figures/            # Tracked generated PDF figures
├── LICENSE             # All-rights-reserved project license
└── README.md
```

Generated sweep data is written to `.worker_results/`, which is ignored by git. The checked-in PDFs in `figures/` are the generated figures from the analysis workflow.

## What The Code Does

The notebook builds a complex starting grid

```python
w0_arr = (w0_real_arr_by_a[idx][None, :] + 1j * w0_imag_values[:, None]).ravel()
```

and uses the Numba-compiled iterator in `core.py` to apply the recurrence

```text
g0 = i / (omega + i * eps)
g  = 1 / (i * a * g - i * omega)
omega = omega - i * m
```

for each starting point. The result is stored as `log10(abs(G))` on the full grid.

Pole candidates are selected from local maxima in the log-scale grid after applying strength, prominence, and minimum-separation cuts. Poles with imaginary part below the default cutoff `-1e-2` are used to build a summed decay curve, which is fit to a single exponential to estimate `gamma`. The sweep then fits

```text
gamma = C * a**x
```

in log-log space.

## Requirements

There is no dependency manifest in this repository. Install the runtime packages manually:

```bash
pip install numpy matplotlib mplhep numba scipy jupyter
```

## Usage

Open the notebook and run it from top to bottom:

```bash
jupyter lab greens_func.ipynb
```

or

```bash
jupyter notebook greens_func.ipynb
```

The notebook is stateful. Later cells depend on variables created in the import and configuration cells.

## Current Notebook Configuration

The checked-in notebook currently uses these main settings:

| Setting | Value | Notes |
|---|---:|---|
| `NUMBA_NUM_THREADS` | `32` | Set before importing Numba in the notebook |
| `n_parallel_a` | `16` | Number of `a` values submitted concurrently |
| `threads_per_job` | `2` | Numba threads used inside each sweep worker |
| `selected_a_values` | `[0.01, 0.1, 1.0, 10.0]` | Representative values used for heatmaps |
| `a_values` | `np.linspace(0.01, 10.0, 50)` plus selected values | 52 unique sweep points |
| `n_w0` | `10_000` | Real-axis grid density for each `a` |
| `w0_imag_values` | `np.linspace(10.0, 10.1, 25)` | Imaginary starting lines |
| `m` | `0.05` | Imaginary step size |
| `iter_range` | `20` | Total imaginary range traversed |
| `n_step` | `401` | Computed as `int(iter_range / m) + 1` |
| `denom_floor` | `1e-16` | Stops unstable recurrence denominators |
| `g_abs_max` | `1e16` | Stops unstable Green's-function magnitudes |

Pole-detection sensitivity is piecewise in `a`:

| Coupling range | Sensitivity |
|---|---:|
| `a <= 0.5` | `0.65` |
| `0.5 < a < 3.0` | `0.15` |
| `a >= 3.0` | `0.0001` |

For a quick exploratory run, reduce `n_w0`, shorten `w0_imag_values`, and sweep fewer `a_values` before running the expensive sweep cell.

## Outputs

The notebook writes two classes of generated artifacts:

| Path | Contents |
|---|---|
| `.worker_results/` | Ignored `.npy` scratch arrays and compressed full-grid sweep `.npz` files |
| `figures/` | Tracked generated PDF plots |

Each `.worker_results/sweep_scan_a_<value>.npz` file contains:

| Key | Meaning |
|---|---|
| `a_value` | Coupling value |
| `scan_w0_arr` | Flattened complex starting grid |
| `scan_log_abs_g` | Full `log10(abs(G))` grid with shape `(len(scan_w0_arr), n_step)` |
| `pole_w` | Detected pole-candidate locations |
| `gamma`, `gamma_err` | Decay-rate estimate and fit uncertainty |
| `n_poles` | Number of detected pole candidates |
| `m`, `n_step` | Recurrence step size and number of steps |

The notebook helper `sweep_scan_plot_grid(data, xlim=None, ylim=None)` reconstructs cell edges and a 2D log-scale grid from a saved `.npz` file so heatmaps can be redrawn without rerunning the recurrence.

## Figures

The current notebook regenerates these main PDFs:

| Figure | Description |
|---|---|
| `figures/w0_starting_lines.pdf` | Initial complex-frequency starting lines |
| `figures/decay_fit_a_1.pdf` | Exponential decay fit for the representative `a = 1` run |
| `figures/gamma_vs_a_power_law_fit.pdf` | Log-log power-law fit of `gamma(a)` |
| `figures/sweep_pole_heatmap_a_<value>.pdf` | Saved-sweep heatmaps for `a = 0.01`, `0.1`, `1`, and `10` |

The repository also contains older diagnostic PDFs named `g_heatmap_a_<value>.pdf` and `pole_candidates_a_<value>.pdf` from the same analysis workflow.

## Verification

Cheap syntax check:

```bash
python -m py_compile core.py
```

Do not use the full notebook as a routine smoke test unless production results or final figures are required. Use a reduced grid for quick runtime checks.

## License

Copyright (c) 2026 yygarypeng and Amigo. All rights reserved.

See [`LICENSE`](LICENSE) for the full terms.
