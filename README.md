# Thermalization in a Black Hole Model with an Inverted Harmonic Potential

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red)](#license)
[![Jupyter](https://img.shields.io/badge/notebook-Jupyter-orange?logo=jupyter)](greens_func.ipynb)

Research code accompanying the thesis of Amigo on thermalization in a black hole model governed by an inverted harmonic potential.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Method](#method)
- [Requirements & Installation](#requirements--installation)
- [Usage](#usage)
- [Reference Output](#reference-output)
- [Configuration Guide](#configuration-guide)
- [License](#license)

---

> **🚧 Work in Progress** — This repository accompanies an ongoing thesis. Details on the overview and physics background will be added as the work progresses.

---

## Repository Structure

```
.
├── greens_func.ipynb   # Main analysis notebook
├── LICENSE
└── README.md
```

### `greens_func.ipynb`

A single self-contained Jupyter notebook that:

1. **Configures** a dense complex-frequency grid and model/runtime parameters.
2. **Computes** the Green's-function recurrence over the complex *ω*-plane (Numba-JIT, multi-threaded).
3. **Detects** pole candidates as prominent local maxima in log₁₀|g| with adaptive thresholds.
4. **Estimates** the decay rate *γ* from poles with sufficiently negative imaginary part.
5. **Sweeps** the coupling *a* and fits the power law *γ = C · a^x*.

---

## Method

### Recurrence iteration

Starting from a flattened set of starting lines in the complex-frequency plane,

```python
w0_real_arr = np.linspace(w0_min, w0_max, n_w0)
w0_imag_values = np.linspace(10.0, 10.1, 50)
w0_arr = (w0_real_arr[None, :] + 1j * w0_imag_values[:, None]).ravel()
```

the notebook iterates the two-step map

```
g0 ← i / (ω + i·eps)
g  ← 1 / (i·a·g − i·ω)
ω  ←  ω − i·m
```

for `n_step` steps. Each trajectory is independent, enabling Numba `prange` parallelism over all starting values in `w0_arr`.

### Pole detection

The absolute value |g| is evaluated on the (Re ω, Im ω) grid, converted to log₁₀ scale, then filtered with:

- a local-maximum mask (`maximum_filter`)
- a local background estimate (`median_filter`)
- percentile-based strength/prominence thresholds
- a minimum complex-plane separation between retained peaks

A single `pole_sensitivity` parameter (0 = strict, 1 = permissive) maps to the detection configuration.

### Decay-rate estimation

Poles with Im ω below an imaginary cutoff (named `tor` in the notebook, default `-1e-2`) contribute exponentially decaying modes. Their imaginary parts are pooled into a summed decay curve and fitted to an exponential via `scipy.optimize.curve_fit` to extract *γ*.

### Power-law fit

Over a sweep of *a* values the resulting *γ(a)* data are fitted in log-log space to

```
γ = C · a^x
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

Install with pip:

```bash
pip install numpy matplotlib numba scipy
```

or with conda:

```bash
conda install numpy matplotlib numba scipy
```

> **Note:** The notebook kernel metadata uses the display name `torch`, but no PyTorch dependency is required. Any Python 3.11 environment with the four packages above is sufficient.

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

---

## Reference Output

The following results are produced by the default configuration saved in the notebook state. They are provided as a sanity-check reference, not a formal benchmark.

### Representative runs (selected `a` values)

```text
a = 0.1: found 11 pole candidates
a = 1: found 23 pole candidates
a = 10: found 58 pole candidates
```

### Decay fit (`a = 10.0`)

```text
gamma = 2.767641 +/- 0.067101
```

### Coupling sweep (`a ∈ [0.1, 10.0]`, 50 points; `scan_stride = 10`)

```text
log-space power-law fit: gamma = 0.971582 * a^0.449186
C = 0.971582 +/- 0.011287
x = 0.449186 +/- 0.006659
```

---

## Configuration Guide

All tuneable parameters live in the **configuration cell** near the top of the notebook.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_threads` | `32` | Numba thread count; reduce to match available CPU cores |
| `n_w0` | `5000` | Grid density along Re ω (per starting line); major cost driver |
| `w0_min / w0_max` | `−10 / 10` | Real-axis scan range |
| `w0_imag_values` | `linspace(10.0, 10.1, 50)` | Imaginary offsets for starting lines |
| `a_values` | `linspace(0.1, 10.0, 50)` | Coupling values for the sweep |
| `selected_a_values` | `[0.1, 1.0, 10.0]` | Representative couplings for 2D plots |
| `m` | `0.05` | Imaginary step per iteration |
| `iter_range` | `20` | Total imaginary range traversed (`n_step = iter_range/m + 1`) |
| `pole_sensitivity` | `0.2` | Pole-detection permissiveness (0 = strict, 1 = sensitive) |
| `denom_floor` | `1e-16` | Early-stop threshold for near-singular recurrence denominator |
| `g_abs_max` | `1e16` | Early-stop threshold for runaway `|g|` values |
| `scan_stride` | `10` | Subsampling factor for the `a`-sweep (higher = faster) |

**For a quick exploratory run**, reduce `n_w0` and/or the number of `w0_imag_values`, and use fewer `a_values` before running the full sweep.

---

## License

Copyright © 2026 yygarypeng and Amigo. All rights reserved.

See [`LICENSE`](LICENSE) for the full terms.
