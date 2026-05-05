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

1. **Configures** the numerical grid and model parameters.
2. **Computes** the Green's-function recurrence over the complex *ω*-plane (Numba-JIT, multi-threaded).
3. **Detects** pole candidates as prominent local maxima in log₁₀|g|.
4. **Estimates** the decay rate *γ* from the poles with non-positive imaginary part.
5. **Sweeps** the coupling *a* and fits the power law *γ = C · a^x*.

---

## Method

### Recurrence iteration

Starting from an initial complex frequency grid

```python
w0_arr = np.linspace(w0_min, w0_max, n_w0) + 1j * w0_imag
```

the notebook iterates the two-step map

```
g  ←  1 / (i·a·g − i·ω)
ω  ←  ω − i·m
```

for `n_step` steps. Each trajectory is independent, enabling Numba `prange` parallelism over the `n_w0` starting values.

### Pole detection

The absolute value |g| is evaluated on the (Re ω, Im ω) grid, converted to log₁₀ scale, and scanned for local maxima that stand out above a rolling background. A single `pole_sensitivity` parameter (0 = strict, 1 = permissive) controls the detection thresholds.

### Decay-rate estimation

Poles with Im ω ≤ 0 contribute exponentially decaying modes. Their imaginary parts are pooled to extract a single decay rate *γ* via a weighted least-squares exponential fit.

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

### Single run (`a = 1.0`)

```text
Found 28 pole candidates
gamma = 1.271340 +/- 0.082232
```

### Coupling sweep (`a ∈ [1, 100]`, 51 points)

```text
log-space power-law fit: gamma = 1.445093 * a^0.295870
C = 1.445093 +/- 0.030071
x = 0.295870 +/- 0.005016
```

---

## Configuration Guide

All tuneable parameters live in the **configuration cell** near the top of the notebook.

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_threads` | `16` | Numba thread count; reduce to match available CPU cores |
| `n_w0` | `5000` | Grid density along Re ω; major cost driver |
| `w0_min / w0_max` | `−5 / 5` | Real-axis scan range |
| `w0_imag` | `10.0` | Imaginary offset of the initial grid |
| `a_values` | `linspace(1, 100, 51)` | Coupling values for the parameter sweep |
| `m` | `0.05` | Imaginary step per iteration |
| `iter_range` | `20` | Total imaginary range traversed (`n_step = iter_range/m + 1`) |
| `pole_sensitivity` | `0.01` | Pole-detection permissiveness (0 = strict, 1 = sensitive) |
| `scan_stride` | `10` | Subsampling factor for the `a`-sweep (higher = faster) |

**For a quick exploratory run**, reduce `n_w0` to ~500 and set `a_values = np.linspace(1.0, 10.0, 5)` in a scratch copy before running the full production sweep.

---

## License

Copyright © 2026 yygarypeng and Amigo. All rights reserved.

See [`LICENSE`](LICENSE) for the full terms.
