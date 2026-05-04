# thermalization-black-hole-model-with-inverted-harmonic-potential

WIP research code for Amigo's thesis on thermalization in a black hole model with an inverted harmonic potential.

Thesis: WIP

## Contents

- `greens_func.ipynb`: numerical notebook for iterating a Green's-function-like recurrence, finding pole candidates, estimating a decay rate `gamma`, and fitting the scaling `gamma = C * a^x`.

## Method

The notebook builds a complex initial grid

```python
w0_arr = np.linspace(w0_min, w0_max, n_w0) + 1j * w0_imag
```

and iterates

```python
g = 1 / (1j * a * g - 1j * w)
w = w - 1j * m
```

over the grid. The iterator is compiled with Numba and parallelized over starting `w0` values.

The analysis then:

- plots the real part of `g` over the complex `w` plane
- detects pole candidates as prominent local maxima in `log10(|g|)`
- estimates a decay rate `gamma` from poles with non-positive imaginary part
- sweeps over `a_values` and fits a power law `gamma = C * a^x`

## Requirements

No environment file is currently committed. The notebook imports:

- `numpy`
- `matplotlib`
- `numba`
- `scipy`

The notebook metadata records Python `3.11.15` and a kernelspec display name of `torch`, but the code itself only uses the packages above.

## Running

Open `greens_func.ipynb` in Jupyter and run cells from top to bottom. The cells are stateful: later analysis cells depend on variables created by earlier configuration and helper cells.

The default configuration is compute-heavy:

- `num_threads = 16`
- `n_w0 = 5000`
- `a_values = np.linspace(1.0, 100.0, 51)`
- `n_step = 401`

For a quick exploratory run, reduce `n_w0`, `a_values`, or increase `scan_stride` in a scratch copy before running the full sweep.

## Current Reference Output

With the saved notebook state, the single run reports:

```text
Found 28 pole candidates
gamma = 1.271340 +/- 0.082232
```

The saved `a` sweep reports:

```text
log-space power-law fit: gamma = 1.445093 * a^0.295870
C = 1.445093 +/- 0.030071
x = 0.295870 +/- 0.005016
```

These values are not yet backed by automated tests; treat them as notebook-state reference output, not a formal benchmark.

## License

Copyright (c) 2026 yygarypeng and Amigo. All rights reserved.

See `LICENSE` for the full terms.
