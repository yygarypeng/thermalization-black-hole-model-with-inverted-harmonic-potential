"""Numerical, fitting, plotting, and worker helpers for the Green's-function notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.ndimage import maximum_filter, median_filter
from scipy.optimize import curve_fit


# -----------------------------------------------------------------------------
# Green's-function iteration


@njit(parallel=True)
def log_abs_g_iterator(w0_arr, a, eps, m, max_iter, denom_floor=1e-12, g_abs_max=1e12):
    """Iterate the recurrence and return log10(abs(G)) for every starting point."""
    n_w = len(w0_arr) # for a specific a, iterate all imaginary (vertical) and real (horizontal) starting points
    log_abs_g = np.empty((n_w, max_iter), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for j in prange(n_w):
        w = w0_arr[j]
        g = 1j / (w + 1j * eps)
        g_abs = abs(g)
        log_abs_g[j, 0] = np.log10(max(g_abs, tiny)) if np.isfinite(g_abs) else np.nan

        for i in range(1, max_iter): # depending on the previous point (cannot use prange here)
            g_abs = abs(g)
            denom = 1j * a * g - 1j * w
            denom_abs = abs(denom)

            if (
                not np.isfinite(g_abs)
                or not np.isfinite(denom_abs)
                or denom_abs < denom_floor
                or g_abs > g_abs_max
            ):
                for k in range(i, max_iter):
                    log_abs_g[j, k] = np.nan
                break

            g = 1 / denom
            w = w - 1j * m
            g_abs = abs(g)
            log_abs_g[j, i] = np.log10(max(g_abs, tiny)) if np.isfinite(g_abs) else np.nan

    return log_abs_g # w/ shape (num of w_real * num of w_imag, max_iter)


@njit(parallel=True)
def g_iterator_selected_image_rows(
    w0_real_arr,
    w0_imag_values,
    selected_row_lookup,
    n_output_rows,
    a,
    eps,
    m,
    max_iter,
    denom_floor=1e-12,
    g_abs_max=1e12,
):
    """Iterate G and keep only selected physical image rows for CSV export."""
    n_real = len(w0_real_arr)
    n_imag = len(w0_imag_values)
    g_out = np.empty((n_output_rows, n_real), dtype=np.complex128)
    nan_complex = np.nan + 1j * np.nan

    for idx in prange(n_output_rows * n_real):
        row = idx // n_real
        real_idx = idx % n_real
        g_out[row, real_idx] = nan_complex

    for start_idx in prange(n_imag * n_real):
        imag_idx = start_idx // n_real
        real_idx = start_idx % n_real
        w = w0_real_arr[real_idx] + 1j * w0_imag_values[imag_idx]
        g = 1j / (w + 1j * eps)

        row = selected_row_lookup[0, imag_idx]
        if row >= 0:
            g_out[row, real_idx] = g

        for step in range(1, max_iter):
            g_abs = abs(g)
            denom = 1j * a * g - 1j * w
            denom_abs = abs(denom)

            if (
                not np.isfinite(g_abs)
                or not np.isfinite(denom_abs)
                or denom_abs < denom_floor
                or g_abs > g_abs_max
            ):
                break

            g = 1 / denom
            w = w - 1j * m
            row = selected_row_lookup[step, imag_idx]
            if row >= 0:
                g_out[row, real_idx] = g

    return g_out


# -----------------------------------------------------------------------------
# Pole detection


def pole_detection_config(sensitivity):
    """Translate a 0..1 pole-detection sensitivity into peak-filter settings."""
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    return {
        "peak_window": 7 if sensitivity < 0.25 else 5 if sensitivity < 0.65 else 3,
        "background_window": 31 if sensitivity < 0.25 else 21 if sensitivity < 0.65 else 15,
        "strength_percentile": 99.3 - 2.3 * sensitivity,
        "prominence_percentile": 99.0 - 4.0 * sensitivity,
        "min_separation": np.clip(0.24 - 0.28 * sensitivity, 0.06, 0.24),
    }


def pole_sensitivity_for_a(a_value, low_max=0.5, high_min=3.0, low=0.6, mid=0.1, high=0.001):
    """Return the piecewise pole-detection sensitivity for a coupling value."""
    if a_value <= low_max:
        return low
    if a_value < high_min:
        return mid
    return high


def pole_config_for_a(a_value, low_max=0.5, high_min=3.0, low=0.6, mid=0.1, high=0.001):
    """Build a pole-detection config for a coupling value."""
    return pole_detection_config(pole_sensitivity_for_a(a_value, low_max, high_min, low, mid, high))


def pole_config_for_a_from_config(a_value, config):
    """Build a pole-detection config from the notebook/worker config mapping."""
    return pole_config_for_a(
        a_value, # split into 3 parts for different sensitivity
        # splitting points
        config["low_sensitivity_max_a"],
        config["high_sensitivity_min_a"],
        # three sensitivity levels
        config["low_a_pole_sensitivity"],
        config["mid_a_pole_sensitivity"],
        config["high_a_pole_sensitivity"],
    )


def find_poles_from_log_abs_g(log_abs_g, w0_arr, m, config, n_imag, n_real):
    """Find prominent local maxima in a log10(abs(G)) iteration grid."""
    n_step = log_abs_g.shape[1]

    # recover the original 2D starting grid structure:
    # log_abs_g: (n_imag * n_real, n_step) -> (n_imag, n_real, n_step)
    # w0_arr:    (n_imag * n_real,)        -> (n_imag, n_real)
    log_abs_g_by_start_line = log_abs_g.reshape(n_imag, n_real, n_step)
    w0_by_start_line = w0_arr.reshape(n_imag, n_real)

    # Starting lines are fine offsets along Im(omega), not a separate physical axis.
    # Combine starting-line index and recurrence step into one physical Im(omega) image axis.
    log_abs_g_image = np.transpose(log_abs_g_by_start_line, (2, 0, 1)).reshape(n_step * n_imag, n_real)
    omega_image = (w0_by_start_line[None, :, :] - 1j * m * np.arange(n_step)[:, None, None]).reshape(
        n_step * n_imag,
        n_real,
    )
    imag_order = np.argsort(omega_image[:, 0].imag)
    log_abs_g_image = log_abs_g_image[imag_order]
    omega_image = omega_image[imag_order]

    finite = np.isfinite(log_abs_g_image)
    if not np.any(finite):
        return np.array([], dtype=complex)

    # maximum_filter and median_filter do not handle NaN well!!
    finite_floor = np.nanmin(log_abs_g_image[finite]) - 1.0
    finite_log_abs_g_image = np.where(finite, log_abs_g_image, finite_floor)

    # maximum value in a windows of nighborhood
    is_local_max = finite_log_abs_g_image == maximum_filter(
        finite_log_abs_g_image,
        size=(config["peak_window"], config["peak_window"]),
    )

    # estimate what the local non-peak baseline looks like
    local_baseline = median_filter(
        finite_log_abs_g_image,
        size=(config["background_window"], config["background_window"]),
    )

    # identify sharp peaks that stand out above the local background
    peak_prominence = finite_log_abs_g_image - local_baseline

    # the poles need to be large enough
    strength_threshold = np.nanpercentile(
        log_abs_g_image[finite],
        config["strength_percentile"],
    )

    # the poles need to be sharp enough
    prominence_threshold = np.nanpercentile(
        peak_prominence[finite],
        config["prominence_percentile"],
    )

    is_pole_candidate = (
        finite
        & is_local_max
        & (log_abs_g_image > strength_threshold)
        & (peak_prominence > prominence_threshold)
    )

    candidate_row, candidate_col = np.where(is_pole_candidate)

    candidate_omega = omega_image[candidate_row, candidate_col]
    candidate_log_abs_g = log_abs_g_image[candidate_row, candidate_col]
    candidate_prominence = peak_prominence[candidate_row, candidate_col]
    strongest_first = np.lexsort((candidate_log_abs_g, candidate_prominence))[::-1] # first sort by strength, then by prominence, and reverse for strongest first
    # the selected poles need to be separated enough in the complex plane
    selected_poles = []
    for idx in strongest_first:
        pole = candidate_omega[idx]
        if all(abs(pole - kept_pole) >= config["min_separation"] for kept_pole in selected_poles):
            selected_poles.append(pole)

    return np.array(selected_poles, dtype=complex)


# -----------------------------------------------------------------------------
# Fitting


def exp_decay(t, amplitude, gamma):
    """Single-exponential decay model used for curve fitting."""
    return amplitude * np.exp(-gamma * t)


def fit_gamma_from_poles(pole_w, t_max=20, tor=-1e-2):
    """Fit the decay rate gamma from poles below the imaginary-axis cutoff."""
    decaying_poles = pole_w[pole_w.imag < tor]
    if len(decaying_poles) < 3:
        return np.nan, np.nan, None, None, None

    time = np.linspace(0, t_max, 101)
    decay_curves = np.exp(np.outer(time, decaying_poles.imag)).sum(axis=1)
    valid = np.isfinite(decay_curves) & (decay_curves > 0)
    fit_time = time[valid]
    fit_decay = decay_curves[valid]
    if len(fit_decay) < 3:
        return np.nan, np.nan, time, decay_curves, None

    p0 = [fit_decay[0], max(-np.max(decaying_poles.imag), 1e-12)]
    params, covariance = curve_fit(
        exp_decay,
        fit_time,
        fit_decay,
        p0=p0,
        bounds=([0, 0], [np.inf, np.inf]),
        maxfev=10_000,
    )
    gamma = params[1]
    gamma_err = np.sqrt(covariance[1, 1]) # sqrt(var(gamma, gamma))
    return gamma, gamma_err, time, decay_curves, exp_decay(time, *params)


def fit_power_law(a_values, gamma_values, gamma_errors):
    """Fit gamma(a) to C * a**x in log-log space."""
    valid = np.isfinite(gamma_values) & (gamma_values > 0) & (a_values > 0)
    fit_a = a_values[valid]
    fit_gamma = gamma_values[valid]
    fit_gamma_err = gamma_errors[valid]
    if len(fit_gamma) < 3:
        return np.nan, np.nan, np.nan, np.nan, fit_a, fit_gamma

    log_a = np.log(fit_a)
    log_gamma = np.log(fit_gamma)
    log_gamma_err = fit_gamma_err / fit_gamma
    use_weights = np.all(np.isfinite(log_gamma_err) & (log_gamma_err > 0))
    kwargs = {"sigma": log_gamma_err, "absolute_sigma": True} if use_weights else {}

    # log-log is a straight line
    params, covariance = curve_fit(
        lambda log_a, log_C, x: log_C + x * log_a,
        log_a,
        log_gamma,
        p0=[1.0, 0.6],
        maxfev=10_000,
        **kwargs,
    )
    log_C, x = params
    log_C_err, x_err = np.sqrt(np.diag(covariance))
    C = np.exp(log_C)
    C_err = C * log_C_err
    return C, C_err, x, x_err, fit_a, fit_gamma


def power_law(a, C, x):
    """Power-law model used for plotting the fitted gamma(a) curve."""
    return C * a**x


# -----------------------------------------------------------------------------
# Sweep computation and file naming


def compute_sweep_scan_for_a(
    a_value,
    scan_w0_arr,
    config,
    eps,
    m,
    n_step,
    denom_floor,
    g_abs_max,
    n_imag,
    n_real,
    threads=None,
):
    """Compute one full-grid sweep scan and its decay-rate estimate."""
    if threads is not None:
        set_num_threads(threads)
    scan_log_abs_g = log_abs_g_iterator(scan_w0_arr, a_value, eps, m, n_step, denom_floor, g_abs_max)
    scan_pole_w = find_poles_from_log_abs_g(scan_log_abs_g, scan_w0_arr, m, config, n_imag, n_real)
    gamma, gamma_err, _, _, _ = fit_gamma_from_poles(scan_pole_w)
    return gamma, gamma_err, len(scan_pole_w), scan_log_abs_g, scan_pole_w


def format_a_for_filename(a_value):
    """Format a coupling value as a filename-safe token."""
    return f"{a_value:g}".replace("-", "m").replace(".", "p")


def sweep_scan_filename(a_value):
    """Return the saved-scan filename for a coupling value."""
    return f"sweep_scan_a_{format_a_for_filename(a_value)}.npz"


def compute_and_save_sweep_scan(
    a_value,
    scan_w0_arr,
    config,
    eps,
    m,
    n_step,
    denom_floor,
    g_abs_max,
    n_imag,
    n_real,
    output_dir,
    threads=None,
):
    """Compute one full-grid sweep scan, save it, and return summary values."""
    gamma, gamma_err, n_poles, scan_log_abs_g, scan_pole_w = compute_sweep_scan_for_a(
        a_value,
        scan_w0_arr,
        config,
        eps,
        m,
        n_step,
        denom_floor,
        g_abs_max,
        n_imag,
        n_real,
        threads=threads,
    )
    scan_path = Path(output_dir) / sweep_scan_filename(a_value)
    np.savez_compressed(
        scan_path,
        a_value=a_value,
        scan_w0_arr=scan_w0_arr,
        scan_log_abs_g=scan_log_abs_g,
        pole_w=scan_pole_w,
        gamma=gamma,
        gamma_err=gamma_err,
        n_poles=n_poles,
        m=m,
        n_step=n_step,
    )
    return gamma, gamma_err, n_poles, scan_path


# -----------------------------------------------------------------------------
# Plotting


def save_figure(fig, save_path):
    """Save a matplotlib figure when a path is provided."""
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")


def plot_w0_starting_lines(w0_real_arr, w0_imag_values, save_path=None, show=True):
    """Plot the initial horizontal starting lines in the complex omega plane."""
    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    for w0_imag in w0_imag_values:
        ax.plot(w0_real_arr, np.full_like(w0_real_arr, w0_imag), linewidth=0.5)

    ax.set_xlabel(r"$Re(\omega_0)$")
    ax.set_ylabel(r"$Im(\omega_0)$")
    ax.set_title("Starting Lines")
    save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_decay_fit(gamma, time, decay_curves, fit_curve, save_path=None):
    """Plot the summed pole decay and the fitted exponential curve."""
    if time is None:
        print("not enough decaying poles for gamma fit")
        return

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(time, decay_curves, label="summed pole decay")
    if fit_curve is not None:
        ax.plot(time, fit_curve, "--", label=f"fit: gamma = {gamma:.4f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Decay")
    ax.set_title("Decay Fit from Pole Candidates")
    ax.legend()
    ax.grid()
    save_figure(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_a_gamma_fit(a_values, gamma_values, gamma_errors, C, x, fit_a, save_path=None):
    """Plot fitted gamma values against coupling and overlay the power-law fit."""
    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(a_values, gamma_values, yerr=gamma_errors, fmt=".", linestyle="none", capsize=4, label="data")
    if np.isfinite(C) and np.isfinite(x) and len(fit_a):
        a_fit = np.linspace(fit_a.min(), fit_a.max(), 500)
        ax.plot(a_fit, power_law(a_fit, C, x), "--", label=rf"fit: $\gamma = {C:.3f} a^{{{x:.3f}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel("fitted gamma")
    ax.set_title("a vs fitted gamma")
    ax.grid(True, which="both")
    ax.legend()
    save_figure(fig, save_path)
    plt.show()
    plt.close(fig)
