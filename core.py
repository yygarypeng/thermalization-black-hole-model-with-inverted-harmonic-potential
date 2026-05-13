import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.ndimage import maximum_filter, median_filter
from scipy.optimize import curve_fit


@njit(parallel=True)
def g_iterator(w0_arr, a, eps, m, max_iter, denom_floor=1e-12, g_abs_max=1e12):
    n_w = len(w0_arr)
    g_arr = np.empty((n_w, max_iter), dtype=np.complex128)
    nan_value = np.nan + 1j * np.nan

    for j in prange(n_w):
        w = w0_arr[j]
        g = 1j / (w + 1j * eps)
        g_arr[j, 0] = g

        for i in range(1, max_iter):
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
                    g_arr[j, k] = nan_value
                break

            g = 1 / denom
            w = w - 1j * m
            g_arr[j, i] = g

    return g_arr


@njit(parallel=True)
def log_abs_g_iterator(w0_arr, a, eps, m, max_iter, denom_floor=1e-12, g_abs_max=1e12):
    n_w = len(w0_arr)
    log_abs_g = np.empty((n_w, max_iter), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for j in prange(n_w):
        w = w0_arr[j]
        g = 1j / (w + 1j * eps)
        g_abs = abs(g)
        log_abs_g[j, 0] = np.log10(max(g_abs, tiny)) if np.isfinite(g_abs) else np.nan

        for i in range(1, max_iter):
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

    return log_abs_g


def pole_detection_config(sensitivity):
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    return {
        "peak_window": 7 if sensitivity < 0.25 else 5 if sensitivity < 0.65 else 3,
        "background_window": 31 if sensitivity < 0.25 else 21 if sensitivity < 0.65 else 15,
        "strength_percentile": 99.3 - 2.3 * sensitivity,
        "prominence_percentile": 99.0 - 4.0 * sensitivity,
        "min_separation": 0.12 - 0.08 * sensitivity,
    }


def pole_sensitivity_for_a(a_value, low_max=0.5, high_min=3.0, low=0.85, mid=0.5, high=0.15):
    if a_value <= low_max:
        return low
    if a_value < high_min:
        return mid
    return high


def pole_config_for_a(a_value, low_max=0.5, high_min=3.0, low=0.85, mid=0.5, high=0.15):
    return pole_detection_config(pole_sensitivity_for_a(a_value, low_max, high_min, low, mid, high))


def pole_config_for_a_from_config(a_value, config):
    return pole_config_for_a(
        a_value,
        config["low_sensitivity_max_a"],
        config["high_sensitivity_min_a"],
        config["low_a_pole_sensitivity"],
        config["mid_a_pole_sensitivity"],
        config["high_a_pole_sensitivity"],
    )


def compute_log_abs_g(g_arr):
    abs_g = np.abs(g_arr)
    finite = np.isfinite(abs_g)
    log_abs_g = np.full(abs_g.shape, np.nan, dtype=float)
    log_abs_g[finite] = np.log10(np.maximum(abs_g[finite], np.finfo(float).tiny))
    return log_abs_g, finite


def find_poles(g_arr, w0_arr, m, config):
    log_abs_g, _ = compute_log_abs_g(g_arr)
    return find_poles_from_log_abs_g(log_abs_g, w0_arr, m, config)


def find_poles_from_log_abs_g(log_abs_g, w0_arr, m, config):
    finite = np.isfinite(log_abs_g)
    if not np.any(finite):
        return np.array([], dtype=complex), np.array([]), log_abs_g

    finite_floor = np.nanmin(log_abs_g) - 1.0
    filtered_log_abs_g = np.where(np.isfinite(log_abs_g), log_abs_g, finite_floor)
    local_max = filtered_log_abs_g == maximum_filter(filtered_log_abs_g, size=config["peak_window"])
    local_background = median_filter(filtered_log_abs_g, size=config["background_window"])
    prominence = filtered_log_abs_g - local_background

    strength_cutoff = np.nanpercentile(log_abs_g, config["strength_percentile"])
    prominence_cutoff = np.nanpercentile(prominence, config["prominence_percentile"])
    is_peak = finite & local_max & (log_abs_g > strength_cutoff) & (prominence > prominence_cutoff)

    candidate_i, candidate_j = np.where(is_peak)
    candidate_w = w0_arr[candidate_i] - 1j * m * candidate_j
    candidate_strength = log_abs_g[candidate_i, candidate_j]
    order = np.argsort(candidate_strength)[::-1]

    selected_w = []
    selected_strength = []
    for idx in order:
        w = candidate_w[idx]
        if all(abs(w - kept_w) >= config["min_separation"] for kept_w in selected_w):
            selected_w.append(w)
            selected_strength.append(candidate_strength[idx])

    return np.array(selected_w, dtype=complex), np.array(selected_strength), log_abs_g


def exp_decay(t, amplitude, gamma):
    return amplitude * np.exp(-gamma * t)


def fit_gamma_from_poles(pole_w, t_max=20, tor=-1e-2):
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
    gamma_err = np.sqrt(covariance[1, 1])
    return gamma, gamma_err, time, decay_curves, exp_decay(time, *params)


def fit_power_law(a_values, gamma_values, gamma_errors):
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
    return C * a**x


def compute_poles_for_a(a_value, w0_arr, config, eps, m, n_step, denom_floor, g_abs_max, threads=None):
    if threads is not None:
        set_num_threads(threads)
    g_arr = g_iterator(w0_arr, a_value, eps, m, n_step, denom_floor, g_abs_max)
    pole_w, pole_strength, log_abs_g = find_poles(g_arr, w0_arr, m, config)
    return g_arr, pole_w, pole_strength, log_abs_g


def compute_sweep_gamma_for_a(a_value, scan_w0_arr, config, eps, m, n_step, denom_floor, g_abs_max, threads=None):
    if threads is not None:
        set_num_threads(threads)
    scan_log_abs_g = log_abs_g_iterator(scan_w0_arr, a_value, eps, m, n_step, denom_floor, g_abs_max)
    scan_pole_w, _, _ = find_poles_from_log_abs_g(scan_log_abs_g, scan_w0_arr, m, config)
    gamma, gamma_err, _, _, _ = fit_gamma_from_poles(scan_pole_w)
    del scan_log_abs_g
    gc.collect()
    return gamma, gamma_err, len(scan_pole_w)


def compute_sweep_scan_for_a(a_value, scan_w0_arr, config, eps, m, n_step, denom_floor, g_abs_max, threads=None):
    if threads is not None:
        set_num_threads(threads)
    scan_log_abs_g = log_abs_g_iterator(scan_w0_arr, a_value, eps, m, n_step, denom_floor, g_abs_max)
    scan_pole_w, _, _ = find_poles_from_log_abs_g(scan_log_abs_g, scan_w0_arr, m, config)
    gamma, gamma_err, _, _, _ = fit_gamma_from_poles(scan_pole_w)
    return gamma, gamma_err, len(scan_pole_w), scan_log_abs_g, scan_pole_w


def format_a_for_filename(a_value):
    return f"{a_value:g}".replace("-", "m").replace(".", "p")


def sweep_scan_filename(a_value):
    return f"sweep_scan_a_{format_a_for_filename(a_value)}.npz"


def should_save_sweep_scan(a_value, config):
    if not config.get("save_sweep_scan_outputs", False):
        return False
    saved_a_values = config.get("save_sweep_scan_a_values")
    if saved_a_values is None:
        return True
    return np.any(np.isclose(float(a_value), np.asarray(saved_a_values, dtype=float)))


def load_w0_arr_for_job(job):
    grid_dir = Path(job["grid_dir"])
    index = job["index"]
    w0_real_arr_by_a = np.load(grid_dir / "w0_real_arr_by_a.npy", mmap_mode="r")
    w0_imag_values = np.load(grid_dir / "w0_imag_values.npy", mmap_mode="r")
    w0_real_arr = np.asarray(w0_real_arr_by_a[index])
    w0_start_lines = w0_real_arr[None, :] + 1j * np.asarray(w0_imag_values)[:, None]
    return w0_start_lines.ravel()


def save_figure(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")


def visible_plot_indices(w0_arr, n_step, m, xlim, ylim, start_stride, step_stride, x_padding=0.0, y_padding=0.0):
    row_indices = np.arange(0, len(w0_arr), start_stride)
    step_indices = np.arange(0, n_step, step_stride)
    row_w0 = w0_arr[row_indices]
    padded_xlim = (xlim[0] - x_padding, xlim[1] + x_padding)
    padded_ylim = (ylim[0] - y_padding, ylim[1] + y_padding)

    row_mask = (row_w0.real >= padded_xlim[0]) & (row_w0.real <= padded_xlim[1])
    if np.any(row_mask):
        row_indices = row_indices[row_mask]
        row_w0 = row_w0[row_mask]

    step_imag_min = np.min(row_w0.imag) - m * step_indices
    step_imag_max = np.max(row_w0.imag) - m * step_indices
    step_mask = (step_imag_max >= padded_ylim[0]) & (step_imag_min <= padded_ylim[1])
    if np.any(step_mask):
        step_indices = step_indices[step_mask]

    return row_indices, step_indices


def plot_w0_starting_lines(w0_real_arr, w0_imag_values, save_path=None, show=True):
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


def plot_g_heatmap(
    w0_arr,
    g_arr,
    m,
    xlim=(-1.5, 1.5),
    ylim=(-10.0, 10.0),
    title="Iterative Solution of g",
    start_stride=10,
    step_stride=2,
    x_padding=0.0,
    y_padding=0.0,
    save_path=None,
    show=True,
):
    row_indices, step_indices = visible_plot_indices(
        w0_arr, g_arr.shape[1], m, xlim, ylim, start_stride, step_stride, x_padding, y_padding
    )
    z = g_arr[np.ix_(row_indices, step_indices)].real
    finite_z = z[np.isfinite(z)]
    vmax = np.nanpercentile(np.abs(finite_z), 98) if len(finite_z) else 1.0
    w_grid = w0_arr[row_indices, None] - 1j * m * step_indices[None, :]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    heatmap = ax.pcolormesh(
        w_grid.real,
        w_grid.imag,
        z,
        cmap="coolwarm",
        shading="auto",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    fig.colorbar(heatmap, ax=ax, pad=0.02, label=r"$Re(G)$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$Re(\omega)$")
    ax.set_ylabel(r"$Im(\omega)$")
    ax.set_title(title)
    save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def make_pole_heatmap_arrays(
    w0_arr,
    log_abs_g,
    m,
    xlim=(-2, 2),
    ylim=(-3.0, 1.0),
    start_stride=10,
    step_stride=2,
    x_padding=0.0,
    y_padding=0.0,
):
    row_indices, step_indices = visible_plot_indices(
        w0_arr, log_abs_g.shape[1], m, xlim, ylim, start_stride, step_stride, x_padding, y_padding
    )
    z = log_abs_g[np.ix_(row_indices, step_indices)]
    w_grid = w0_arr[row_indices, None] - 1j * m * step_indices[None, :]
    return w_grid.real, w_grid.imag, z


def plot_saved_pole_heatmap(
    heatmap_x,
    heatmap_y,
    heatmap_z,
    pole_w,
    xlim=(-2, 2),
    ylim=(-3.0, 1.0),
    title="Pole Candidates from Saved Sweep Scan",
    save_path=None,
    show=True,
    log_vmin=None,
    log_vmax=None,
):
    finite_z = heatmap_z[np.isfinite(heatmap_z)]
    if log_vmin is None or log_vmax is None:
        vmin, vmax = np.nanpercentile(finite_z, [5, 99.8]) if len(finite_z) else (0.0, 1.0)
    else:
        vmin, vmax = log_vmin, log_vmax

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    heatmap = ax.pcolormesh(
        heatmap_x,
        heatmap_y,
        heatmap_z,
        cmap="magma",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.scatter(pole_w.real, pole_w.imag, s=35, facecolors="none", edgecolors="cyan", linewidths=1.2)
    fig.colorbar(heatmap, ax=ax, label=r"$log_{10}(|G|)$")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$Re(\omega)$")
    ax.set_ylabel(r"$Im(\omega)$")
    ax.set_title(title)
    save_figure(fig, save_path)
    if show:
        plt.show()
    plt.close(fig)


def plot_poles(
    w0_arr,
    log_abs_g,
    pole_w,
    m,
    xlim=(-2, 2),
    ylim=(-3.0, 1.0),
    title="Pole Candidates from Prominent Local Peaks",
    start_stride=10,
    step_stride=2,
    x_padding=0.0,
    y_padding=0.0,
    save_path=None,
    show=True,
    log_vmin=None,
    log_vmax=None,
):
    heatmap_x, heatmap_y, heatmap_z = make_pole_heatmap_arrays(
        w0_arr,
        log_abs_g,
        m,
        xlim=xlim,
        ylim=ylim,
        start_stride=start_stride,
        step_stride=step_stride,
        x_padding=x_padding,
        y_padding=y_padding,
    )
    plot_saved_pole_heatmap(
        heatmap_x,
        heatmap_y,
        heatmap_z,
        pole_w,
        xlim=xlim,
        ylim=ylim,
        title=title,
        save_path=save_path,
        show=show,
        log_vmin=log_vmin,
        log_vmax=log_vmax,
    )


def plot_decay_fit(gamma, time, decay_curves, fit_curve, save_path=None):
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
    fig, ax = plt.subplots(constrained_layout=True)
    ax.errorbar(a_values, gamma_values, yerr=gamma_errors, fmt=".", linestyle="none", capsize=4, label="data")
    if np.isfinite(C) and np.isfinite(x) and len(fit_a):
        a_fit = np.linspace(fit_a.min(), fit_a.max(), 500)
        ax.plot(a_fit, power_law(a_fit, C, x), "--", label=rf"fit: $\gamma = {C:.3f} a^{{{x:.3f}}}$")
    ax.set_xlabel("a")
    ax.set_ylabel("fitted gamma")
    ax.set_title("a vs fitted gamma")
    ax.grid(True, which="both")
    ax.legend()
    save_figure(fig, save_path)
    plt.show()
    plt.close(fig)


def save_selected_figures(a_value, w0_arr, g_arr, log_abs_g, pole_w, config, figure_dir):
    limits = config["selected_plot_limits"][str(a_value)]
    a_name = format_a_for_filename(a_value)
    plot_g_heatmap(
        w0_arr,
        g_arr,
        config["m"],
        xlim=tuple(limits["xlim"]),
        ylim=tuple(limits["ylim"]),
        title=f"a = {a_value:g}: Iterative Solution of $G$",
        start_stride=config["plot_start_stride"],
        step_stride=config["plot_step_stride"],
        x_padding=config["plot_x_padding"],
        y_padding=config["plot_y_padding"],
        save_path=figure_dir / f"g_heatmap_a_{a_name}.pdf",
        show=False,
    )
    plot_poles(
        w0_arr,
        log_abs_g,
        pole_w,
        config["m"],
        xlim=tuple(limits["xlim"]),
        ylim=tuple(limits["ylim"]),
        title=f"a = {a_value:g}: Selected Pole Candidates",
        start_stride=config["plot_start_stride"],
        step_stride=config["plot_step_stride"],
        x_padding=config["plot_x_padding"],
        y_padding=config["plot_y_padding"],
        save_path=figure_dir / f"pole_candidates_a_{a_name}.pdf",
        show=False,
        log_vmin=config.get("plot_log_vmin"),
        log_vmax=config.get("plot_log_vmax"),
    )


def run_selected_job(job):
    config = job["config"]
    a_value = job["a"]
    w0_arr = load_w0_arr_for_job(job)
    pole_config = pole_config_for_a_from_config(a_value, config)
    g_arr, pole_w, pole_strength, log_abs_g = compute_poles_for_a(
        a_value,
        w0_arr,
        pole_config,
        config["eps"],
        config["m"],
        config["n_step"],
        config["denom_floor"],
        config["g_abs_max"],
        threads=job["threads"],
    )
    save_selected_figures(a_value, w0_arr, g_arr, log_abs_g, pole_w, config, Path(job["figure_dir"]))
    del g_arr, log_abs_g
    gc.collect()
    return {
        "a": a_value,
        "pole_w": pole_w,
        "pole_strength": pole_strength,
        "n_poles": len(pole_w),
        "threads": get_num_threads(),
    }


def run_sweep_job(job):
    config = job["config"]
    a_value = job["a"]
    w0_arr = load_w0_arr_for_job(job)
    scan_w0_arr = w0_arr[:: config["scan_stride"]]
    pole_config = pole_config_for_a_from_config(a_value, config)
    gamma, gamma_err, n_poles, scan_log_abs_g, scan_pole_w = compute_sweep_scan_for_a(
        a_value,
        scan_w0_arr,
        pole_config,
        config["eps"],
        config["m"],
        config["n_step"],
        config["denom_floor"],
        config["g_abs_max"],
        threads=job["threads"],
    )
    scan_path = None
    if should_save_sweep_scan(a_value, config):
        scan_path = Path(job["scan_output_dir"]) / sweep_scan_filename(a_value)
        limits = config.get("selected_plot_limits", {}).get(str(a_value))
        if limits is None:
            plot_xlim = (float(np.nanmin(scan_w0_arr.real)), float(np.nanmax(scan_w0_arr.real)))
            plot_ylim = (
                float(np.nanmin(scan_w0_arr.imag) - config["m"] * (config["n_step"] - 1)),
                float(np.nanmax(scan_w0_arr.imag)),
            )
        else:
            plot_xlim = tuple(limits["xlim"])
            plot_ylim = tuple(limits["ylim"])
        heatmap_x, heatmap_y, heatmap_z = make_pole_heatmap_arrays(
            scan_w0_arr,
            scan_log_abs_g,
            config["m"],
            xlim=plot_xlim,
            ylim=plot_ylim,
            start_stride=config["plot_start_stride"],
            step_stride=config["plot_step_stride"],
            x_padding=config["plot_x_padding"],
            y_padding=config["plot_y_padding"],
        )
        plot_log_vmin = config.get("plot_log_vmin")
        plot_log_vmax = config.get("plot_log_vmax")
        save_kwargs = {
            "a_value": a_value,
            "heatmap_x": heatmap_x,
            "heatmap_y": heatmap_y,
            "heatmap_z": heatmap_z,
            "pole_w": scan_pole_w,
            "gamma": gamma,
            "gamma_err": gamma_err,
            "n_poles": n_poles,
            "m": config["m"],
            "n_step": config["n_step"],
            "scan_stride": config["scan_stride"],
            "plot_xlim": np.asarray(plot_xlim),
            "plot_ylim": np.asarray(plot_ylim),
            "plot_start_stride": config["plot_start_stride"],
            "plot_step_stride": config["plot_step_stride"],
            "plot_x_padding": config["plot_x_padding"],
            "plot_y_padding": config["plot_y_padding"],
            "plot_log_vmin": np.nan if plot_log_vmin is None else plot_log_vmin,
            "plot_log_vmax": np.nan if plot_log_vmax is None else plot_log_vmax,
        }
        if config.get("save_sweep_scan_compressed", False):
            np.savez_compressed(scan_path, **save_kwargs)
        else:
            np.savez(scan_path, **save_kwargs)
    del scan_log_abs_g
    gc.collect()
    return {
        "idx": job["index"],
        "a": a_value,
        "gamma": gamma,
        "gamma_err": gamma_err,
        "n_poles": n_poles,
        "threads": get_num_threads(),
        "scan_path": str(scan_path) if scan_path is not None else None,
    }
