"""Reproducible experiments. Each one is registered in ``EXPERIMENTS`` under a CLI name."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from fbmwavelets.estimation import estimate_hurst_cwt, estimate_hurst_dwt
from fbmwavelets.generators import generate_fbm
from fbmwavelets.haar import haar_correlation
from fbmwavelets.plotting import save_figure
from fbmwavelets.processes import BrownianMotion
from fbmwavelets.spectrum import time_frequency_spectrum
from fbmwavelets.synthesis import rmd_synthesis, wavelet_synthesis, wornell_synthesis

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Run-wide options shared by all experiments."""

    save_dir: Path | None = Path("results")
    fmt: str = "png"
    seed: int = 55


def _seed(cfg: Settings) -> np.random.Generator:
    """Seed the NumPy global state (used by the ``fbm`` package) and return a Generator."""
    np.random.seed(cfg.seed)
    return np.random.default_rng(cfg.seed)


# --------------------------------------------------------------------------
# Paths and generators
# --------------------------------------------------------------------------
def brownian_motion(cfg: Settings) -> None:
    """Plot a two-dimensional Brownian path."""
    _seed(cfg)
    path = BrownianMotion(increment=0.01, delta=0.1, dim=2).generate(1000)
    log.info("Brownian path: mean=%s, std=%s", path.mean().round(3), path.std().round(3))
    fig, ax = plt.subplots(figsize=(6, 5))
    path.plot(ax)
    ax.set_title("Two-dimensional Brownian motion")
    save_figure(fig, "brownian_motion", cfg.save_dir, cfg.fmt)


def brownian_animation(cfg: Settings) -> None:
    """Save an animated GIF of a two-dimensional Brownian path."""
    _seed(cfg)
    path = BrownianMotion().generate(500)
    if cfg.save_dir is None:
        log.warning("No save directory: the animation is not written.")
        return
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    path.plot_animation(interval=40, step=5, save_path=str(cfg.save_dir / "brownian_motion.gif"))


def fbm_paths(cfg: Settings) -> None:
    """Davies-Harte fBm paths for several Hurst parameters."""
    _seed(cfg)
    hurst = [0.2, 0.5, 0.8]
    t, series = generate_fbm(hurst, nb_points=1000, nb_series=3, variance=50)
    fig, axes = plt.subplots(len(hurst), 1, figsize=(10, 7), sharex=True)
    for ax, h in zip(axes, hurst, strict=True):
        ax.plot(t, series[h].T, alpha=0.8)
        ax.set_title(f"$H = {h}$")
        ax.set_ylabel("$B^H_t$")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    save_figure(fig, "fbm_paths", cfg.save_dir, cfg.fmt)


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------
def _synthesis_figure(
    cfg: Settings,
    name: str,
    title: str,
    hurst: Sequence[float],
    make: Callable[[float, np.random.Generator], np.ndarray],
    n_sim: int = 2,
) -> None:
    rng = _seed(cfg)
    fig, axes = plt.subplots(len(hurst), 1, figsize=(10, 2.6 * len(hurst)), sharex=True)
    for ax, h in zip(axes, hurst, strict=True):
        for _ in range(n_sim):
            ax.plot(make(h, rng), alpha=0.8)
        ax.set_title(f"$H = {h}$")
        ax.set_ylabel("$B^H[n]$")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sample index $n$")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, name, cfg.save_dir, cfg.fmt)


def synthesis_wavelet(cfg: Settings) -> None:
    """Wavelet-based heuristic synthesis."""
    _synthesis_figure(
        cfg, "synthesis_wavelet", "Wavelet-based fBm synthesis (db4)", [0.2, 0.5, 0.8],
        lambda h, rng: wavelet_synthesis(h, J=8, rng=rng),
    )  # fmt: skip


def synthesis_wornell(cfg: Settings) -> None:
    """Wornell-type synthesis with dyadic interpolation of the wavelet."""
    _synthesis_figure(
        cfg, "synthesis_wornell", "Wornell-type fBm synthesis (db4)", [0.2, 0.5, 0.8],
        lambda h, rng: wornell_synthesis(h, J=6, N=1024, rng=rng),
    )  # fmt: skip


def synthesis_rmd(cfg: Settings) -> None:
    """Random midpoint displacement."""
    _synthesis_figure(
        cfg, "synthesis_rmd", "Random midpoint displacement", [0.2, 0.5, 0.8],
        lambda h, rng: rmd_synthesis(h, J=10, rng=rng),
    )  # fmt: skip


# --------------------------------------------------------------------------
# Hurst estimation
# --------------------------------------------------------------------------
HURST_GRID = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)


def latex_hurst_table(hurst: Sequence[float], estimates: Sequence[float]) -> str:
    """Build a horizontal LaTeX table of true values, estimates and absolute errors."""
    n = len(hurst)
    head = " & ".join(f"${h:.4g}$" for h in hurst)
    est = " & ".join(f"${e:.4g}$" for e in estimates)
    err = " & ".join(f"${abs(h - e):.4g}$" for h, e in zip(hurst, estimates, strict=True))
    return "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            r"\begin{tabular}{l" + "c" * n + "}",
            r"\hline",
            rf"$H$ & {head} \\",
            r"\hline",
            rf"$\hat{{H}}$ & {est} \\",
            rf"$|H - \hat{{H}}|$ & {err} \\",
            r"\hline",
            r"\end{tabular}",
            r"\caption{Hurst estimation results}",
            r"\label{tab:hurst}",
            r"\end{table}",
        ]
    )


def hurst_dwt(cfg: Settings) -> None:
    """DWT estimation on one path per Hurst value; log-variance diagram."""
    _seed(cfg)
    _, series = generate_fbm(HURST_GRID, nb_points=2**12, nb_series=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for h, paths in series.items():
        est = estimate_hurst_dwt(paths[0])
        ax.plot(est.scales, est.log_variance, "o-", label=f"$H={h}$, $\\hat H={est.hurst:.3f}$")
        log.info("H=%.2f -> estimate %.4f", h, est.hurst)
    ax.set_xlabel("Octave $j$")
    ax.set_ylabel(r"$\log_2\,\mathrm{Var}(d_j[n])$")
    ax.set_title(r"Log-variance diagram of $d_j[n]$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, "hurst_dwt_diagram", cfg.save_dir, cfg.fmt)


def hurst_table(cfg: Settings) -> None:
    """LaTeX table of DWT Hurst estimates."""
    _seed(cfg)
    _, series = generate_fbm(HURST_GRID, nb_points=2**12, nb_series=1)
    estimates = [estimate_hurst_dwt(series[h][0]).hurst for h in HURST_GRID]
    table = latex_hurst_table(HURST_GRID, estimates)
    log.info("LaTeX table:\n%s", table)
    if cfg.save_dir is not None:
        cfg.save_dir.mkdir(parents=True, exist_ok=True)
        (cfg.save_dir / "hurst_table.tex").write_text(table + "\n")


def hurst_error(cfg: Settings, n_points: int = 100) -> None:
    """Absolute error of the DWT and CWT estimators over a fine grid of H (slow)."""
    _seed(cfg)
    grid = np.linspace(0, 1, n_points, endpoint=False)[1:]
    _, series = generate_fbm(grid, nb_points=2**12, nb_series=1)
    dwt = np.array([estimate_hurst_dwt(series[h][0]).hurst for h in grid])
    cwt = np.array([estimate_hurst_cwt(series[h][0]).hurst for h in grid])
    for label, est in (("DWT", dwt), ("CWT", cwt)):
        log.info("%s: mean absolute error = %.4f", label, np.mean(np.abs(est - grid)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, np.abs(dwt - grid), label="DWT (db4)")
    ax.plot(grid, np.abs(cwt - grid), label="CWT (morlet)")
    ax.set_xlabel("$H$")
    ax.set_ylabel(r"$|H - \hat{H}|$")
    ax.set_title("Absolute estimation error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "hurst_error", cfg.save_dir, cfg.fmt)


# --------------------------------------------------------------------------
# Haar correlation and spectrum
# --------------------------------------------------------------------------
def haar_correlation_figures(cfg: Settings) -> None:
    """3D surface and 1D slices of the Haar coefficient correlation."""
    h_values = np.linspace(0, 1, 80)[1:]
    lags = np.arange(-50, 50)
    j_values = np.arange(-3, 4)
    h0, k0 = 0.7, 3

    lag_grid, h_grid = np.meshgrid(lags, h_values, indexing="ij")
    corr = haar_correlation(0, lag_grid, h_grid)
    corr = corr / np.max(np.abs(corr[:, -1]))  # normalize by the largest-H column
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(lag_grid, h_grid, corr, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Lag $k = n - m$")
    ax.set_ylabel("$H$")
    ax.set_zlabel(r"$\rho_j(n, m)$")
    ax.set_title(r"Haar coefficient correlation $\rho_j(n, m)$ (normalized)")
    save_figure(fig, "haar_correlation_3d", cfg.save_dir, cfg.fmt)

    fig, ax = plt.subplots(figsize=(6, 4))
    for h in (0.1, 0.3, 0.5, 0.7, 0.9):
        ax.plot(lags, haar_correlation(0, lags, h), lw=2, label=f"$H={h}$")
    ax.set(xlabel="$k$", ylabel=r"$\rho_j(n, m)$", title="Slices in $k$ for different $H$ ($j=0$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "haar_correlation_slices_k_H", cfg.save_dir, cfg.fmt)

    fig, ax = plt.subplots(figsize=(6, 4))
    for j in j_values:
        corr_k = haar_correlation(j, lags, h0)
        ax.plot(lags, corr_k / corr_k[lags == 0], lw=2, label=f"$j={j}$")
    ax.set(
        xlabel="$k$",
        ylabel=r"Normalized $\rho_j(n, m)$",
        title=f"Slices in $k$ for different $j$ ($H={h0}$)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, "haar_correlation_slices_k_j", cfg.save_dir, cfg.fmt)

    fig, ax = plt.subplots(figsize=(6, 4))
    for j in j_values:
        corr_h = haar_correlation(j, k0, h_values)
        ax.plot(h_values, corr_h / corr_h.max(), lw=2, label=f"$j={j}$")
    ax.set(
        xlabel="$H$",
        ylabel=r"Normalized $\rho_j(n, m)$",
        title=f"Slices in $H$ for different $j$ ($k={k0}$)",
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save_figure(fig, "haar_correlation_slices_H_j", cfg.save_dir, cfg.fmt)


def fbm_spectrum(cfg: Settings) -> None:
    """Time-frequency spectrum of fBm for several Hurst parameters."""
    t = np.linspace(0.01, 1, 300)
    omega = np.linspace(0.5, 50, 300)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, h in zip(axes, (0.1, 0.5, 0.9), strict=True):
        s = np.abs(time_frequency_spectrum(t[None, :], omega[:, None], h))
        mesh = ax.pcolormesh(t, omega, s, norm=LogNorm(), shading="auto", cmap="magma")
        ax.set_title(f"$H = {h}$")
        ax.set_xlabel("$t$")
        fig.colorbar(mesh, ax=ax, label=r"$|\mathrm{WVSD}(t, \omega)|$")
    axes[0].set_ylabel(r"$\omega$")
    fig.tight_layout()
    save_figure(fig, "fbm_spectrum", cfg.save_dir, cfg.fmt)


EXPERIMENTS: dict[str, Callable[[Settings], None]] = {
    "brownian-motion": brownian_motion,
    "brownian-animation": brownian_animation,
    "fbm-paths": fbm_paths,
    "synthesis-wavelet": synthesis_wavelet,
    "synthesis-wornell": synthesis_wornell,
    "synthesis-rmd": synthesis_rmd,
    "hurst-dwt": hurst_dwt,
    "hurst-table": hurst_table,
    "hurst-error": hurst_error,
    "haar-correlation": haar_correlation_figures,
    "fbm-spectrum": fbm_spectrum,
}

# Experiments run by "all": the animation and the slow error sweep are opt-in.
DEFAULT_EXPERIMENTS = [n for n in EXPERIMENTS if n not in {"brownian-animation", "hurst-error"}]


def run(name: str, cfg: Settings) -> None:
    """Run one experiment by its registered name."""
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}. Available: {', '.join(EXPERIMENTS)}")
    log.info("Running experiment: %s", name)
    EXPERIMENTS[name](cfg)
