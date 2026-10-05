"""Command-line fBm generator: choose the Hurst parameters, the method and the number of series.

Examples
--------
    python scripts/generate_fbm.py -H 0.3 0.6 -n 5
    python scripts/generate_fbm.py -r 0.4 0.85 0.05                  # H = 0.4, 0.45, ..., 0.85
    python scripts/generate_fbm.py -H 0.5 -n 10 -l single -c series
    python scripts/generate_fbm.py -H 0.3 0.6 -n 5 -m rmd -l single
    python scripts/generate_fbm.py -H 0.7 -m wavelet --levels 10 --wavelet db8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fbmwavelets.logger import setup_logging
from fbmwavelets.plotting import configure_matplotlib, save_figure
from fbmwavelets.simulate import add_generation_arguments, generate_series, resolve_hurst

log = logging.getLogger("generate_fbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_generation_arguments(parser)
    parser.add_argument(
        "-l",
        "--layout",
        default="auto",
        choices=["auto", "subplots", "single", "separate"],
        help="One panel per H (subplots), all series on one axis (single), or one figure per H "
        "(separate). auto: subplots for up to 3 Hurst values, single otherwise.",
    )
    parser.add_argument(
        "-c",
        "--color-by",
        default="hurst",
        choices=["hurst", "series"],
        help="With --layout single: one color per Hurst value, or one color per series.",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    parser.add_argument("--name", default="fbm_series", help="Output file name (no extension).")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--save-data", action="store_true", help="Also save the series as .npz.")
    parser.add_argument(
        "--usetex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render text with LaTeX (default). Use --no-usetex to fall back to mathtext.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()
    args.hurst = resolve_hurst(parser, args)
    if args.layout == "auto":
        args.layout = "subplots" if len(args.hurst) <= 3 else "single"
    return args


def plot_panels(t: np.ndarray, series: dict[float, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.8 * len(series)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (h, paths) in zip(axes, series.items(), strict=True):
        ax.plot(t, paths.T, alpha=0.8)
        ax.set_title(f"$H = {h}$")
        ax.set_ylabel("$B^H_t$")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("$t$")
    fig.tight_layout()
    return fig


def plot_single(t: np.ndarray, series: dict[float, np.ndarray], color_by: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # matplotlib default cycle
    if color_by == "series":
        k = 0
        for paths in series.values():
            for path in paths:
                ax.plot(t, path, color=colors[k % len(colors)], lw=1.2)
                k += 1
        ax.set_title(", ".join(f"$H = {h}$" for h in series))
    else:
        for i, (h, paths) in enumerate(series.items()):
            color = colors[i % len(colors)]
            ax.plot(t, paths.T, color=color, alpha=0.6 if paths.shape[0] > 1 else 0.9, lw=1.2)
            ax.plot([], [], color=color, label=f"$H = {h}$")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$B^H_t$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    matplotlib.use("Agg")
    configure_matplotlib(usetex=args.usetex)
    t, series = generate_series(args, args.hurst)
    log.info(
        "Generated %d series for each H in %s (method: %s)",
        args.nb_series,
        list(series),
        args.method,
    )

    if args.layout == "subplots":
        save_figure(plot_panels(t, series), args.name, args.save_dir, args.format)
    elif args.layout == "single":
        save_figure(plot_single(t, series, args.color_by), args.name, args.save_dir, args.format)
    else:
        for h, paths in series.items():
            save_figure(plot_panels(t, {h: paths}), f"{args.name}_H{h}", args.save_dir, args.format)

    if args.save_data:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        path = args.save_dir / f"{args.name}.npz"
        np.savez(path, t=t, **{f"H_{h}": paths for h, paths in series.items()})
        log.info("Saved data to %s", path)


if __name__ == "__main__":
    main()
