"""Animated GIF of fBm paths drawn progressively, for one or several Hurst parameters.

Generation options (-H/-r, -n, -N, -m, --levels, ...) are the same as in generate_fbm.py.

Examples
--------
    python scripts/make_gif.py -r 0.4 0.85 0.05
    python scripts/make_gif.py -r 0.4 0.85 0.05 --dim 2
    python scripts/make_gif.py -H 0.3 0.7 -n 3 --fps 30 --hold 2
    python scripts/make_gif.py -H 0.5 -n 10 -c series --no-legend -m rmd
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fbmwavelets.logger import setup_logging
from fbmwavelets.plotting import configure_matplotlib
from fbmwavelets.simulate import add_generation_arguments, generate_series, resolve_hurst

log = logging.getLogger("make_gif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_generation_arguments(parser)
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        choices=[1, 2],
        default=1,
        help="1: B^H_t against t. 2: planar paths (independent coordinates).",
    )
    parser.add_argument(
        "-c",
        "--color-by",
        default="hurst",
        choices=["hurst", "series"],
        help="One color per Hurst value (default, with a legend) or one color per series.",
    )
    parser.add_argument("--frames", type=int, default=80, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--hold", type=float, default=1.0, help="Seconds the last frame stays on screen."
    )
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=(9.0, 5.2), metavar=("W", "H"), help="Inches."
    )
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument(
        "--legend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the legend (default: yes when coloring by Hurst value).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Default depends on --dim.")
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
    if args.output is None:
        name = "fbm_animation.gif" if args.dim == 1 else "fbm_animation_2d.gif"
        args.output = Path("results") / name
    if args.legend is None:
        args.legend = args.color_by == "hurst"
    return args


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    matplotlib.use("Agg")
    configure_matplotlib(usetex=args.usetex)

    # A planar path needs two independent coordinates per series.
    t, series = generate_series(args, args.hurst, nb_series=args.nb_series * args.dim)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # (x, y) coordinates of every path: t against B^H_t in 1D, (B^H_t, B'^H_t) in 2D.
    paths: list[tuple[float, np.ndarray, np.ndarray]] = []
    for h, arr in series.items():
        for k in range(args.nb_series):
            if args.dim == 1:
                paths.append((h, t, arr[k]))
            else:
                paths.append((h, arr[2 * k], arr[2 * k + 1]))

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    lines = []
    for i, (h, _, _) in enumerate(paths):
        index = args.hurst.index(h) if args.color_by == "hurst" else i
        label = f"$H = {h}$" if args.color_by == "hurst" and i % args.nb_series == 0 else None
        (line,) = ax.plot([], [], color=colors[index % len(colors)], lw=args.linewidth, label=label)
        lines.append(line)
    ax.grid(True, alpha=0.3)
    if args.legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    # Same plot area for every animation, so that they line up in the README.
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.12, top=0.96)

    if args.dim == 1:
        ymax = max(np.abs(y).max() for _, _, y in paths) * 1.05
        ax.set(xlim=(t[0], t[-1]), ylim=(-ymax, ymax), xlabel="$t$", ylabel="$B^H_t$")
    else:
        # Equal scale on both axes, with limits widened to fill the frame.
        box = ax.get_position()
        ratio = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
        xs = np.concatenate([x for _, x, _ in paths])
        ys = np.concatenate([y for _, _, y in paths])
        half = max(1.1 * np.ptp(ys), 1.1 * np.ptp(xs) / ratio) / 2
        cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
        ax.set(
            xlim=(cx - half * ratio, cx + half * ratio),
            ylim=(cy - half, cy + half),
            xlabel="$x$",
            ylabel="$y$",
        )

    ends = np.linspace(2, t.size, args.frames).astype(int)

    def update(frame: int):
        end = ends[frame]
        for line, (_, x, y) in zip(lines, paths, strict=True):
            line.set_data(x[:end], y[:end])
        return lines

    frames = list(range(args.frames)) + [args.frames - 1] * round(args.hold * args.fps)
    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    log.info("Saved animation to %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
