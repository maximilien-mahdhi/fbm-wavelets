"""Shared series generation for the command-line scripts.

Both ``scripts/generate_fbm.py`` and ``scripts/make_gif.py`` expose the same generation
options (Hurst parameters, method, number of series, ...); they are defined once here.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from fbmwavelets.generators import generate_fbm, rescale_variance
from fbmwavelets.synthesis import rmd_synthesis, wavelet_synthesis, wornell_synthesis

log = logging.getLogger(__name__)

EXACT_METHODS = ("daviesharte", "cholesky", "hosking")
SYNTHESIS_METHODS = ("rmd", "wavelet", "wornell")
DEFAULT_LEVELS = {"rmd": 10, "wavelet": 8, "wornell": 6}


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that describe which fBm series to generate."""
    hurst = parser.add_mutually_exclusive_group(required=True)
    hurst.add_argument(
        "-H", "--hurst", type=float, nargs="+", metavar="H", help="Hurst parameters in (0, 1)."
    )
    hurst.add_argument(
        "-r",
        "--hurst-range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help="Hurst parameters START, START + STEP, ..., STOP (STOP included).",
    )
    parser.add_argument("-n", "--nb-series", type=int, default=1, help="Series per Hurst value.")
    parser.add_argument(
        "-N",
        "--nb-points",
        type=int,
        default=None,
        help="Samples per series (default 1000 for exact methods, 1024 for wornell). "
        "Ignored by rmd and wavelet, whose length is set by --levels.",
    )
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument(
        "-v", "--variance", type=float, default=None, help="Rescale each series to this variance."
    )
    parser.add_argument(
        "-m",
        "--method",
        default="daviesharte",
        choices=[*EXACT_METHODS, *SYNTHESIS_METHODS],
        help="Exact sampling (daviesharte, cholesky, hosking) or approximate synthesis "
        "(rmd, wavelet, wornell).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Number of scales J of a synthesis method: rmd gives 2**J + 1 samples, wavelet "
        "gives 2**J, wornell sums J octaves (defaults: rmd 10, wavelet 8, wornell 6).",
    )
    parser.add_argument(
        "--wavelet", default="db4", help="Mother wavelet of the wavelet and wornell methods."
    )
    parser.add_argument("--seed", type=int, default=55)


def resolve_hurst(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[float]:
    """Return the list of Hurst parameters requested by ``-H`` or ``-r``, validated."""
    if args.hurst_range is not None:
        start, stop, step = args.hurst_range
        if step <= 0 or stop < start:
            parser.error("--hurst-range: STEP must be positive and STOP >= START.")
        hurst = [round(float(h), 10) for h in np.arange(start, stop + step / 2, step)]
    else:
        hurst = list(args.hurst)
    if not all(0 < h < 1 for h in hurst):
        parser.error("Hurst parameters must lie in (0, 1).")
    return hurst


def _synthesize(
    args: argparse.Namespace, hurst: list[float], nb_series: int
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    levels = args.levels or DEFAULT_LEVELS[args.method]
    if args.method == "rmd":
        n = 2**levels + 1
    elif args.method == "wavelet":
        n = 2**levels
    else:
        n = args.nb_points or 1024
    if args.nb_points is not None and args.method in ("rmd", "wavelet"):
        log.warning(
            "--nb-points is ignored by %s: using %d samples (J=%d).", args.method, n, levels
        )

    rng = np.random.default_rng(args.seed)
    series = {}
    for h in hurst:
        if args.method == "rmd":
            paths = [rmd_synthesis(h, J=levels, rng=rng) for _ in range(nb_series)]
        elif args.method == "wavelet":
            paths = [
                wavelet_synthesis(h, J=levels, wavelet=args.wavelet, rng=rng)
                for _ in range(nb_series)
            ]
        else:
            paths = [
                wornell_synthesis(h, J=levels, N=n, wavelet=args.wavelet, rng=rng)
                for _ in range(nb_series)
            ]
        paths = np.stack(paths)
        series[h] = rescale_variance(paths, args.variance) if args.variance else paths
    return np.linspace(args.t0, args.t1, n), series


def generate_series(
    args: argparse.Namespace, hurst: list[float], nb_series: int | None = None
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Generate ``nb_series`` paths per Hurst parameter with the method chosen in ``args``.

    Returns the time grid and a dictionary mapping each H to an array (nb_series, n).
    """
    nb_series = args.nb_series if nb_series is None else nb_series
    np.random.seed(args.seed)
    if args.method in SYNTHESIS_METHODS:
        return _synthesize(args, hurst, nb_series)
    return generate_fbm(
        hurst,
        nb_points=args.nb_points or 1000,
        nb_series=nb_series,
        t0=args.t0,
        t1=args.t1,
        variance=args.variance,
        method=args.method,
    )
