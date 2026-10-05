"""Command-line entry point: run one or several experiments.

Examples
--------
    python scripts/run_experiment.py hurst-dwt
    python scripts/run_experiment.py all --format pdf
    python scripts/run_experiment.py all --no-usetex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

from fbmwavelets.experiments import DEFAULT_EXPERIMENTS, EXPERIMENTS, Settings, run
from fbmwavelets.logger import setup_logging
from fbmwavelets.plotting import configure_matplotlib

log = logging.getLogger("run_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiments",
        nargs="+",
        metavar="EXPERIMENT",
        choices=[*EXPERIMENTS, "all"],
        help=f"One or more of: {', '.join(EXPERIMENTS)}, all.",
    )
    parser.add_argument("--save-dir", type=Path, default=Path("results"))
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument(
        "--usetex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render text with LaTeX (default). Use --no-usetex to fall back to mathtext.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    matplotlib.use("Agg")
    configure_matplotlib(usetex=args.usetex)

    names: list[str] = []
    for name in args.experiments:
        for expanded in DEFAULT_EXPERIMENTS if name == "all" else [name]:
            if expanded not in names:
                names.append(expanded)
    cfg = Settings(save_dir=args.save_dir, fmt=args.format, seed=args.seed)
    for name in names:
        run(name, cfg)


if __name__ == "__main__":
    main()
