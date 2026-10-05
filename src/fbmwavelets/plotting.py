"""Matplotlib configuration and figure saving helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

LATEX_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"


def configure_matplotlib(usetex: bool = False) -> None:
    """Set the global matplotlib style; ``usetex`` requires a LaTeX installation."""
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams["text.latex.preamble"] = LATEX_PREAMBLE if usetex else ""
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 200


def save_figure(fig: plt.Figure, name: str, save_dir: Path | None, fmt: str = "png") -> None:
    """Save ``fig`` as ``<save_dir>/<name>.<fmt>`` and close it. No-op if save_dir is None."""
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure to %s", path)
