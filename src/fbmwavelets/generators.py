"""Exact fBm sampling (Davies-Harte) for several Hurst parameters."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from fbm import FBM

log = logging.getLogger(__name__)


def rescale_variance(paths: np.ndarray, variance: float) -> np.ndarray:
    """Rescale each series (row) so that its empirical variance equals ``variance``."""
    return paths * np.sqrt(variance / paths.var(axis=-1, keepdims=True))


def generate_fbm(
    hurst: float | Sequence[float],
    nb_points: int = 1000,
    nb_series: int = 1,
    t0: float = 0.0,
    t1: float = 1.0,
    variance: float | None = None,
    method: str = "daviesharte",
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Generate fBm realizations on a regular time grid.

    Parameters
    ----------
    hurst:
        One Hurst parameter or a sequence of them, each in (0, 1).
    nb_points:
        Number of samples per series.
    nb_series:
        Number of independent series per Hurst parameter.
    t0, t1:
        Time interval; the series are sampled on ``linspace(t0, t1, nb_points)``.
    variance:
        If given, each series is rescaled so that its empirical variance equals this value.
    method:
        Sampling method of the ``fbm`` package (``daviesharte``, ``cholesky`` or ``hosking``).

    Returns
    -------
    t:
        Time grid, shape (nb_points,).
    series:
        Dictionary mapping each Hurst parameter to an array of shape (nb_series, nb_points).
    """
    hurst_values = [float(h) for h in np.atleast_1d(hurst)]
    t = np.linspace(t0, t1, nb_points)
    series: dict[float, np.ndarray] = {}
    for h in hurst_values:
        generator = FBM(n=nb_points - 1, hurst=h, length=t1 - t0, method=method)
        paths = np.stack([generator.fbm() for _ in range(nb_series)])
        if variance is not None:
            paths = rescale_variance(paths, variance)
        series[h] = paths
        log.debug("Generated %d series for H=%.2f", nb_series, h)
    return t, series
