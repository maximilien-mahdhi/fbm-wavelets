"""Wavelet-based time-frequency spectrum of fractional Brownian motion."""

from __future__ import annotations

import numpy as np


def time_frequency_spectrum(
    t: np.ndarray | float, omega: np.ndarray | float, hurst: float, sigma_h: float = 1.0
) -> np.ndarray:
    """Wavelet-based spectrum (WVSD) of an fBm.

    Evaluates ``sigma_H**2 (1 - 2**(1 - 2H) cos(4 pi omega t)) / |2 pi omega|**(2H + 1)``.
    ``t`` and ``omega`` are broadcast against each other; ``omega`` must not contain zero.
    """
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    numerator = 1 - 2 ** (1 - 2 * hurst) * np.cos(4 * np.pi * omega * t)
    return sigma_h**2 * numerator / np.abs(2 * np.pi * omega) ** (2 * hurst + 1)
