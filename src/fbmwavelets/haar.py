"""Correlation of Haar wavelet coefficients of fractional Brownian motion."""

from __future__ import annotations

import numpy as np


def xi(lag: float | np.ndarray, hurst: float | np.ndarray) -> np.ndarray:
    """Discrete Haar kernel :math:`\\xi_H[k]`, defined for every integer lag ``k``.

    For ``k != 0`` it is the normalized fourth-order finite difference of ``|k|**(2H + 2)``;
    for ``k = 0`` it equals ``(1 - 2**(-2H)) / ((H + 1)(2H + 1))``.
    """
    k = np.abs(np.asarray(lag, dtype=float))
    h = np.asarray(hurst, dtype=float)
    p = 2 * h + 2
    safe_k = np.where(k == 0, 1.0, k)  # avoid evaluating the off-diagonal formula at k = 0
    off_diagonal = (
        (safe_k - 1) ** p
        - 4 * (safe_k - 0.5) ** p
        + 6 * safe_k**p
        - 4 * (safe_k + 0.5) ** p
        + (safe_k + 1) ** p
    ) / ((2 * h + 1) * (2 * h + 2))
    diagonal = (1 - 2 ** (-2 * h)) / ((h + 1) * (2 * h + 1))
    return np.where(k == 0, diagonal, off_diagonal)


def haar_correlation(
    j: float | np.ndarray,
    lag: float | np.ndarray,
    hurst: float | np.ndarray,
    sigma_h: float = 1.0,
) -> np.ndarray:
    """Correlation :math:`\\rho_j(n, m) = E[d_j[n] d_j[m]]` of Haar coefficients of an fBm.

    Equals ``sigma_H**2 / 2 * xi_H[n - m] * (2**j)**(2H + 1)``. Arguments are broadcast
    against each other.
    """
    amplitude = (2.0 ** np.asarray(j, dtype=float)) ** (2 * np.asarray(hurst, dtype=float) + 1)
    return sigma_h**2 / 2 * xi(lag, hurst) * amplitude
