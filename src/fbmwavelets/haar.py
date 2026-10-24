"""Correlation of Haar wavelet coefficients of fractional Brownian motion."""

from __future__ import annotations

import numpy as np


def xi(lag: float | np.ndarray, hurst: float | np.ndarray) -> np.ndarray:
    """Discrete Haar kernel ξ_H[k], defined for every integer lag k.

    For k ≠ 0:

        ξ_H[k] = ( |k−1|^(2H+2) − 4|k−½|^(2H+2) + 6|k|^(2H+2)
                   − 4|k+½|^(2H+2) + |k+1|^(2H+2) ) / ((2H+1)(2H+2))

    and for k = 0:

        ξ_H[0] = (1 − 2^(−2H)) / ((H+1)(2H+1))
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
    """Correlation of the Haar wavelet coefficients of an fBm at scale j.

        ρ_j(n, m) = E[d_j[n] d_j[m]] = (σ_H² / 2) · ξ_H[n − m] · (2^j)^(2H+1)

    with ξ_H given by xi. Arguments are broadcast against each other.
    """
    amplitude = (2.0 ** np.asarray(j, dtype=float)) ** (2 * np.asarray(hurst, dtype=float) + 1)
    return sigma_h**2 / 2 * xi(lag, hurst) * amplitude
