"""Approximate fBm synthesis: wavelet synthesis, Wornell-type synthesis, midpoint displacement."""

from __future__ import annotations

import logging

import numpy as np
import pywt

log = logging.getLogger(__name__)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def wavelet_synthesis(
    hurst: float,
    J: int = 8,
    wavelet: str = "db4",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Heuristic wavelet synthesis of fBm on N = 2**J samples.

    The ``2**m`` detail coefficients at resolution level m (m = 0 is the coarsest) are
    i.i.d. Gaussian with standard deviation ``2**(-(H + 1/2) * m)``; the path is rebuilt
    with an inverse periodized DWT and shifted to start at zero.
    """
    gen = _rng(rng)
    n = 2**J
    coeffs = [np.zeros(1)]  # coarsest approximation: zero mean
    for j in range(J, 0, -1):
        m = J - j  # resolution level of the 2**m coefficients
        sigma = 2.0 ** (-(hurst + 0.5) * m)
        coeffs.append(sigma * gen.standard_normal(2**m))
    x = pywt.waverec(coeffs, wavelet, mode="periodization")[:n]
    return x - x[0]


def wornell_synthesis(
    hurst: float,
    J: int = 6,
    N: int = 1024,
    wavelet: str = "db4",
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Wornell-type synthesis: a sum of dilated and shifted wavelets on [0, 1].

    ``B(t) = sum_j sum_k 2**(j/2) * d_j[k] * psi(2**j t - k)`` where the detail
    coefficients ``d_j[k]`` are independent centered Gaussians of variance
    ``2**(-(2H + 1) j)``, i.e. a net amplitude ``2**(-j H)`` on ``psi(2**j t - k)``
    (Wornell, 1990). The wavelet is evaluated by linear interpolation of its tabulated
    version. The result starts at zero and has unit standard deviation.
    """
    gen = _rng(rng)
    _, psi, grid = pywt.Wavelet(wavelet).wavefun(level=10)
    t = np.linspace(0, 1, N)
    x = np.zeros(N)
    for j in range(J):
        scale = 2**j
        weight = 2.0 ** (-j * hurst)  # 2**(j/2) normalization times std 2**(-j (H + 1/2))
        for k in range(-2, scale + 2):  # range chosen to cover the support of psi
            x += weight * gen.standard_normal() * np.interp(scale * t - k, grid, psi, 0, 0)
    x -= x[0]
    return x / np.std(x)


def rmd_synthesis(
    hurst: float,
    J: int = 10,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Random midpoint displacement on N = 2**J + 1 samples.

    At each level the midpoint of a segment is the average of its endpoints plus a
    Gaussian displacement whose standard deviation is ``scale**H`` (``scale = 1`` at the
    coarsest level), the scale being halved after every level.
    The result is normalized to unit standard deviation.
    """
    gen = _rng(rng)
    n = 2**J + 1
    x = np.zeros(n)
    scale = 1.0
    x[-1] = gen.standard_normal() * scale**hurst
    step = n - 1
    while step > 1:
        half = step // 2
        sigma = scale**hurst
        mid = np.arange(half, n - 1, step)
        x[mid] = 0.5 * (x[mid - half] + x[mid + half]) + sigma * gen.standard_normal(mid.size)
        step = half
        scale /= 2
    return x / np.std(x)
