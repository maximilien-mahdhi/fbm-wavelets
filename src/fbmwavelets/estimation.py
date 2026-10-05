"""Hurst parameter estimation from the scale behavior of wavelet coefficients."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pywt
from scipy.stats import linregress

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HurstEstimate:
    """Result of a log-variance regression.

    For fBm, ``E[d_j^2] ~ 2**(j (2H + 1))``, hence ``H = (slope - 1) / 2``.
    """

    hurst: float
    slope: float
    intercept: float
    scales: np.ndarray  # octaves j used in the regression
    log_variance: np.ndarray  # log2 of the mean squared coefficients at each octave
    r_value: float


def _regress(octaves: np.ndarray, log_var: np.ndarray) -> HurstEstimate:
    if octaves.size < 2:
        raise ValueError("At least two scales are required for the regression.")
    fit = linregress(octaves, log_var)
    return HurstEstimate(
        hurst=(fit.slope - 1) / 2,
        slope=fit.slope,
        intercept=fit.intercept,
        scales=octaves,
        log_variance=log_var,
        r_value=fit.rvalue,
    )


def estimate_hurst_dwt(
    x: np.ndarray, wavelet: str = "db4", j_max: int = 8, mode: str = "symmetric"
) -> HurstEstimate:
    """Estimate H of an fBm path by regression on DWT detail coefficients.

    The finest octave (j = 1) and the coarsest one are discarded because of
    discretization and boundary effects.

    Parameters
    ----------
    x:
        One-dimensional fBm path.
    wavelet:
        Discrete wavelet name.
    j_max:
        Maximal decomposition level (reduced if the signal is too short).
    mode:
        Signal extension mode. ``symmetric`` avoids the artificial jump that the
        periodized extension creates between the two ends of a non-periodic path.
    """
    x = np.asarray(x, dtype=float)
    level = min(j_max, pywt.dwt_max_level(x.size, pywt.Wavelet(wavelet).dec_len))
    details = pywt.wavedec(x, wavelet, level=level, mode=mode)[1:][::-1]
    octaves = np.arange(1, level + 1)
    log_var = np.log2([np.mean(d**2) for d in details])
    keep = slice(1, -1)  # drop finest and coarsest octaves
    result = _regress(octaves[keep], log_var[keep])
    log.debug("DWT estimate: H=%.3f (slope=%.3f, levels=%d)", result.hurst, result.slope, level)
    return result


def estimate_hurst_cwt(
    x: np.ndarray, wavelet: str = "morl", j_max: int = 8, trim: float = 0.1
) -> HurstEstimate:
    """Estimate H of an fBm path by regression on CWT coefficients.

    The scales are ``2**1, ..., 2**j_max``. A fraction ``trim`` of the samples is
    removed at both ends to limit boundary effects.
    """
    x = np.asarray(x, dtype=float)
    octaves = np.arange(1, j_max + 1)
    coefs, _ = pywt.cwt(x, 2.0**octaves, wavelet)
    margin = int(trim * x.size)
    inner = coefs[:, margin : x.size - margin]
    log_var = np.log2(np.mean(np.abs(inner) ** 2, axis=1))
    result = _regress(octaves, log_var)
    log.debug("CWT estimate: H=%.3f (slope=%.3f)", result.hurst, result.slope)
    return result
