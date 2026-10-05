import numpy as np
import pytest

from fbmwavelets import haar_correlation


def _numerical_correlation(lag: int, hurst: float, m: int = 600) -> float:
    """E[d[n] d[m]] for psi = +1 on [0, 1/2), -1 on [1/2, 1), by midpoint-rule integration."""
    s = (np.arange(m) + 0.5) / m
    psi = np.where(s < 0.5, 1.0, -1.0)
    kernel = np.abs(s[:, None] - (s[None, :] + lag)) ** (2 * hurst)
    return -0.5 * psi @ kernel @ psi / m**2


@pytest.mark.parametrize("hurst", [0.3, 0.7])
@pytest.mark.parametrize("lag", [0, 1, 3])
def test_matches_double_integral(hurst, lag):
    assert haar_correlation(0, lag, hurst) == pytest.approx(
        _numerical_correlation(lag, hurst), rel=1e-3
    )


def test_symmetric_in_lag():
    lags = np.arange(1, 10)
    assert np.allclose(haar_correlation(0, lags, 0.6), haar_correlation(0, -lags, 0.6))


def test_scale_factor():
    ratio = haar_correlation(2, 1, 0.7) / haar_correlation(0, 1, 0.7)
    assert ratio == pytest.approx(2 ** (2 * (2 * 0.7 + 1)))


def test_broadcasting_shape():
    out = haar_correlation(np.arange(3)[:, None], np.arange(-5, 5)[None, :], 0.5)
    assert out.shape == (3, 10)
