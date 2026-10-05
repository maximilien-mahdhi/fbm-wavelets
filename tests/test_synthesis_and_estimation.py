import numpy as np
import pytest

from fbmwavelets import (
    FractionalBrownianMotion,
    estimate_hurst_cwt,
    estimate_hurst_dwt,
    generate_fbm,
    rmd_synthesis,
    wavelet_synthesis,
    wornell_synthesis,
)


@pytest.mark.parametrize("hurst", [0.3, 0.7])
def test_dwt_estimator_on_exact_fbm(hurst):
    np.random.seed(0)
    _, series = generate_fbm(hurst, nb_points=2**12, nb_series=10)
    estimates = [estimate_hurst_dwt(x).hurst for x in series[hurst]]
    assert np.mean(estimates) == pytest.approx(hurst, abs=0.08)


@pytest.mark.parametrize("hurst", [0.3, 0.5])
def test_cwt_estimator_on_exact_fbm(hurst):
    np.random.seed(0)
    _, series = generate_fbm(hurst, nb_points=2**12, nb_series=10)
    estimates = [estimate_hurst_cwt(x).hurst for x in series[hurst]]
    assert np.mean(estimates) == pytest.approx(hurst, abs=0.08)


@pytest.mark.parametrize("hurst", [0.3, 0.7])
def test_wavelet_synthesis_has_requested_hurst(hurst):
    rng = np.random.default_rng(0)
    estimates = [
        estimate_hurst_dwt(wavelet_synthesis(hurst, J=12, rng=rng)).hurst for _ in range(20)
    ]
    assert np.mean(estimates) == pytest.approx(hurst, abs=0.12)


def test_synthesis_shapes_and_normalization():
    assert wavelet_synthesis(0.5, J=8, rng=0).shape == (256,)
    w = wornell_synthesis(0.5, J=4, N=256, rng=0)
    assert w.shape == (256,) and w[0] == 0 and np.std(w) == pytest.approx(1)
    r = rmd_synthesis(0.5, J=8, rng=0)
    assert r.shape == (257,) and np.std(r) == pytest.approx(1)


def test_synthesis_is_reproducible():
    assert np.array_equal(wavelet_synthesis(0.4, rng=3), wavelet_synthesis(0.4, rng=3))


def test_generate_fbm_shape_and_variance():
    t, series = generate_fbm([0.2, 0.8], nb_points=500, nb_series=3, variance=50)
    assert t.shape == (500,)
    assert series[0.2].shape == (3, 500)
    assert np.allclose(series[0.8].var(axis=1), 50)


@pytest.mark.parametrize("hurst", [0.3, 0.5, 0.8])
def test_process_increment_scale(hurst):
    np.random.seed(0)
    path = FractionalBrownianMotion(increment=0.01, delta=0.1, hurst=hurst).generate(5000)
    expected = 0.1 * 0.01**hurst
    assert np.diff(path.x, axis=0).std() == pytest.approx(expected, rel=0.1)
    assert path.x.shape == (5000, 2)
    assert np.all(path.x[0] == 0)
