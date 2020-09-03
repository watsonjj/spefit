from spefit.estimate import (
    find_spe_peaks,
    calculate_peak_ratio,
    estimate_spe_parameters,
)
from spefit import PMTSingleGaussian
import numpy as np
import pytest


def get_test_x_y(disable_pedestal=False):
    pdf = PMTSingleGaussian(1, disable_pedestal)
    parameters = np.array([0, 0.3, 2, 0.1, 2])
    x = np.linspace(-5, 100, 10000)
    y = pdf(x, parameters, 0)
    return x, y, parameters


def test_find_spe_peaks():
    x, y, parameters = get_test_x_y()

    peak_x, peak_y, peak_sigma = find_spe_peaks(x, y)
    assert len(peak_x) == 5
    assert len(peak_y) == 5
    assert len(peak_sigma) == 5
    np.testing.assert_allclose(peak_x[0], parameters[0], atol=1e-2)
    np.testing.assert_allclose(peak_sigma[0], parameters[1], rtol=1e-2)

    peak_x, peak_y, peak_sigma = find_spe_peaks(x, y, max_n_peaks=2)
    assert len(peak_x) == 2
    assert len(peak_y) == 2
    assert len(peak_sigma) == 2

    with pytest.raises(ValueError):
        find_spe_peaks(x, np.zeros_like(x), min_n_peaks=1)

    with pytest.raises(ValueError):
        find_spe_peaks(x, y, min_n_peaks=100)


def test_calculate_peak_ratio():
    ratio = calculate_peak_ratio(k=1, lambda_=3, sigma0=1, sigma1=1)
    np.testing.assert_allclose(ratio, np.sqrt(2) / 3)

    ratio = calculate_peak_ratio(k=np.array([1, 1]), lambda_=3, sigma0=1, sigma1=1)
    assert ratio.size == 2
    np.testing.assert_allclose(ratio, np.sqrt(2) / 3)


@pytest.mark.parametrize("disable_pedestal", [True, False])
def test_estimate_spe_parameters(disable_pedestal):
    x, y, parameters = get_test_x_y(disable_pedestal)
    estimates = estimate_spe_parameters(x, y, disable_pedestal)

    np.testing.assert_allclose(estimates[0], parameters[0], atol=1e-2)
    np.testing.assert_allclose(estimates[1], parameters[1], rtol=1e-3)
    np.testing.assert_allclose(estimates[2], parameters[2], rtol=1e-3)
    np.testing.assert_allclose(estimates[3], parameters[3], rtol=1e-3)
    np.testing.assert_allclose(estimates[4], parameters[4], rtol=1e-3)
