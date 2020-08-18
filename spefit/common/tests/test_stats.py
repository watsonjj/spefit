from spefit.common.stats import poisson, normal_pdf, poisson_logpmf
import scipy.stats as scipy_stats
import numpy as np
from numpy.testing import assert_allclose


def test_poisson_pmf():
    k = np.arange(100) - 10
    mu = np.arange(100) - 10
    with np.errstate(divide='ignore', invalid='ignore'):
        assert_allclose(poisson_logpmf(k, mu), scipy_stats.poisson.logpmf(k, mu), rtol=1e-5)
        assert_allclose(poisson_logpmf(k, mu), scipy_stats.poisson.logpmf(k, mu), rtol=1e-5)
        assert_allclose(poisson_logpmf(np.nan, mu), scipy_stats.poisson.logpmf(np.nan, mu), rtol=1e-5)
        assert_allclose(poisson_logpmf(k, np.nan), scipy_stats.poisson.logpmf(k, np.nan), rtol=1e-5)
        assert_allclose(poisson_logpmf(np.nan, np.nan), scipy_stats.poisson.logpmf(np.nan, np.nan), rtol=1e-5)


def test_poisson():
    k = np.arange(1, 100)
    mu = np.arange(1, 100)
    assert_allclose(poisson(k, mu), scipy_stats.poisson.pmf(k, mu))


def test_normal_pdf():
    x = np.linspace(-10, 10, 100)
    mean = 0
    std = 5
    assert_allclose(normal_pdf(x, mean, std), scipy_stats.norm.pdf(x, mean, std))
