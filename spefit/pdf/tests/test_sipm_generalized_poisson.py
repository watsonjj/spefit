from spefit.pdf.sipm_generalized_poisson import (
    generalized_poisson,
    sipm_generalized_poisson,
)
import numpy as np


def test_generalized_poisson():
    y = np.zeros(100)
    for i in range(y.size):
        y[i] = generalized_poisson(i, 1, 0.4)
    np.testing.assert_allclose(np.sum(y), 1, rtol=1e-3)


def test_sipm_generalized_poisson():
    x = np.linspace(-1, 20, 1000)
    y = sipm_generalized_poisson(x, 0.0, 0.2, 1.0, 0.1, 0.2, 1.0)
    np.testing.assert_allclose(np.trapz(y, x), 1, rtol=1e-3)
