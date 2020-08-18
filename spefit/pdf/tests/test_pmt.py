from spefit.pdf.pmt_single_gaussian import pmt_single_gaussian
import numpy as np


def test_pmt():
    x = np.linspace(-1, 20, 1000)
    y = pmt_single_gaussian(x, 0., 0.2, 1., 0.1, 1.)
    np.testing.assert_allclose(np.trapz(y, x), 1, rtol=1e-3)
