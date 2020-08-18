from spefit.pdf.sipm_gentile import sipm_gentile
import numpy as np


def test_sipm_gentile():
    x = np.linspace(-1, 20, 1000)
    y = sipm_gentile(x, 0.0, 0.2, 1.0, 0.1, 0.2, 1.0)
    np.testing.assert_allclose(np.trapz(y, x), 1, rtol=1e-3)
