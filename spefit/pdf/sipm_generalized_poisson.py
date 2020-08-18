from spefit.pdf.base import PDF, PDFParameter
from spefit.common.stats import normal_pdf
import numpy as np
from numba import njit, vectorize, float64, int64
from math import exp, sqrt, lgamma, log

__all__ = ["SiPMGeneralizedPoisson", "generalized_poisson", "sipm_generalized_poisson"]


class SiPMGeneralizedPoisson(PDF):
    def __init__(self, n_illuminations: int):
        """SPE PDF for a SiPM utilising a modified Poisson to describe the
        optical crosstalk

        Parameters
        ----------
        n_illuminations : int
            Number of illuminations to simultaneously fit
        """
        function = sipm_generalized_poisson
        parameters = dict(
            pe0=PDFParameter(initial=0, limits=(-2, 2)),
            pe0_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            pe=PDFParameter(initial=1, limits=(-2, 3)),
            pe_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            opct=PDFParameter(initial=0.2, limits=(0, 1)),
            lambda_=PDFParameter(initial=0.7, limits=(0, 5), multi=True),
        )
        super().__init__(n_illuminations, function, parameters)


@vectorize([float64(int64, float64, float64)], fastmath=True)
def generalized_poisson(k, mu, opct):
    """Generalized Poisson probabilities for a given mean number per event
    and per opct event.

    Parameters
    ----------
    k : int
        Photoelectron peak number
    mu : float
        The mean number per event
    opct : float
        The mean number per opct event

    Returns
    -------
    probability : float
    """
    mu_dash = mu + k * opct
    return mu * exp((k - 1) * log(mu_dash) - mu_dash - lgamma(k + 1))


@njit(fastmath=True)
def sipm_generalized_poisson(x, pe0, pe0_sigma, pe, pe_sigma, opct, lambda_):
    """SPE spectrum PDF for a SiPM using Gaussian peaks with amplitudes given by
    a modified Poisson formula

    TODO: Explanation/derivation

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    pe0 : float
        Distance of the zeroth peak (electronic pedestal) from the origin
    pe0_sigma : float
        Sigma of the zeroth peak, represents spread of electronic noise
    pe : float
        Distance of the first peak (1 photoelectron post opct) from the origin
    pe_sigma : float
        Sigma of the 1 photoelectron peak
    opct : float
        Optical crosstalk probability
    lambda_ : float
        Poisson mean (average illumination in p.e.)

    Returns
    -------
    spectrum : ndarray
        The y values of the total spectrum.
    """
    spectrum = np.zeros_like(x)
    p_max = 0
    for k in range(100):
        p = generalized_poisson(k, lambda_, opct)

        # Skip insignificant probabilities
        if p > p_max:
            p_max = p
        elif p < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        total_sigma = sqrt(k * pe_sigma ** 2 + pe0_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += p * normal_pdf(x, pe0 + k * pe, total_sigma)

    return spectrum
