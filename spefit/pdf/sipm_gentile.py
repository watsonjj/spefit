from spefit.pdf.base import PDF, PDFParameter
from spefit.common.stats import poisson, normal_pdf
from spefit.common.basic import binom
from numba import njit
from math import exp, pow, sqrt

__all__ = ["SiPMGentile"]


class SiPMGentile(PDF):
    def __init__(self, n_illuminations: int):
        func = sipm_gentile
        param = dict(
            eped=PDFParameter(initial=0, limits=(-2, 2)),
            eped_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            pe=PDFParameter(initial=1, limits=(-2, 3)),
            pe_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            opct=PDFParameter(initial=0.2, limits=(0, 1)),
            lambda_=PDFParameter(initial=0.7, limits=(0, 5), multi=True),
        )
        super().__init__(n_illuminations=n_illuminations, function=func, parameters=param)


@njit(fastmath=True)
def sipm_gentile(x, eped, eped_sigma, pe, pe_sigma, opct, lambda_):
    """
    PDF for the SPE spectrum of a SiPM as defined in Gentile 2010
    http://adsabs.harvard.edu/abs/2010arXiv1006.3263G

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    eped : float
        Distance of the zeroth peak (electronic pedestal) from the origin
    eped_sigma : float
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
        The y values of the total spectrum
    """
    # TODO: Handle cases where pedestal not included (dark noise spectrum)
    # TODO: SES
    # Obtain pedestal peak
    p_ped = exp(-lambda_)
    spectrum = p_ped * normal_pdf(x, eped, eped_sigma)

    pk_max = 0

    # Loop over the possible total number of cells fired
    for k in range(1, 100):
        pk = 0
        for j in range(1, k+1):
            pj = poisson(j, lambda_)  # Probability for j initial fired cells

            # Skip insignificant probabilities
            if pj < 1e-4:
                continue

            # Sum the probability from the possible combinations which result
            # in a total of k fired cells to get the total probability of k
            # fired cells
            pk += pj * pow(1-opct, j) * pow(opct, k-j) * binom(k-1, j-1)

        # Skip insignificant probabilities
        if pk > pk_max:
            pk_max = pk
        elif pk < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        total_sigma = sqrt(k * pe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += pk * normal_pdf(x, eped + k * pe, total_sigma)

    return spectrum
