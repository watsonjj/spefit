from spefit.pdf.base import PDF, PDFParameter
from spefit.common.stats import poisson, normal_pdf
from numba import njit
from math import exp, sqrt

__all__ = ["PMTSingleGaussian"]


class PMTSingleGaussian(PDF):
    def __init__(self, n_illuminations: int):
        func = pmt_single_gaussian
        param = dict(
            eped=PDFParameter(initial=0, limits=(-2, 2)),
            eped_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            pe=PDFParameter(initial=1, limits=(-2, 3)),
            pe_sigma=PDFParameter(initial=0.1, limits=(0, 2)),
            lambda_=PDFParameter(initial=0.7, limits=(0, 5), multi=True),
        )
        super().__init__(n_illuminations=n_illuminations, function=func, parameters=param)


@njit(fastmath=True, parallel=True)
def pmt_single_gaussian(x, eped, eped_sigma, pe, pe_sigma, lambda_):
    """
    Simple description of the SPE spectrum PDF for a traditional
    Photomultiplier Tube, with the underlying 1 photoelectron PDF described by
    a single gaussian

    Parameters
    ----------
    x : ndarray
        The x values to evaluate at
    eped : float
        Distance of the zeroth peak (electronic pedestal) from the origin
    eped_sigma : float
        Sigma of the zeroth peak, represents spread of electronic noise
    pe : float
        Distance of the first peak (1 photoelectron) from the origin
    pe_sigma : float
        Sigma of the 1 photoelectron peak
    lambda_ : float
        Poisson mean (average illumination in p.e.)

    Returns
    -------
    spectrum : ndarray
        The y values of the total spectrum.
    """
    # Obtain pedestal peak
    p_ped = exp(-lambda_)
    spectrum = p_ped * normal_pdf(x, eped, eped_sigma)

    pk_max = 0

    # Loop over the possible total number of cells fired
    for k in range(1, 100):
        p = poisson(k, lambda_)  # Probability to get k avalanches

        # Skip insignificant probabilities
        if p > pk_max:
            pk_max = p
        elif p < 1e-4:
            break

        # Combine spread of pedestal and pe peaks
        total_sigma = sqrt(k * pe_sigma ** 2 + eped_sigma ** 2)

        # Evaluate probability at each value of x
        spectrum += p * normal_pdf(x, eped + k * pe, total_sigma)

    return spectrum
