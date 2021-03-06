"""spefit - Fitting of Single Photoelectron Spectra."""
from .container import ChargeContainer
from .cost import Cost, UnbinnedNLL, BinnedNLL, LeastSquares
from .pdf import (
    PDF,
    PDFParameter,
    PMTSingleGaussian,
    SiPMModifiedPoisson,
    SiPMGentile,
)
from .fitter import minimize_with_iminuit, CameraFitter

__all__ = [
    "ChargeContainer",
    "Cost",
    "UnbinnedNLL",
    "BinnedNLL",
    "LeastSquares",
    "PDF",
    "PDFParameter",
    "PMTSingleGaussian",
    "SiPMModifiedPoisson",
    "SiPMGentile",
    "minimize_with_iminuit",
    "CameraFitter",
]


__version__ = "1.0.0"
