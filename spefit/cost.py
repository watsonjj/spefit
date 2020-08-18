"""Common minimization cost functions altered to handle n_illuminations."""
from spefit.pdf.base import PDF
from spefit.common.basic import xlogy
from spefit.container import ChargeContainer
from typing import List
from numba import njit, vectorize, float64
import numpy as np
from abc import abstractmethod, ABCMeta
from scipy.stats import distributions

__all__ = [
    "_sum_log_x",
    "_bin_nll",
    "_total_binned_nll",
    "_least_squares",
    "Cost",
    "UnbinnedNLL",
    "BinnedNLL",
    "LeastSquares",
]


@njit(fastmath=True)
def _sum_log_x(x):
    return np.sum(np.log(x))


@vectorize([float64(float64, float64)], fastmath=True)
def _bin_nll(f_y, d_y):
    """Negative log-likelihood for N counts in a charge bin given the expected
    value, assuming the bin counts are Poisson distributed.
    Formulated with the likelihood ratio (such that the minimum of the NLL * 2
    is distributed according to the χ2 distribution (with (n − m) degrees of
    freedom).

    Parameters
    ----------
    f_y : ndarray or float
        Expected bin counts from the PDF
    d_y : ndarray or float
        Measured bin counts

    Returns
    -------
    ndarray or float
    """
    return f_y if d_y == 0 else f_y - d_y - xlogy(d_y, f_y / d_y)


@njit(fastmath=True)
def _total_binned_nll(f_y, d_y):
    """Sum over the likelihood of all bins, after scaling the PDF result to the
    expected number of counts in the bins.

    Parameters
    ----------
    f_y : ndarray
        Expected bin counts from the PDF
    d_y : ndarray
        Measured bin counts

    Returns
    -------
    float
    """
    scale = np.sum(d_y) / np.sum(f_y)
    return np.sum(_bin_nll(f_y * scale, d_y))


@njit(fastmath=True)
def _least_squares(f_y, d_y):
    """Extract the least squared chi2, after scaling the PDF result to the
    expected number of counts in the bins.
    Require more than 5 counts per bin, in order to enable the Gaussian
    approximation of the Poisson distribution.

    Parameters
    ----------
    f_y : ndarray
        Expected bin counts from the PDF
    d_y : ndarray
        Measured bin counts

    Returns
    -------
    float
    """
    scale = np.sum(d_y) / np.sum(f_y)
    f_ys = f_y * scale
    gt5 = d_y > 5
    return np.sum((d_y[gt5] - f_ys[gt5]) ** 2 / d_y[gt5])


class Cost(metaclass=ABCMeta):
    errordef = None

    def __init__(self, pdf: PDF, charges: List[ChargeContainer]):
        """Cost definition for minimization, to find the parameter values of
        the assumed PDF that is most likely to have produced the observed
        charge distribution.

        Errordef defines the increment above the minimum that corresponds to
        one standard deviation (as per the iminuit definition). Errordef should
        be 1.0 for a least-squares cost function and 0.5 for negative
        log-likelihood function.

        Parameters
        ----------
        pdf : PDF
            PDF class to use in the fit
        charges : List[ChargeContainer]
            List of ChargeContainers with length n_illuminations
        """
        self._pdf = pdf
        self._n_free_parameters = self._pdf.n_free_parameters
        self._charges = charges
        self._n_illuminations = len(charges)
        if self._n_illuminations != pdf.n_illuminations:
            raise ValueError("Charges must be a list of length n_illuminations")

    @abstractmethod
    def __call__(self, parameters: np.ndarray) -> float:
        """Evaluate the cost function for a particular set of parameter values

        Parameters
        ----------
        parameters : ndarray
            Array of the parameter values for the fit function (all illuminations)
            Must be ordered according to the `pdf._parameters`

        Returns
        -------
        float
        """

    @property
    def iminuit_kwargs(self):
        return self._pdf.iminuit_kwargs

    @property
    def parameter_names(self):
        return self._pdf.parameter_names

    @property
    @abstractmethod
    def dof(self):
        pass

    @abstractmethod
    def chi2(self, parameters: np.ndarray):
        pass

    def reduced_chi2(self, parameters: np.ndarray):
        return self.chi2(parameters) / self.dof

    def p_value(self, parameters: np.ndarray):
        return distributions.chi2.sf(self.chi2(parameters), self.dof)

    @classmethod
    def from_name(cls, name: str, *args, **kwargs):
        """Factory method to obtain subclass by name
        """
        for subclass in cls.__subclasses__():
            if subclass.__name__ == name:
                return subclass(*args, **kwargs)
        raise ValueError(f"No Cost class with the name: {name}")


class UnbinnedNLL(Cost):
    """Unbinned negative log-likelihood. Slower than the BinnedNLL, but no
    features are lost due to the binning.
    """

    errordef = 0.5

    def __call__(self, parameters):
        likelihood = 0
        for i in range(self._n_illuminations):
            f = self._pdf(self._charges[i].values, parameters, i)
            likelihood += -_sum_log_x(f)
        return likelihood

    @property
    def dof(self):
        n = sum([d.values.size for d in self._charges])
        m = self._n_free_parameters
        return n - m

    def chi2(self, parameters):
        """J. Heinrich, PHYSTAT2003, arXiv:physics/0310167
        "The method is fatally flawed in the unbinned case. Don’t use it.
        Complain when you see it used."
        """
        raise ValueError("Chi2 is not defined for UnbinnedNLL")


class BinnedNLL(Cost):
    """Binned negative log-likelihood, formulated with the likelihood ratio"""
    errordef = 0.5

    def __call__(self, parameters):
        likelihood = 0
        for i in range(self._n_illuminations):
            f_y = self._pdf(self._charges[i].between, parameters, i)
            likelihood += _total_binned_nll(f_y, self._charges[i].hist)
        return likelihood

    @property
    def dof(self):
        n = sum([d.hist.size for d in self._charges])
        m = self._n_free_parameters
        return n - m

    def chi2(self, parameters: np.ndarray):
        return self(parameters) * 2


class LeastSquares(Cost):
    """Least squares/chi-square (for Poisson-distributed bin counts)"""
    errordef = 1

    def __call__(self, parameters):
        chi2 = 0
        for i in range(self._n_illuminations):
            f_y = self._pdf(self._charges[i].between, parameters, i)
            chi2 += _least_squares(f_y, self._charges[i].hist)
        return chi2

    @property
    def dof(self):
        n = sum([(d.hist > 5).sum() for d in self._charges])
        m = self._n_free_parameters
        return n - m

    def chi2(self, parameters):
        return self(parameters)
