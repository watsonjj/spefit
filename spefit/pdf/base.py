from copy import copy
from typing import Callable, Tuple, Dict, List
import numpy as np

__all__ = ["PDFParameter", "PDF"]


class PDFParameter:
    def __init__(
        self,
        initial: float,
        limits: Tuple[float, float],
        fixed: bool = False,
        multi: bool = False,
    ):
        """Parameter of the PDF

        TODO: Convert to a dataclass when 3.7 is adopted as minimum Python version

        Parameters
        ----------
        initial : float
            Starting value for the parameter during the fit
        limits : Tuple[float, float]
            Range of allowed values for the paramter during the fit
        fixed : bool
            If True, the parameter is fixed to its initial value and will not
            be optimised in the fit
        multi : bool
            Indicates if the parameter has a unique value per illumination.
            Utilised for in simultaneous multi-illumination fits
        """
        self.initial = initial
        self.limits = limits
        self.fixed = fixed
        self._multi = multi

    @property
    def initial(self) -> float:
        return self._initial

    @initial.setter
    def initial(self, value: float):
        self._initial = value

    @property
    def limits(self) -> Tuple[float, float]:
        return self._limits

    @limits.setter
    def limits(self, value: Tuple[float, float]):
        self._limits = value

    @property
    def fixed(self) -> bool:
        return self._fixed

    @fixed.setter
    def fixed(self, value: bool):
        self._fixed = value

    @property
    def multi(self) -> bool:
        return self._multi


class PDF:
    def __init__(
        self,
        n_illuminations: int,
        function: Callable = None,
        parameters: Dict[str, PDFParameter] = None,
    ):
        """Defines a function to be fit, and its corresponding parameters

        Parameters
        ----------
        function : Callable
            Function defining the SPE spectra PDF relevant for the data to be fit
        parameters : dict
            Dict of ``PDFParameter``, with entries corresponding to the
            arguments of the pdf function
        n_illuminations : int
            Number of illuminations to fit simultaneously
        """
        if function is None or parameters is None:
            raise ValueError("PDF class must define the function and parameters")

        self._n_illuminations = n_illuminations
        self._function = function
        result = self._prepare_parameters(parameters, n_illuminations)
        self._parameters, self._parameter_is_multi, self._lookup = result

    def __call__(self, x: np.ndarray, parameters: np.ndarray, i_illumination: int):
        """Evaluates the PDF of the fit function for a particular illumination

        Parameters
        ----------
        x : ndarray
            Values to evaluate the fit function at
        parameters : ndarray
            Array of the parameter values for the fit function (all illuminations)
            Must be ordered according to the `self._parameters`
        i_illumination : int
            Illumination index to evaluate the fit function for

        Returns
        -------
        ndarray
        """
        return self._function(x, *self._lookup_parameters(parameters, i_illumination))

    def _lookup_parameters(self, parameters: np.ndarray, i_illumination: int):
        """Extract the correct parameters from the array that correspond to the
        current illumination
        """
        return parameters[self._lookup[i_illumination]]

    def _update_parameter(self, attribute, name, value):
        if name not in self._parameters:
            if self._parameter_is_multi.get(name, False):
                for i_illumination in range(self._n_illuminations):
                    multi_name = f"{name}{i_illumination}"
                    setattr(self._parameters[multi_name], attribute, value)
            else:
                raise ValueError(f"No parameter named {name} in PDF")
        else:
            setattr(self._parameters[name], attribute, value)

    def update_parameters_initial(self, **parameter_initial: float):
        for name, value in parameter_initial.items():
            self._update_parameter("initial", name, value)

    def update_parameters_limits(self, **parameter_limits: Tuple[float, float]):
        for name, value in parameter_limits.items():
            self._update_parameter("limits", name, tuple(value))

    def update_parameters_fixed(self, **parameter_fixed: bool):
        for name, value in parameter_fixed.items():
            self._update_parameter("fixed", name, value)

    @staticmethod
    def _prepare_parameters(parameters, n_illuminations):
        parameter_dict = {}
        parameter_is_multi = {}
        lookup = np.zeros((n_illuminations, len(parameters)), dtype=np.int)
        i_lookup = 0
        for i_param, (name, param) in enumerate(parameters.items()):
            if param.multi:
                parameter_is_multi[name] = True
                for i_illumination in range(n_illuminations):
                    lookup[i_illumination, i_param] = i_lookup
                    parameter_dict[f"{name}{i_illumination}"] = copy(param)
                    i_lookup += 1
            else:
                parameter_is_multi[name] = False
                for i_illumination in range(n_illuminations):
                    lookup[i_illumination, i_param] = i_lookup
                parameter_dict[name] = copy(param)
                i_lookup += 1
        return parameter_dict, parameter_is_multi, lookup

    @property
    def function(self) -> Callable:
        """Provides access to internal PDF function"""
        return self._function

    @property
    def n_illuminations(self) -> int:
        return self._n_illuminations

    @property
    def parameters(self) -> Dict[str, PDFParameter]:
        return self._parameters

    @property
    def parameter_names(self) -> List[str]:
        return list(self._parameters.keys())

    @property
    def initial(self) -> Dict[str, float]:
        return {n: p.initial for n, p in self.parameters.items()}

    @property
    def n_free_parameters(self) -> int:
        return sum([not p.fixed for p in self.parameters.values()])

    @property
    def iminuit_kwargs(self) -> Dict[str, float or Tuple[float, float]]:
        kwargs = {}
        for name, param in self._parameters.items():
            kwargs[name] = param.initial
            kwargs["limit_" + name] = param.limits
            kwargs["fix_" + name] = param.fixed
        return kwargs

    @classmethod
    def from_name(cls, name: str, *args, **kwargs):
        """Factory method to obtain subclass by name"""
        for subclass in cls.__subclasses__():
            if subclass.__name__ == name:
                return subclass(*args, **kwargs)
        raise ValueError(f"No PDF class with the name: {name}")
