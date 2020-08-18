from typing import Tuple
import numpy as np

__all__ = ["ChargeContainer"]


class ChargeContainer:
    def __init__(self, values: np.ndarray, n_bins: int, range_: Tuple[float, float]):
        """
        Container to pass the event charges to the Cost class. Internally, this
        class excludes the charges outside the range, and bins the charges -
        ready for binned Cost methods.

        Parameters
        ----------
        values : ndarray
            Array of charges to fit
        n_bins : int
            Number of histogram bins
        range_ : tuple
            Define range of charges to consider
        """
        self.values = values[(values >= range_[0]) & (values <= range_[1])]
        self.hist, self.edges = np.histogram(values, bins=n_bins, range=range_)
        self.between = (self.edges[1:] + self.edges[:-1]) / 2
