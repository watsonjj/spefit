from spefit.container import ChargeContainer
import numpy as np


def test_charge_container():
    rng = np.random.default_rng(seed=0)
    samples = rng.uniform(0, 100, 10000)
    bins = 100
    range_ = (40, 60)
    charge = ChargeContainer(samples, n_bins=bins, range_=range_)
    assert (charge.values >= range_[0]).all()
    assert (charge.values <= range_[1]).all()
    assert charge.hist.size == bins
