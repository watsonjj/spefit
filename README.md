# spefit 

![tests](https://github.com/watsonjj/spefit/workflows/test_dev/badge.svg) [![codecov](https://codecov.io/gh/watsonjj/spefit/branch/master/graph/badge.svg)](https://codecov.io/gh/watsonjj/spefit) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/43250a5b5ee54103be45d26de93bdca1)](https://www.codacy.com/manual/watsonjj/spefit?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=watsonjj/spefit&amp;utm_campaign=Badge_Grade) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> ![PyPI](https://img.shields.io/pypi/v/spefit) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/watsonjj/spefit/master?filepath=tutorials)

Optimised framework for the fitting of [Single Photoelectron Spectra](https://github.com/watsonjj/spefit/wiki/Single-Photoelectron-spectra) (SPE) in order to characterize the properties of photomultipliers which influence the measured illumination response.

* Supported Python versions: 3.6+
* Supported platforms: Linux, OSX
* Source: <https://github.com/watsonjj/spefit>
* License: [BSD-3-Clause](LICENSE)
* Citation: _pending_

## Package Features

* Basic [numpy](https://numpy.org/) API
* Runtime-selectable Probability Density Functions (PDFs), optimised using [numba](http://numba.pydata.org/)
* PDFs for the SPE spectra of both Photomultiplier Tubes and Silicon Photomultipliers
* Configuration of PDFs for the case where no pedestal peak exists (e.g. dark counting)
* Estimation of SPE parameters for improved initial fit values
* Runtime-selectable minimization cost definitions, optimised using numba
* Simultaneous fitting of multiple datasets (e.g. containing different average illuminations) for better parameter constraining
* Minimization provided by [iminuit](https://github.com/scikit-hep/iminuit) - Python frontend to the MINUIT2 C++ library
* Calculation of parameter errors and resulting p-value
* Extendable to allow the inclusion of any additional SPE description and minimization cost definitions
* Compatible with other minimization routines
* Convenience class provided for the parallel processing of cameras containing multiple photomultiplier pixels

## Installation

`pip install spefit`

## Usage

With a numpy array of size (n_events) called `charge_array`, containing the measured charges from the low illumination of a photomultiplier, the parameters of the SPE spectra can be extracted with:

```python
from spefit import ChargeContainer, PMTSingleGaussian, BinnedNLL, minimize_with_iminuit

charges = [ChargeContainer(charge_array, n_bins=100, range_=(-3, 6))]
pdf = PMTSingleGaussian(n_illuminations=1)
cost = BinnedNLL(pdf, charges)
values, errors = minimize_with_iminuit(cost)
```

Jupyter notebook demonstrations are provided in [tutorials](tutorials).
