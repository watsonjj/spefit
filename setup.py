from setuptools import setup, find_packages

PACKAGENAME = "spefit"
DESCRIPTION = "Fitting of Single Photoelectron Spectra"
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@desy.de"
VERSION = "1.0.0"

setup(
    name=PACKAGENAME,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    license='BSD3',
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'tqdm',
        'numba',
        'iminuit',
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL
)
