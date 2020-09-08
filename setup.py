from setuptools import setup, find_packages

setup(
    name="spefit",
    packages=find_packages(),
    version="1.0.0",
    description="Fitting of Single Photoelectron Spectra",
    license="BSD3",
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "tqdm",
        "numba>=0.50.1",
        "iminuit>=1.4.9",
    ],
    python_requires="~=3.6",
    setup_requires=["pytest-runner",],
    tests_require=["pytest",],
    author="Jason J Watson",
    author_email="jason.watson@desy.de",
    url="https://github.com/watsonjj/spefit",
)
