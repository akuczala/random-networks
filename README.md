# random-networks

Python libraries developed for research in recurrent neural networks

## Random matrix spectra
Includes libraries for generating N x N random matrices with structured correlations, and their spectra for large N, as described in https://arxiv.org/abs/1610.09353. Includes a Jupyter notebook reproducing the figures.

genRandom.py generates finite N realizations from the matrix distributions described in the paper.

genSpec.py produces the Green's functions and mean eigenvalue densities for large random matrices using the semi-analytical approach described in the paper.

The data folder contains precomputed eigenvalues for various matrix distributions. These are used to approximate eigenvalue densities for comparison with theory.

## Information in recurrent networks
fisherlib.py computes the Fisher information (signal-to-noise ratio) of coupled random recurrent networks. An example notebook is included.

## Stochastic network dynamics
Dynamics.jl simulates random symmetric recurrent networks using the [DifferentialEquations](https://github.com/JuliaDiffEq/DifferentialEquations.jl) package. It also provides functions that uniformly sample from the [Gaussian orthogonal ensemble](https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles). An example notebook is included.
