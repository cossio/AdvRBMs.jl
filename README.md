# AdvRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/AdvRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/AdvRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/AdvRBMs.jl/branch/master/graph/badge.svg?token=X9C4L2QCHH)](https://codecov.io/gh/cossio/AdvRBMs.jl)

Train and sample [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (**RBMs**) in Julia, with 1st and 2nd order **adv**ersarial constraints on the weights, that promote concentratation of information about labeled features into selected hidden units.

## Installation

This package is not registered.
Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/AdvRBMs.jl")
```

This package does not export any symbols.

## Related packages

This package builds on top of <https://github.com/cossio/RestrictedBoltzmannMachines.jl> which implements standard Restricted Boltzmann Machines in Julia.

Also: https://github.com/cossio/AdvRBMs_App.jl.git

# Paper

This repository hosts code accompanying the paper: https://arxiv.org/abs/2206.11600. If you use this code in a publication, please cite:

* Fernandez-de-Cossio-Diaz, Jorge, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." arXiv preprint arXiv:2206.11600 (2022)
