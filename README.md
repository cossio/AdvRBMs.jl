# RBMs12Con Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RBMs12Con.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/RBMs12Con.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RBMs12Con.jl/branch/master/graph/badge.svg?token=X9C4L2QCHH)](https://codecov.io/gh/cossio/RBMs12Con.jl)

Train and sample [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (**RBMs**) in Julia, with **1**st and **2**nd order **con**straints on the weights, that promote concentratation of information about labeled features into selected hidden units.

## Installation

This package is not registered.
Install with:

```julia
using Pkg
Pkg.add(url="https://github.com/cossio/RBMs12Con.jl")
```

This package does not export any symbols.

## Related packages

This package builds on top of <https://github.com/cossio/RestrictedBoltzmannMachines.jl> which implements standard Restricted Boltzmann Machines in Julia.
