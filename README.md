# AdvRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/AdvRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/AdvRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/AdvRBMs.jl/branch/master/graph/badge.svg?token=X9C4L2QCHH)](https://codecov.io/gh/cossio/AdvRBMs.jl)

Train and sample [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia, with additional constraints on the weights restricting information about data labels.

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
