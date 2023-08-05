# AdvRBMs Julia package

Train and sample [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (**RBMs**) in Julia, with 1st and 2nd order **adv**ersarial constraints on the weights, that promote concentratation of information about labeled features into selected hidden units.

## Installation

This package is registered. Install with:

```julia
using Pkg
Pkg.add("AdvRBMs")
```

This package does not export any symbols.

## Related packages

Implementation of standard Restricted Boltzmann Machines in Julia:

- https://github.com/cossio/RestrictedBoltzmannMachines.jl

Ising model simulations were carried out using [IsingModels.jl](https://github.com/cossio/IsingModels.jl).

# Citation

If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." [Physical Review X 13, 021003 (2023)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021003).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).