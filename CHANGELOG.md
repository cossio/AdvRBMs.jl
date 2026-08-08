# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Fix the adversarial constraints in `advpcd!` for `StandardizedRBM`: the 1st-order hard projection and the 2nd-order penalty were applied to the standardized weights, ignoring the standardization scales. Constraints computed from data (`calc_q`/`calc_Q`) are now rescaled internally by `scale_v` (and the 2nd-order penalty additionally by the current `scale_h`), so that inputs to constrained hidden units carry no 1st-order label information, as intended. The hard projection now runs before the zerosum gauge fix, so for Potts visible layers with zerosum constraints the gauge and the constraint hold simultaneously; the projection is additionally re-imposed after the gauge fix, so the constraint always holds at exit even for Potts hidden layers whose sites are split across constraint groups.
- Fix `calc_qs` and `calc_Qs` dropping the trailing constraint dimension of each entry, which made their outputs unusable as the `qs`/`Qs` keyword arguments of `advpcd!` (an error for vector visible layers and for the 2nd-order penalty, and a silently over-constrained projection for multi-dimensional visible layers). `calc_Qs(T, u, v)` now also honors the requested element type `T`.
- Require Julia 1.12 or later (was 1.8).
- Raise dependency lower bounds to tested versions: RestrictedBoltzmannMachines 5.6, FillArrays 1.9, Optimisers 0.4.
- Move the unused internal helpers `sylvester_projection` and `project∂!` out of the module, into `src/sylvester.jl` (not included).

## v2.1.0

- Add support for RestrictedBoltzmannMachines v4, v5.

## v2.0.0

- BREAKING: Use `StandardizedRBM` and `CenteredRBM` from RestrictedBoltzmannMachines.jl package.

## v1.1.0

- Support StandardizedRestrictedBoltzmannMachines.jl package.

## v1.0.1

- Some compat updates.

## v1.0.0

- Release v1.0.0.
- Support `CenteredRBM` (from https://github.com/cossio/CenteredRBMs.jl).

## v0.4.0

- Support RBMs.jl v1, 2.
