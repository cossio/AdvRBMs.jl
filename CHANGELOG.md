# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Fix the adversarial constraints in `advpcd!` for `StandardizedRBM`: the 1st-order hard projection and the 2nd-order penalty were applied to the standardized weights, ignoring the visible standardization scales. Constraints computed from data (`calc_q`/`calc_Q`) are now rescaled by `scale_v` internally, so that inputs to constrained hidden units carry no 1st-order label information, as intended.
- Require Julia 1.12 or later (was 1.8).
- Raise dependency lower bounds to tested versions: RestrictedBoltzmannMachines 5.6, FillArrays 1.9, Optimisers 0.4.

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
