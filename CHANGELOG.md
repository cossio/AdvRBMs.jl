# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Require Julia 1.10 or later (was 1.8).
- Raise dependency lower bounds to tested versions: RestrictedBoltzmannMachines 5, FillArrays 1.9, Optimisers 0.3.2.
- Remove unused explicit imports from the module (no behavior change).

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
