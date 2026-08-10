# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- BREAKING: Support RestrictedBoltzmannMachines 7.1, which is now required (was 3.8, 4, 5). All RestrictedBoltzmannMachines symbols used by this package are `public` as of that release, so this package no longer depends on upstream internals. Following the upstream change, `wts = nothing` is no longer accepted: the `wts` keyword of `advpcd!`, `calc_q`, `calc_Q`, `calc_qs`, and `calc_Qs` must be an `AbstractVector{<:Real}` and defaults to lazy uniform weights, so the unweighted and weighted cases share a single code path. As upstream, `advpcd!` validates weights once per training run (finite, positive, matching the number of samples, with at least one sample), uses them exactly as given, and for `CenteredRBM` now also weights the initial centering. Unweighted callbacks receive lazy uniform weight slices as `wd` (previously `nothing`).
- A `batchsize` larger than the number of samples now clamps to the number of samples in both `advpcd!` methods (matching the upstream trainers), instead of silently performing zero iterations.
- Fix the adversarial constraints in `advpcd!` for `StandardizedRBM`: the 1st-order hard projection and the 2nd-order penalty were applied to the standardized weights, ignoring the standardization scales. Constraints computed from data (`calc_q`/`calc_Q`) are now rescaled internally by `scale_v` (and the 2nd-order penalty additionally by the current `scale_h`), so that inputs to constrained hidden units carry no 1st-order label information, as intended. The hard projection now runs before the zerosum gauge fix, so for Potts visible layers with zerosum constraints the gauge and the constraint hold simultaneously; the projection is additionally re-imposed after the gauge fix, so the constraint always holds at exit even for Potts hidden layers whose sites are split across constraint groups.
- Fix `calc_qs` and `calc_Qs` dropping the trailing constraint dimension of each entry, which made their outputs unusable as the `qs`/`Qs` keyword arguments of `advpcd!` (an error for vector visible layers and for the 2nd-order penalty, and a silently over-constrained projection for multi-dimensional visible layers). `calc_Qs(T, u, v)` now also honors the requested element type `T`.
- `advpcd!` for `CenteredRBM` now updates the hidden offsets by estimating `<h>` from the minibatch via `center_hidden_from_data!` and its `damping` keyword, like the upstream centered trainer, instead of reading it off the free-energy gradient with `grad2ave` (removed upstream).
- Fix the `advpcd!` docstrings, which documented nonexistent `q` and `Q` keyword arguments; they now describe the actual `qs`, `Qs`, and `ℋs` keywords.
- Require Julia 1.12 or later (was 1.8).
- Raise dependency lower bounds to tested versions: FillArrays 1.9, Optimisers 0.4.
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
