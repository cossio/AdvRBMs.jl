# Repository instructions

This file provides guidance to coding agents working in this repository.

## Project overview

AdvRBMs.jl is a Julia package for training Restricted Boltzmann Machines
(RBMs) with 1st- and 2nd-order adversarial constraints on the weights. The
constraints promote concentration of information about labeled features
into selected hidden units (see CITATION.bib). It builds on
RestrictedBoltzmannMachines.jl and supports Julia 1.8 and later.

## Repository workflow

- Run commands from the repository root. Use `--project=.` for the package
  and `--project=test` for standalone test files.
- Run the narrowest relevant test file first, then
  `julia --project=. -e 'using Pkg; Pkg.test()'` when the change crosses
  subsystems or affects public behavior.
- Load the package with `julia --project=. -e 'import AdvRBMs'`.

## Workspace and environment

- The root `Manifest.toml` is committed. Keep it in sync when Project.toml
  changes and commit the resolved result.
- The test project needs the root package developed into it for standalone
  runs. If that setup is missing, run
  `julia --project=test -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'`.
- No external services or GPU hardware are required; CI runs the CPU test
  suite on Linux (`.github/workflows/ci.yml`).
- Tests in `test/runtests.jl` are organized as independent modules. Test
  files run standalone with `--project=test`; the suite includes Aqua
  quality checks and gradient checks with Zygote.

## Package architecture and invariants

- The package exports no symbols. Prefer `import AdvRBMs` and qualified
  names, or explicit `using AdvRBMs: ...`. Internally the code imports
  `RestrictedBoltzmannMachines as RBMs`.
- Array conventions follow RestrictedBoltzmannMachines.jl: layer dimensions
  come first and the trailing dimension indexes samples (batch). `rbm.w`
  has shape `(size(visible)..., size(hidden)...)`. Preserve this convention
  rather than assuming vector layers or matrix weights.
- `calc_q`/`calc_qs` in `src/calc_qQ.jl` compute the 1st-order statistics
  `q` correlating visible units with labels `u`; `calc_Q`/`calc_Qs` compute
  the 2nd-order tensors `Q`. Labels are `Bool` vectors (binary) or one-hot
  `Bool` matrices (categorical) with samples along the last dimension, and
  all statistics accept optional sample weights `wts`.
- `advpcd!` trains an RBM with persistent contrastive divergence under the
  constraints. The 1st-order constraint (`q' * w ≈ 0` on constrained hidden
  units) is hard: weights and weight gradients are projected onto the
  kernel of `q` with `kernelproj` (`src/proj.jl`). The 2nd-order constraint
  (`w' * Q * w ≈ 0`) is a soft penalty with strength `λQ` via `∂wQw`. Any
  weight update on constrained units must preserve the hard constraint.
- Constraints are lists `qs`, `Qs` applied to disjoint groups of hidden
  units `ℋs` (given as `CartesianIndices`); `default_qs`, `default_Qs`, and
  `default_ℋs` in `src/default_q.jl` provide the unconstrained defaults.
- `src/advpcd.jl` handles `RBM` and `CenteredRBM`; `src/advpcd_std.jl`
  implements the analogous `advpcd!` for `StandardizedRBM`. Keep the two
  methods consistent when changing training behavior.
- Prefer broadcast and linear-algebra formulations that also work for GPU
  arrays; `src/proj.jl` documents CUDA-friendly parenthesizations. Avoid
  scalar-indexing loops over parameter-sized arrays.
- `src/wip/` holds work-in-progress code not included in the module.

## GitHub operations

- A network-restricted sandbox can make `gh auth status` look like an
  invalid token. If it fails in the sandbox, retry it with host/network
  access before asking the user to reauthenticate; treat credentials as
  invalid only if that host-level check also fails.

## Changes and pull requests

- Add CHANGELOG.md entries only for user-facing package changes to source,
  APIs, behavior, or dependencies. Do not add entries for CI, workflows,
  agent plumbing, or other repository tooling.
- PRs receive automated review comments from the Claude Code review
  workflow in `.github/workflows/claude-code-review.yml`. Address each
  actionable finding or explain the disagreement in its thread, reply to
  every thread, and resolve it once addressed.
- Follow REVIEW.md; flag substantial avoidable complexity only when a
  materially simpler design satisfies the current requirements.
- Never merge a PR or enable auto-merge unless the repository owner
  explicitly instructs it.

## Releases

- During development, the version in Project.toml carries a `-DEV` suffix
  (for example, `2.1.1-DEV`), and changes accumulate under `## Unreleased`
  in CHANGELOG.md.
- Choose release numbers using ColPrac's Julia package SemVer guidance. For
  this post-1.0 package, breaking changes bump major, non-breaking features
  bump minor, and bug fixes bump patch. Suggest one version with a brief
  explanation, but always leave the final decision to the user and wait for
  explicit confirmation before making release changes.
- Use `$register-new-version` for release, registration, tagging, or
  publishing tasks. The shared workflow lives at
  `.claude/skills/register-new-version/SKILL.md` and is exposed to Codex at
  `.agents/skills/register-new-version/SKILL.md`. It covers the release
  commit, triggering Registrator directly on that commit, and monitoring
  the General registry PR.
