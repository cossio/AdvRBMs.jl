# Review instructions

This is a Julia package for training Restricted Boltzmann Machines with
adversarial constraints on the weights. Prioritize numerical and
statistical correctness above everything else: a subtly wrong formula is
worse than a crash, because tests may not catch it and users get silently
wrong results.

Report real problems only. Do not comment on style, formatting, or
preferences; do not restate what the diff does. If you find nothing
significant, post no findings.

## What Important means here

Reserve Important (blocking-severity) for findings that would make the
package compute wrong results or break users:

- Bugs and logic errors: wrong edge-case behavior, or crashes on valid
  input.
- Incorrect math or statistics: wrong sign, missing normalization or
  centering, mishandled sample weights `wts`, mishandled log/exp
  (overflow where a logsumexp/`log1p` formulation is needed), biased
  sampling.
- Violations of the constraint semantics: the 1st-order constraint is
  hard — weights and weight gradients of constrained hidden units must be
  projected with `kernelproj` so `q' * w ≈ 0` survives every update; the
  2nd-order constraint is a soft `λQ` penalty through `∂wQw`; the
  hidden-unit groups `ℋs` must stay disjoint.
- Violations of the array dimension convention (inherited from
  RestrictedBoltzmannMachines.jl): layer dimensions first, batch dimension
  last; weights `w` shaped `(size(visible)..., size(hidden)...)`. Code
  that only works for 1-dimensional layers, a single batch, or non-Potts
  layers — while claiming to be general — is a bug, not a nit.
- Divergence between the `RBM`/`CenteredRBM` training loop in
  `src/advpcd.jl` and the `StandardizedRBM` one in `src/advpcd_std.jl`
  when a change should apply to both.
- Code that breaks generic array backends: scalar indexing into
  parameter-sized arrays, constructing `Array`/`Vector` where `similar` or
  broadcasting is needed, or GPU-hostile linear-algebra formulations
  (`src/proj.jl` documents CUDA-friendly parenthesizations).
- Breaking changes to public API or behavior not reflected in tests and in
  the `## Unreleased` section of CHANGELOG.md.
- Security issues in changes to CI workflows.
- Substantial avoidable complexity when a materially simpler design
  satisfies the current requirements. Identify the unnecessary structure
  and a concrete, behavior-preserving alternative; do not block on vague
  preferences.

Style, naming, refactoring suggestions, and docstring wording are Nit at
most.

## Verification bar

A claim that math is wrong needs a short derivation or a citation of the
correct form elsewhere in this codebase (`file:line`) — not an inference
from function names. A claim that code breaks on GPU arrays or on
higher-dimensional layers needs the concrete failing call shape, not a
suspicion.

## Cap the nits

Report at most five Nits per review. If you found more, say "plus N similar
items" in the summary instead of posting them inline. If everything you
found is a Nit, lead the summary with "No blocking issues."

## Do not report

- Anything CI already enforces: the test suite
  (`.github/workflows/ci.yml`) and the agent-docs linter
  (`.github/scripts/lint_agent_docs.py`).
- Formatting and code style preferences.
- Manifest files: they are gitignored and resolved fresh from Project.toml.
- Missing CHANGELOG.md entries for CI, workflow, or repo-tooling changes —
  the changelog records only user-facing package changes.

## Always check

- Changes to training behavior keep `advpcd!` in `src/advpcd.jl` and
  `src/advpcd_std.jl` consistent.
- New tests are added as independent modules in `test/runtests.jl` and can
  run standalone with `--project=test`.
- New tests do not require a physical GPU (GitHub CI has none).
- Changes to public functions keep docstrings and tests in sync with the
  new behavior.
- The version in Project.toml keeps its `-DEV` suffix outside of release
  PRs.

## Agent-instruction files

When the diff touches CLAUDE.md, AGENTS.md, REVIEW.md, or anything under
`.agents/` or `.claude/`, also review those files for: contradictions with
each other or with the actual repository (spot-check commands, paths, and
factual claims against the code); substantial redundancy that can drift
apart; context bloat (content that does not earn its place); and skill
frontmatter descriptions that fail to say what the skill does and when to
use it. Consult the best-practice guides with WebFetch if helpful:
https://code.claude.com/docs/en/best-practices and
https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
The deterministic linter (`.github/scripts/lint_agent_docs.py`) already
enforces sizes, frontmatter constraints, and path existence — focus on
semantics.

## Re-reviews

After the first review of a PR, suppress new Nits and post Important
findings only, so follow-up pushes converge instead of accumulating style
rounds.
