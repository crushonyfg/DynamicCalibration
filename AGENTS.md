# AGENTS.md

## Purpose

This file records project-specific working conventions for the DynamicCalibration repository.
It is meant to keep code changes easy to review, easy to ablate, and easy to roll back.

## Environment

- Use the `jumpGP` conda environment for runtime smoke tests.
- Do not treat the default shell Python as the validation environment for model code.
- Preferred runtime check pattern:
  - `conda run -n jumpGP python -m py_compile ...` for syntax checks.
  - `conda run -n jumpGP python -c "..."` for small targeted smoke tests.
- When a change affects a runner or a new method path, run at least one minimal smoke test in `jumpGP` before closing the task.

## Change style

- Prefer adding a new interface, config flag, or method entry instead of overwriting an old method.
- Preserve old behavior by default. New logic should be opt-in through config or a new runner method name.
- Avoid changing the execution logic of unrelated methods.
- If a new patch fails, the old paths should still run normally.
- When extending an existing implementation, prefer safe fallback behavior over hard failure if the extension cannot initialize.

## Ablation-friendly rules

- Do not rename or silently repurpose an existing method if a new variant is being tested.
- Add new variants with explicit names such as:
  - `..._STDGate`
  - `..._particleGP`
  - `..._particleBasis`
- Keep runner wiring explicit so ablation tables map directly to code paths.
- When a new modeling option is added, expose it through runner metadata instead of hard-coding it inside the BOCPD implementation.

## Validation expectations

- For pure code-structure changes, run `py_compile` on all touched Python files.
- For modeling-path changes, also run one small runtime smoke test in `jumpGP`.
- If a change adds a new discrepancy mode or restart mode, test at least one representative synthetic path that actually enters the modified branch.
- When a runtime bug is fixed, prefer reproducing it with a minimal command and then rerunning the same command after the patch.

## Documentation expectations

- Keep `rolled_cusum_modeling_workdoc.md` up to date whenever you change:
  - `restart_bocpd_rolled_cusum_260324_gpytorch.py`
  - `particle_specific_discrepancy.py`
  - `likelihood.py` discrepancy-related interfaces
  - runner method tables that expose new rolled-CUSUM variants
- Update the workdoc when any of the following change:
  - available method names
  - config switches
  - discrepancy parameterization
  - restart / refresh semantics
  - mathematical interpretation of the predictive law
- Prefer documenting the math object and the code hook together, so the implementation remains readable to humans.

## Modeling guardrails

- PF weighting is discrepancy-free unless a future change explicitly introduces and documents a different design.
- BOCPD-side discrepancy changes should be described as predictive-law or memory-refresh changes, not as PF weighting changes, unless that is truly what the code does.
- If a change needs a stronger theoretical reinterpretation, document the exact formula in the workdoc before or together with the code change.

## Suggested workflow for future edits

1. Inspect the current runner entry points and identify the exact ablation branch to add.
2. Add the new code path behind a fresh config switch or method name.
3. Keep backward compatibility in shared interfaces where possible.
4. Update `rolled_cusum_modeling_workdoc.md` with the new option and formula.
5. Run `py_compile`.
6. Run a minimal `jumpGP` smoke test.
7. Only then report the change as complete.

## Optional future maintenance

- If the project accumulates more specialized workflows, it may be worth adding small local Codex skills for:
  - runner smoke-test recipes
  - rolled-CUSUM interface lookup
  - discrepancy-model extension checklist
- Until then, this `AGENTS.md` plus `rolled_cusum_modeling_workdoc.md` should serve as the main maintenance guide.


## Runner Documentation Rule

- Whenever a new `run_*` script, runner profile, or experiment entrypoint is added, document its command-line usage in `rolled_cusum_modeling_workdoc.md`.
- At minimum, document:
  - the exact `conda run -n jumpGP python -m ...` command,
  - required and optional CLI flags,
  - supported `--profile` values if any,
  - which outputs are expected to be written.
- If a runner has multiple intended invocation modes such as `main`, `ablation`, or `appendix`, list each mode explicitly with one example command.
- Do not leave a new runner discoverable only from code; the invocation path should be written down in repo docs the same turn the runner is introduced or changed.
