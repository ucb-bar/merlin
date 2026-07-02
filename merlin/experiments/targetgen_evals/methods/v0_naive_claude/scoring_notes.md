# Scoring Notes: v0_naive_claude

## Role in the ablation

v0 is the unconstrained baseline. It represents what you get from handing Claude Code
the source snapshot and saying "write the dialect." Any structured method that scores
below v0 is worse than the baseline. Any method that matches v0 at lower cost is more
efficient.

## Expected failure modes

- **Architecture violations**: Unconstrained generation often produces R3 violations
  (TableGen before xDSL), R5 violations (ops without evidence), or R8 violations
  (scheduling policy in semantics).
- **Low evidence coverage**: Without a structured evidence-grounding step, the method
  may assert capabilities not present in the source snapshot.
- **High cost**: Claude Code with no budget constraint will use many tokens exploring
  the source.
- **Human interventions**: May require multiple prompt rounds to fix verifier errors.

## What a good v0 result looks like

- `schema_valid: true`
- `arch_rules_passed >= 6` (R1, R3, R4, R8, R9, R10 are easiest)
- `xdsl_files >= 2` (dialect + lowering pass)
- `pass_tests_pass > 0` (at least one positive test accepted)
- `evidence_coverage >= 0.5`

## What a poor v0 result looks like

- `arch_rules_failed >= 5` (suggests the method produced TableGen or modified Merlin)
- `xdsl_files: 0` (method did not produce xDSL artifacts at all)
- `pass_tests_pass: 0` and `pass_tests_total > 0` (dialect exists but is incorrect)

## Human intervention tracking

Record every prompt round required to fix failures after the initial generation:
- Each diagnostic-feedback-rewrite cycle counts as 1 intervention.
- Minor typo fixes in prompts do not count.
