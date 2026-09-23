---
title: Generating capsules for a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-09-21
related: [adding_a_target, gemmini_experiment, capsule_bench, integrations]
code_refs:
  - experiments/catalog.yaml
  - packages/merlin-experiments/src/merlin_experiments/phase0/__main__.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/generation.py
  - packages/merlin-experiments/src/merlin_experiments/corpus/preparation.py
---

# Generating capsules for a target

Phase 0 derives tests; it does not certify a compiler. Start with an experiment in
[the catalog](../../experiments/catalog.yaml), not a script in a legacy corpus directory.
Install Merlin and the optional `merlin-experiments` distribution. Provision selected
OOT support, RTL facts and capture/toolchain dependencies explicitly; an example recipe
alone does not make them available. See [integrations](integrations.md).

## Find the inputs

The [examples index](../../examples/README.md) maps targets to definitions and phase inputs.
[Gemmini's Phase 0 example](../../examples/gemmini/phase0/README.md) is one concrete starting point.

| Input or output | Owner |
| --- | --- |
| Public target recipe | `examples/<target>/phase0/recipe.yaml` |
| Target descriptor and input selection | The example's `target/` directory and `experiment.yaml` |
| Shared performance-family policy | `experiments/templates/phase0/performance.yaml` |
| Synthesis/SMT profiles | Explicit definition inputs; retained locations vary during migration |
| Hidden profiles, holdouts and answers | Host-private inputs, never public examples or candidate grants |
| Generated capsules and generation receipts | `<run-dir>/phase0/capsules/` |
| Prepared grading release and review evidence | A fresh operator-selected artifact directory |

The installed generator lives in `merlin_experiments.phase0`; shared derivation primitives
remain in core. There is no need to copy generation scripts into `out/`. Generated capsules
are artifacts, not new library code or files to sync into a wheel.

## Inspect, preflight, then generate

These commands use catalog ID `gemmini-functional` as an example. Choose your definition
from the catalog and replace `/configured/out` with your configured output root. Use a
fresh run directory; retain earlier runs for inspection.

```sh
merlin experiment inspect gemmini-functional --phase 0
merlin experiment preflight gemmini-functional --phase 0
merlin experiment run gemmini-functional --phase 0 \
  --run-dir /configured/out/runs/gemmini/phase0/example-1
```

Only `run` starts generation. Inspect/preflight are not proof that framework capture,
numerical oracles or hardware will work. The definition selects the descriptor, public
recipe, shared performance template and optional profiles. Explicit recipe mode does not
discover sibling profiles; frozen runs bind optional input absence as well as present bytes.
The adapter supplies the output destination, never the descriptor's source corpus.

For standalone invocation, `python -m merlin_experiments.phase0 --help` describes the
installed generator's explicit inputs, including required `--output-root`. Do not use the
legacy native command to regenerate every target.

Generation can retain completed members when another member fails; rejected synthesized
members can also be removed. Read the logs and receipts, not merely the directory count.
Missing facts, skipped builders and partial output are not successful grading inputs.
Preparation requires a successful attempt with matching immutable output identity. Correct
failed inputs and use a fresh run rather than copying answer files into public directories
or relabeling an outcome.

## Review before Phase 1

A completed generation run does not automatically become the grading corpus. Prepare a
new release, inspect it, and stop for an operator's review:

```sh
merlin experiment corpus prepare /configured/out/runs/gemmini/phase0/example-1 \
  --output /configured/out/artifacts/protocols/gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/gemmini-review-1
```

Preparation combines the descriptor-selected source pool with receipt-declared generated
members, retains classified hand-authored members, and stages selected resources. It refuses
unresolved provenance, removals, collisions and admission-count changes; it does not edit
the source corpus or silently approve a different grading population.

Inspection reports aggregate counts and commitments. Detailed diagnostics and review records
are owner-only under `private/`. Keep hidden capsules, goldens and private weights out of public
examples, shared packages and agent-visible bundles. Being gitignored is not access control.

Only after reviewing the prepared inputs and private diagnostics, acknowledge the exact
digest returned by inspection:

```sh
merlin experiment corpus seal /configured/out/artifacts/protocols/gemmini-review-1 \
  --expected-digest DIGEST_FROM_INSPECT --reviewed-by OPERATOR --review-note REVIEW_SUMMARY
```

The seal records a review acknowledgement, not numerical or hardware certification. Do not
edit a sealed release. Follow the [reviewed Phase 0 handoff](../../experiments/README.md#reviewed-phase-0-handoff)
to select its released descriptor and `corpus_seal` explicitly in a Phase 1 definition,
retaining the chosen treatment and budgets. Phases run separately; private review evidence
stays host-only. Never rewrite historical receipts to attribute old runs to new inputs.

## Interpret coverage from evidence

Capsule counts describe a particular run, not a guarantee attached to a target name.
Inspect generation receipts and release admission results. Missing facts, unsupported
operations, failed builders and absent private answers must remain visible; none establishes
correctness or full coverage. Compare requirements and actual members, not just percentages:
two targets can report the same ratio over different sets.

For definition-based cross-target comparison, use
`merlin experiment corpus compare DEFINITION DEFINITION`; the standalone generator's
`--comparison-manifest` is for legacy profile collections only.

Legacy corpus trees remain migration inputs where explicitly selected. They are not the
entry point for a new experiment, and stale files there cannot substitute for the newly
generated, reviewed and sealed population.
