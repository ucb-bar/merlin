---
title: Generating capsules for a target
kind: guide
status: current
owner: targetgen
last_verified: 2026-10-06
related: [adding_a_target, gemmini_experiment, capsule_bench, integrations, phase0_specification]
code_refs:
  - experiments/catalog.yaml
  - packages/merlin-experiments/src/merlin_experiments/phase0/__main__.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/writer.py
  - packages/merlin-experiments/src/merlin_experiments/corpus/preparation.py
  - packages/merlin-experiments/src/merlin_experiments/corpus/release.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/freeze.py
  - packages/merlin-experiments/src/merlin_experiments/runner.py
  - src/merlin/targetgen/corpus_spec.py
  - src/merlin/targetgen/group_capsule_entries.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/model_forms.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/form_perf.py
  - src/merlin/kernels/endpoints.py
  - src/merlin/perf/isa_prohibition.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/instruction_roles.py
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
The matching [Atlas example](../../examples/atlas/phase0/README.md) follows the
same input/artifact structure. Read [the SW-spec guide](phase0_specification.md)
to separate authored behavior from extracted RTL and workload policy.

An `attention_qk` entry can state independent query length `M`, key length `N`,
and reduction depth `K` (or the corresponding `_tiles` extents). The generated
operands are `Q[M,K]` and `K[N,K]`, with scores `[M,N]`; omitting `N` retains the
square `[M,M]` default. For integer arithmetic, the capsule writer checks its
actual stimulus and reduction depth against the selected internal-width policy.
That mathematical bound is not target execution evidence. A generated tier cap
also needs an explicit cheaper sibling; it does not certify an unrun simulator.

| Input or output | Owner |
| --- | --- |
| Public target recipe | `examples/<target>/phase0/recipe.yaml` |
| Software semantics and operation signatures | Explicit `target/software-spec.yaml`; selected by recipe and experiment |
| Required hardware evidence | Explicit `target/hardware.yaml`, selected OOT support and exact extracted facts |
| Target descriptor and input selection | The example's `target/` directory and `experiment.yaml` |
| Shared performance-family policy | `experiments/templates/phase0/performance.yaml` |
| Synthesis/SMT profiles | Explicit definition inputs; retained locations vary during migration |
| Hidden profiles, holdouts and answers | Host-private inputs, never public examples or candidate grants |
| Generated capsules and generation receipts | `<run-dir>/phase0/capsules/` |
| Raw facts and exact consumer views | `<run-dir>/phase0/hardware/` and `evidence-manifest.json` |
| Selected SW inputs and generation gaps | `<run-dir>/phase0/software/` and `coverage/generation.json` |
| Prepared grading release and review evidence | A fresh operator-selected artifact directory |

The installed generator lives in `merlin_experiments.phase0`; shared derivation primitives
remain in core. There is no need to copy generation scripts into `out/`. Generated capsules
are artifacts, not new library code or files to sync into a wheel.

The tracked `merlin/contract/capsules/` tree is retained historical benchmark input,
not the destination for a new Phase 0 run. Existing descriptors and frozen input
bundles still name its exact paths; it is excluded from the installed core wheel.
The generator rejects output or evidence destinations that overlap either this
legacy tree or the descriptor-selected source corpus. Keep new derived capsules in
the run artifact, then prepare and seal a reviewed release for Phase 1 and Phase 2.

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

Both target examples select `evidence_mode: diagnostic`. Pass
`--phase0-rtl-facts /absolute/generated/facts.json` to select exact extraction bytes
and inspect `evidence-manifest.json` for the actual raw/derived consumer inputs.
Unresolved evidence cannot be promoted by selecting verified mode. A changed
software spec or recipe requires new synthesis commitments; legacy references
remain diagnostic, while genuine digest-bound mismatches refuse generation.
Regenerate and explicitly select the new requirement/synthesis pair, never edit
old evidence to match new input bytes.

For standalone invocation, `python -m merlin_experiments.phase0 --help` describes the
installed generator's explicit inputs, including required `--output-root`. Do not use the
legacy native command to regenerate every target.

Generation can retain completed members when another member fails; rejected synthesized
members can also be removed. Read the logs and receipts, not merely the directory count.
Missing facts, skipped builders and partial output are not successful grading inputs.
Preparation requires a successful attempt with matching immutable output identity. Correct
failed inputs and use a fresh run rather than copying answer files into public directories
or relabeling an outcome.

## Stages derived from the iteration captures

`merlin experiment corpus derive` also groups each declared ITERATION capture with the one
compute-group pass (`compute_groups` + `group_command`, stated by `group_capsule_entries`) and
writes two derived stages into the byte-bound plan:

- **Model forms** (`MF_<workload>_...`, Phase 1): one capsule per distinct op-form at the
  workload's own multipliers, reduction depth and features, positions reduced to two tiles;
  host-region trailing-window sums (row sums, window means) are offered in their device form.
  A member too large to certify is capped at the loop tier and `extends` a certified sibling.
- **Form-perf scope** (`requirements.yaml` → `scope.performance.forms`, Phase 2): every form class
  with its per-workload predicted issue-cycle share. The shared template's `PW` family mints one
  `_perf` member per class, at its costliest group's extents, with a candidate arm held to the
  experiment's `prohibited_instruction_roles` and an unrestricted vendor-reference arm.
  `phase2-capsule-coverage.json` → `form_perf_coverage` lists every class at or above the
  template's share threshold and whether it has a member and a vendor bar.

Both stages refuse held-out models by name (the descriptor's claim roster and the reviewed
evaluation-only models in `merlin/contract/claim_models.yaml`). A group capsule reaches disk only
through `phase0.group_forms.write_group_capsule`, under `group_binding` -- the corpus binding with an
accumulator the recipe, contract or RTL facts state, never a literal.

An experiment's `policy.prohibited_instruction_roles` names closed-vocabulary roles
(`merlin.kernels.roles`). Phase 0 resolves them against the target's derived instruction taxonomy
and records the result as `instruction_policy` in `MANIFEST.yaml` and `coverage/generation.json`;
Phase 1 receives it in `MERLIN_PROHIBITED_INSTRUCTION_ROLES` and every phase command carries it.
Declaring a role does not by itself scan a program; enforcement is a separate whole-ELF check.

The policy fails closed at every step, because a rule that forbids nothing reads exactly like an
enforced one:

- The taxonomy binds roles through `compute_endpoints.yaml`, read from the selected contract
  (`MERLIN_CONTRACT_DIR`). A missing declaration raises, and a target that no endpoint binds has an
  `UNKNOWN` taxonomy.
- A declared role that matches none of the target's instructions makes the policy `UNKNOWN`, with a
  `reason`. Verified Phase 0 refuses it, and `corpus prepare` refuses to release a verified corpus
  whose policy cannot be enforced.
- Phase 1 scans every ELF that a capsule grade links, including code that nothing calls. A prohibited
  instruction fails the capsule (`PROHIBITED_INSTRUCTION`). A scan that could not run, or whose roles
  name no instruction, leaves the capsule `incomplete`.
- Phase 2 measured mode carries the sealed policy by value. It reads it from `--phase0-manifest`, the
  config's `phase0_manifest`, or the descriptor's corpus manifest. A run, cell or group arm whose
  policy is not enforceable is refused before any build. So is a scan that prohibits fewer
  instructions than Phase 0 sealed.

## Review before Phase 1

A completed generation run does not automatically become the grading corpus. Prepare a
new release, inspect it, and stop for an operator's review:

```sh
merlin experiment corpus prepare /configured/out/runs/gemmini/phase0/example-1 \
  --output /configured/out/artifacts/protocols/gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/gemmini-review-1
```

The default preparation mode is historical: it combines the descriptor-selected source pool
with receipt-declared generated members and retains classified hand-authored members. For a
generated-only public release, select `--generated-only` explicitly. That mode copies public
capsules only from the selected, immutable Phase 0 run output; it never falls back to the
descriptor's legacy capsule tree. If Phase 0 emitted no hidden cohort, provide a separate
operator-owned `--private-baseline /absolute/hidden-category` or preparation refuses the
empty hidden grade. A `--retirements` review is incompatible with generated-only mode.
Neither mode edits the source corpus or silently approves a different grading population.
Generated-only preparation still requires nonempty native admission and an operator seal;
it does not assert numerical, model-wide, or hardware correctness.
Preparation may reuse a completed, dedicated frozen Phase 0 run after its historical host
capture environment changes: it checks the recorded successful attempt and output digest,
run-owned frozen inputs, copied producer/provider source, generation lineage, and selected
capture attestations. This
consumes the historical producer's bytes; it neither reruns Phase 0 nor requalifies that old
environment. New execution or resume still requires the selected live runtime to match its
freeze. A newly prepared Phase 1 bundle separately freezes the **current** installed compiler
and tool sources while retaining the Phase 0 producer's historical identity. The unsigned run
receipt establishes recorded byte consistency, not cryptographic proof of execution; operator
review and sealing remain separate. Combined plans with non-run-owned input pins remain
outside this artifact-only admission path.
`--phase1-policy-descriptor FILE` retains an explicit Phase-1-only gate policy beside the release
(`phase1-policy-descriptor.yaml`, owner-only and read-only, recorded by digest). It may differ from
the frozen Phase 0 descriptor only in `phase1_gates`; any other difference refuses, so the overlay
never changes the Phase 0 source identity.
For a sealed Phase 1 run, select the released descriptor and use its complete descriptor
cohort. A raw capsule-root override is diagnostic only and cannot inherit the review,
even when its files are under the reviewed release.

Preparation also copies the exact effective capability view exported by Phase 0
(`software/contract.json`) into the release's `experiment/contracts/target_contract.yaml`.
The released descriptor selects it, every experiment level receives a read-only
grant, and Phase 1 automatically binds it as `MERLIN_TARGET_CONTRACT`. Support plugins
still come from the separately selected OOT provider: changing a capability view
cannot inject executable support code. Conflicting contract selections or changed
bytes refuse launch/resume. Older releases lacking this handoff need a newly frozen
run for verified execution; do not edit their manifests.

Inspection reports aggregate counts and commitments. Detailed diagnostics and review records
are owner-only under `private/`. Keep hidden capsules, goldens and private weights out of public
examples, shared packages and agent-visible bundles. Being gitignored is not access control.
For a run with selected evidence, `private/preparation.json` also records the exact
requirement, evidence, generation receipt and generated manifest digests, the
capsule roster, omissions and both cohort-coverage commitments. `corpus inspect`
shows only `generation_lineage_sha256`; sealing rechecks the private record against
the frozen run. A missing or changed member requires a new preparation, not a
manual manifest edit.

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
