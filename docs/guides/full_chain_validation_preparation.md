---
title: Preparing private full-session numerical validation
kind: guide
status: current
owner: experiments
last_verified: 2026-10-23
code_refs:
  - examples/gemmini/phase1/launch/prepare_full_chain_scope.py
  - examples/gemmini/phase1/launch/external_validation.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_numerical_selection.py
---

# Prepare full-session validation after capsule authoring

Keep authoring inputs public and the initial target backend empty. Full-model
validation consumes the exact frozen authored compiler through the existing
capsule qualification handoff. A retained reference compiler may independently
produce protected reference evidence; it cannot supply candidate kernels or
candidate support when authored lowering is missing.

Prepare the scope without executing or opening private tensor payloads:

```sh
python examples/gemmini/phase1/launch/prepare_full_chain_scope.py \
  --private-spec "$PRIVATE_SPEC" --out "$NEW_PRIVATE_PREPARATION"
```

Use the selected environment with Merlin and merlin-experiments installed. The
output directory is created with mode 0700 and the proposal with mode 0600. Keep
it outside author grants. The proposal pins the private spec, reference manifest,
SHA256SUMS, model configuration and root session contracts. It retains exact
existing input digest declarations and recurrent input/output bindings without
reading arrays. Payload digest declarations are not fresh payload verification.

This command intentionally produces `blocked_not_selected`, never an executable
numerical selection, a static receipt or a successful model result. It compares
session and selected-reference step counts as one early mismatch detector;
matching counts do not establish correct state transfer or complete output scope.
The ordinary chain verifier must prove those properties from actual execution.
An explicit `paper_ready: false` or synthetic-input declaration limits the claims
supported by that capture; it does not itself prevent numerical execution or
create a new permission requirement. Preserve it in the proposal's limitations.

Before activating `selected_quantized_full_chain`, resolve:

- Session scope: image streams, prefill and recurrent decode/cache, or prefix,
  recurrent denoise and action decoding must match the selected references.
- Reference coverage: an admitted `merlin.operator.all_output_plan.v1` must select
  each complete-output producer manifest, its sums and every declared output at
  every required step. A selected subset of judged outputs is insufficient.
- Native launch bindings: choose the source/runtime-bound declarative launch,
  infrastructure receipt, private numerical criterion and bounded final deadline.
- Candidate attribution: after freezing, select its capsule qualification/handoff
  and compiler package. Produce original native static/build receipts for that
  package; do not copy receipt claims from a retained compiler.

Use `external_validation.py` for a separate finite post-capsule evaluation. Its
required arguments are `--config`, `--task-root`, `--prep`, `--capsule-prep`,
`--capsule-run`, `--private-spec-template`, `--infrastructure-template`, `--run-id`
and `--authorization-text`. An explicit `--capsule-qualification` may select the
ordinary handoff. The template needs actual numerical and all-output selections;
there is no safe runnable command while those artifacts are missing. The wrapper
rebinds run-owned identities and deadlines without regenerating references.

The existing supervisor produces `static-index.json`, complete chain reports,
all-output audits and `full-chain-qualification.json`. Six stage ELFs establish
neither stateful session execution nor a full model pass. `--validation-scope full`
alone does not select these numerical checks. No full-model pass is required to
begin construction from an empty backend; the final evaluation must remain an
explicit unfulfilled obligation until its actual evidence exists.
