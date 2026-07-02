# Decision record — Phase D / task #135: keep the general backends (NOT the converged agentic submissions)

**Decision:** The cross-approach perf benchmark keeps pointing approaches (b) and (c) at the **general**
OOT backend packages — baseline = `agent_spec_v0_mlir_oot`, merlin-targetgen = `agent_spec_v1_mlir_oot`.
We do **not** swap them for the per-capsule converged agentic submissions (`rb_full_*`, `merlin_full_*`).
Phase D is closed **by design**, not deferred.

## Why

1. **Apples-to-apples comparison subject.** The benchmark's unit is "drive the *same* kernel through each
   code-gen *approach*." An approach must be a general, shape-agnostic backend that can lower *any*
   harvested shape. `agent_spec_v0/v1` are exactly that. The golden C library and the IREE dialect are
   also general. So the four columns are all the same *kind* of thing.

2. **The agentic submissions are run states, not backends.** `runs/merlin_assisted/merlin_full_*` and
   `rb_full_*` are agentic *authoring* artifacts (rounds/, TASK.md, per-capsule converged output). They
   are the product of the *other* axis of this project — the baseline-vs-merlin **authoring-effort** A/B
   (tokens/cost/rounds). They are not reusable general backend packages.

3. **Swapping them in would overfit the capsule set.** Pointing the perf arms at per-capsule converged
   outputs would measure code that was tuned *to those exact capsules*, turning a general-compiler
   comparison into a shape-overfit one. That directly contradicts the standing project principle
   (memory: `abstract-into-compiler-not-overfit`): the goal is to abstract *why* experts win into
   **general compiler capabilities**, never to ship per-shape hand/agent-tuned kernels as if they were
   the compiler. The general backends are therefore the *more correct* — not merely the more convenient —
   comparison subjects.

## What remains possible (no action needed now)

The runner takes a package path per arm, so IF a converged backend is ever packaged as a **general OOT
target** (one that lowers arbitrary shapes, not a per-capsule artifact), pointing an arm at it is a
one-line config change. Until such a general package exists, there is nothing to swap.

## Cross-reference

- `reports/AUTONOMOUS_RUN_SUMMARY.md` (§#135) — original decision capture.
- The authoring-effort A/B (tokens/cost/rounds) remains the home of the `rb_full`/`merlin_full`
  submissions; this decision does not touch that comparison.
