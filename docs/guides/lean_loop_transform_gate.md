---
title: Check ordered loop unrolling and segmentation with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [verification, lean_pass_gate]
code_refs:
  - src/merlin/verify/loop_transform_intake.py
  - src/merlin/verify/loop_transform_bridge.py
  - src/merlin/verify/loop_transform_gate.py
  - src/merlin/verify/loop_transform_cli.py
  - src/merlin/verify/lean_graph_state_pipeline_emit.py
  - src/merlin/verify/lean/Merlin/GraphStatePipelineLoops.lean
  - merlin/tests/ir/loop_transform_qualification.py
---

# Check ordered loop unrolling and segmentation with Lean

This direct checking path runs the selected producer, reconstructs a proof from
the complete original, unrolled and segmented IR, and delivers the final bytes
only after fresh Lean compilation and theorem audits. It is separate from the
[compiler pass gate](lean_pass_gate.md); it does not install compiler-wide loop
enforcement or establish numerical or hardware fidelity.

The separate [compiler integration](lean_compiler_stack.md) captures current
compiler buffers and supports individual unrolling, individual tiling and their
composition. It reuses this checker without requiring the historical producer
checkout. The standalone API described below retains its pinned-producer contract.

The compiler also has separately qualified bounded generic-value singleton
UNROLL and TILE routes; generic UNROLL→TILE composition is excluded. See the
[case catalogue](lean_bounded_pass_profiles.md) for their exact source grammar.
Compiler source exclusions run ordinary transforms and are logged unverified
at invocation end. Once admitted, unsuccessful target/proof/tool checks block.
Standalone checks below retain strict refusal and live-delivery rules.

The reference checkout must contain revision
`4499eebb959c0f92cefd92cc6d80646ba66474ed`. The bridge checks its revision and the
bytes of imported producer sources. Use the repository Python environment with
xDSL and the project-selected Lean executable. Each check needs a new destination.

```sh
mkdir -p out/artifacts/verification/loop-example
python -m merlin.verify.loop_transform_cli \
  --source input.mlir --factor 3 --tile 2 \
  --reference /path/to/pinned-producer-checkout \
  --lean /path/to/selected/lean \
  --destination out/build/loop-example/check-1 \
  --output out/artifacts/verification/loop-example/after.mlir
```

Alternatively, `--buffers before.mlir unrolled.mlir segmented.mlir` supplies three
complete modules. The bridge still reruns both transforms and checks that their
actual outputs match. Both boundaries must change. The first two identity stages
inside the existing proof emitter are compatibility steps, not CSE or LICM runs.

The theorem `Merlin.GraphStatePipelineInstance.complete` establishes equal
returned values, modeled errors and terminal worlds for every defined entry in
the projected modules. It covers arbitrary argument vectors and initial worlds,
including modeled wrong-arity and absent-entry behavior. Body grouping, SSA,
control, carries, continuation and entry-roster obligations are reconstructed
from actual IR; source success or whole-program equality is not assumed.

The model uses a common total pure-primitive interpretation, explicit index
control meanings and a common possibly failing stateful external provider whose
successful results have their declared arity. Complete IR parsing/projection,
source binding, host transport and correspondence with native execution remain
trusted boundaries. `checked` records proof acceptance and audits; independent
specification review and native fidelity are separate.

Supported inputs have one changed top-level loop, two arguments to that entry,
one to four word carries, a region-free source body, static nonnegative bounds,
positive stride, index64, factor2..4, and a trip count greater than and divisible
by the factor. Tiles are2..16. At most16 retained external calls occur across the
unrolled body. Ordered one-dimensional segmentation permits a short final
segment. General multidimensional, reordered, parallel or dynamic tiling,
unroll tails, empty/inline modes and internal calls are outside this API's scope.

Successful checks retain raw stages, the generated query, exact source/tool/object
bindings, import audits and printed theorem types under the destination. The
Python API's live completion is required for delivery; saved JSON cannot authorize
a consumer. Changed artifacts invalidate that completion.

```sh
python -m merlin.verify.loop_transform_cli \
  --replay out/build/loop-example/check-1/REPORT.json \
  --reference /path/to/pinned-producer-checkout \
  --lean /path/to/selected/lean \
  --destination out/build/loop-example/check-2
```

Replay reconstructs the request and builds fresh proof objects. It produces
evidence; output delivery requires a fresh actual request. Unsupported domains,
unproved obligations and unavailable execution are distinct failure statuses,
all with nonzero exit and no consumer delivery. A processing-limit or native
crash is not a semantic counterexample or a successful mutation refusal.
