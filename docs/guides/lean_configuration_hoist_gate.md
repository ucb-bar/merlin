---
title: Check scoped configuration loop hoisting with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-28
related: [verification, lean_configuration_elision_gate, lean_loop_transform_gate]
code_refs:
  - src/merlin/verify/configuration_hoist_intake.py
  - src/merlin/verify/configuration_hoist_capture.py
  - src/merlin/verify/configuration_hoist_bridge.py
  - src/merlin/verify/configuration_hoist_gate.py
  - src/merlin/verify/compiler_configuration_hoist.py
  - src/merlin/verify/compiler_configuration_hoist_intake.py
  - src/merlin/compile/scheduling/configuration_loops.py
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationHoistChecker.lean
---

# Check scoped configuration loop hoisting with Lean

The historical direct API checks complete Kernel IR before/after moving one
invariant configuration setter immediately before a positive counted loop.
It does not install compiler-wide enforcement or establish native ISA fidelity.

The current compiler also supplies the explicit selector
`configuration_loop_hoisting`. It guards the actual public producer, joins the
original active or explicitly supplied validation, and holds the result before
an ordinary Kernel consumer receives it:

```python
result = hoist_loop_configuration(
    kernel, adapter, selected_contracts=original_selected_contracts,
    pass_validation=validation,
)
consume(result.kernel)
```

`validation.passes` contains the selector. The original typed facts/software
selection and adapter identity must match; caller labels do not supply
independent qualification. The static profile supports one positive top-level
loop and constant setter/body operands. The theorem preserves whole partial
outcomes, including failures and terminal state, under its original indexed
`Models` premise for every index. Finite native examples do not discharge that
premise or establish hardware fidelity. Default producer calls remain free of
verification dependencies when no validation is selected.

The current-only V2 profile also admits retained data operands expressed with
the loop IV, sums, scaled terms and selects. Configuration field indices and
values, and every moved pure-setter operand, remain literal constants. Trusted
host analysis checks signed-index64 rendering, selected encoding ranges and
alignment; it checks literals in unreachable branches too. Actual expression
and binder tokens are retained in the Lean obligation. These finite bounds do
not weaken or discharge the theorem's all-index `Models` premise.

Use `compiler_configuration_hoist_indexed.proposal_adapter` explicitly to ask
the producer to consider IV-dependent data. This is an untrusted proposal: the
canonical worker independently checks the actual original/output pair. The
ordinary selected adapter keeps its conservative behavior. V1 is attempted
first and retains its original domain; V2 has a distinct bound profile identity.

The current V3 profile handles a nonempty prefix of pure setters, including
overlapping writes and a loop containing only that prefix. For example:

```text
repeat 3 times:                 set a = 7
  set a = 7                    set a = 3
  set a = 3          →         set b = 5
  set b = 5                    repeat 3 times:
  work(i)                        work(i)
```

The checker derives the final assignment of the whole prefix. Retained calls
may write the same final configuration values while keeping their other effects;
conflicting writes and clobbers refuse. Every moved setter is still a total
pure overwrite under `Models`. The new ten-module canonical proof preserves
complete outcomes, including failure state; finite native comparisons remain
separate qualification. V1/V2 keep their original nine-module proof closures.
Installed qualification covers 53 distinct current-source cases, including four
V3 public-producer/worker/native paths, V1/V2 compatibility and a real reversed
setter-order proof refusal. Multiple/nested Kernel loops, completion fields and
IV-dependent configuration values still require further engineering.

This current route has no historical producer-checkout requirement. The
standalone API below retains its original revision and source-manifest guard.

```python
from pathlib import Path
from merlin.verify.configuration_hoist_intake import prepare
from merlin.verify.configuration_hoist_gate import check, deliver, replay

candidate = prepare(
    Path("source.mk").read_text(), Path("final.mk").read_text(),
    reference="/path/to/detached-e048-producer",
    facts="/path/to/original-facts.json",
    software="/path/to/original-software.yaml",
    target="selected-target",
    source_manifest="/path/to/PRODUCER-SOURCE-MANIFEST-e0484f488.json",
)
lean = Path("/path/to/selected/lean")
result = check(candidate, lean=lean, destination="out/build/hoist/check-1")
if result.status != "checked":
    raise RuntimeError(result.reason)
deliver(result.completion, candidate, consume_checked_kernel)
again = replay(result.completion.report, lean=lean,
               destination="out/build/hoist/replay-1")
assert again.status == "checked", again.reason
```

Destinations must be fresh. The read-only producer must remain detached at
e0484f4886ae0780d0325cdfa998842b61bb647c. The source manifest has pinned SHA256
bb35450b5374e4fea125d0f766588830eea7a771131de7d1e961d5a4a4ea2626.
Its 2,009 Python source blobs were independently compared with that Git revision.
Used file origins and alias helpers, original facts/software, fresh Python-cache
absence, and complete typed Kernel text/parse roundtrips are checked. These are
source-attribution controls, not proofs of Python, parser or ISA correctness.

The initial domain has one positive static top-level loop, one leading pure
constant setter, a nonempty flat ordered body, and no body writes/clobbers of
setup fields. Even identical overlapping writes are outside this subset.
Recognized instruction operands are integer constants. Completion lists and
call asynchronous/unit/staging annotations must be empty. Zero-trip, nested,
multi-prefix, IV-dependent and unsupported operand/control forms are refused.
Kernel headers, arguments, descriptors, call fields and metadata stay bound to
the complete actual endpoints.

`ActualConfigurationHoist.complete_kernel` instantiates the proved partial-loop
checker. Under one common indexed call interpretation, a total exact pure setter
and the declared successful body frames, it preserves complete successful and
error/terminal states from arbitrary initial state. The checker derives setup
fixedness and the body frame condition; it assumes neither source success nor
whole-run equality. Surrounding partial calls and arbitrary modeled observers
inherit equality. The interpretation must represent all relevant caller, memory,
return and observation state with a nonaliasing configuration mapping.

Native checking uses the installed package's nine canonical proof modules,
independent of caller `MERLIN_REPO_ROOT`, plus a generated query in a fresh
private tree. All fifteen roots/types/imports and standard-three-axiom evidence
are audited. Live delivery revalidates source/tool/process/query/object bindings;
saved or copied tokens do not authorize consumption. Replay rebuilds fresh.

Human effect-policy approval, primitive/native/hardware fidelity, actual-model
coverage and composition with adjacent passes remain separate obligations.
This is a scoped conditional result, not verification of the complete hoisting
algorithm, all scheduling transforms, completion-token loops or numerical policy.
