---
title: Check scoped configuration loop hoisting with Lean
kind: guide
status: current
owner: verification
last_verified: 2026-10-29
related: [verification, lean_configuration_elision_gate, lean_loop_transform_gate]
code_refs:
  - src/merlin/verify/configuration_hoist_intake.py
  - src/merlin/verify/configuration_hoist_capture.py
  - src/merlin/verify/configuration_hoist_bridge.py
  - src/merlin/verify/configuration_hoist_gate.py
  - src/merlin/verify/compiler_configuration_hoist.py
  - src/merlin/verify/compiler_configuration_hoist_intake.py
  - src/merlin/verify/compiler_configuration_hoist_multi.py
  - src/merlin/verify/compiler_configuration_hoist_recursive.py
  - src/merlin/verify/configuration_hoist_recursive_emit.py
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationRecursiveHoist.lean
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationHoistFocus.lean
  - src/merlin/compile/scheduling/configuration_loops.py
  - src/merlin/compile/scheduling/configuration_c_consumer.py
  - src/merlin/verify/compiler_configuration_hoist_wide.py
  - src/merlin/verify/lean/Merlin/GraphStateConfigurationHoistChecker.lean
---

# Check scoped configuration loop hoisting with Lean

The current compiler wrapper follows the [bounded profile policy](lean_bounded_pass_profiles.md):
known original-source exclusions run ordinary hoisting and are logged unverified
at invocation end. Supported sources retain blocking actual-target/proof/tool
checks and the original selected ISA/effect/C64 owner guards. The direct
proof/replay APIs below remain strict; unsupported cases receive no proof credit.

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
At its recorded source cut, installed qualification covered 53 distinct cases, including four
V3 public-producer/worker/native paths, V1/V2 compatibility and a real reversed
setter-order proof refusal. Completion fields and IV-dependent configuration
values remain outside this profile. The V6 profile below adds nested Kernel loops.

The current V4 profile composes prefix motions across multiple flat top-level
loops. Each checked stage retains the actual earlier/later command context,
global frame and full atom roster; a closed terminal join binds the complete
actual output. It does not move a setter across another loop. The eleven-module
canonical proof preserves the same complete outcomes under all-index `Models`.
At its recorded source cut, 90 distinct cases passed, including four multi-loop public/
worker/native paths, fresh replay, four kernel refusals and V1/V2/V3 compatibility.
The 153 native initial/failure observations remain finite qualification.

Counted Kernel IR requires positive extents. Zero-extent Kernel inputs are
rejected by the existing structure checker and producer before optimization;
they receive no coverage. V1–V5 do not decode nested Kernel commands; completion
fields remain unsupported. Ordinary calls retain the inherited extent cap of
1,000,000, including the current nested profile below.

An explicit original C64 consumer supports a separate current-only V5 wide
profile; smaller counts retain V1–V4 selection:

```python
from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

consumer = ConfigurationCConsumer(index_bits=64)  # select before the producer
result = consumer.hoist(
    kernel, adapter, selected_contracts=original_selected_contracts,
    pass_validation=validation,
)
c_source = result.emit_c(instruction_set, symbol="kernel")
```

The original validation also selects index64. The consumer checks both actual
Kernel endpoints against the builtin emitter's signed integer domain, including
literals in dead expression branches. Its controlled emission uses the same
captured width and revalidates original selected contracts, endpoint text and
checked source/proof products before and after emission. Constructed, copied,
serialized or foreign-thread result carriers do not own that operation.
A caller-supplied width or an unchecked default C emitter is not a domain proof.

Positive extents through 2^63−1 are admitted by V5; the existing Focus theorem
checks counts algebraically. No huge loop is compiled or executed. V1–V4 math,
default behavior and legacy job/request/report shapes remain unchanged. The
versioned C-bound route reconstructs a fresh consumer for worker/replay requests;
a saved report is descriptive input, never live completion authority. For fresh
controlled replay use a newly selected original consumer:

```python
fresh = ConfigurationCConsumer(index_bits=64)
replayed = fresh.replay(report_path, lean=selected_lean, destination=fresh_directory)
c_source = replayed.emit_c(instruction_set, symbol="kernel")
```

At its recorded source cut, installed independent qualification passed 129 distinct cases: 45 consumer
controls, 79 legacy controls and five canonical cases (six SDK invocations,
including MAX63 replay). It covers 1,000,001 and 2^63−1 symbolically, a real Lean
changed-count refusal, preserved legacy V4 schemas and 15 small compiled
state/failure comparisons. This establishes the declared consumer-domain binding;
all-index `Models`, actual instruction/rendering/encoding/native fidelity and
human effect-policy review remain separate.

The current V6 profile checks recursive positive counted Kernel loops through
that same public producer and original consumer interface. For example, a pure
setup can move out of the inner loop while remaining inside the outer loop:

```text
repeat 2 times with i:             repeat 2 times with i:
  repeat 3 times with j:             set a = 3
    set a = 3             →          repeat 3 times with j:
    work(i, j)                        work(i, j)
  observe(i)                        observe(i)
```

Every enclosing loop, sibling and call suffix is interpreted in the checked
command tree. The parent index environment is restored for the suffix after a
loop; retained work can refer to both outer and inner IVs. Multiple pure setup
writes, including overlapping writes with the same final-value discipline as
V3, may move at several nesting levels. Each move stays at its original lexical
scope. The checker compares exact actual frame, atom, operand, binder, loop
metadata and full terminal command trees; a proposed stage or a host
`actual_final_join` flag supplies no proof authority.

Select `configuration_loop_hoisting` in the original `PassValidation` and call
`hoist_loop_configuration` as above. A nested source selects the fixed domain
`current-compiler-recursive-positive-loops-pure-prefix-hoist-v6`; a caller cannot
relabel it as V1–V5. To bind actual C emission, create
`ConfigurationCConsumer(index_bits=64)` before the producer, call
`consumer.hoist(...)`, and consume only the original result's `emit_c(...)`.
This is the same immutable original-consumer, live worker, delivery and fresh
replay protocol shown above. A plain `PassValidation` index-width setting does
not select that consumer or lift the ordinary 1,000,000 cap.

The default cap is a compatibility policy, not a count-enumerating proof limit.
With the original C64 consumer, each positive static loop extent may reach
2^63−1, subject to the actual original and final integer-domain checks, retained
expression ranges and the finite structural budgets. Full expression literals,
including factors, comparison values and unreachable branches, must fit the
selected rendering domain. The current source limits are 4,096 statements,
64 levels of recursive inspection, 256 configuration fields, 65,536 proposed
stages, and 32 MiB aggregate stage capture. These bound checking work; they are
not effect or model premises. Large-count qualification is symbolic/text only.
Native examples use small loops.

`GraphStateConfigurationRecursiveHoist.lean` derives recursive body fixedness,
loop congruence, full-context congruence and stage-chain composition from the
existing pure-prefix laws. The closed
`ActualConfigurationHoist.complete_kernel` theorem preserves the whole partial
outcome, including the error and its terminal state, for every initial state
and every IV environment under one common
`PartialHoist.Recursive.Models fields literal before.atoms run` premise. It
assumes neither source success, local simulation nor whole-run equality.
Successful-body fixedness is derived from the admitted call effects; failures
short-circuit from equal states. All relevant memory, return and observer state
must be represented by the original interpretation with nonaliasing
configuration fields. Host range checks and finite native observations do not
discharge this all-environment model premise.

The canonical V6 closure has eleven modules. Its private qualification compiles
the exact library, explicit first/later inner/outer failure probes, six actual
positive endpoint queries and seven genuine closed-negative endpoint queries.
Positive obligations audit 31 roots and 502 declarations with only the standard
three axioms. At the recorded V6 qualification cut, 73 installed host cases passed, and five freshly
reconstructed legacy profiles retain the same semantic payloads while rejecting
stale source-cut identities. Five installed public tests passed, exercising three
accepted original-owner workers, one fresh original-C64 replay and two genuine
operand/frame query refusals.
The controlled C fixture contributes 27 finite state/failure observations from
small 2×3 loops; exact emitted text joins the compiled fixture. This was 78
distinct installed pytest cases (73 host plus five public), with five legacy
reconstruction checks counted separately. The public cohort completes in
369.28 seconds. Evidence is retained under
`out/build/configuration-hoist-nested-20261029/installed-qualification-v1/`.
These cohorts retain separate exact source cuts.

The supported tree is ordinary static positive counted loops and ordered
integer-operand calls with admitted effects. Retained data expressions can use
multiple active IVs, sums, scales and selects through the existing explicit
untrusted `proposal_adapter`. The default selected adapter retains its
conservative behavior. Moved pure-setter operands and configuration effect
indices/values remain literal constants. A motion refuses when its retained
body has unknown or unordered effects, a conflicting configuration write or a
clobber. Unchanged surrounding calls remain exact partial calls; they do not
need a fictitious body frame. Pointer/noninteger operands, carried completion
fields, async/unit/staging annotations, IV shadowing and unknown expression or
control forms remain outside this profile. No producer math, default emitter
behavior, V1–V5 domain/schema,
ISA/effect approval or numerical policy changes. Native instruction semantics,
encoding, downstream MLIR/LLVM and executable/hardware fidelity remain separate.

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
