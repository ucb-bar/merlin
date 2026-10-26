# One-call caller-context successor — SOURCE ONLY, 2026-10-26

**All new Lean remains UNCOMPILED; all Lean positive/negative outcomes are
UNOBSERVED. No new qualification, replay or native result exists.** The private
runtime hold is unchanged. Primary is `/home/jeremy/Projects/merlin`,
`feat/pass-verification`, observed HEAD `9d7fe0846c43574ca6f83a5cccc5dc147b06ca6e`.
Root alone publishes/integrates. No owned live jobs remain.

This source batch extends the previous precise refusal for an unchanged wrapper
calling the transformed kernel. It preserves the prior **50 files**,
`source-entries-v1`, `entries-source-review-v1` and historical capsules unchanged.
It adds **11 sources**: seven Lean modules, three Python leaves and one test.

## Audit: what the kernel result actually exposes

`After.checked_actual_chain` and the entry successor compare **possible finite
successful observations** from entry allocation clock zero. The observation is:
exact returned values, the entire heap outside the private allocation partition,
and user state. It hides private cells and the allocation clock. The retained
source plan allocates twice; the reused expansion allocates once. Those clocks
cannot be identified by the observation theorem.

The new `PublicRelated` relation is precisely equality of user state and
`visibleHeap`. It does not imply equal MemoryWorlds. Its load/store laws reconstruct
concrete heap accesses/updates. A suffix cannot inspect hidden state through an
unrestricted opaque operation: only concrete binary, literal, bounded-index,
load/store actions are admitted; casts/dimensions/constants are interpreted by
the existing intrinsic Code constructors. Calls, regions, allocation and release
are excluded from this continuation grammar.

## Supported wrapper grammar and actual theorem chain

The **normalized** wrapper begins with exactly one direct call to the selected
kernel. Arguments are checked reads from typed formals; aliasing/repeated operands
are allowed. The call returns scalar/index results. After it, a finite sequence
of admitted scalar/index operations, identical view aliases, dimensions and
public-memory loads/stores ends in a scalar/index return. Multiple inline public
observer updates are supported by the same constructor induction. A subsequent
`func.call @touch` is not admitted by this first call-free suffix grammar.

An identical descriptor alias before the call in the raw source is supported when
ordinary shape normalization removes it: the source→normalized module theorem
supplies that edge. Other retained prefix operations are explicitly refused by
the current first-call route. The descriptor-alias test's independently parsed
final wrapper omits that alias; it is not inferred to be equivalent by Python.

| New root | What is derived |
|---|---|
| `PublicOpcode.transfer` | Per-constructor action simulation on related public worlds, with identical explicit arguments/results |
| `suffix_transfer` | Whole call-free continuation simulation, maintaining external-descriptor support after every local SSA write |
| `CallerContext.complete_iff` | Exact complete entry decomposition into the one callee run and suffix; actual smaller call depth, no guessed fuel or monotonicity premise |
| `CallerContext.typed_arguments` | Callee argument typing from typed wrapper formals, freshness, checked argument membership/kinds and plain callee formals |
| `ContextBoundary.kernel_intermediate` | Intermediate kernel observation equivalence derived from actual checked source/final endpoints, normalization and materialization |
| `CallerContext.checked_boundary` | Original raw wrapper→normalized wrapper→related callee worlds→suffix→actual final wrapper |
| `checked_context_roster` / generated `every_definition` | Complete source/final entry roster, selecting kernel, closed unchanged entry or supported caller-context route |

No final root takes whole-callee/run/world equality as a premise. Kernel observation
preservation is derived from existing checked roots inside the composition. The
callee's result equality and public-world relation come from the resulting
observation; the continuation consumes only that relation. Intermediate register
support is derived at entry and after local writes, not supplied as a global
intermediate typing assumption.

Expanded code still contains eliminated-allocation aliases. Source review caught
that a normalizer check comparing expanded code to itself would be inappropriate.
The implementation instead uses the actual materialization theorem in reverse
for the intermediate edge. No extra self-normalization premise remains.

## Domain, observations and explicit refusals

Typed wrapper inputs are required. Every input descriptor's allocation lies
outside the selected kernel's private partition; the caller arguments passed to
the kernel satisfy its original canonical singleton-view contract. Additional
external descriptors remain fully typed: allocation/base/offset/shape/strides/
element/space are not erased. Caller aliases are not assumed disjoint.

The suffix receives matching scalar/index kernel results and preserved captured
arguments. Local typed result checks plus non-view output restrictions prevent
private pointers from entering its register support. Actual coordinates and
address calculations are used for public loads/stores. Return descriptors are
outside this chosen suffix domain, including public descriptor returns.

Explicit refusals retain full actual raw programs and prospective finite checks:

- **Repeated kernel call:** missing stability from RELATED initial caller heaps
  and nonzero/different private clocks. The clock-zero theorem cannot be reused.
- **Transitive kernel caller:** missing contextual congruence through the extra
  call layer; identical local text alone is insufficient.
- **Allocation/other retained prefix:** the kernel would not necessarily start
  at the checked clock-zero boundary; no allocator reset/fresh-scope promise added.
- **Private-pointer escape, allocator/unknown suffix effects:** outside the
  constructorwise public frame grammar.
- **External declaration/unknown caller:** no external-call/ABI model is silently
  inferred; declarations remain in the complete header roster and force refusal.

Allocation is still total, private and unobservable **in the declared model**.
No allocator resource/failure guarantee, error classification, divergence,
native ABI, numerical fidelity or hardware behavior is proved.

## Actual example and nonvacuity candidate

The raw source and final are **new source-level wrapper test extensions** of the
existing retained two-epoch modules. They are not relabeled original 4499 producer
captures, and no compiler pass was rerun for these extensions. Both entire raw
modules are independently parsed/decoded; all original functions plus the wrapper,
interfaces/visibility, calls, SSA, operands, returns and memory/control fields are
serialized into complete Lean terms. A new untrusted normalization/expansion hint
is checked against those terms; the old Python pair matcher is forbidden by tests.

Base wrapper:

```
(value, extent) = kernel(a, b, caller)
continued = add(value, bias)
old = load(caller[0])
observed = add(old, continued)
store(observed, caller[0])
return (observed, extent)
```

The constructed nonzero witness uses the **existing** Nat-addition interpretation
as an inhabited abstract relation. This is not a new numerical law or native i32
execution claim. Inputs are `17,29`, caller cell starts at `91`, bias is `7`, and
private cells start uninitialized. The frozen epoch witness constructor derives
full initialization, kernel return `308,2`, and caller cell `183`. The new candidate
constructs the actual scalar/public load/add/store/return continuation, returning
`498,2` and storing `498`. No source-success premise is supplied. Generic Unit
sample inputs also provide typed caller-domain candidates, including argument and
descriptor aliases; those alias cases have no new concrete execution witness.

The same named scalar/literal interpretation is used on both sides. Scalar
literals retain the previous explicit partial effect-free interpretation and
closed-kernel-support check. Numerical/primitive/native fidelity remains separate.

## Exact delivery and source/data validation

- `../source-context-v1/`: immutable integrity-only successor. **61 owned files**
  total, of which 11 are new; **118 project Lean / 51 Python sources** in the
  static dependency closure (overlapping counts). `successor-source.patch` is
  additive over the unchanged previous 50-source cut.
- `NEW-OWNED-SOURCES.json`, `OWNED-SOURCES.json`, `NEW-DECLARATIONS.json`, import
  graphs and closure manifests identify source dependencies, not runtime objects.
- `QUERIES.json`: **14 distinct complete queries**, each bound to original raw
  `source.mlir`, `final.mlir`, `Context.lean`, preparation report and full proposed
  root roster. The base positive proposes 239 support roots plus `every_definition`.
  These are requested roots, not kernel-audited roots.
- `evidence/<case>/` retains the complete raw buffers/query/report. New wrapper
  provenance is explicit; original source/final capture hashes are also retained.
- `../tests-context-v4/`: **14 SDK-free tests passed**, all input modules pass
  Python/xDSL verification, all source bindings unchanged, no old pair matcher.
  Three expected positives: base, repeated scalar argument, descriptor alias.
  Six final-binding mutations: arguments, return, order, observer value, changed
  kernel, caller buffer. Five precise unsupported variants are listed above.
  This is preparation/field-binding evidence, not Lean acceptance/rejection or
  a semantic wrong-result/detection-rate claim.
- Optional Z3 dlopen attempts were blocked; two expected pytest timeout-plugin
  warnings remain. No SDK/runtime reads or native proof/solver launch occurred.
- Earlier source runs v1 (10), v2/v3 (13) remain retained. v4 adds the descriptor
  alias and removes an unnecessary raw-wrapper-equals-normalized-body obligation
  from the generic route; only the concrete execution witness checks that body.
- Original runtime incidents, all failed W profiles, prior source/counterexample
  capsules and the 8→16B countermodel remain unchanged. No failed profile was retried.

Machine-readable frame/domain/root/field/refusal/evidence maps accompany this README.
Lexical/AST checks are source review only, not a std3/Lean kernel audit.

## Claim and next gates

**Available now:** coherent source candidates for the supported one-call context,
full actual-IR preparation and explicit refusal/witness candidates.

**Proposed after successful checking/review:** equal possible finite successful
return/public-heap observations for every checked entry of this extended module,
within the stated wrapper/input/primitive/allocator domain. This is not all shared
pass inputs or all compiler graph transformations.

Remaining gates: justified new runtime context; compile the exact dependency
closure and root queries; inspect theorem types and transitive std3 axioms; run
actual positives and changed finite bindings; freeze a new qualification and one
fresh replay; finish independent Graham/Prepare projection/assumption review.
Any source correction must create a new cut. Root alone publishes. Historical
Linux receipts are not laptop/portability qualification. gSIM remains separate.
