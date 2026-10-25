# Pure scalar graph transformations — conditional whole-program correspondence

PRIMARY `/home/jeremy/Projects/merlin`, `feat/pass-verification`. Owned additive
GraphStateScalar / lean_graph_state_scalar leaves and matching tests only.
All 37 previous v2/chain/elision source bytes and capsules remain read-only.
No publication, shared integration, production edit, specification/lock change,
numerical gate discharge or operation approval is issued.

## Existing anchor and actual gap closed

The pinned4499 Merlin shared transformations are `scalar_redundancy_elimination`
(`eliminate_scalar_redundancy`: CSE + dead-chain elimination) and
`scalar_loop_invariant_motion` (`hoist_scalar_invariants`) in pure_scalar.py.
Existing `Merlin.eliminateDuplicate`, `eliminateUnused`, and `hoistInvariant` in
Foundation, selected by lean_obligations.extended_obligations, establish pure
functional identities. Their old scalar validator trusts structural optimizer
replay. Those identities are not whole-program SSA/effect/control proofs.

New roots: `Merlin.GraphState.Scalar.normalize_sound`, `check_sound`, `check_typed`,
`check_initialized`, and `check_operands_initialized`. They establish operational
returned/persistent observations from a sound executable graph checker consuming
both complete actual programs. No optimizer witness count or rerun-equality is
proof authority. The old Foundation/LeanMLIR import route is unavailable without
a matching selected manifest. This work supplies an independent pure-core graph
extension; it does not assume a numerical or xDSL↔Kernel semantics bridge.

## Exact language, domain and observation relation

Every source/target function uses its own ordered lexical SSA stack. Static
checks resolve all operand/result types, previous definitions, entry arguments,
loop captures, IV/carry/result/yield slots and exact return slots. A malformed or
missing block never becomes an empty successful boundary. Declarations,
function/module attributes, symbol interfaces and all surrounding functions are
completely bound by the trusted actual-IR projection.

Pure nodes are closed, unflagged signless fixed-width integer constants,
add/sub/mul, signed/unsigned extension and truncation, widths 1..256. Exact
properties, attributes, effects, arities and widths are independently checked
against actual operation classes. Index arithmetic, floats, poison flags and
unmodeled metadata are not admitted pure primitives. Retained leaf operations
are opaque steps with exact descriptors, ordered operands and successful-result
arity contracts; they may fail and expose their terminal error/world.

Supported control is one-block functions containing nested one-block SCF for
loops with checked positive constant stride, possibly dynamic bounds, zero trips
and multiple carries. Bounds/control map to a common finite iteration sequence
under an explicit interpretation premise. The theorem is independent of trip
count and never assumes a nonempty loop. Unknown region/CFG semantics and calls
to locally defined functions refuse the whole boundary. External declarations
and calls remain explicit common partial-operation contracts.

The observation is the entire returned value vector plus persistent world on
success, or the same error and terminal world on failure. Private SSA temporary
cells are excluded. No independently numbered SSA cells are equated by fiat.
Pure deterministic TOTAL value functions have no world access. This local law
is necessary for deletion and zero-trip speculation; backend fidelity and
operation approval do not follow from a whitelist, hash or trait.

## Proof mechanism

Normalization inlines pure expression trees into every retained consumer,
terminator, observer and loop yield. Retained results introduce common logical
binders; checked shifting preserves exact captures across scopes. Opaque steps,
ordered operands, loop geometry, carries and returns remain in the normalized
code. Acceptance checks equality of complete observation skeletons and actual
static typing on both sides. `normalize_sound` derives equality from independent
operational evaluators for raw SSA and normalized code, including partial failure.
Consequently CSE needs identical actual descriptors/ordered values and available
bindings; dead chains can disappear only when no retained observation depends
on them. Loops keep opaque/effect order, and total invariant computations can be
speculated even when the iteration list is empty.

Typing is proved inductively, not reported as an inventory. Given initialized
entry slots and LOCAL retained-result count contracts, typed execution derives
result count, excludes internal arity faults, and proves every actually reached
SSA/IV/carry/capture read is initialized. It does not prove external calls
succeed, runtime value representations have their declared MLIR type, or values
are poison free. Those are separate local semantic/representation obligations.
No whole-source-success, final-equality or intermediate-typing assumption is
used in place of these proofs.

Actual chains have exact source/middle/final buffers and one common descriptor
and type registry constructed from all three programs. Independent middle
projections must agree; source/middle/final joins are also Lean checked. Component
theorems instantiate `check_sound` separately and compose via `preserves_trans`.
A separate original→final check supplies the direct route. These are distinct
proof routes in one disclosed kernel object, not a claim of separately imported
stage-object custody. No semantic renumbering is assumed from equal integers.

## Evidence and countermodels

Actual cases cover i1/i8/i16/i32/i128/i256, duplicate constants, complete dead
chains, casts, opaque reads/consumers, partial operations, separate functions,
external signatures, dynamic/nonunit/negative-lower/zero-trip loops, several
carries and nested motion. Different textual -1/255 i8 constants are compared
only after the actual parser produces the SAME canonical IntegerAttr data;
no new numerical equality law is assumed. Identity/flagged boundaries are
labeled separately from changed programs.

Compiler-valid mutations are verified by the pinned parser/module verifier
before checking: opcode/operand order, same-typed wrong values, constants,
interfaces, poison/unmodeled metadata, dropped observer inputs/effects/partial
steps, effect order, variant motion, zero-trip trapping speculation, carry/yield
order, external binding and untouched functions. Typed data mutations and
malformed/unsupported scopes have separate categories. Concrete typed
countermodels retain differing observer worlds and unsafe empty-loop speculation.

Bound actual-roster Int models prove nonzero initialized source execution and
the target result through the generic graph theorem. For example inputs 2,3 give
25; loop/carry and nested examples include ordered persistent observations.
A separate typed source model fails in an opaque partial call and preserves the
same error/world. These are concrete model executions, not approved primitive
fidelity, actual source-definedness or poison freedom. Numeric interpretations
are explicit fixture data and remove none of Merlin's old numerical gates.

The first monolithic four-iteration/two-carry closed conversion witness timed
out at 60 seconds. It is preserved UNQUALIFIED, with no raised limit or retry.
A NEW per-operation proof route uses checked local equations for every source
instruction, loop guard and iteration, composed against the SAME evaluator.
This route does not assume source success and does not repair/requalify the
old elision SIGSEGV. Earlier mutation fixture setup errors and all first proof
elaboration failures are retained as development evidence, never detections.

## Trust, review and limits

| Boundary | Established here | Remaining obligation |
|---|---|---|
| Complete actual IR | All functions/interfaces/typed use-def/control/metadata bound and checked | Pinned parser, Python complete projection and actual producer execution trust |
| Pure arithmetic | Generic deterministic total world-free primitive theorem; closed source guard | Actual primitive/representation/poison semantics implements the admitted meaning |
| Retained calls | Same complete descriptor, inputs, order and partial outcome | External/state/effect/arity contracts and runtime delivery are truthful |
| Control | Exact finite common sequence and carry/yield mapping, including empty loops | Actual SCF index/control representation satisfies the selected domain |
| SSA initialization | Every reached slot exists; returned counts; no internal arity fault | Value typing, success and absence of poison remain distinct |
| Composition | Checked joins, separate component composition and direct proof | No local-call fuel, unsupported CFG/region or foreign semantics bridge supplied |
| Kernel/replay | Actual proof objects, exact roots, std3 closure, live original-bound replay | Pinned kernel/runtime and explicit host transport trust; JSON alone grants no authority |

No native MLIR/device comparison or retained-model/full-model execution is
claimed. This packet exercises actual pinned production APIs on independently
chosen valid modules. The full LeanMLIR/native backend route remains unavailable
without a selected matching manifest; no downloads or substituted implementation
are used. Actual-model, native, definedness, numerical and compiler flags stay
false. Existing scalar-square counterexamples, policy nonapproval and all old
numerical/native failures remain separate unchanged evidence.

Review should establish the pure TOTAL/state-free primitive law, complete opaque
runtime contracts, SCF control interpretation and Python projection coverage for
the intended paper scope. Do not approve a full compiler/source-equivalence
claim from a test count or from initialized SSA typing alone.

## Fixed context, replay and delivery

The shared Std.Time.Format.Basic incident was checked ONCE against this task's
36 actual audit inventories / 707 unique proof objects. The affected module is
absent; every recorded object and executable/shared-library binding remained
exact. The result is retained. New checks use only those preexisting core pins,
not new identities; an extra/changed standard import becomes UNQUALIFIED.
No toolchain writes/rebuilds or old failed-profile retry occurred. This narrow
result does not explain the incident or establish global toolchain innocence.

The final PLAN freezes all owned source bytes, tool/profile/core pins, actual
positive/refusal roster and tests before measurement. Tests retain passing
artifacts. Actual originals are followed by ONE separate fresh-process replay
batch, each reprojecting complete original buffers and reproducing the retained
object from its exact query path. Stale and consistently rehashed source, middle,
model, query, object, tool, dependency, imported-object and audit substitutions
must refuse. The host receipt protocol itself is not claimed proved in Lean.

See sealed delivery.json / MANIFEST.json for final measured counts and exact
patch/closure identities, independent inventory verification, reproducible recipe
and paper rows for both transformations. All approval flags remain false.
