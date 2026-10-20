---
title: Compiler verification architecture
kind: design
status: current
owner: verification
last_verified: 2026-10-12
related: [verification, lowering_pipeline, capture_execution_attestation, verified_compiler_tooling, verified_compiler_roadmap]
code_refs:
  - src/merlin/verify/receipts.py
  - src/merlin/verify/scalar_ir.py
  - src/merlin/verify/outline_ir.py
  - src/merlin/verify/split_reduction.py
  - src/merlin/verify/cli.py
  - src/merlin/verify/lean_checker.py
  - src/merlin/verify/lean_export.py
  - src/merlin/verify/lean_sat_runtime.py
  - src/merlin/verify/lean/Merlin/Foundation.lean
  - src/merlin/transforms/pipeline.py
  - src/merlin/perf/whole_model_chunks.py
  - src/merlin/compile/mesh.py
  - src/merlin/targetgen/operation_numerics.py
  - src/merlin/targetgen/contraction_egraph.py
  - src/merlin/xdsl_dialects/lowering/outline.py
  - src/merlin/xdsl_dialects/lowering/dispatch_program.py
  - src/merlin/llvmlower/pipeline.py
---

# Compiler verification architecture

Merlin should verify **each actual compilation** against the exact input and output
of its transformations. This is translation validation, not a claim that every
possible run of a pass is correct. The release claim is a conjunction: frontend
faithfulness, pass preservation, placement completeness, target instruction
semantics, ABI/memory behavior, and numerical conformance. An `unsat` result for
one modeled boundary cannot discharge a different one.

The core refinement obligation is: for every input satisfying a declared domain
`D`, every defined observable behavior of the emitted program is permitted by
the source. `D` must name shapes, dtypes, layouts, aliases, initialization,
rounding, target capabilities, and memory effects. Observable behavior includes
declared model outputs and host-visible memory, not merely an accumulator value.
If source behavior is undefined, the refinement direction and the allowed target
behavior need to be stated; plain output equality is insufficient. If any side
cannot be modeled, the result is `unsupported` or `unknown`, never `verified`.

## Boundary obligations and current evidence

The [shared transformation route](../guides/verified_compiler_tooling.md) adds
host-selected Lean checking for a deliberately narrow xDSL subset. Scalar
wrapping integers use a typed Lean-MLIR interpretation. Ordered loop partition
arithmetic and buffer lifetime checks use parameterized lemmas together with
trusted correspondence of the actual emitted structure. These are explicit
local obligations, not complete operational semantics for SCF, memory, LLVM
or target instructions. Strict coverage is fixed by the frozen host profile;
unknown covered checks refuse acceptance. Independent numerical and hardware
checks still apply. The [roadmap](verified_compiler_roadmap.md) lists the
remaining models and pass groups.

Lean receipts distinguish `lean_kernel` from `lean_native_reflection`. Generated
bit-vector obligations using `bv_decide` require the original toolchain's pinned
SAT executable and explicit `allow_native_reflection` permission. Lean checks the
solver's LRAT certificate using native reflection; that mode trusts the selected
Lean compiler and native checker and admits only its two additional reflection
axioms. It is not kernel-only assurance. Missing selection, stale bytes or
unapproved axioms refuse; a candidate cannot choose this mode in a request.
Neither method proves that a final object implements a checked IR pair.

| Boundary | What must be preserved | Verification approach | Status in Merlin |
| --- | --- | --- | --- |
| PyTorch/export → model2MLIR capture | Input/weight identity, state, operator semantics, dtype, shape and numerical policy | Sealed source/data closure; independent PyTorch vs MLIR differential tests; formal rules for bounded decompositions | Capture attestation is narrower than semantic equivalence; no full-model formal proof |
| Frontend normalization → linalg | Defined init, quantization/dequantization, casts, rounding, shapes | Per-rewrite SMT or exact arithmetic theorem; differential tests for float/unsupported ops | xDSL verification and numerical tests exist; no general pass-by-pass proof |
| Linalg → outlined kernels + driver | Inlining kernels reproduces values and effects; no missed region captures, reordered effect, or dropped call | Independently reconstruct/inlining comparison, SSA and effect checks, numerical differential; SMT on bounded pure slices | Structural `module.verify()` and host parity tests; no formal effect/memory theorem |
| Driver → dispatch DAG | Same call symbol, operand/result mapping, view semantics, order and host/device transfers | Independent correspondence checker over actual driver and DAG, followed by runtime equivalence | `verify_program` proves DAG well-formedness, not semantic equivalence |
| Linalg → interface | Same typed function over shared symbolic inputs | Existing QF_BV translation validator, replay-bound source/target/tool receipts | Implemented for narrow concrete rank-2 integer subset only |
| Interface → OOT target IR/command buffer | Value semantics, legal placement, no unaccounted op or host recomputation, physical layouts | SMT for modeled integer commands; operation-denominator and placement audit; independent device execution | In-tree interface→command-buffer validator exists; target IR, physical packing and full OOT compiler are not proved |
| Scheduling / DMA / memory planning | Dependency order, alias/lifetime safety, waits, bank/resource legality and forward progress | Bounded state-machine model checking or SMT happens-before proof, with target facts and RTL conformance; stress variable DMA latency | Structural checks and simulator/RTL tests, not a universal scheduler proof |
| LLVM-dialect MLIR → LLVM IR → object | ABI, pointer/address-space and alignment semantics, instruction mapping, no new undefined behavior | MLIR verifier at every pass; source/target semantic checks for Merlin rewrites; Alive2 on supported LLVM-IR optimizations; disassembly/ISA and execution checks for machine code | IR and host/device differential checks; no end-to-end formal machine-code theorem |
| Host/device runtime → model output | Transfer ownership, coherence, invocation order, output completeness and numerical acceptance | Compositional boundary receipts plus representative kernel and held-out model execution | Numerical/hardware gates exist; a successful model is a witness, not a theorem for arbitrary models |

`src/merlin/verify/receipts.py` currently supports only `linalg_to_interface` and
`interface_to_command_buffer`. It binds actual before/after hashes, typed
signatures, semantics-encoder hash, `mlir-translate` hash, Z3 version and SMT
query. `qualify_receipt` replays the obligation. The source encoder requires a
defined `linalg.fill` accumulator; it refuses bare `tensor.empty`. This proves
all integer input bit patterns **at one concrete shape under the encoded value
semantics**, not packing, aliasing, DMA, floats, hardware, or arbitrary models.
The saved OOT `merlin_iface` text bridge has the same narrow scope. An IR verifier
checks syntax/types and is not a semantic proof.

The existing `contract.prove` token verifies a matching requirement string, not
the requirement's truth; `src/merlin/verify/proofs.py` correctly distinguishes
`asserted` from `verified`. A pass invocation log proves reachability, not
correctness. Likewise, a source-to-preprocessed result-type map checks type
correspondence but not value preservation.

## E-graphs: optimization search, not automatic proof

The real-contraction e-graph in `src/merlin/targetgen/contraction_egraph.py`
retains a `linalg.generic` and a microkernel `func.call` in one e-class, then
extracts a lower-cost choice. The type checker establishes both alternatives
have the same result type; it **does not establish that the call implements the
contraction**. Its PDL rule also cannot express all legality conditions. Until
the call's exact implementation and preconditions have a certificate, this is a
candidate-selection mechanism, not a formally justified equivalence class.

The proposed rule admission protocol is:

1. Record a rewrite's source/target semantics, side conditions and observable
   effects. Separate exact integer equality from FP reassociation or a stated
   error bound. Never assert equality solely from a cost, common type, or op name.
2. Prove the rule on bounded concrete typed domains with SMT, or provide a
   reviewed proof of a parameterized rule. A hardware primitive additionally
   needs an ISA/RTL conformance link; an abstract call model is an assumption.
3. Apply only after checking its side conditions on the exact IR, including
   shape, layout, aliasing, accumulator initialization, quantization and target
   capability. Save the matched IR and the rule/condition digests.
4. Extract a candidate, then translation-validate the **actual emitted
   candidate** against the original source. Replay the certificate when the
   compiler, rule, model or target facts change.

Keep e-graphs bounded to pure regions until effects, memory and control-flow
semantics are modeled. Distinct algorithms may be mathematically equal yet
observationally different under IEEE floating-point reassociation. Do not
insert a host/device call and a linalg op into the same *proved* class merely
because small sampled tests match.

## Executable first pilot: split-K arithmetic

`merlin.verify.split_reduction.verify_split_reduction` builds both exact
bit-vector expressions over the **same** symbolic signed input pairs. The
source accumulates in its declared output width. The candidate accumulates
each chunk in a narrower partial width, sign-extends each partial result and
adds it to the output. Z3 checks whether any input makes them differ:

```sh
merlin-verify split-reduction --k 72 --operand-width 8 --partial-width 20 \
  --output-width 32 --chunks 31,31,10
python -m pytest -q merlin/tests/ir/test_split_reduction.py
```

The command prints the verdict, any counterexample, and the conservative
`maximum_safe_chunk` for the widths. Exit 0 is a proof, 1 a counterexample, 2 a
solver abstention or an invalid partition.

The test proves a safe 3-term split for every 4-bit input pair at that shape,
refutes an overflowing narrower partial accumulator with a counterexample,
and rejects an omitted tail before solving. The widths are parameters. For
signed 8-bit products and a signed 20-bit partial accumulator, the conservative
full-domain bound is 31 products per chunk; a K=72 source therefore needs at
least three such chunks *if* its output accumulation width and actual device
instruction semantics match this model. This is an **algebraic pilot**, not a
proof that a generated kernel split K correctly or that hardware implements
20-bit signed accumulation. The next gate must parse the exact source and
emitted target IR, derive its chunks and widths from target facts, and bind the
proof receipt to those bytes; then run the target conformance witness.

### First actual-transform receipt: integer function chunking

`merlin.perf.whole_model_chunks.chunk_forward` is an exercised Merlin pass:
it turns one large SSA function into sequential helper calls (rematerializing
cheap producers after each call). The
`merlin.verify.scalar_ir.verify_scalar_transform` pilot takes the printed
before-pass and after-pass **bytes** from that invocation, reparses both,
verifies the modules, and symbolically executes the entry over shared
bit-vector arguments. It follows the emitted helper call graph; it does not
trust the pass's live-out bookkeeping. It then proves no return differs for
any input at the concrete integer widths. The receipt binds both byte hashes,
the verifier/parser hashes, xDSL and Z3 versions, query digest and verdict;
qualification reruns it on the supplied bytes. A changed helper arithmetic
instruction produces a counterexample and changed target bytes invalidate
an older receipt:

```sh
python -m pytest -q merlin/tests/ir/test_scalar_ir.py
```

This pilot accepts only pure, single-block scalar integer functions with
`arith.constant`, unflagged `arith.addi/subi/muli`, calls to defined functions
and `func.return`. It rejects unknown operations, external calls, recursion,
overflow flags, unmodeled attributes, floating point and effects. It covers
one emitted transformation at one concrete signature, not all uses of
`chunk_forward`, tensor/memref chunking, LLVM lowering, or the generated
object. The separate split-K theorem above is **not** attributed to
`chunk_forward`: that pass does not split a reduction. A future split-K pass
must expose its exact before/after IR and bind its emitted chunk boundaries,
widths and target primitive semantics to the algebraic obligation.
The mesh runtime already splits one such case: when a selected integer policy
declares a signed MAC result narrower than the full reduction needs,
`merlin.compile.mesh.run_matmul_on_mesh` derives the per-dispatch K bound from
`merlin.targetgen.operation_numerics.integer_split_k_limit`, aggregates the
partial results exactly in i64 checked against i32, and records an
`integer_split_k` witness naming every primitive call. The witness states the
split it performed; it does not prove the primitive's arithmetic.

### Actual outlining boundary: syntactic call-expansion proof

`merlin.verify.outline_ir.verify_outline_transform` checks the **printed input
and output of the exercised** `outline_dispatches` pass. It parses and verifies
both modules independently, expands the emitted private kernel calls, and
compares typed SSA result graphs including each `linalg.matmul` region body,
operands, result positions, attributes and properties. The receipt binds exact
before/after bytes, checker/parser versions and both graph hashes; replay
requires the same bytes and checker. A changed call operand produces a
`mismatch` witness identifying the differing graph path. This is a syntactic
congruence proof: if unchanged pure MLIR primitives have their declared
deterministic semantics, identical expanded graphs have identical results.
It does **not** independently prove matmul arithmetic, provide a numerical
counterexample for a mismatch, or prove memory/ABI/device behavior.

The current domain is single-block, call-free source `@forward` and a target
with only reachable, private `$kernel_` helpers. It accepts statically shaped,
unencoded integer-tensor `linalg.matmul` with explicitly supplied accumulator
tensors, scalar integer constants/arithmetic in the matmul body, and no
effectful operations. It abstains on `tensor.empty`/`linalg.fill`, generic
regions, captures, aliases, floats, quantization, unknown operations and
function metadata it cannot account for. This leaves the common initialized
matmul outline with an explicit fill **unproved** until the checker can model
full-overwrite initialization and cloned producers. The test exercises one-
and two-dispatch actual pass output and a type-correct, wrong operand rewiring:

```sh
python -m pytest -q merlin/tests/ir/test_outline_ir.py
```

## Qualification sequence

1. Freeze exact capture, SW spec, compiler/OOT, LLVM/CIRCT, RTL and simulator
   revisions, with materialized source and emitted-stage byte hashes. A version
   string or historical unsigned receipt is not a source-execution proof.
2. Enumerate every exercised source op, dtype, shape, layout, alias/effect and
   placement row. Make unsupported rows and missing tool/timeout states visible.
3. At each Merlin-authored boundary, run the applicable formal validator on the
   actual before/after artifacts, plus an independent numerical oracle. A
   counterexample becomes a new development reproduction, not a hidden holdout
   leaked into agent feedback.
4. At the OOT seam, check that every supported accelerator opportunity is
   actually offloaded and every other op has a declared host path. Prove or test
   transfers, packed layout, DMA ordering and output visibility separately from
   abstract arithmetic equality.
5. Validate LLVM optimizations only where the tool's IR semantics apply; check
   generated object/ELF ISA and ABI against the selected target, then run
   representative kernels on simulator/RTL and periodically run sealed held-out
   full models. The last checks catch assumptions in the semantic model and
   unmodeled integration behavior; they do not generalize by themselves.

Release evidence should contain a per-boundary coverage matrix: `verified`,
`refuted`, `unsupported`, `unknown`, `unavailable`, or `not_executed`, each
referencing exact artifacts and assumptions. A compiler is not fully certified
while any required row is unproved or unexecuted. Formal per-shape theorems,
property/differential tests, RTL conformance and blind model validation remain
different kinds of evidence; none substitutes for another.

## Tooling notes

MLIR's [pass manager](https://mlir.llvm.org/docs/PassManagement/) can run a
verifier after each pass and save pass-local crash reproducers. These establish
IR validity/reproducibility, not value preservation. [Alive2](https://github.com/AliveToolkit/alive2/blob/master/README.md)
performs bounded translation validation of supported LLVM IR optimizations;
its own documentation warns against treating it as an interprocedural proof,
and it does not prove MLIR-to-machine-code lowering. [Equality saturation](https://arxiv.org/abs/2004.03082)
retains alternatives and selects by cost; the correctness claim still depends
on valid rewrite rules and a faithful extraction. MLIR's [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
can describe and constrain transformations, but its handle/IR checks are not
the semantic equality relation required above.
