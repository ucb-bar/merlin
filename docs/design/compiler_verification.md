---
title: "Verifying the compiler: a working log"
kind: design
status: draft
owner: core
last_verified: 2026-09-05
related: [dialect_test_bar, derived_capsule_axes, rtl_derived_compiler_tooling, compiler_plane]
code_refs: [merlin/python/merlin/xdsl_dialects/lowering/passes.py, merlin/python/merlin/xdsl_dialects/contract.py, merlin/python/merlin/targetgen/rtl_check_compiler.py, merlin/python/merlin/targetgen/conformance.py, merlin/python/merlin/targetgen/capability_manifests.py, build_tools/scripts/check_pass_obligations.py]
---

# Verifying the compiler: a working log

**This document is a living log, not a finished design.** It is written as the work happens, so that
what ends up in a paper is the thing that was actually built and measured rather than a
reconstruction. Entries are dated and append-only; when a claim here turns out to be wrong, the entry
is corrected in place *and* the correction is recorded in the log, because the wrong turns are part
of the result.

It answers one question: **how do we know a compiler pass is correct?**

---

## 1. The question, and why the existing answer is incomplete

The capsule bench grades *outcomes*. A capsule is compiled by the backend under test, and the result
is checked against an independent golden, against a decoded instruction stream, and finally against
cycle-accurate RTL. That machinery is careful — a tier that could not run leaves a capsule
`incomplete` rather than `pass`, certificates are bound to the digest of the bytes that earned them,
and every RTL-derived literal is fail-closed. See [the cross-target dialect test bar](dialect_test_bar.md).

But grading outcomes is not the same as verifying passes, and the gap is easy to state:

> If a capsule passes, we know *this program* compiled correctly *this time* on *this target*.
> We do not know that the pass which tiled it is correct — only that its output happened to agree
> with a golden on the shapes we happened to run.

That is exactly the objection a programming-languages reader will raise, and it is fair. Two findings
from the September 2026 audit make it concrete, and both are about things the repo already declares
but never checks.

### Finding 1 — `contract.prove` is not a proof

The `contract` dialect (`merlin/python/merlin/xdsl_dialects/contract.py`) is, by design, a dialect of
obligations: `contract.assume`, `contract.require`, `contract.prove`, producing a
`!contract.proof<"requirement">` token. It reads like a proof system.

Its verifier checks that the proof token's requirement *string* equals the op's requirement *string*.
That is the entire check. A "discharged obligation" in our IR today is one string matching another
string. Nothing establishes that the property holds.

### Finding 2 — `compiler_obligations` has no consumer

Every capability manifest carries a per-target list of what that target's compiler must do. These are
derived, reviewed, and *required* by the manifest validator (`capability_manifests.py`), and they are
exactly the right granularity:

| target | `compiler_obligations` |
|---|---|
| `gemmini` | `must_tile_to_mesh_shape`, `must_commit_accumulator_before_reuse`, `must_prove_rhs_immutable_for_residency`, `must_respect_scratchpad_capacity` |
| `atlas`, `mx_gemmini` | `must_tile_to_mesh_shape`, `must_supply_e8m0_block_scales` |
| `radiance` | `must_map_to_warps` |

Grep for consumers and you find one docstring mention in `target_lowering.py`. Nothing tests them.
The obligations are stated and then dropped.

### What *is* already checked

To be fair to the existing system, two things are genuinely measured and should not be re-invented:

- **Pass invocation.** `build_tools/scripts/check_pass_obligations.py` reads a JSONL log written by
  `passes.install_pass_recorder()` and reports passes that are catalogued but never reached, or
  reached but transform nothing. With no log it exits 2 — "cannot decide" — never 0. That
  UNMEASURED-is-not-clean discipline is the model everything below follows.
- **The emitted instruction stream.** `rtl_check_compiler.py` / `rtl_check_runner.py` compile
  RTL-derived facts into FileCheck assertions over the *decoded* instruction stream. Tile counts,
  opcode legality, instruction-class coverage and ordering are all checked against literals derived
  from the target's own elaborated RTL.

So the missing layer is specifically **the pass in isolation**: nothing runs a single pass on a
single input and asserts what it should have done, and nothing states what any pass preserves.

---

## 2. Vocabulary (this is the part that must be precise)

"Capsule" is not a term of art in MLIR or in PL, and using it without definition invites exactly the
ambiguity it has caused. The repo's own code already implements the right distinction; it simply was
never named. The vocabulary below is the naming, not a change.

**Obligation.** A parameterized statement of what the compiler must do, quantified over shapes.
Implemented as `conformance.Cell` — the tuple `(semantic_family, dtype, tile_alignment)`, plus the
derived axes in [derived capsule axes](derived_capsule_axes.md) (shape geometry, rank, memory regime,
epilogue, lane). `semantic_family` ranges over five primitives (`contraction`, `reduction`,
`elementwise_map`, `movement`, `synchronization`) and three composites (`attention`, `normalization`,
`softmax`). An obligation is *derived*, never written down: `conformance.spec(target)` computes
`required = admitted ∩ observed`, where `admitted` comes from the target's capability manifest and
`observed` from a family census over real captured models, with tile and block-scale boundaries read
from the target's RTL facts.

**Witness.** One concrete instance of an obligation at one concrete shape: a compilable program, a
stimulus, an independent golden, and the evidence required of it.

**Capsule.** *The on-disk encoding of a witness* — a directory holding `capsule.yaml`,
`capsule.interface.mlir` (the program the backend under test compiles), and a withheld golden. That
is all the word means. Where this document or a paper needs precision, it says obligation or witness.

**Evidence vector.** A witness is not checked once. It carries several independent checks at
different fidelities, and the fidelity ladder is `L0`–`L5`:

| tier | what it establishes |
|---|---|
| `L0` | the emitted command buffer reproduces an independent golden |
| `L1` | the command buffer is internally consistent (reference == simulate) |
| *trace* | the decoded instruction stream issues the required instruction classes (not an L-tier) |
| `L2` | a functional or RTL-derived model agrees |
| `L3`–`L5` | elaborated RTL, cycle-accurate (Verilator / GSIM / VCS / FireSim) |

A tier index is a *fidelity*, not a simulator: one target's `L3` is Verilator, another's is an
RTL-derived model, and each result records which. A mandatory tier that could not run leaves the
witness `incomplete`, never `pass`.

**Obligation family vs. witness.** The generalization claim is at the obligation level: a single
witness checks one instance at several fidelities; the *set* of witnesses covering an obligation is
what shows the pass generalizes across shapes and edge cases. Held-out witnesses are drawn from
obligations the public set already covers, at shapes it does not.

### Terminology hazards

These are recorded so they do not leak into external writing. Four different things in this repo are
called a **family**: `semantic_family` (the computation class), the compute-unit kind profile, the
performance-claim family, and a sweep id. **Pass** is triply overloaded: a witness *passes*, a
compiler *pass* runs, and `pass_requirements` names obligation classes. **L0–L5** is two unrelated
ladders — oracle fidelity tiers, and performance optimization levels. And **cell** is the obligation
while **capsule** is the witness; they are not synonyms.

---

## 3. The verification model: three layers

| layer | question | cost | established practice it follows |
|---|---|---|---|
| **static** | did this pass do the structural thing it exists to do, on this input? | milliseconds | `lit` + `FileCheck`, the standard MLIR pass-test flow |
| **formal** | is this pass semantics-preserving, for all inputs at this shape? | seconds to verify; refutation costs far more and is shape-bounded | translation validation against an SMT encoding |
| **dynamic** | does the compiled program match an independent golden on real hardware? | minutes–hours | the capsule oracle ladder, as today |

The layers are complementary and none subsumes another. The static layer is cheap enough to run on
every commit and catches structural regressions immediately; it says nothing about arithmetic. The
formal layer proves the arithmetic at a bounded shape; it says nothing about the hardware. The
dynamic layer is the only one that touches RTL; it is far too expensive to cover a shape space.

The formal layer follows **"First-Class Verification Dialects for MLIR"** (Fehr, Fan, Pompougnac,
Regehr, Grosser; PLDI 2025), which gives MLIR dialects a formal semantics by *lowering them to
semantic dialects* that bottom out in SMT-LIB, and then builds semantics-agnostic tools on top —
translation validation, `pdl` rewrite verification, and dataflow transfer-function soundness. The fit
is close because that work is built on xDSL, which is already our IR framework, and because its `smt`
dialect is now upstream in both MLIR and xDSL. Section 5 records exactly which of its three tools we
adopt and which we do not, with reasons.

### What may be FileCheck'd, and what may not

This constraint is load-bearing and is easy to get wrong twice, so it is written down.

An earlier version of the RTL check compiler matched op mnemonics over the *generated target
dialect's* MLIR text. It was corroborated against 383 real backend-generation runs and **removed**:
a generated out-of-tree dialect invents its own op names per run, so those patterns have no
derivation source, and they false-failed on legal MLIR surface forms while the decoded instruction
stream never did.

The descent makes the reason structural. Exactly one layer is unstable:

```
linalg → contract → schedule → interface → ⟨ target dialect: UNSTABLE ⟩ → runtime → command buffer
         └──────── stable, in-tree, bare namespaces ────────┘             └── stable, in-tree ──┘
```

The generated dialect is *sandwiched* between two stable surfaces. Every structural property worth
checking is visible on one or both, so nothing needs to name a generated mnemonic.

| surface | stable? | checked by |
|---|---|---|
| `contract.*`, `schedule.*`, `interface.*`, `runtime.*`, `dse.*` — hand-authored xDSL dialects, fixed namespaces | yes | the new static layer |
| `merlin_iface.*` — frozen grammar with an IRDL definition (`merlin/contract/merlin_iface.irdl.mlir`) | yes | the new static layer, via `mlir-opt --irdl-file` |
| the generated target dialect | **no** — per-run invention | deliberately nothing |
| decoded instruction stream / kernel words | yes, RTL-derived | the existing check compiler, unchanged |

For the target-lowering obligation specifically, the MLIR-surface row is **deliberately empty**. That
is the honest consequence of the decision above, and it is stated here so nobody re-introduces
mnemonic matching in six months.

---

## 4. How pass tests are derived rather than written

The same principle that governs the capsule corpus governs the pass tests: a fact about a target is
extracted from that target's own sources, never written down. A pass test is generated from a triple,
every element of which already exists and is derived:

```
capability manifest              conformance cell                 pass catalog
compiler_obligations       ×     (family, dtype, alignment)   ×   PassInfo.obligation
"what this compiler         "the shape class it must         "which pass owes
 must do"                    do it for"                       this obligation"
```

The pass catalog closes the third axis with four obligations and nothing else:
`partition/eligibility`, `target transformation`, `target lowering`, `boundary materialization`.
`corpus_synth.pass_requirements_for()` already derives which of these a given shape demands.

Mapping the manifest obligations onto checkable properties, on the stable surfaces only:

| manifest obligation | surface | what is checked | derived from |
|---|---|---|---|
| `must_tile_to_mesh_shape` | `runtime`, `interface` | tile count `⌈M/edge⌉·⌈N/edge⌉` | the mesh edge in RTL facts — dropped if the edge is a software default rather than a hardware fact |
| `must_commit_accumulator_before_reuse` | `interface` | exactly one commit per named output; no commit between accumulations | the frozen interface grammar |
| `must_prove_rhs_immutable_for_residency` | `contract`, `interface` | an immutability assumption reaches the pack; no second pack; evict after last use | the interface grammar and the contract dialect's registered predicates |
| `must_supply_e8m0_block_scales` | `interface` | a scale operand per reduction group | the target's declared block-scale quantum |
| `must_map_to_warps` | `schedule` | placement present, lane count matching the declared geometry | the manifest's warp geometry |
| `must_respect_scratchpad_capacity` | — | **not FileCheck-able** — a numeric bound, already covered by the existing numeric screen | — |

The last row matters as much as the others. A check that cannot be grounded is **omitted and
recorded**, never defaulted. The generator reports `emitted / omitted / reason`, so coverage is a
measured number rather than a claim — the same discipline the RTL check compiler already applies when
it flags its one ungrounded axis instead of presenting it as rigorous.

### Coverage is bounded by derivability, and that is uneven today

Stated plainly, because the alternative is an empty suite that looks green:

| target | RTL facts cache | capability manifest |
|---|---|---|
| `gemmini` | rich (~55 KB) | yes, 4 obligations |
| `atlas` | rich (~40 KB) | yes, 2 |
| `radiance` | thin — ISA classes only | yes, 1 |
| `mx_gemmini` | **empty facts block** | yes, 2 |
| `saturn_opu` | **empty facts block** | **no** — fails closed, no capability residual |
| `saturn_opu_rvv` | **empty facts block** | **no** — same |

An empty facts block is a silent extractor failure, not a derivation limit; it is fixed by re-running
elaboration, not by weakening the generator. This is why the design layers two tiers: the
**obligation-derived** tier needs only the capability manifest and the conformance cells and so covers
four targets today, while the **facts-grounded** tier layers on where facts exist. Both record what
they dropped and why.

---

## 5. The formal layer, and which of the paper's three tools apply

**Translation validation — adopted.** Take the module before and after a pass, give both a semantics
by lowering to the `smt` dialect, assert the negation of a refinement relation, and solve. `unsat`
means the pass preserved semantics on that input; `sat` yields a counterexample. The chosen pass pair
is `interface → runtime`, which straddles the unstable generated dialect: it validates a generated
lowering without ever naming its ops.

The obligation is **quantifier-free**, which is the whole tractability story. Because the conformance
cells supply *concrete* extents, dimensions are never symbolic — only element values are. The query
is `unsat(∃ inputs. out_src ≠ out_tgt)`, in QF_BV.

It is also the right tool for our situation in a way `pdl` is not, for a reason that only became clear
on inspection: **the tile/accumulate rewrite is not an in-tree pass.** `interface.accumulate` is
declared in the interface dialect and has no producer anywhere in the lowering package; the K-tiling
is emitted by the generated out-of-tree backend, which differs on every run. Translation validation
validates a compilation *instance* regardless of who wrote the compiler, so it covers code we did not
write. That is a weaker theorem (per compilation, not per rewrite) but a wider one.

**`pdl` once-and-for-all rewrite verification — deferred, with a prerequisite.** Verifying a rewrite
for all inputs up to a bitwidth proves something about a rewrite *we own*. We do not own this one. The
correct sequence is to first move tiling in-tree as a real catalogued pass expressed as a `pdl`
pattern — independently worthwhile, since it converts a per-run agent artifact into a merlin-owned
pass with an obligation — and only then verify it. xDSL ships `pdl` and can apply patterns; the
`pdl`-to-SMT lowering is the piece that lives only in the paper's own tooling.

**Dataflow transfer-function soundness — does not apply, and we say so.** The paper's third tool
proves transfer functions sound against a Galois connection. We checked: the cross-op checks in the
lowering package are syntactic def-use and ordering walks, the one fixpoint is a reachability closure,
and the only genuine abstract interpretation in the tree is a frontend index analysis, not a
compiler-pass obligation. There is no abstract domain here, so there is nothing to prove sound.
Claiming this pillar would be imitating the paper's shape rather than its method.

**Floating point is verified structurally, not bit-exactly — on purpose.** For a float datapath,
reassociation is a *legal* backend choice; the dialect test bar already establishes this, and it is
why a float capsule's golden is a tolerance rather than an equality. A bit-exact float refinement
check would therefore reject *correct* backends: it is not merely expensive, it is the wrong
specification. Float contractions are encoded as an uninterpreted function over tiles, and what is
verified is structure — which contributions reach which accumulator, exactly once. That covers the
entire class of tiling bugs (a double-counted tile, a dropped K tail, a wrong accumulator) without
any float arithmetic. Integer datapaths get the real bitvector encoding.

---

## 6. Working log

### 2026-09-04 — baseline audit, and the toolchain turns out to be already present

The audit findings are in §1. The pleasant surprise is that essentially none of this needs new
infrastructure; the following were each verified by execution, not assumed.

**The SMT chain works today, with no external verification package.** xDSL 0.68 ships the `smt`
dialect, and its printed syntax is accepted by upstream MLIR's own SMT-LIB exporter, which is present
in the in-tree LLVM build. Constructing a module in Python, printing it, exporting it, and solving it
with the installed z3 works end to end:

```
xDSL smt module  →  mlir-translate --export-smtlib  →  z3
```

```mlir
builtin.module {
  smt.solver() : () -> () {
    %0 = smt.declare_fun "x" : !smt.bv<8>
    %1 = smt.bv.constant #smt.bv<3> : !smt.bv<8>
    %2 = smt.eq %0, %1 : !smt.bv<8>
    smt.assert %2
    smt.yield
  }
}
```
exports to
```smtlib
; solver scope 0
(declare-const x (_ BitVec 8))
(assert (let ((tmp (= x #x03))) tmp))
(reset)
```
and solves to `sat`, model `x = 3`.

This matters for how the work can be described: our SMT-LIB emission is *upstream MLIR's own
`ExportSMTLIB` translation*, not a bespoke encoder. Two practical notes, both learned the hard way in
the ten minutes it took to run the above:

- xDSL 0.68 has **no `smt.solver` / `smt.check` op** — the dialect is there, the solver scope is not.
  The exporter needs the scope, so a local wrapper op (a region terminated by the existing
  `smt.yield`) is required. It is about twenty lines. `smt.check` is not needed; the solver API
  supplies the check.
- The exporter emits a trailing `(reset)` per solver scope. Handed to z3 verbatim, the query is `sat`
  but the model comes back **empty** — the reset has already discarded it. Strip the trailing reset
  before solving, or the counterexample silently disappears. This is precisely the failure shape the
  repo has a standing rule about: a check that ran, reported success, and told you nothing.

**A frozen dialect can be verified with no new code.** The in-tree `mlir-opt` accepts
`--irdl-file=`, so `merlin/contract/merlin_iface.irdl.mlir` — the IRDL definition of the frozen
interface grammar — can be registered into a real `mlir-opt` with one flag. That gives an
IRDL-grounded structural verifier for the one dialect that is contractually frozen, and makes
negative tests (a malformed interface module must be *rejected*) trivial to express.

**`FileCheck` and `llvm-lit` are already in the tree**, in the LLVM build under `third_party/`. The
existing RTL check runner already resolves `FileCheck` through a candidate list; the lit harness
reuses that resolver rather than adding a second one. No PyPI `lit` or `filecheck` package is needed.

**What is missing is smaller than expected.** There is no `merlin-opt`: no tool registers merlin's
dialects with an opt-style driver, which is why every `// RUN:` line currently in the tree names a
tool that does not exist. There is no `lit.cfg` anywhere. And the one lit-shaped asset — eight `.mlir`
files under the target-generation eval datasets — names passes that were never written and is "run"
by a harness that admits in its own docstring that it only counts files. That last one is the failure
class this repo keeps re-encountering: a check that could not run, reporting success.

**Corrections to earlier drafts of this plan, recorded rather than quietly fixed:**
- An earlier draft proposed depending on the paper's external verification package for SMT-LIB
  emission and solving. Unnecessary — upstream `mlir-translate` and z3 cover it. That package is
  needed only for the `pdl`-to-SMT lowering, which is deferred anyway.
- An earlier draft proposed checking the tile/accumulate rewrite as an in-tree pass. It is not one;
  `interface.accumulate` has no producer. Corrected in §5.
- An earlier draft listed pysmt as a candidate solver interface. Dropped — z3 consumes the exporter's
  output directly, and a second solver abstraction is a second thing to keep in sync.

*Next entry will cover `merlin-opt` and the first executable lit suite.*


### 2026-09-04 (later) — the three layers exist and are measured

All three layers now run. What follows is measured, not projected; commands to reproduce are in §7.

**Built.** `merlin-opt` (`merlin/python/merlin/xdsl_dialects/opt.py`), an xDSL opt-style driver that
reflects over the pass catalog — 10 of the 12 catalogued transforms are registered as `-p` passes, and
the other 2 are reported as unregistrable *with the reason*: they consume a serialized dispatch
program rather than a module, which their declared dialects (`func -> <dispatch-program>`) already
said. A tool that silently exposed 10 of 12 would be indistinguishable from one that had 10.

A `lit` suite (`merlin/tests/data/lit`, driven by `merlin/tests/ir/test_lit_suite.py`) with six seed
tests: core-dialect lowering through `merlin-opt`, and frozen-grammar conformance through upstream
`mlir-opt --irdl-file`. Runs in **0.27 s**.

The SMT chain, and translation validation on top of it (`merlin/python/merlin/verify/`). The real
output of `merlin-materialize-interface` is verified — `unsat`, i.e. it computes the declared
contraction for **every input** at that shape.

The derived per-target generator (`merlin/python/merlin/targetgen/lit_check_compiler.py`,
`lit_suite.py`), which compiles each target's declared `compiler_obligations` into checks or into
recorded omissions.

**The detection matrix** (`merlin.verify.evaluate`, 4x4x4, eight seeded faults, solver bound 60 s;
every number below is read from `out/artifacts/verification/v1/latest/detection_matrix.json`):

| fault | static | formal | dynamic |
|---|---|---|---|
| miswired commit | miss (3.1 ms) | **DETECTED** (3.93 s) | **DETECTED** (10.7 ms) |
| swapped matmul operands | miss (5.1 ms) | **DETECTED** (3.57 s) | **DETECTED** (7.1 ms) |
| dropped activation | miss (2.9 ms) | **DETECTED** (4.73 s) | **DETECTED** (7.4 ms) |
| dropped evict | **DETECTED** (4.4 ms) | miss | miss |
| evict before last use | **DETECTED** (6.0 ms) | miss | miss |
| duplicate pack | **DETECTED** (4.7 ms) | miss | **DETECTED** (10.7 ms) |
| duplicate commit | **DETECTED** (5.0 ms) | *abstained* | **DETECTED** (9.6 ms) |
| commit after reuse | **DETECTED** (5.9 ms) | miss | miss |

No layer flags the unmutated program. Every fault is caught by something. Three faults —
`dropped_evict`, `evict_before_last_use` and `commit_after_reuse` — are caught by the **static layer
alone**: they are lifetime and ordering defects that change no computed value, so neither a
refinement check nor a numeric golden can see them. That is the non-redundancy result, and it is what
justifies the cheapest layer existing. The RTL tiers are recorded as `not_measured` rather than
estimated.

*abstained* is a third state, not a miss: on `duplicate_commit` the SMT **encoder** refused the
program ("3 commits but only 2 activation arguments") and the solver never ran. Recording that as a
miss would have credited the formal layer with looking and finding nothing, which it did not do.

**An honest negative, and it is worse than first written.** At 4x4x4 the formal layer catches nothing
the dynamic golden misses, and costs 338x the static layer. Its distinct value is the *quantifier* —
all inputs versus one stimulus — and that value is real but **not yet demonstrated as a detection
difference**. It is easy to see where it would bite: the default stimulus is degenerate (measured: an
8x8 activation has 64 elements and only **4 distinct values**, the known period-4 issue), so a fault
that only manifests on values outside that set would evade the dynamic layer entirely while the
formal layer refutes it. Building such a fault is open work; until it exists, the claim for the
formal layer is scope, not extra detections.

The stronger caveat is **shape**, and it was missing from earlier revisions of this section. Re-running
the identical matrix at 16x16x16 — one gemmini mesh tile, i.e. a *real* hardware shape rather than a
toy one — the formal column goes to **zero detections**: all three numeric faults return `unknown`
after 73–88 s against a 60 s bound, independently reproduced at 72.5 / 78.8 / 77.3 s. So "the formal
layer catches numeric fault X" is established at 4x4x4 and **not established at a mesh tile**.

**Verification cost is not refutation cost.** This is the distinction the scaling table below does
*not* measure, and the reason it must not be cited as evidence that the formal layer is usable at a
mesh tile:

| direction | question | at 16x16x16 |
|---|---|---|
| verification (`unsat`) | is this correct program correct for all inputs? | ~1.8–3.8 s |
| refutation (`sat`) | here is a broken program — produce a counterexample | `unknown` at 60 s |

Every point in the scaling table is a **correct** program, so the curve prices verification only.
Nothing in the harness has yet measured how refutation scales.

**Formal-layer VERIFICATION cost.** All `unsat`, all correct programs; time is the full
lower-encode-export-solve loop. Absolute seconds move between runs; cite the curve, not a cell.

| shape | time | note |
|---|---|---|
| 8x8x8 | 0.32 s | |
| 16x16x16 | 3.76 s | one gemmini mesh tile |
| 16x32x16 | 4.47 s | two K tiles |
| 32x32x32 | 18.27 s | one atlas mesh tile |
| 64x16x64 | 34.11 s | |

**Obligation coverage: 2 of 11 declared obligations across six targets are now checked; the baseline
was 0 of 11**, because nothing consumed `compiler_obligations` at all. The other nine each carry a
recorded reason, per target, in `out/artifacts/verify/lit/<target>/coverage.json`:

| target | declared | checked | why the rest are not |
|---|---|---|---|
| gemmini | 4 | 2 | tiling is not in-tree (below); scratchpad capacity is a numeric bound, not a structural one |
| atlas | 2 | 0 | same tiling reason; block scales are not representable on the interface plane |
| mx_gemmini | 2 | 0 | empty RTL facts block — mesh edge not derivable |
| radiance | 1 | 0 | no warp/lane geometry declared in the capability manifest |
| saturn_opu, saturn_opu_rvv | 1 each | 0 | no capability residual: fails closed |

### Three findings the layer produced before it was finished

**1. The frozen grammar cannot currently verify a real capsule, for two independent reasons.**

`merlin_iface.irdl.mlir` is generated by `tblgen-to-irdl`, which writes a custom type's symbol name
with its sigil included — `irdl.type @"!resident"`. That name is unspellable in MLIR text. Measured on
both the in-tree LLVM build and install trees (identical, 23.0.0git), in generic assembly syntax:

| IRDL | valid module | `evict` handed a tensor | undeclared `!merlin_iface.NOT_A_REAL_TYPE` |
|---|---|---|---|
| tracked (sigil) | **rejected** | rejected | rejected |
| sigil stripped | accepted | **rejected** | **rejected** |

So the tracked IRDL rejects everything, valid modules included; stripping the sigil from the
declarations and the `::@` references makes the constraints genuinely bite — an `evict` given a tensor
fails with *"expected base type 'merlin_iface.resident' but got 'builtin.tensor'"*. That is a real,
two-substitution fix, in a generator this work does not own.

The second reason is worse, and was found by a concurrent session challenging the first result.
Capsule interface files are written in **custom (pretty) assembly** — `%W = merlin_iface.tensor {…} :
tensor<…>` — and an IRDL-registered dialect has **no custom parser**; only the generic form parses.
Run against a real capsule, `mlir-opt --irdl-file` exits 1 with **completely empty stderr**: no
diagnostic, on either IRDL. A silent non-zero exit is the worst of the three outcomes, because a
harness that only checks "did it pass" would read it as a working check.

Consequently the generator's own claim — that a C++ out-of-tree tool can parse the frozen
`*.interface.mlir` grammar with zero hand-written dialect code — does not hold today. The seed and
generated pass tests are written in generic form, which is why they do enforce; the tracked capsule
corpus is not checkable this way as it stands. Closing that needs capsules emitted in generic form, a
real dialect with a parser, or an accepted and documented limitation.

A small confirmation that the layer does real work, found by running it: the positive test in this suite was
written with a bare `!merlin_iface.acc`, and the IRDL rejected it — that type carries the accumulator's
element width, and the bare spelling drops it. The layer caught a genuine mistake in its own test on the
first run, and that case is now a negative control.

*Method note, recorded because it nearly produced a wrong result in both sessions:* the first
measurement of this on both sides reported "rc=0 everywhere, nothing is enforced". The cause was
reading `$?` after a `$(...)` substitution in the same line, so the reported status was `basename`'s.
Capture the exit status on the line immediately after the command.

**2. Tiling is not an in-tree pass, so `must_tile_to_mesh_shape` cannot be checked on the MLIR
plane.** `interface.accumulate` is declared in the interface dialect and has **no producer anywhere**
in the lowering package; no staged pass splits K. The tiling is emitted by the generated out-of-tree
backend, which differs per run. The obligation is therefore recorded as omitted with that reason —
even for gemmini, where the mesh edge *is* derived (`mesh_dim=16`). An earlier draft of this work
asserted a tile count here; it was wrong, and a wrong check is worse than a recorded gap. The
obligation is already checked today on the decoded instruction stream, and becomes checkable here the
moment tiling moves in-tree.

**3. A concrete case for pass-level checking, from a concurrent audit.** The atlas ISA model and its
RTL disagree on eight instructions: the model declares `DMA_CONFIG_CH0..7` with `funct7=1` while the
RTL decode pattern says `DMA_CONFIG` is `funct7=0` and `DMA_WAIT` is `funct7=1`. A backend deriving
from the model emits a `DMA_CONFIG` word the hardware executes as `DMA_WAIT`, so the base register is
never written. End-to-end capsule execution surfaces that as a mysterious wrong answer, many minutes
downstream; a pass-level check of emitted encodings against RTL-derived facts catches it directly.
This is the shape of defect the layers above exist for.

*Next: joining these verdicts to the obligation gate, and an input-dependent fault that separates the
formal layer from the dynamic one.*


### 2026-09-04 (evening) — the loop closes: a refuted obligation becomes a witness

**Counterexamples now rejoin the corpus.** When translation validation refutes an obligation, z3
returns a concrete input at a concrete shape. `merlin.verify.witness` writes that out as a witness —
`capsule.yaml` + `capsule.interface.mlir` in the frozen `merlin_iface` grammar + the counterexample
values — carrying a new first-class provenance, `source_role: smt_counterexample`, so a
solver-generated shape can never be mistaken for one an author chose. The emitted witness **validates
against the real capsule schema**, which is the bar: a counterexample capsule the corpus cannot load
would be a demo, not a result.

This is the direct answer to *"the capsules are very case-specific"*. The corpus stops being bounded
by what an author thought to write down. It also sidesteps the degenerate-stimulus problem by
construction: the witness carries the values that actually break the program, and a partial model is
refused rather than silently written out as a smaller tensor.

**`contract.prove` is now audited rather than believed.** The verifier still compares two strings —
that is all a verifier can do, and tightening it would reject every module that legitimately carries a
token. Instead `merlin.verify.proofs.audit_proofs` classifies each token as `verified` (a layer
discharged this requirement for the producing pass), `asserted` (it exists, nothing discharged it), or
`unattributed` (it names no producer, so nothing could). Evidence is scoped to the producing pass, so
a requirement discharged for a different pass cannot credit this one. Measured baseline on the
reference workload: **2 asserted, 0 verified** — which is the honest starting number, and the one that
should move as coverage grows. The op's own docstring now says the verifier checks a name match, so
nobody reads a `contract.prove` in the IR as evidence.

**A verify arm was added, not grafted onto arm 4.** The new tooling is granted through a separate
`merlin_verify` arm (assisted base + `rtl_generators`, `rtl_facts`, `verify_seam`); the existing
rtlchecks arm is unchanged. An arm that gains two capabilities at once produces a delta attributable
to neither, and would retroactively decouple every arm-3-vs-arm-4 number already reported. Wiring it
surfaced a real leak: the existing `xdsl_kit` grant covers `xdsl_dialects/` as a whole directory, so
`opt.py` was *already* reachable by arms 3/4/5 — the seam therefore had to be explicitly **denied**
there, not merely omitted, or half the new arm's declared treatment would have been nonexistent.

Two follow-ups are recorded rather than fixed, both in files this work does not own: the harness's
bundle-id-to-arm resolver matches by longest stem over the default ladder only, so a
`merlin_assisted_verify_*` bundle would silently resolve to arm 3 and run with arm-3's tools under the
verify arm's name; and the tracked assisted-arm deny manifests are stale by exactly the two new seam
entries until bundles are regenerated.


### 2026-09-04 (late) — verdicts become evidence, and the gate immediately says what is missing

Both layers now write to a shared verification log (`MERLIN_VERIFY_LOG`), beside the existing
invocation log, and `check_pass_obligations.py` joins them. A run over all six targets records **2
`verified` by FileCheck, 9 `abstracted` with their recorded reasons, and 1 `verified` by SMT**.

Three properties of the join are deliberate. A solver `unknown` maps to `unmeasured`, never to a
pass. A **refutation fails the gate unconditionally and carries no ratchet key at all** — a ratchet is
for absent evidence, never for a disproof. And with either log missing the gate exits 2, because
"reached" and "verified" come from different logs and an axis that cannot be decided must not report
clean.

**The first thing the joined gate did was tell us what we had not verified.** Its report:

```
verdict logs read: ['…/verify.jsonl']
verified by a static or formal layer: 0 / 4
verdicts against names the catalog does not carry (evidence that stopped counting):
  ['merlin-materialize-interface']
```

`merlin-materialize-interface` is in the **prototype** catalog — the staged research pipeline — not in
the production catalog of four whole-model boundary passes. The catalog's own rule is that a
prototype pass is "independently tested and never credited to production", and the gate enforced it:
our verdict was quarantined rather than counted.

That is the correct behaviour and an uncomfortable result, so it is stated plainly: **the formal layer
today verifies a staged pipeline pass. The four production passes have no static or formal verdict.**
Extending it there is real work — those passes operate on a dispatch program rather than on
value-typed tensors, and two of them are not MLIR passes at all — and it is now a named gap with a
gate that will keep reporting it, rather than an impression left by a green test run.


### 2026-09-04 (figures) — the evaluation, plotted from records

> **Superseded in part by the 2026-09-05 entry below.** The numbers here were correct for the run
> they describe, but three statements in this entry did not survive audit: the fault corpus was six,
> not eight; the `506x` ratio carried no shape; and the F4 "annotates two mesh-tile points" claim
> described an intended behaviour the code did not have. Read the correction entry before citing any
> figure number from this section.

Four figures, in `out/artifacts/verification/v1/latest/`, each generated from a JSON record so no
measured number is ever typed into plotting code. The tests prove that property directly: each plotter
is fed a synthetic record full of values that appear nowhere in the repo, and the rendered canvas text
is asserted to contain *those* values.

- **F1 detection matrix** — the headline. Six faults x three measured layers, each cell annotated with
  its wall cost, and an explicit hatched **RTL — NOT MEASURED** column so the gap is visible rather
  than absent. The callout carries the result: two faults are caught by the static layer alone.
- **F2 cost-to-detect** — log-scale seconds per layer. This run: static mean 4.1 ms, dynamic 9.0 ms,
  formal 2.08 s — **the formal layer costs 506x the static one**. The RTL row is hatched full width
  with its reason.
- **F3 obligation coverage** — checked vs omitted per target, all nine omission reasons in the caption.
- **F4 formal scaling** — log-log solve time against M·N·K, with an empirical slope computed from the
  points. It annotates **two** mesh-tile points, not one: the derived edges are `{gemmini: 16,
  atlas: 32}`, so 16³ and 32³ are both real tile sizes.

Scaling this run, all `unsat`: 2³ 0.05 s · 4³ 0.07 · 8³ 0.25 · **16³ 3.11 (gemmini mesh tile)** ·
16x32x16 4.49 · **32³ 15.85 (atlas mesh tile)** · 64x16x64 32.54. Absolute times move between runs —
the 2³ point absorbs one-time import cost and varies by an order of magnitude — so cite the shape of
the curve and the layer ratio, not a single second count.

### 2026-09-05 — an audit of our own claims, and five defects it found

The question that started this was blunt: *even when a layer detects something, what do we do with
it?* Answering it end to end meant re-running the evaluation at a **real** shape rather than a toy
one, and that surfaced defects in the verification layer itself. All five were in code written for
this work; all five are fixed; the corrected numbers are above.

**1. An abstention was recorded as a clean miss.** The detection record had no field for solver
status and never stored the timeout, so a 73 s timeout and "the layer ran and found nothing" were the
same row: `detected: false`, distinguished only by free text. That directly violates this package's
own first invariant — *a layer that cannot run must never look like one that ran clean* — and it is
the third time this repo has been bitten by a check that could not run reporting as one that passed.
The record is now `verify_detection_matrix/v2`: every attempt carries an explicit
`outcome ∈ {detected, clean, abstained, error}`, the solver bound is stored alongside it, and the
dataclass refuses a row whose `outcome` and `detected` disagree.

**2. The formal layer credited an encoder refusal as a detection.** `duplicate_commit` reported
DETECTED in 2 ms. That was `UnsupportedSemantics: 3 commits but only 2 activation arguments` — the
encoder declining the program, caught by a blanket `except` and scored as a refutation. The solver
never ran. It now abstains.

**3. The figures would have turned three timeouts into a coverage claim.** F1 drew an abstention in
the same pale cell as a miss, and its "caught by the static layer ALONE" callout counted faults that
were exclusive only because another layer timed out — at 16³ that reads "5 faults caught by static
ALONE", which is an artifact of the budget, not a coverage result. Abstentions are now hatched and
excluded from the callout, and F1/F2 both carry the shape and the solver bound on the canvas.

**4. The derived check was strictly weaker than the hand-written one it was modelled on.** The
generated `must_prove_rhs_immutable_for_residency` ended at `CHECK: interface.resident_evict` with no
trailing `CHECK-NOT: interface.matmul`, so it **passed** a genuine use-after-evict that the
hand-written check in `merlin/tests/data/lit/core/` caught. A residency obligation without the
trailing negative is a presence check, not a lifetime check.

**5. Nothing in the fault corpus was commit-shaped.** All three structural faults attacked residency,
so the commit half of the obligation — the half `must_commit_accumulator_before_reuse` compiles to —
had never been falsified by anything. A check that no fault can make fail is not evidence, however
green it looks. Two operators were added (`duplicate_commit`, `commit_after_reuse`), and closing this
also revealed that the *hand-written* check needed a `CHECK-NOT` between its two commits to assert
commit-once at all. `commit_after_reuse` is now a third static-only detection, so the cheap layer's
non-redundancy result spans both halves of its obligation rather than only residency.

**The shape finding.** Verification and refutation are not the same cost, and the scaling figure only
ever measured the first. Proving a correct program correct at one gemmini mesh tile takes seconds;
producing a counterexample for a broken one at that shape exceeds a 60 s bound. Every claim about the
formal layer's *detections* is therefore scoped to the small shapes where refutation terminates, and
the write-up now says so wherever it makes one.

**What a detection actually does** — the part the question was really about. Three consequences, in
descending order of how wired they are:

- **It fails a gate.** A `refuted` verdict reaches `check_pass_obligations.py` and is a hard failure
  with no flag and no ratchet, carrying its counterexample. Every other axis can be ratcheted, because
  a ratchet forgives *absent* evidence; a refutation is evidence we have.
- **It becomes a graded test case.** A `sat` model is emitted as a schema-valid capsule with
  `source_role: smt_counterexample`, and the existing independent oracle grades it with no
  special-casing. This is the concrete answer to "the capsules are very case-specific": the solver
  picks the case, not us.
- **It reaches an agent mid-loop** — designed, not working. `_arm_from_bundle_id` resolves the verify
  arm's bundle id against the default ladder only, so it silently falls through to the assisted arm.
  One line, in a file another session holds (VER-26). A campaign launched before it lands produces an
  unattributable result.

The honest ceiling on all three is unchanged: the pass we formally validate is in the *prototype*
catalog, so the gate correctly refuses to credit it to production and the four production passes still
read `0 / 4 verified` (VER-28).

### 2026-09-05 (encoding) — the refutation wall was ours, not the solver's

Following the shape finding above: refutation at a mesh tile was **a solver wall, not a timeout**.
Raising the budget 30x (60 s -> 1800 s) still returned `unknown` after 1829 s. So the fix was never
going to be patience.

A controlled experiment isolated the cause. Holding the multiply **term count fixed** at 16 384 and
varying only the multiplier **bit-width**:

| multiplier width | 16x16x16 refutation |
|---|---|
| 8-bit | **sat in 37 s** |
| 16-bit | unknown at 616 s |
| 32-bit | unknown at 1829 s |

The dominant cost variable is width, not term count — bit-blasted partial-product area scales as
`terms x width^2`. And the width was 32 because of a decision in our own encoder: `symbolic_tensor`
declared every element at the **accumulator** width and constrained it back to the element range with
a shift identity, because xDSL 0.68 ships no extract/concat/extend op. Every i8 x i8 product was a
full 32x32 multiplier carrying eight meaningful bits.

The same experiment also explained why the *clean* direction is cheap, and it is not what the earlier
entries assumed: for a correct program the target-side and spec-side terms are **syntactically
identical**, so every disjunct is `(not (= t t))` and z3's rewriter collapses the query in
preprocessing without bit-blasting a single multiplier. Fast `unsat` was never evidence that the
solver could handle these shapes.

**The fix.** Upstream's exporter does accept `smt.bv.concat` (verified by running it — it emits
`(concat a b)`), so a ~20-line local op supplies the missing widening, the same workaround already
used for `smt.solver`. Sign extension needs no further op: `concat(ashr(x, w-1), x)` *is*
sign-extension, since the high half of an arithmetic right shift is all sign bits. Elements are now
declared at their native width, multiplied at `2w` (exact — an i8 x i8 product always fits in 16
bits), and widened to the accumulator only for the sum. A product too wide for its accumulator raises
rather than truncating.

A second inefficiency turned up in the same code while measuring the first: `matmul` widened each
element once **per use** rather than once, emitting `M*N*K` widenings where `M*K + K*N` suffice — an
8x term blowup at 16³ for no semantic gain. Hoisting it out of the `k` loop is folded into the numbers
below.

**Measured effect** (same fault `swapped_matmul_operands`, `reuse=2`, `acc_width=32`):

| shape | before (32-bit multiply) | after (16-bit multiply, widening hoisted) |
|---|---|---|
| 4x4x4 | sat 5.56 s | **sat 1.89 s** |
| 8x8x8 | sat 439.14 s | **sat 16.19 s** (27x) |
| 12x12x12 | not reached (10³ already `unknown`) | **sat 652 s** |
| 16x16x16 | unknown @ 1829 s | **unknown @ 916 s** |

The refuting boundary moved from **8 to 12**. That is a real gain, and it does **not** reach a gemmini
mesh tile: 16x16x16 still does not refute at any budget measured, across three encodings and budgets
up to 30 minutes. The mesh-tile limitation stands.

**The trade, stated rather than buried.** Narrowing the multiplier makes the query textually larger
(a concat and a shift per element), so the *verification* direction gets more expensive even as
refutation gets cheaper: 32x32x32 verifies in 55 s under the new encoding against 18 s under the old.
Refutation is the direction detection depends on, so this is the right trade — but it is a trade, and
the scaling figure shows the regression rather than hiding it.

**On citing these seconds.** The host is shared and was under load throughout; 12³ measured 489 s in
one run and 652 s in another with only the hoist between them, which is contention, not signal. The
boundary shapes are robust — the gaps are one to two orders of magnitude — but individual second
counts want a quiet host before they go in a paper.

**So the standing claim is unchanged in kind, only in degree.** The formal layer refutes at small
shapes, now somewhat larger ones, and not at a real tile. Anyone citing a detection from it must cite
the shape. Two invariants are recorded in `merlin/python/merlin/verify/AGENT.md` so this cannot
silently regress: multiply at the data's width, never the accumulator's; and never cite a scaling
curve measured on correct programs as evidence that fault detection is tractable at that shape. Both
are pinned by tests.

### 2026-09-05 (production) — the gate reads 1 / 4 instead of 0 / 4

Until now every check in this work exercised `merlin-materialize-interface`, which lives in the
**prototype** catalog. The obligation gate correctly refused to credit that to production, so the four
production passes read **`0 / 4 verified`** — the difference between "we verify a pass" and "we verify
the compiler", and the gap a PL reviewer would press on first.

The core lit suite (`merlin/tests/data/lit/core/`) already drove production passes and **nothing was
reading it**. `lit_suite.record_core_verdicts()` closes that. The pass under test is parsed from each
file's own `RUN:` line — structurally, no regex and no hardcoded pass list — and its requirement class
comes from the catalog, so neither is asserted by the recorder: a test that stops exercising a pass
stops crediting it, and a `RUN:` line naming a pass the catalog does not define is surfaced as a
broken test rather than skipped.

With a new check for `merlin-add-c-interface` (boundary materialization — the host seam, without which
the symbol the runtime `dlsym`s does not exist) the gate now reports:

```
verified by a static or formal layer: 1 / 4
  merlin-add-c-interface   ... unmeasured/verified
verdicts against names the catalog does not carry (evidence that stopped counting):
  ['merlin-apply-schedule', 'merlin-infer-contract-facts', 'merlin-materialize-interface']
```

That last line is the gate staying honest about its own denominator: three real verdicts exist and are
deliberately **not** counted toward production, because those passes are not in the production catalog.

**And writing the check found a defect in the pass.** `merlin-add-c-interface`'s docstring says "Mark
public funcs" and its catalog summary says "each public func gets a ciface wrapper"; the implementation
walks every `func.func` and marks it regardless of `sym_visibility`, so a private helper gets a C
wrapper it has no caller for. The lit file pins what the pass **actually** does — so the suite is a
true regression test rather than red against a pass nobody agreed to change — and records the
intent/behaviour mismatch as **VER-29** for the owner of `merlin/python/merlin/llvmlower/`. It is a
small defect. It is also the first production pass anyone checked, and it did not survive the check.

### 2026-09-05 (the seam) — we were checking the wrong side of it

Everything above validates the ``interface`` plane. But ``interface`` is the **input** a backend
receives, produced by our own pass ``merlin-materialize-interface``; a capsule-bench agent writes the
code **downstream** of it. So the formal layer verified our passes and said nothing about the compiler
actually under evaluation. That is the gap this entry closes, and it is the difference between a
tooling result and a paper result.

**Why the command buffer, and not the `runtime` dialect.** There are two backend paths and they are
not the same shape. The in-tree ``TargetPackage`` path lowers ``interface -> target -> runtime ->
command_buffer`` with the ``runtime.*`` ops constructed by core code. The capsule-bench path — what an
agent actually writes — is a **subprocess** with CLI entrypoints that emits ``command_buffer.json``
DIRECTLY, and ``target_dialect_contract.yaml`` states in so many words that the intermediate MLIR is
*"a recommendation, not a gate"*. A checker built on the ``runtime`` dialect would therefore have been
structurally unable to see the thing we want to check. The command buffer is the ABI both paths
converge on, it is schema-validated, and the in-tree path yields it free via ``emit_command_buffer``.

**What it does.** ``merlin/python/merlin/verify/cb_semantics.py`` symbolically executes the buffer,
mirroring ``runtime/simulator.py`` opcode for opcode, and ``refine.validate_compilation`` encodes the
interface program and the buffer over the SAME symbolic leaves and asks whether any declared output
can differ. ``unsat`` means the backend's buffer computes what the program specified, for every
integer input at that shape.

Measured, on the in-tree pipeline's own buffer at 2x2x2:

| command buffer | verdict |
|---|---|
| unmutated | ``unsat`` — no false positive |
| ``lhs``/``rhs`` swapped on a MATMUL | ``sat``, 8-element counterexample |
| ``output_dtype`` narrowed to i8 | ``sat``, 8-element counterexample |
| second COMMIT reads the first accumulator | ``sat``, 12-element counterexample |
| a MATMUL deleted | **abstained** — the buffer references an undefined tensor, i.e. it is malformed rather than semantically wrong, and saying "refuted" would mischaracterise it |

**The differential test is the load-bearing one.** A checker that disagrees with the engine the corpus
actually grades against would refute CORRECT backends, which is worse than not checking at all. So
before any refutation is trusted, the encoder is pinned against ``merlin.runtime.simulate``: bind every
symbolic leaf to the concrete value the simulator was given, assert the encoded output differs from
what it actually produced, and require ``unsat``. Any disagreement is an encoder bug until proven
otherwise.

**A soundness trap, and its guard.** ``Tensor.matmul`` documents "accumulated in i32" but accumulates
in **unbounded Python ints** — it tags the dtype without enforcing it — while this encoder wraps mod
2^32. The two agree exactly while no sum leaves the accumulator's range, which is a derivable side
condition, not a hope: ``K <= (2^(acc-1) - 1) / 2^(2w-2)``, i.e. **K <= 131071** for i8 into i32.
Beyond it the honest verdict is an abstention, because the two engines are then answering different
questions. ``safe_k_bound`` computes it and the encoder refuses rather than guessing.

**Two more findings the layer surfaced.** ``capsule_golden._apply_epilogue`` defaults ``output_dtype``
to ``i32`` and narrows any width, while ``simulator.py`` and ``reference.py`` default to ``i8`` and
narrow on ``i8`` only. They agree on every dtype the corpus currently emits and diverge on an absent
attribute or ``i16`` — a latent inconsistency between two engines that both claim to define the same
readout. Separately, ``merlin_iface`` capsule MLIR has no Python round-trip (custom assembly, no xDSL
parser), so the ``compile`` CLI accepts the in-tree ``interface`` plane and a capsule's own file cannot
be passed to it today.

### 2026-09-05 (correction) — what a counterexample capsule actually contributes

An earlier claim in ``verify/witness.py``, and in how this work was described, was **wrong**:
that a counterexample witness "carries the values that actually break the program, so it cannot be
degenerate by construction". It does not, and three facts settle it:

* ``capsule.schema.json``'s ``inputs[]`` has no values field and sets ``additionalProperties: false``;
* ``capsule_golden.materialize_capsule_leaves`` fills every leaf unconditionally with
  ``Tensor.deterministic(name, shape, dtype)``;
* nothing outside this package's own tests reads ``counterexample_inputs.json``.

So a counterexample capsule contributes the **shape and configuration** the corpus was missing, and is
then graded on the corpus's own deterministic fill — which is the degenerate stimulus. Carrying the
solver's values into grading needs a stimulus channel in ``capsule_runner``. The docstring is corrected
in place and the correction recorded here, because the claim was load-bearing for "the capsules stop
being case-specific": the SHAPES stop being hand-picked; the VALUES do not, yet.

### 2026-09-05 (lattice) — the verified set is generated, not curated

``Boundaries.extent_probes()`` has always emitted extents that straddle each real hardware boundary —
the degenerate 1, a mostly-empty tile (edge/4), edge/2, the tail (edge-1), the exact tile, the overflow
(edge+1), two tiles — and **nothing iterated them**: ``corpus_synth.extents_for`` reads only the
``edge``. ``merlin/python/merlin/verify/lattice.py`` sweeps them, verifying the compilation at each.

For gemmini the lattice is ``[1, 4, 8, 15, 16, 17, 32]``, derived from ``mesh_dim=16`` in that target's
own RTL facts. The cost is affordable for the reason established earlier: sweeping a lattice means
verifying CORRECT programs, which is the cheap ``unsat`` direction.

**The count is deliberately not inflated.** The three ``contraction`` cells differ only by ALIGNMENT —
aligned / partial / sub_tile — and alignment is exactly what the extent expresses (16 is the aligned
tile, 15 the partial tail, 4 a sub-tile occupancy). Sweeping each cell separately would issue the
identical query three times and report three verified points for one solved query. They are grouped,
and the record says how many distinct query groups produced the coverage. Cells whose family has no
program builder are recorded with that reason rather than dropped.

### 2026-09-05 (advisory) — the checker an agent runs on its own output

``merlin-verify compile --interface <f.mlir> --command-buffer <cb.json>`` needs no in-tree lowering and
no simulator, so it works on a submission from a backend nobody has seen. Three exit codes, and the
middle one is why this is safe to put in front of an agent: 0 verified, 1 refuted **with the concrete
inputs printed**, 2 abstained — and an abstention is explicitly reported as a limitation of the
checker, not a defect in the backend. Refuting correct work because our encoder is incomplete is the
one outcome that would make this worse than useless.

Measured: the in-tree pipeline's own buffer exits 0; the same buffer with a MATMUL's operands swapped
exits 1 and prints the eight input values that expose it.

Feedback is **advisory by design** — the agent sees the verdict and can still submit. That measures
whether the signal helps without changing what counts as a pass, and avoids confounding the arm
comparison. A blocking mode would attach in the locked harness, and is deliberately not built.

### 2026-09-05 (completion) — finishing the half that was left, and one real detection

The previous entries described work that was half done. An audit against the plan found: five of the
ten planned opcodes unencoded, the command-buffer fault corpus with **zero consumers**, and the
counterexample-to-capsule path not built at all. This entry records finishing them, and what that
turned up.

#### The command-buffer detection matrix, and a fault the golden cannot see

Wiring the fault corpus to a consumer produced the first case in this work where the formal side
detects something the numeric golden does not. Measured at 4x4x4, solver bound 120 s:

| fault | compilation check | numeric golden |
|---|---|---|
| `cb_swapped_matmul_operands` | DETECTED 0.99 s | DETECTED |
| `cb_crosswire_commit` | DETECTED 1.93 s | DETECTED |
| **`cb_narrow_output`** | **DETECTED 0.64 s** | **miss** |

`cb_narrow_output` sets `output_dtype: i8` on a COMMIT whose program declared a wider result — which
is also what an emitter that simply FORGETS the attribute produces, because the reference defaults it
to `i8` when absent. Why the golden misses it, concretely:

```
the stimulus the golden actually uses:
   A0 = [1, 2, 1, 2]    A1 = [2, 2, 1, 0]    W = [2, 1, 0, 1]

golden  Y0 = [[2, 3], [2, 3]]
faulty  Y0 = [[2, 3], [2, 3]]   identical
```

Every value already fits in i8, so the injected clamp is a no-op on that stimulus and the comparison
passes. The solver, quantifying over all i8 inputs, returns `sat` with:

```
   arg0_0_1 = -44   arg0_1_0 = 65    arg2_1_0 = -109   arg2_0_1 = 70   ...
```

where the accumulation reaches roughly 4600, the clamp bites, and the outputs differ. This is the
degenerate-stimulus mechanism that earlier entries predicted but could not demonstrate. It is now one
measured fault, not a general claim: the other two faults in the corpus are caught by both layers.

Note also what the matrix does NOT say. The static layer is recorded as **not applicable** rather than
as a miss: its checks are FileCheck patterns over MLIR text and a command buffer is JSON, so it cannot
look at the artifact at all. A layer that cannot see the subject has not looked and found nothing.

#### Opcode coverage, stated as a number

Of the 23 opcodes in the command-buffer schema:

| class | count | opcodes |
|---|---|---|
| encodable here | 11 | RES_PACK, MATMUL, MATMUL_RESIDENT, COMMIT, EVICT, VECTOR_MAP, BIAS_ADD, VREDUCE, ATTENTION_QK, ATTENTION_PV, MOVEMENT |
| float in the reference itself | 5 | RMSNORM, SOFTMAX, GELU, SOFTCAP, ROPE |
| no branch in the reference at all | 5 | LAYERNORM, GEGLU, ATTENTION_FULL, CONV, MATMUL_BATCHED |
| encodable in principle, not built | 2 | CONV2D, BATCHED_MATMUL |

Every schema opcode is in a named class, and a test asserts that — a 24th opcode added upstream fails
the test instead of silently abstaining as "unknown". The four classes give four different
diagnostics, because "unknown" tells a reader nothing about whether to extend the encoder, use a
different layer, or ignore the message.

#### A refuted shape becomes a capsule — verified through the real generator

The earlier `emit_witness` wrote a capsule directory itself, which got none of what the generator
provides: no `golden.yaml` for the grader, and `update_provenance_manifest` would have classified a
solver-produced capsule as `hand_authored`. The path now emits a PROFILE ENTRY into
`profiles/<target>.smt.yaml`, the sidecar `load_profile` already merges, so the golden, the MANIFEST
provenance and the scrubbing all come from `generate_target`.

Verified by building one through the real builder rather than asserting it:

```
entry:   CX_contraction_i8_15x17x15   source_role: smt_counterexample
built:   inputs W[17,15] i8, A0[15,17] i8   numeric_policy: exact_int/i32
         VALID against capsule.schema.json
```

The `CX` prefix cannot collide with `corpus_synth`'s `SY`, and entries de-duplicate by name so the
same shape refuting twice does not grow the profile.

#### Shape space, and the comparison that does NOT favour us

This is the number that answers "the capsules are very case-specific", and it is smaller than earlier
entries implied:

| | shapes | inputs per shape |
|---|---|---|
| dynamic ladder (gemmini) | **144 distinct**, across 460 capsules | one deterministic stimulus |
| formal lattice sweep | **7**, derived from mesh_dim=16 | every integer input |

The formal side covers **far fewer shapes**, not more. What it adds is the quantifier, and only at the
shapes it can reach. Any framing that implies the formal sweep broadens shape coverage is wrong; the
honest statement is that it deepens a small number of derived boundary shapes while the dynamic ladder
remains the only layer that touches hardware and the only one with real breadth.

#### Advisory feedback

The `verify_seam` ToolSpec already granted the whole `verify/` package, so the new checker reached the
`merlin_verify` arm without a grant change — but its description named only the old translation
validation. It now names the agent-facing command, its three exit codes, and that an abstention is a
limit of the checker rather than a defect in the buffer. The seam remains granted to `merlin_verify`
alone; arms 3/4/5 do not have it, so arm-to-arm attribution is unaffected.

### 2026-09-05 (findings) — three defects the layer surfaced, none of them in the layer

Pointing checks at things nobody had checked keeps finding things. Three from this pass, in
descending severity. None is in the verification code; all are in what it was pointed at.

#### 1. Two harness engines define the same readout differently, and the contract does not adjudicate

`capsule_golden._apply_epilogue` ends `_narrow_to_dtype(t, attrs.get("output_dtype", "i32"))` —
default **i32**, narrowing **any** integer width below the accumulator. `runtime/simulator.py` and
`runtime/reference.py` end their COMMIT with `if attrs.get("output_dtype", "i8") == "i8"` — default
**i8**, narrowing on an **exact** match. Verified by reading both lines, and by execution on one
capsule with M,K,N = 2,32768,2 (accumulator 74192):

| `output_dtype` | golden | reference | simulator | |
|---|---|---|---|---|
| absent | 74192 | 127 | 127 | **diverge** |
| `i16` | 32767 | 74192 | 74192 | **diverge** |
| `i4` / `u8` | 7 / 255 | 74192 | 74192 | **diverge** |
| `i8` | 127 | 127 | 127 | agree |
| `i32` | 74192 | 74192 | 74192 | agree |

The divergence is value-dependent — at K=4096 the accumulator is 9047 and `i16` agrees — which is why
it has never surfaced by accident.

**Which way the error goes.** At L0 the golden is compared against `reference_outputs(agent_cb)`. A
submission that omits `output_dtype` — legal, since nothing requires it — gets the reference's i8
clamp applied to a result the capsule declared i32, and fails with *"your command buffer does not
compute the declared operation"*. The failure blames the agent for a disagreement between two harness
engines. L1 cannot catch it, because both of its sides apply the same rule. Measured: **85 of 130**
shipped integer contraction capsules would fail L0 for a correct backend that simply omits the
attribute.

**Latent today, and the guard keeps it that way.** All 317 shipped `merlin_iface.commit` ops declare
`output_dtype`, so nothing currently reaches it. A test now fails on any capsule added without it —
the tripwire fires on the capsule that would arm the defect, not months later on a submission.

**Not fixed here, deliberately.** The contract is silent: `command_buffer.schema.json` lists the
attribute as an optional string with no default, and `command_buffer_abi.yaml` describes its values
but not its absence, in a block that spells out `REQUIRED` and `no default` for the pooling attributes
immediately below. Neither engine is wrong by the declared ABI, so picking one in shared library code
would be inventing a target fact — and `i8` in particular is one target's `requant_output_dtype`,
which `corpus_spec` correctly derives and never assumes. This needs a contract decision plus one
shared narrow function, not a unilateral edit. Recorded, pinned, and left for that decision.

Our own encoder mirrors the RUNTIME rule, because the runtime is what the corpus grades against. That
is deliberate and means it inherits the divergence rather than fixing it; a test asserts the encoder
says so, so this package does not quietly become a sixth answer to the question.

#### 2. A production pass can compile the wrong function and delete the model, silently

`merlin-outline-dispatches` is documented as splitting `func @forward`, and `run_dialect_plane` calls
it with `forward=None`. With no name it takes the FIRST func with a body and rebuilds the module as
driver-plus-kernels, discarding the rest. Reproduced directly:

```
in:   func.func private @helper(...)   +   func.func @forward(...) { linalg.matmul }
out:  builtin.module { func.func @helper(%0) { return %0 } }
```

`@forward` — the whole model — is gone. Zero kernels, no exception, no diagnostic, and `@helper` lost
its `private`. This appears to be the mechanism behind an incident already recorded in the obligation
gate's own docstring: *"the boundary capstone invoked the outliner once and outlined ZERO kernels."*

Pinned by `merlin/tests/data/lit/core/outline_dispatches.mlir`, which asserts what the pass ACTUALLY
does so a fix shows up as a red test rather than landing silently. Not fixed here: `capsule_runner`
calls it this way and is under another session's lock with runs in flight.

#### 3. A cost model that misses the dominant contraction spellings

`schedule_dispatch.node_cost` gates its M·N·K branch on `prov.op in ("matmul", "batch_matmul")`.
Across captured MLIR in `out/artifacts/`: `matmul` 7453, `batch_matmul` 5540 — but also
`convolution_im2col_matmul` **8701** and `int_matmul` **3834**, all real contractions with a K
dimension, none matched. A 256x1024x1024 GEMM prices at 268,435,456 as `matmul` and **262,144** as
`int_matmul` — 1024x under. For the int8 corpus every GEMM is priced as a copy, so hart balancing and
the reported speedup are wrong for exactly the models the multicore path exists to serve.

#### What the gate reads now

**2 of 4** production passes verified, up from 1. The remaining two are not reachable by lit at all:
`merlin-emit-dispatch-program` and `merlin-partition-dispatches` consume and produce Python
dataclasses rather than IR, so there is no stdout for FileCheck to read, and `merlin-opt` already
refuses them with that reason. Reaching 4/4 would need a text-output seam on `merlin-opt` — a code
change, not a test. Recording them as `abstracted` would document the reason but would NOT satisfy
`--fail-on-unverified`, and saying otherwise would overstate the coverage.

### 2026-09-05 (source) — validating the pass against its own input

Until now the formal layer compared the emitted ``interface`` program against a **re-derived
specification** — ``enc.matmul`` over the same symbolic inputs. That is conformance against a model of
what the pass should have done, not translation validation, and this log has been careful never to
claim otherwise. ``verify/linalg_semantics.py`` closes it: the specification side is now the ACTUAL
``linalg`` module the pass consumed.

**Zero points are the part that needed care.** ``linalg.quantized_matmul`` carries lhs/rhs zero points
as operands. When both resolve to constant zero the contraction delegates to ``Encoder.matmul`` (the
2x-width multiply). When either is a non-zero constant it does the exact ``linalg`` arithmetic —
sign-extend, subtract as add-of-negative since the ``smt`` dialect ships no subtract, accumulate mod
2^32. A zero point that is not a resolvable constant **abstains**, and that is the right verdict
rather than a gap: the ``interface`` plane carries no zero-point operand at all, so a runtime zp means
the leaf correspondence between the two artifacts does not exist.

Measured at 2x2x2, reuse 2:

| control | verdict | time |
|---|---|---|
| real pass output vs its own linalg source | ``unsat`` | 0.03 s |
| the same at 4³ / 8³ | ``unsat`` | 0.08 s / 0.50 s |
| `miswired_commit` / `swapped_operands` / `dropped_activation` | ``sat`` with counterexample | 0.07–0.12 s |
| source zero point changed 0 -> 7, output unchanged | ``sat`` with counterexample | 0.17 s |
| symbolic (block-argument) zero point | abstains | — |

That last positive-side case was added beyond the brief and is the one that matters: without it, an
encoder that silently assumed zero would have passed every other test here.

**What ``unsat`` from ``validate_pass`` proves, stated narrowly.** For THIS source module and THIS
interface module at THEIR shape, every commit equals the corresponding ``linalg`` result for every
integer assignment to the source's arguments. It proves nothing about other shapes or other programs.
And both sides bottom out in the same ``Encoder.matmul`` when the zero points are zero, so what the
query genuinely checks is that the pass emitted a contraction over the same operands, in the same
pairing, at the same shape, with nothing added, dropped or mis-wired — the negative controls show it
is not vacuous — but it is not two independent derivations of what a matmul means. A shared
misunderstanding of ``interface.matmul`` would survive it. Residency is invisible to this check and
remains the static layer's job.

### 2026-09-05 (arm) — a treatment the agent was never told about

Preparing the verify-arm campaign surfaced a measurement hazard worth more than the campaign. The
freshly generated verify-arm ``STARTER_PROMPT.md`` was **byte-identical** to arm 4's, and
``ALLOWED_MERLIN_TOOLS.md`` rendered only ``ToolSpec.note`` — a policy label — while ``ToolSpec.blurb``,
which carries what the tool answers and how to invoke it, was rendered **nowhere**.

An arm whose single treatment is a tool the agent is never told about produces a null result that
cannot distinguish *"the tool does not help"* from *"the agent never knew it was there"*. That is how
an earlier campaign lost its ISA grounding.

Fixed: the bundle manifest now records **which tools it granted** — it carried paths but no capability
names, so ``tools.txt`` and the manifest were two half-descriptions of one grant — and the generated
doc renders each granted tool's blurb.

A second, smaller problem was found the same way: **this checkout has zero ``merlin-*`` console
scripts**. The package is used through ``PYTHONPATH``, so both ``merlin-verify ...`` and
``merlin-opt ...`` were `command not found` in the very environment the agent runs in. Both blurbs now
name the module form, and a test walks every blurb failing on any ``merlin-*`` token absent from PATH.
It caught the second one on its first run.

**Still open before any campaign** (not fixed here, deliberately): the committed arm-3/4/5 bundles
predate the verify deny lines, and ``xdsl_kit`` grants the whole ``xdsl_dialects/`` directory, so those
arms can currently read ``opt.py``. Deny wins in the sandbox binder, so two missing lines are the only
thing that would mask it. Those are SERVED bundles with a live run in flight; regenerating them mid-run
is itself a measurement change, so it needs a window rather than a patch.

### 2026-09-05 (4 of 4) — the last two passes were checkable all along

The obligation gate read **2 of 4**, and the reason given here for the other two was that they
"consume and produce Python dataclasses, so there is no stdout for FileCheck to read". That was true
of the PASSES and false as a conclusion.

`merlin-emit-dispatch-program` takes an `OutlineResult` and returns a `DispatchProgram`;
`merlin-partition-dispatches` takes that and returns a `PartitionResult`. Neither can ever be a
ModulePass, and `--list-merlin-passes` was right to report them unregistrable. But `DispatchProgram`
already had `to_dict`, and `schedule_dispatch` already had `emit_schedule_c` rendering a stable C
table — a text emitter nobody had connected to a driver. `merlin-opt --emit` runs the stage and
prints it. The gate could not rise above 2 of 4 not because those passes were unverifiable, but
because nothing exposed their results.

```
verified by a static or formal layer: 4 / 4
```

**The RUN lines still carry `-p <pass-name>` in `--emit` mode, deliberately.** `record_core_verdicts`
credits a pass by reading that token, so a check whose subject is invisible to the gate discharges
nothing. Handing an unregistrable name to `PassPipeline.parse_spec` raises, so `setup_pipeline` is
skipped in emit mode; the emit path never builds a pipeline anyway.

Both checks are non-vacuous, and measured rather than argued. The dispatch-program check falls to a
renamed kernel, a changed entry symbol, and a rewired result buffer. The schedule check falls to a
broken **level barrier** — a consumer moved down to its producer's level, which is a race, since harts
within a level run concurrently — and to a dropped kernel. That barrier is the strongest invariant of
the three previously-unverified passes, and pinning it on the EMITTED table matters because the table
is what a multicore runtime consumes: a schedule correct in memory and wrong on emission is still a
wrong schedule.

**How the first draft failed, since it is the same lesson twice in one day.** The initial checks
asserted a `"kernel"` key the emitter does not produce — the field is `"op"` — so the check failed
against *correct* output. Rewritten by reading the emitter's real output. A check written from a guess
about a format tests the guess, not the code.

---

### 2026-09-05 — the readout fix over-corrected, and 37 tests said so

The COMMIT readout contract fix (`95e636ae`) routed both runtime engines through one shared
`_narrow_int_readout`, replacing a `== "i8"` test that had diverged from the golden for 77 days. The
integer half of that change is right and stands: absent means `i32`, `i16` saturates to the `i16` range,
and `merlin/tests/ir/test_readout_dtype_divergence.py` pins all three engines against each other for
`i32`/`i16`/`i8`/`i4`/`u8`.

The change also made the two runtime helpers **raise** on a non-integer `output_dtype`, reasoning that
an integer engine has no definition for a float readout and should say so rather than pass it along.
That reasoning ignored who the authority is. `capsule_golden._narrow_to_dtype` passes a non-integer
token through unchanged, so raising did not remove a divergence — it created a second one, pointing the
same direction as the first: a correct backend gets an error where the oracle gets a value.

It fired immediately, on 37 tests across four files (`test_xdsl_vector_ops`,
`test_xdsl_whole_model_chain`, `test_gemmini_native_pooling`, `test_rtl_checks`). All of them commit an
`f32` tensor produced by a VECTOR_MAP chain — a shape the old `== "i8"` test had passed through for as
long as those tests had existed. Fixed by matching the golden in every branch, and
`test_all_three_engines_agree_on_every_declared_dtype` now carries `f32` and `bf16` cases so the
strictness argument fails the suite instead of the corpus.

Worth recording because the earlier claim in that commit — "since the golden already treated absent as
`i32`, any buffer relying on the old clamp was already failing L0, so this can only fix and not break" —
was true of the *default*, which is what it was reasoning about, and silently untrue of the *dtype
vocabulary*, which the same edit widened without saying so. The safe-direction argument covered one axis
of a two-axis change.

### 2026-09-05 — a regenerated corpus lost the store dtype, and the backend caught it

Found while chasing the failures above, and unrelated to them. Commit `2f06e353` regenerated the gemmini
corpus; the three hand-authored pooling capsules (`GP0_matmul_maxpool_i8`, `GP1_matmul_maxpool_tail_i8`,
`GP2_conv2d_maxpool_i8`) came back with `output_dtype = "i32"` where their tracked bytes had said `"i8"`,
and `SY_epilogue_maxpool` had been `i32` from the start.

`corpus_spec._resolve_output_dtype` read only the epilogue: `acc_scale` resolved to the target's declared
requant width and everything else to the accumulator width. A profile entry's own `output_dtype: i8` —
which all three hand-authored entries carry, with the reason written beside it — was never read. Nothing
failed at generation time. A well-formed capsule came out; just not the one the entry described.

It surfaced two layers down, where the fact lives: gemmini's native max-pool runs in the store DMA at the
input width, so `gemmini_codegen_mlir._native_pool_spec` refuses any commit that is not an i8 store. Four
capsules became uncompilable and 16 tests went red.

Two resolution rules now, and the second is the load-bearing one:

* an entry's explicit `output_dtype` wins, and an unknown token raises rather than falling back to a
  default (a typo must be a generation failure, not a silent substitution — the original defect's shape);
* a `maxpool` epilogue otherwise resolves to the target's **operand** dtype, derived from the descriptor.

Honouring the declaration alone would have fixed only the three hand-authored entries. The synthesized
one has no author to declare anything: `corpus_synth.declare_pool_window` supplies the window geometry,
and the axis that builds it knows nothing about a store path. Deriving the width is what makes the
generated capsule and the hand-authored ones agree.

This is the same failure family the verification work exists to expose, arriving from the other
direction: a generator that drops a declared fact produces output that is *well-formed and wrong*, and
only a consumer that happens to be strict about that fact ever notices. Here one was — the gemmini
backend refuses the store outright. Nothing in the corpus tooling would have.

### 2026-09-05 — the negative controls were weaker than they looked

The three `merlin_iface` negative tests each ran `not %mlir-opt --irdl-file=%iface-irdl %s | %filecheck`
against `CHECK: error` plus a `CHECK-SAME` fragment. That shape has a specific weakness: it passes when
an error appears ANYWHERE in the output. A parse failure on an unrelated line, a diagnostic about a
different op, a typo in the module — each satisfies it. So the test can go on passing after the
constraint it is named for stops being enforced, which is the one thing a negative control exists to
prevent.

Converted to the upstream form, `--split-input-file -verify-diagnostics`, which binds each expectation
to a line and to its message text and fails on unexpected diagnostics as well as unproduced ones.
Verified it bites by moving one expectation one line off its op: the run then fails twice, once for
`unexpected error` and once for `expected error ... was not produced`. The `not | FileCheck` form does
not distinguish those two situations at all.

The three files became one `iface/invalid.mlir` with ten cases — the original three plus a non-string
`role`, a matmul whose weight operand was never packed, a `resident_pack` with no `layout`, a commit
reading a tensor instead of an accumulator, a missing `output_dtype`, an `output_dtype` given as an
integer, and an `epilogue` given as a bare string. Each message was obtained by running the case, not
written from the constraint: the one message that was guessed (`but got 0`, where mlir-opt says `but
had 0`) failed against correct output — the third time that specific mistake has been recorded here.

**And a second file that pins what the grammar does NOT check.** `merlin_iface.irdl.mlir`'s generated
header already lists three ODS constraints IRDL cannot express — a tensor element type that is a token,
and element-wise constraints on `commit`'s `epilogue` and `conv2d`'s geometry arrays. They are absent
from the IRDL rather than present-as-`c_pred`, deliberately: mlir-opt drops a `c_pred` from its
enclosing `all_of` without a diagnostic, so a constraint carried that way can never fail, and one that
cannot fail reads as enforcement while providing none.

`iface/unchecked_by_irdl.mlir` now asserts that all three malformed modules are ACCEPTED, each with the
layer that does catch it named beside it. This is the honest counterpart to the ten rejections: without
it, a green suite is indistinguishable from a suite that rejects only what someone happened to write a
test for. If a case there starts failing, the gap closed and it belongs in `invalid.mlir`.

`test_negative_tests_are_present` accepts both spellings now, and a new assertion catches the trap the
conversion introduces: a file carrying `-verify-diagnostics` with no `expected-error` asserts the input
is CLEAN — a positive test wearing a negative test's clothes, which would pass on the day the verifier
stops rejecting anything.

Suite: 10 tests, all passing (7 core pass tests, the positive grammar test, `invalid.mlir` with its 10
split cases, and `unchecked_by_irdl.mlir` with its 3).

### 2026-09-05 — the detection rate, measured: 1 of 21, and not by the formal layer

The honest replacement for "six of seven hand-picked fixes caught". That number was a demonstration
chosen after the layers existed by someone who knew what they check; it is not a rate and should never
be cited as one.

**Method.** Population fixed before any draw: every `fix(` commit touching a path a layer can observe
(`runtime/`, `xdsl_dialects/`, `capsule_golden.py`, `corpus_spec.py`, `verify/`) — 102 commits at
`a8a89545`. Sample of 25 drawn by a seeded shuffle-then-take, so raising `n` extends the sample rather
than replacing it and a disappointing result cannot be quietly rerolled. The layers did not exist when
these commits landed, so the defect is brought forward instead: the parent's version of each file the
commit touched is written over a copy of the package and the five layers run against that copy.

**Result.**

| | |
|---|---|
| replayable | 24 of 25 (1 commit's parent files no longer exist; reported, not dropped) |
| detected, all commits | 2 / 24 |
| **detected, fixes predating the layers** | **1 / 21** |

The single historical detection is `6314d4fc fix(runtime): stop dropping a declared bias epilogue in
silence`, and the attribution matters more than the count: **all five layers caught it, including the
numeric golden that already existed.** The formal layer added nothing there. Across 21 historical
fixes, the number caught by the formal layer and not by the pre-existing dynamic check is **zero**.

**What this means, stated plainly.** The layers model one thing — whether a command buffer computes what
its interface program specified — and most of this repo's `fix(` history is not that. Reading the missed
list: nested OpenMP regions, worker stack sizing, a Zephyr vector-state enable, gate thresholds, tier
record shapes, a discovery sentinel, weight selection by linked footprint. None of those is a semantic
divergence in the lowering, and no layer here claims to see one. The claim this work supports is about a
CLASS of defect, and this measurement is what bounds the claim.

**The refinement that is not allowed yet.** The obvious response is to narrow the population to defects
in the modelled surface, which would raise the rate. Choosing that scope now, after seeing 1/21, is
exactly the move the whole design guards against. If it is worth running it is worth pre-registering:
write the narrower `OBSERVED_ROOTS` down, commit it, then draw. Both records then stand side by side
rather than one replacing the other.

**Three instrument bugs found on the way, each of which had produced a number.** Worth recording because
every one of them made the instrument look like it was working.

1. `repo_root()` resolves from the merlin package's own file location, so inside the shadow it pointed
   at the temp directory — no capsule corpus, no lit suite, no llvm-build. Every layer found nothing to
   check and exited clean, and 25 commits were recorded as misses for defects nothing had looked at.
   Caught by running the shadow at the parent of the COMMIT readout fix, a defect whose effect is known
   and large, and getting green from everything.
2. Two of the five layers were wired to names that were never created (`merlin.verify.replay_lit`,
   `test_golden_engines_agree.py`). Both reported `red`/`error` at baseline, were disqualified, and the
   run reported a rate for the remaining three while the surrounding text said five. `render` now warns
   loudly when any layer is unusable, and a test resolves every `LAYERS` entry to a real file or
   importable module.
3. The population grows as work lands, and the sample is a shuffle of the population, so two runs
   minutes apart drew different commits. The population is pinned to a resolved commit now and the sha
   is in the record.

Each is the same shape as the failure this package exists to prevent: a check that could not run,
reporting success.

### 2026-09-05 — "the rules are collapsed" was true of two of five

Closing out the `output_dtype` contract item. The earlier entries said both runtime engines now route
their readout through one shared `_narrow_int_readout`. That was true of `COMMIT` and `CONV2D`, and I
reported it as done. Deriving the narrowing set from the engines instead of from that claim found three
more sites still carrying the exact-`i8` test: `BIAS_ADD`, `ATTENTION_QK` and `ATTENTION_PV`, two in
`simulator.py` and one shared `_attention_epilogue` in `reference.py`.

They carry the identical divergence. All three route through the golden's single `_apply_epilogue`,
which ends in `_narrow_to_dtype(t, attrs.get("output_dtype", "i32"))` and narrows at ANY integer width,
while the runtime narrowed only on an exact `"i8"` — so an `i16` or `u8` readout on an attention or
bias-add path saturated in the oracle and passed through in the engines graded against it. All five sites
now share the rule, and `test_no_engine_keeps_a_private_readout_rule` fails on the sixth: the exact-i8
test is a literal, so its absence is checkable rather than assertable only by reading.

**The declaration, not the default.** `validate_command_buffer` now reports a narrowing command that
declares no `output_dtype` as a problem, naming the command index and the opcode, and `simulate` raises
rather than guessing. Agreement on a default is a weaker guarantee than a declaration: it makes the
buffer's meaning depend on a convention the submission never stated, and the next divergence would be
as silent as the last. All 421 narrowing ops in the shipped corpus already declare it, so this states an
invariant that already holds. `NARROWING_OPCODES` is pinned by a test to the set the simulator actually
passes to `_narrow_int_readout` — demanding it for fewer leaves the silent case open, for more rejects
correct buffers.

**And the change found a live encoder bug.** Adding a `u8` case to the encoder's differential against
`merlin.runtime.simulate` failed immediately: `cb_semantics._narrow` derived the WIDTH from the dtype
spelling but always saturated to the SIGNED range, so a `u8` readout was clamped to [-128, 127] where
all three engines clamp it to [0, 255]. The encoder disagreeing with its own oracle is the one outcome
that makes this tool worse than nothing — it refutes correct backends. Signedness now comes from the
`i`/`u` prefix, as it does in the engines.

Two incidental breakages, both repaired rather than worked around. `test_a_produced_intermediate_need_not_be_a_declared_tensor`
filtered problems on the substring `"declares no"`, which the new readout message also matches; it now
filters on `"declares no 'tensors'"`, the complaint it actually means. And the encoder's ABSENT
differential case can no longer run end to end, since such a buffer is refused before either engine
reads it — so the property it protected (the encoder's default equals the engines') is asserted directly
against the engines' source instead of being dropped.

### 2026-09-05 — VER-26 was still live, and worse than "the arm cannot run"

The plan recorded a hard precondition: *a verify-arm campaign must not launch until VER-26 lands*,
because `run_baseline_qa_loop._arm_from_bundle_id` matched against the default arm list and
`merlin_assisted_verify_*` would resolve to arm 3. Another session has since rewritten that function to
longest-matching-stem and made it raise on no match, so I checked whether the precondition had cleared.

It had not, and the failure is not the one the note describes. `merlin_assisted_verify_public_v0`
resolves to **`merlin_assisted`** — not to a `KeyError`. Every opt-in arm's stem deliberately contains
`merlin_assisted`, because `generate_prompt._is_assisted_arm` is a substring test and the arm is meant
to inherit the assisted prompt without a prompt edit. So resolving against the default ladder alone does
not refuse an opt-in bundle; it silently mis-resolves it to the longest stem that IS in the ladder.

The consequence is the specific thing this experiment cannot survive: the verify arm would run with
arm-3's tool grants — **no verification seam at all** — under the verify arm's name, and the result
would be read as evidence about the seam. Nothing downstream can detect that. A gemmini verify bundle
already exists on disk (`targets/gemmini/input_bundles/merlin_assisted_verify_public_v0`), so this was
live rather than hypothetical.

One line: resolve against `_ALL_ARMS` (default ladder plus opt-in arms) rather than `_ARMS`.
`merlin/tests/infra/test_arm_resolution.py` now derives the check from the arm table rather than a list
typed into the test, so a seventh arm is covered when it is added, and asserts separately that
`verify_seam` reaches exactly one arm — an arm gaining two things at once produces a result
attributable to neither.

**The precondition is cleared; the launch is not made.** The two grounds I held it on are also gone:
the spend ceiling is enforced (`MERLIN_MAX_SPEND_USD` stops the batch through a shared ledger), and the
grading engine has settled. What remains is a paid campaign, which is a decision rather than a task.

---

## 7. Reproducing what is claimed here

NOTE: the shared `.venv` may resolve `merlin` from a different worktree, so pin `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/merlin/python

# the toolchain facts asserted in the log
third_party/llvm-build/bin/mlir-translate --help | grep export-smtlib
third_party/llvm-build/bin/mlir-opt        --help | grep irdl-file
.venv/bin/python -c "from xdsl.dialects import smt; print(hasattr(smt, 'SolverOp'))"   # False

# the two audit findings. NOTE the second no longer reproduces as first written: at the time
# `compiler_obligations` had exactly one consumer, and that consumer was a DOCSTRING mention.
# This work added real ones (lit_check_compiler, plots), so the count is now higher by
# construction. Kept, with the correction, rather than quietly re-tuned to still look right.
sed -n '167,183p' merlin/python/merlin/xdsl_dialects/contract.py       # prove compares strings
grep -rln compiler_obligations --include=*.py merlin/python build_tools  # was 1 file, now several

# the layers themselves
.venv/bin/python -m merlin.xdsl_dialects.opt --list-merlin-passes      # 10 registered, 2 explained
third_party/llvm-build/bin/llvm-lit -sv merlin/tests/data/lit          # the static layer
.venv/bin/python -m merlin.verify.evaluate --m 4 --k 4 --n 4           # the detection matrix
.venv/bin/python -m merlin.targetgen.lit_suite --all --write           # the obligation ledger
third_party/llvm-build/bin/llvm-lit -s out/artifacts/verify/lit        # the derived suite

# the command-buffer layer: what a BACKEND emits, not what it receives
.venv/bin/python -m merlin.verify.cli --help
.venv/bin/python -c "
from merlin.verify.evaluate import run_cb_matrix, render
print(render(run_cb_matrix(m=4, k=4, n=4, timeout_ms=120000)))"   # cb_narrow_output: golden misses

# the agent-facing check, on files alone -- no in-tree lowering, no simulator.
# Exit 0 verified / 1 refuted with the counterexample printed / 2 abstained.
.venv/bin/python -m merlin.verify.cli compile \
    --interface <interface.mlir> --command-buffer <command_buffer.json>

# the derived extent lattice, and the shape-space comparison that does not favour us
.venv/bin/python -m merlin.verify.lattice --target gemmini            # 7 shapes, all unsat
.venv/bin/python -m merlin.verify.lattice --target gemmini --emit-counterexamples

# the tests, including the mutation and refutation controls, the differential test against
# merlin.runtime.simulate, and the assertion that every schema opcode lands in a named class
.venv/bin/python -m pytest merlin/tests/ir -q -m "not slow"
```

Two commands above are worth running for what they DISPROVE rather than what they show. The
`run_cb_matrix` line prints `cb_narrow_output` as caught by the compilation check and missed by the
numeric golden — the only fault in the corpus where that happens. The `lattice` line prints the shape
space, where the formal sweep's 7 shapes sit against the dynamic ladder's 144; the formal layer is the
narrower of the two and the output says so.

## 8. Open questions

- How much of the formal layer is worth claiming externally before the `pdl` prerequisite lands?
  Translation validation alone is a real result — it is what the PLDI'25 authors used to find five
  miscompilations upstream — but it is a per-compilation theorem, and the difference should be stated
  rather than blurred.
- Should tiling move in-tree? It would make the strongest obligation verifiable once and for all, but
  it also moves work out of the generated backend, which is the thing under evaluation. This is a
  measurement-design question, not just an engineering one.
- The three targets with empty RTL facts blocks need re-elaboration before their derived suites mean
  anything. Until then their coverage numbers are honest zeros, and should be reported as such.
