---
title: "Design: Triton as a target-independent kernel frontend"
kind: design
status: draft
owner: ir
last_verified: 2026-08-10
related: [lowering_pipeline, core_dialects, target_agnostic_core, target_resolution]
code_refs:
  - merlin/python/merlin/triton/__init__.py
  - merlin/python/merlin/triton/source.py
  - merlin/python/merlin/triton/addressing.py
  - merlin/python/merlin/triton/bridge.py
  - merlin/python/merlin/triton/spec.py
  - merlin/python/merlin/compile_core.py
  - merlin/python/merlin/xdsl_dialects/lowering/pipeline.py
  - merlin/python/merlin/xdsl_dialects/lowering/contract_facts.py
  - merlin/python/merlin/xdsl_dialects/lowering/interface_lowering.py
  - merlin/python/merlin/xdsl_dialects/interface.py
  - merlin/python/merlin/frontends/linalg_mlir.py
  - merlin/python/merlin/llvmlower/lower.py
  - merlin/python/merlin/llvmlower/kernel_backend.py
  - merlin/python/merlin/targetgen/target_registry.py
---

# Triton as a target-independent kernel frontend

## Why

Merlin can *generate* a compiler target from an RTL repo. It cannot today be *programmed* by a
kernel author: the only two ways into the staged pipeline are `build_input_module()` — a synthetic
`repeated_rhs_matmul` — and `lower_model()`, which takes a whole captured model. There is no way to
hand Merlin "this kernel" and get it compiled for a target.

The goal is to make a standard `@triton.jit` kernel a legal Merlin input **above the
target-selection boundary**, so that:

> once Merlin learns how to compile generic computation to a new target, that target automatically
> becomes programmable from Triton.

The thesis is architectural, not about performance: *the same* Triton source compiles to RVV, to a
generated Gemmini target, and to a generated Radiance target, and target #3 costs **0** Triton LOC,
**0** Triton prompts, **0** TargetGen edits.

## The architecture

```
                        @triton.jit  (author's kernel, unmodified Triton)
                                       │
                                       ▼
                        upstream Triton frontend  (stock wheel)
                              Python AST → TTIR
                                       │
                                       ▼
                         target-independent bridge
                     TTIR → linalg / tensor / arith / scf / math
                        (pointer args RE-RAISED to tensor args)
                                       │
                     ┌─────────────────┴─────────────────┐
                     │                                   │
             standalone kernel                    model kernel body
                     │                             (replace outlined func — later)
                     └─────────────────┬─────────────────┘
                                       ▼
                                MERLIN COMPILER
                          contract → schedule → interface
                                       │
                                       ▼
                          generated Target Dialect
                                       │
                            runtime / LLVM / binary
                                       │
                 ┌─────────────────┬───┴────────────┬──────────────────┐
                 ▼                 ▼                ▼                  ▼
                RVV             Gemmini          Radiance        Unknown target
```

The convergence point is **linalg-on-tensors**, not the target dialect. Lowering TTIR straight to a
target dialect would bypass contract inference, scheduling, interface materialization and residency
decisions — i.e. everything that makes Merlin a compiler rather than a code emitter.

## Invariants

These are asserted by `merlin/tests/infra/test_triton_target_independence.py`, not merely documented.

```
INV-1  zero Triton-specific LLM calls / prompts per target
INV-2  zero target-specific code or target-name literals in merlin.triton
INV-3  no schema is widened for this frontend (TargetGen: target-blind changes only)
INV-4  TTGIR is never a Merlin boundary
INV-5  no direct TTIR → gemmini/saturn/<any> lowering
INV-6  linalg-on-tensors is the single convergence point
INV-7  the generated target dialect stays the target-legality boundary
INV-8  unsupported Triton semantics FAIL CLOSED with a located diagnostic
INV-9  standalone and model-replacement paths share one bridge
INV-10 source → TTIR → core MLIR is reproducible and inspectable on disk
```

INV-3 was originally "zero TargetGen / schema modifications", scoped to Milestones 0–4 because
finding 4 predicted a *general* Merlin gap would need a core dialect addition. It did (M5 added
`interface.elementwise` and a target-spec seam), so the invariant is now stated in the form that
survives: the two schemas must never be widened by a commit touching the frontend — that is the
concrete carve-out failure mode — while TargetGen *code* may change provided it stays target-blind,
which `check_no_target_name.py` enforces repo-wide.

## Four findings from the existing code

These were established against the tree before any code was written; each one invalidates an
otherwise-reasonable version of this design.

### 1. A post-bufferization bridge output cannot feed any Merlin ingest

The obvious bootstrap — Microsoft's `triton-shared` and its `--triton-to-linalg` pass — emits
post-bufferization IR (`memref.reinterpret_cast`, `memref.copy`, `bufferization.to_tensor`,
`affine.store`) wrapped around tensor-level `linalg`. Merlin cannot consume that:

- `interface_lowering.lower_to_interface` maps matmul operands through a `value_map` built **only**
  from `src_block.args`, so an operand that is not a function block argument raises `KeyError`.
- `contract_facts._trace_to_block_arg` walks only `linalg.transpose` and
  `tensor.{cast,reshape,collapse_shape,expand_shape}` / `linalg.copy`, so a bufferized RHS traces to
  nothing and residency is never inferred.

Therefore the bridge's real job is **pointer → tensor re-raising**: its output must present
tensor-typed function arguments feeding the matmul directly, which is exactly the shape model2MLIR
already emits. Op-by-op TTIR translation is the easy part.

**Evaluation, on the record.** `triton-shared`'s README states verbatim: *"This repository is no
longer maintained. It remains available for reference."* There is no pip install path; it builds as
an out-of-tree `third_party` codegen backend inside a **source build of Triton pinned to the commit
in `triton-hash.txt`** (`TRITON_PLUGIN_DIRS`, binaries under
`triton/build/<cmake>/third_party/triton_shared`). So it cannot run against an unmodified Triton
wheel, and adopting it means owning a pinned Triton source build — the "custom version we maintain"
that was ruled out. Its `--triton-to-linalg` output spans linalg-on-tensors **plus memref, affine and
bufferization**, which is precisely the shape finding 1 shows no Merlin ingest can consume.

Verdict: **not adopted.** The bridge is owned in-tree in Python/xDSL and never patches Triton.
`Cambricon/triton-linalg` is the maintained C++ alternative and is rejected for the same two reasons
(source build, post-bufferization output); it stays noted in case the bridge ever needs a reference
implementation to compare against.

### 2. `lower_to_interface` silently drops non-matmul payload — a false-PASS generator

`lower_to_interface` rebuilds the function body from scratch as pack / matmul / commit / evict +
return. A kernel carrying a masked store, a grid loop, or an elementwise epilogue would descend,
`verify()` at every stage, emit a command buffer, and compute *something else*. Nothing in the
pipeline would complain. A payload-completeness guard that fails closed on any unaccounted-for op
must land **before** the Gemmini milestone is treated as evidence of anything.

### 3. There are two backends, and the split is by payload, not by target

`merlin-compile`'s own docstring puts it plainly — "genuinely different pipelines, not a false
unification". Matmul-family payloads descend the staged pipeline, and `toy_npu`, `saturn` **and**
`gemmini` all take it. Non-matmul payloads (a vector add) reach `lower_to_contract`, find no matmul,
and then `lower_to_interface` raises `no matmul payload`; those must go through
`llvmlower.lower_model`. So "RVV = LLVM path, accelerator = staged path" is the wrong model.

The router that chooses between them belongs in **core**, keyed off the resolved target contract and
dialect-plan coverage via `targetgen.target_registry` — never in `merlin.triton`, which would
violate INV-2. It is now `merlin.compile_core.choose_route` / `compile_core_mlir`, and it routes on
the *intersection* of what a target's dialect plan declares with what the interface layer can
actually build (`STAGED_MATERIALIZABLE`), because a plan may legitimately declare coverage for an op
`interface` has no way to materialize — which is exactly finding 4. Coverage is read from two
committed plan spellings (`from: interface.matmul` and `op: matmul`) structurally.

One consequence worth stating: an unreadable dialect plan **fails closed**. Reading it as "this
target accelerates nothing" would silently demote an accelerator to the generic path and still
report success — and it is the common case rather than an exotic one, since a target whose plan
lives in its out-of-tree package has no in-tree plan to read and must be passed as
`target_package=`.

### 4. The `interface` dialect had no elementwise op, and Radiance is where that bit

Radiance's committed dialect plan declares both `radiance.matmul` and `radiance.elementwise`, and its
contract lists `capabilities.ops: [matmul, elementwise]`. But `interface.py` registered only
ResidentPack / ResidentEvict / Matmul / AccumulatorCreate / Accumulate / Commit / AsyncCopy / Await /
Fifo\* / CommandRegion — no `interface.elementwise` — and `lower_to_interface` materialized matmuls
only. A Triton vector add therefore could not reach `radiance.elementwise` through the staged pipeline.

The decision criterion is whether an equivalent **non-Triton** program hits the same wall. A
hand-written `linalg.add` did, identically. So this was a *Merlin interface-abstraction gap, not a
Triton gap*, and the fix is an `interface.elementwise` every frontend benefits from rather than a
Triton special case. **Closed in M5b** — see the results section for what was and was not included.

## How pointers become tensors

The bridge's hard half is undoing what Triton's frontend did: recovering "multiply these two
matrices" from "one program instance computes these addresses and loads through them". It rests on
two observations.

*Addresses are affine.* Every index-space value in TTIR is `constant + Σ cᵍ·program_id[g] +
Σ kᵈ·iota[d]`, built by `tt.make_range` / `tt.splat` / `tt.expand_dims` / `tt.broadcast` /
`arith.{addi,muli}`. `addressing.Affine` is exactly that form; anything that does not fit — a product
of two varying indices, a bitwise index computation — is refused rather than approximated.

*The grid is enumerated, not lowered.* Every program instance's accesses are collected, and if they
tile a declared argument exactly — each element once, **in order** — then the whole launch is
equivalent to naming that argument as a tensor and the grid has vanished without any parallelism
decision being taken. Full-but-reordered coverage (a transposed tile) is reported as its own case
rather than accepted, because it covers perfectly and means something different.

Doing that check *concretely* is what makes masks impossible to ignore. A masked tail is precisely
what stops `ceil(N/BLOCK)` instances from running past the end, so dropping a mask turns exact
coverage into over-coverage and the check fails. The pair
`vector_add_unmasked` **accepted** at n=1024 and **refused** at n=1000 is the regression test for
that: same kernel, same grid arithmetic, only the declared extent differs — and it is exactly the bug
that hides whenever the block size happens to divide the tensor.

Two consequences worth stating. A runtime scalar used as a mask bound must be given a compile-time
value in the spec's `assumptions`, because a bound that cannot be checked is not assumed. And a
contraction under a multi-program grid is refused even when coverage is perfect: a batched matmul
covers every argument exactly once, and folding its grid away would turn a stack of small products
into one large one.

## Grid / SPMD normalization, and why the two accelerators differ

A Triton program describes **one program instance** in an SPMD launch grid. The bridge's job is to
turn (program body + grid) into ordinary whole-function semantics — *normalization*, not lowering.
Execution order is sequential at this boundary because choosing target parallelism is Merlin's
decision, not Triton's.

This is what the Radiance arm actually tests. Gemmini's proof runs at `grid=1` and sequentializes.
Radiance carries `compiler_obligations: [must_map_to_warps]` and `capabilities.simt.lanes_per_warp:
16`, so the same grid must become *warps*. Both have to fall out of one normalized form: bake
"grid → threads" into the bridge and Gemmini breaks; bake "grid → sequential loop" and Radiance can
never use warps. A single accelerator cannot detect either mistake.

Relatedly, GPU-specific autotuning knobs are not portable target semantics. `num_warps` and
`num_stages` are accepted only to satisfy the Triton frontend and recorded as provenance;
`BLOCK_*` / `GROUP_*` are kept as portable meta-parameters that Merlin may later treat as schedule
candidates.

## Conventions this work must respect

- **Tests** go in the bucket of the subsystem they exercise; `merlin/tests/triton/` is not a legal
  bucket (`check_structure.py` fixes the set). Bridge/IR tests → `ir/`, guards → `infra/`, the RVV
  arm → `rvv/`, Gemmini → `gemmini/`, generated-target descent → `targetgen/`.
- **Artifacts** go through `artifacts.new_product("triton-kernel", target=…, version=0)` and
  `cache_dir("triton")`; a top-level `out/triton/` would violate the one-root convention.
- `merlin/python/merlin/triton/**` is inside both gate scan roots: **no `import re`**
  (`check_no_regex.py` — parse structurally) and **no target-name literals**
  (`check_no_target_name.py`).
- Triton itself is a stock wheel behind an optional extra, used read-only. All dependence on Triton
  internals is confined to one module (`source.py`) so that the rest of the package is insulated
  from Triton's fast-moving compiler API.

## Results, and where the Radiance arm stopped

Measured on this branch:

| arm | result |
| --- | --- |
| host | vector add exact vs NumPy at 15 extents; int8 and fp32 matmul exact / within f32 tolerance |
| RVV (spike) | bit-identical to host; `vsetvli` + `vfadd.vv` in the emitted object; **0** frontend lines |
| Gemmini | full staged descent, three-way bit-exact on **Verilator RTL** (`derived_from_rtl: true`), 576 cycles |
| convergence | Triton vs hand-authored linalg: byte-identical interface, target, runtime and command buffer — no canonicalization, on gemmini *and* saturn |
| portability | identical TTIR and core MLIR across every stageable target; each lowers to its own dialect |

**Radiance (M5a) is done.** It needed a target-dialect package, which did not exist — `radiance_oot`
is contracts-only and TargetGen synthesizes a deliberately *empty* xDSL dialect for it. One was
authored at `out/artifacts/targets/radiance/hand_v0`, and the byte-identical kernel now descends to
`radiance.stage` / `radiance.matmul` / `radiance.commit` / `radiance.release` with the command buffer
gated against an independent integer reference. It is deliberately **not** a renamed Gemmini: a
systolic array packs an operand into a feed, a SIMT cluster stages it into shared scratchpad, and
Radiance's `must_map_to_warps` obligation makes `radiance.matmul` *require* a warp width that
`gemmini.matmul` has no property for at all.

Nothing in the package is typed in. The scratchpad capacity is `1 << SMEM_LOG_SIZE` read from
Radiance's own kernel headers; the warp width comes from its committed manifest; and the
abstraction-surface features the staged pipeline checks (`resident_packed_tensor`,
`accumulator_commit`, `command_buffer`, `metrics`) are each derived from a specific contract fact,
because a feature string with no justifying fact is a fabricated capability that every later check
believes. `derive_facts.py` records the provenance per fact.

That needed one general seam in core: a `TargetSpec` may carry properties its own contract supplies,
which the rebuild loop merges without interpreting. Core therefore never learns that any target maps
to warps — the package derives it and fails closed on a contract demanding the mapping without
declaring a width.

**The grid claim (M5b) follows from it.** The bridge normalizes the SPMD grid away entirely — no loop,
no lanes, no warps — so the parallelism decision is still unmade when Merlin takes over. From one
identical module Gemmini emits no warp mapping and Radiance emits `lanes_per_warp = 16`. Had the
frontend chosen threads, Gemmini would break; had it chosen a sequential loop, Radiance could never
reach warps. Target #3 cost **0** frontend lines.

**`interface.elementwise` (M5b) is closed**, on the evidence that decided where it belonged: a
hand-authored `linalg.add` was routed to the generic path identically to the Triton one, making it a
Merlin abstraction gap rather than a Triton gap. The op now exists, `lower_to_interface` materializes
an elementwise payload, and an integer Triton vector add descends to `radiance.elementwise` and a
`VECTOR_MAP` command whose output matches an independent reference. The runtime tier already
implemented `VECTOR_MAP`; only the compiler path was missing, exactly as the finding said.

Four boundaries were held rather than papered over:

- **Coverage stays per target.** Gemmini, saturn and toy_npu declare no elementwise coverage, so the
  same kernel still routes to LLVM there. The route is the intersection of what a plan declares with
  what the interface layer can build.
- **`linalg.sub` is not expressible.** The runtime implements add and mul; lowering a subtract as an
  add would be a miscompile.
- **A fused matmul-plus-elementwise payload is refused.** Materializing it means deciding whether the
  combine folds into the commit epilogue or becomes its own dispatch — a scheduling question with a
  measurable answer, not one to guess at in a rebuild loop.
- **Float elementwise still cannot descend.** The command-buffer ABI is integer; that is a runtime-tier
  gap, not an elementwise one, and it is not claimed as fixed.

### Two fail-opens found on the way

`load_curated_contract` returned the *default* contract whenever a target had no in-tree contract
file. Asking for any out-of-tree or misspelled target therefore lowered the whole module for
`toy_npu` — and produced a module that verified at all six stages, simulated correctly, and emitted a
command buffer naming a target nobody asked for. It now resolves through the registry and raises
otherwise.

The bridge's dead-code detection was only one level deep. Triton guards every arithmetic expression
with an overflow check (widen to i64, compare, and), and that chain is dead only *transitively*: a
single level leaves the widening looking live, and the bridge then has to interpret an i64 shadow of
the real computation — which for a data value it cannot. It now computes backward liveness to a
fixpoint from the side-effecting ops.

Both are the same shape as the mask-dropping the bridge exists to prevent: a wrong answer that passes
every check.

## What is explicitly not being built yet

Tiled/multi-program GEMM, masked tails on the accelerator path, softmax, model kernel replacement,
`torch.library.triton_op` integration, autotuning, Autocomp and kernel mining are all gated behind
the standalone end-to-end result. `merlin/python/merlin/kernels/ingest/triton.py` already
text-mines `@triton.jit` for the kernel index — that is mining, a different abstraction, and it stays
untouched. `merlin.triton` also stays out of `frontends.registry`, which maps checkpoints to
`CaptureBundle`s.
