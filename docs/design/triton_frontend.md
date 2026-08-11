---
title: "Design: Triton as a target-independent kernel frontend"
kind: design
status: draft
owner: ir
last_verified: 2026-08-10
related: [lowering_pipeline, core_dialects, target_agnostic_core, target_resolution]
code_refs:
  - merlin/python/merlin/triton/__init__.py
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
INV-3  zero TargetGen / schema modifications through Milestone 4
INV-4  TTGIR is never a Merlin boundary
INV-5  no direct TTIR → gemmini/saturn/<any> lowering
INV-6  linalg-on-tensors is the single convergence point
INV-7  the generated target dialect stays the target-legality boundary
INV-8  unsupported Triton semantics FAIL CLOSED with a located diagnostic
INV-9  standalone and model-replacement paths share one bridge
INV-10 source → TTIR → core MLIR is reproducible and inspectable on disk
```

INV-3 is scoped to Milestones 0–4 deliberately: see finding 4 below, where a *general* Merlin gap is
expected to require a core dialect addition, justified by evidence rather than by convenience.

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

`triton-shared` also states in its README, verbatim: *"This repository is no longer maintained."* It
is not pip-installable — it builds as a `third_party` codegen backend inside a **source build of
Triton at a pinned commit**. So it is evaluated on the record (build cost, output dialect inventory,
whether it runs against an unmodified Triton) rather than assumed, and the default is to own a small
Python/xDSL bridge that never patches Triton. `Cambricon/triton-linalg` is the maintained C++
alternative, noted in the same evaluation.

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
violate INV-2.

### 4. The `interface` dialect has no elementwise op, and Radiance is where that bites

Radiance's committed dialect plan declares both `radiance.matmul` and `radiance.elementwise`, and its
contract lists `capabilities.ops: [matmul, elementwise]`. But `interface.py` registers only
ResidentPack / ResidentEvict / Matmul / AccumulatorCreate / Accumulate / Commit / AsyncCopy / Await /
Fifo\* / CommandRegion — there is no `interface.elementwise` — and `lower_to_interface` materializes
matmuls only. A Triton vector add therefore cannot reach `radiance.elementwise` through the staged
pipeline today.

The decision criterion is whether an equivalent **non-Triton** program hits the same wall. A
hand-written `linalg.add` does, identically. So this is a *Merlin interface-abstraction gap, not a
Triton gap*: the fix is an `interface.elementwise` that every frontend benefits from, never a Triton
special case. It is scheduled as the last milestone and is the only one permitted to touch core
dialect surface, and then only with that equivalence demonstrated first.

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

## What is explicitly not being built yet

Tiled/multi-program GEMM, masked tails on the accelerator path, softmax, model kernel replacement,
`torch.library.triton_op` integration, autotuning, Autocomp and kernel mining are all gated behind
the standalone end-to-end result. `merlin/python/merlin/kernels/ingest/triton.py` already
text-mines `@triton.jit` for the kernel index — that is mining, a different abstraction, and it stays
untouched. `merlin.triton` also stays out of `frontends.registry`, which maps checkpoints to
`CaptureBundle`s.
