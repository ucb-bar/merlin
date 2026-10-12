---
title: "Design: Triton as a target-independent kernel frontend"
kind: design
status: draft
owner: ir
last_verified: 2026-08-10
related: [lowering_pipeline, core_dialects, target_resolution]
code_refs:
  - src/merlin/triton/__init__.py
  - src/merlin/triton/source.py
  - src/merlin/triton/addressing.py
  - src/merlin/triton/bridge.py
  - src/merlin/triton/spec.py
  - src/merlin/compile_core.py
  - src/merlin/xdsl_dialects/lowering/pipeline.py
  - src/merlin/xdsl_dialects/lowering/contract_facts.py
  - src/merlin/xdsl_dialects/lowering/interface_lowering.py
  - src/merlin/xdsl_dialects/interface.py
  - src/merlin/frontends/linalg_mlir.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/kernel_backend.py
  - src/merlin/targetgen/target_registry.py
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

## Convergence and routing constraints

The bridge produces linalg-on-tensors, with tensor-typed function arguments and
complete operation semantics. Post-bufferization pointer or memref operations
cannot be treated as an equivalent ingest without their own supported conversion.

Core routing uses the intersection of declared dialect-plan coverage and the
interface layer's actual materialization support. Missing or unreadable target
plans refuse; they do not silently select a different target or imply that the
target accelerates nothing. Select an OOT plan explicitly when required.

Every payload operation must be accounted for. A verified module containing
only a subset of the source work is not a successful lowering. Shared interface
gaps belong in generic compiler infrastructure; target-specific realization
belongs in the target package. Unknown effects or unsupported compositions stay
refused.

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
- `src/merlin/triton/**` is inside both gate scan roots: **no `import re`**
  (`check_no_regex.py` — parse structurally) and **no target-name literals**
  (`check_no_target_name.py`).
- Triton itself is a stock wheel behind an optional extra, used read-only. All dependence on Triton
  internals is confined to one module (`source.py`) so that the rest of the package is insulated
  from Triton's fast-moving compiler API.

## Target realization and supported boundaries

Target lowering consumes independently selected resource, ISA and capability
facts. Memory address windows are not physical capacity, and a feature name
without justifying evidence is not an admitted capability. Missing target
requirements remain explicit refusals.

The generic frontend does not choose warp width, accelerator layouts or device
schedules. Elementwise descent requires both an expressible interface payload
and matching target coverage. Unsupported arithmetic, fused dispatch choices or
runtime precisions cannot be guessed from a related operation.

## Scope of the standalone bridge

Tiled/multi-program GEMM, masked tails on the accelerator path, softmax, model kernel replacement,
`torch.library.triton_op` integration, autotuning, Autocomp and kernel mining require separately admitted semantics and execution paths; standalone
bridge success does not establish them. `src/merlin/kernels/ingest/triton.py` already
text-mines `@triton.jit` for the kernel index — that is mining, a different abstraction, and it stays
untouched. `merlin.triton` also stays out of `frontends.registry`, which maps checkpoints to
`CaptureBundle`s.
