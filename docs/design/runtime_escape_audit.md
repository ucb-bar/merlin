---
title: "Design note: auditing for runtime escapes in emitted compute regions"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [expert_gap_attribution, compiler_plane]
code_refs: [merlin/python/merlin/kernels/escape_audit.py, merlin/python/merlin/rvvgen/escape_sweep.py, merlin/python/merlin/llvmlower/selfcopy.py, merlin/python/merlin/kernels/cca.py, build_tools/scripts/k1_escape_cost.py]
---

# Runtime escapes: making a whole defect CLASS visible

## The class

A *runtime escape* is a call the compiler emitted where it should have emitted code. MLIR's
`memrefCopy` is the archetype: a rank-generic strided copy that walks elements with dynamic rank and
stride handling, correct for anything and ruinous for the 4x8 tile it usually gets handed.

One such escape — a `memref.copy %x, %x`, a buffer copied onto itself, that bufferization left behind
per output tile — accounted for **77% of the f32 GEMM's retired instructions**. Erasing it
(`llvmlower/selfcopy.py`, feature `erase_self_copy`) gave a 1.88x bit-exact speedup.

Two properties make this worth an instrument rather than a patch:

1. **It was invisible.** It did not appear in wall time as anything but "we are slow". No existing
   diagnostic named it. It survived to LLVM as an opaque call that no upstream fold touched.
2. **It does not transfer.** Each lowering path can have its own escapes, so "did we fix it?" has to
   be asked per (op, dtype, model), not once.

## What the audit measures, and the two ways it can lie

`kernels/escape_audit.py` attributes each escape to a call site inside the compiler-emitted
functions, tagged with whether a loop encloses it. **Depth, not site count, is the cost signal**: the
original defect was a *single* call site, which any count-based screen ranks as noise, sitting inside
a loop nest where it ran once per output tile.

Building this surfaced two ways an escape audit can produce a confident, wrong clean bill of health.
Both are now guarded, and both are pinned by tests:

**Loop structure must be read from the LINKED ELF.** On the K1 Linux build path the emitted
`model.o` still carries unrelocated branch displacements — every branch literally targets itself — so
`loop_spans()` reads empty and a per-tile call is filed as a harmless one-off. Measured on one build:
**0 back-edge spans in the object, 6017 in the linked ELF**. A scope with no loop at all is therefore
reported `unknown`, never clean. (The spike path does resolve its branches, so the CCA beam, which
lifts from a spike-built `objdump.txt`, is unaffected.)

**Escapes must be scoped to the compiler-emitted functions**, read from the object's defined symbols.
The linked ELF statically contains libc, whose internals call `memcpy`/`malloc` constantly; counting
over the whole binary drowns the signal.

A third limit is reported rather than fixed. Counting enclosing back-edge spans is a sound *depth*
only while a loop body is address-contiguous, which after loop rotation and block reordering it need
not be. When the enclosing spans do not form a containment chain the depth number is flagged
`depth_reliable=False`. Membership (`in_loop`) still stands; only the count is untrustworthy.

## What the sweep found

`rvvgen/escape_sweep.py` fans the audit across {op} x {dtype} x {model} x {baseline, feature}. Over
42 cells (7 ops x 2 dtypes, plus 7 whole models, each with and without `erase_self_copy`), all
readable, no unknowns:

**The escape is matmul-shaped, not universal.** Every elementwise and reduction cell — `gelu`,
`sigmoid`, `silu`, `softmax` — is clean in both dtypes, re-checked at a second and third shape each
(one outlier: `gelu` at N=1024 int8 shows a single in-loop `memcpy`). The whole matmul family —
`matmul`, `batch_matmul`, and `conv2d_as_matmul` — carries the per-tile `memrefCopy` in both dtypes.

That last point corrects a result this sweep initially produced itself, and it is the reason shapes
are now chosen deliberately. `batch_matmul` first came back **clean**, because the default cell used
N=8: exactly one `[1,4,8,1]` vector tile wide, so there is no N-tiling loop and no per-tile copy. At
N=64 the same op carries the escape like every other matmul. **A clean cell is worth re-checking at a
second shape before it is believed** — a degenerate shape produces a real, reproducible, and entirely
misleading negative.

**A prior belief was wrong.** The int8 no-op had been recorded as "int8 lowers differently and has no
self-copy". The int8 matmul *does* emit the in-loop `memrefCopy`, and `erase_self_copy` *does* remove
it from the emitted code. Whatever explains the 0.03% is not the escape being absent.

**Whole models carry many.** Baseline in-loop `memrefCopy` sites: small_llama 17, bitvla 19, openvla
35, rdt2 26 — cleared to zero by `erase_self_copy` in every case.

**One open suspect, deliberately not fixed.** The int8 path at **M < 8** emits `malloc`+`memset`
inside a loop body; at M >= 8 it is clean, and f32 is clean at every M. Minimal reproducer: an int8
matmul with M in {1,2,4}, N=K=64. This is the small-M regime of openvla/rdt2, and it is *not* reached
by `erase_self_copy`. It is reported as a screened suspect, not a finding: the allocations are of
constant size (so they look hoistable), but this is exactly the case whose depth is flagged
unreliable, and **redundancy was not proven**. Erasing an allocation that does real work is a
correctness bug, so nothing was changed.

## Method note: why the fit, not the wall clock

Cost is quantified with retired instructions on the same bracket as the timing, fitted as

    instret(N) = a*N^3 + b*N^2

over a square NxNxN GEMM. `a` is per-MAC (real arithmetic), `b` is per-output-element, which is where
per-tile overhead lands. On the original defect this gave a=0.197 ins/MAC and b=79 ins/output-element
and predicted a held-out size to 0.4% — that separation is what turned "we are slow" into a located
bug. `build_tools/scripts/k1_escape_cost.py` runs it, holding out the largest size as the check,
gated on `VERIFY PASS`.

## Using it

    # host-only screen, no board
    .venv/bin/python -m merlin.rvvgen.escape_sweep --models-f32 <bundles> --models-int8 <bundles>

    # board cost of a suspect, with the scaling fit
    .venv/bin/python build_tools/scripts/k1_escape_cost.py --dtype int8 --sizes 64,96,128,160

The general capability being pursued is "no redundant runtime escapes in compute regions". The sweep
is the screen; proving redundancy at the IR level and measuring the delta remains the bar for turning
a suspect into a fix, and any fix stays a default-off feature so the frozen `hand_v0` control keeps a
byte-identical lowering.
