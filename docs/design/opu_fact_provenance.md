---
title: "Design: which OPU facts come from the ISA, and which only from measurement"
kind: design
status: draft
owner: targetgen
last_verified: 2026-08-17
related: [incremental_target_evolution_opu, target_onboarding, reproducibility]
code_refs:
  - merlin/python/merlin/targetgen/rtl/opu_isa.py
  - merlin/python/merlin/kernels/opu_cert.py
  - merlin/python/merlin/kernels/census.py
  - merlin/python/merlin/llvmlower/opu_shim.py
  - merlin/python/merlin/targetgen/routing.py
  - merlin/python/merlin/targetgen/corpus_spec.py
  - merlin/contract/matrix_units.yaml
---

# Which OPU facts come from the ISA, and which only from measurement

## Why this document exists

"The OPU is basically an extension of the ISA" is true, and it is the reason capsule generation for this
unit can be automated: legality and geometry are *derivable* from the target's own sources. But it invites
a second, false inference — that everything needed to make a layer OPU-friendly is therefore derivable
too. It is not. Every fact that decides whether routing a contraction to this unit is a *good idea* came
from measurement, and three of them were only findable by running the whole model on real RTL.

This document draws that line explicitly, names the extraction point for each fact, and says which tooling
is needed versus already present. It exists because the line is not obvious from the code: the ISA
derivation and the measurement paths live in different modules with no shared vocabulary, so a reader can
easily conclude that one subsumes the other.

## The two OPU surfaces — read this first

Two different endpoints are both called "the OPU", and the ISA claim applies to only one of them.

| | RVV **matrix extension** | **command buffer** |
|---|---|---|
| software endpoint | vector instructions (`.insn`) on opcode `0x57` | one-hot command ports (`macc`/`mvin`/`shift`) |
| configs | `KodiakOPU1CoreConfig`, `OPUV256D128ShuttleConfig` | `saturn_opu_mxv256d128` (arc model) |
| has an instruction decode? | **yes** — so instruction classes derive | **no** — classes derive to `[]` |
| tile edge notion | VLMAX, from `vLen`/SEW (`opu_cert.logical_tile_edge`) | `capabilities.mesh.rows` = 16 (`spatial_introspect`) |
| bench corpus | none (31 frozen cases in `kernels/opu_corpus.py`) | 31 capsules from `profiles/saturn_opu.yaml` |

These are the same silicon and different contracts. The tile-edge row is the trap: both are "the tile", the
numbers differ by construction, and nothing in `corpus_spec._tile_dim` distinguishes them — it takes
`capabilities.mesh.rows` when present. A generator that assumes one tile per target silently produces
capsules for the wrong geometry.

## Derivable from the ISA / RTL

All of the following come out of `targetgen/rtl/opu_isa.py`, which reads Saturn's own Chisel
(`Consts.scala`, `Instructions.scala`, `Parameters.scala` — located via `merlin/contract/matrix_units.yaml`)
and **cross-checks every field against an independent witness**, the expert C header's `.insn r` macros.
`crosscheck()` fails closed on any disagreement, and `IsaDerivation.ok` requires no gaps *and* every
crosscheck agreeing, so a partial derivation cannot masquerade as a complete one.

- **Encodings.** `funct6` is a `ChiselEnum` ordinal, so it is the count of `Value` slots declared before the
  name: `opmacc = 40`, `opmvin = 42`, `opmvinbcast = 44`, `opmvout = 46`, giving `funct7 = (funct6<<1)|1`
  → `0x51`/`0x55`/`0x59`/`0x5d` on opcode `0x57` (OP-V). Nothing here is a literal in our code.
- **Tile geometry, as a formula rather than a number.** A cell is one int8 MAC/cycle; a cluster is
  `(cWidth/aWidth) x (cWidth/bWidth)` cells; the array is `(dLen/aWidth)/clusterY x (dLen/bWidth)/clusterX`
  clusters. So int8 peak is `(dLen/8)^2` — 1024 MACs/cycle at `dLen=256`, 256 at `dLen=128`. The logical
  tile is `vLen/SEW`, held across `(vLen/dLen)^2` MRF sub-tiles, which is why a full-tile accumulate cannot
  beat 4 cycles at Kodiak's geometry.
- **Register-file depth.** `nMrfRegs`, and `regsPerCell = (vLen/dLen)^2 * nMrfRegs`.
- **Datapath type pairs.** int8 → i32 and fp8 → f32 accumulate.
- **Ordering legality.** mvin before macc; the op categories a command buffer may contain.

That set is sufficient to *generate* a corpus: instruction classes, legal operand shapes, tile edges and
brackets around them. It is not sufficient to know what to do with any of it.

## Measurement-only: the transformation economics

Four facts, none of them expressible in an encoding, each with the place it actually comes from.

### 1. Rate and per-tile overhead — and why one shape cannot give them

`kernels/opu_cert.solve_unit_rate` recovers **142.8 MACs/cycle** and **2079.5 cycles per tile-pair** from
the certification run itself. The two terms are separable only with **at least two K points**: a single
measurement conflates the rate with the fixed overhead, and a corpus built at one reduction depth will
price a tiled unit confidently and wrongly. Against the derived 1024 MACs/cycle peak, the certified
microkernel sits at 13.95% — a 7.2x gap, attributable to 7 instructions plus 3 `vsetvli` per K-step. The
ISA tells you the peak; only the run tells you the fraction of it a real kernel reaches.

### 2. Layout-conversion cost — 87% of the routed region

The ISA states that the accumulate consumes a K-major LHS. It says nothing about what reaching that layout
costs, and the answer dominated everything else: the LHS pack was **87% of the routed region**. The
signature that identified it is what matters methodologically — the excess was a **constant 21.3–26.6
cycles per packed element across five shapes whose pack sizes differ 4x**, i.e. a per-byte cost, not a
shape-dependent one, which is what proved it cache-line behaviour (`pack[kk*M + i]` written down a column
of k, so every store opens a new line and each line is refetched once per row of A) rather than arithmetic.
Extracted by joining `op_profile` PROF ticks to per-contraction M/N/K in `kernels/census.py`.

Two corollaries worth keeping: blocking that loop took the model from 4,010,140,489 to 3,960,882,937 cycles
bit-exact, and **the routed region must not be priced by subtraction** — control-matmul-share minus
whole-model-saving gave 172,389,266 cycles where per-contraction measurement gives 114,276,780, a 33%
error, because the two legs do not spend equal time elsewhere.

### 3. The memory-ordering hazard — found only by the whole model on real RTL

Batched contractions route correctly and produced a **1.586x speedup that was wrong** (`cos 0.99995,
max_rel 0.83`). Every slice packs into the same scratch buffer, and Saturn is **decoupled** — the scalar
core runs ahead of the load/store unit — so slice i+1's stores landed on bytes slice i's kernel was still
reading. A `fence rw, rw` between slices restored bit-exactness for 0.005% of the cycles.

This is the load-bearing example for the whole document. Decoupling is a microarchitectural property that
appears in no encoding, and **neither existing guard could have caught it**: the acceptance corpus hands
the kernel pre-transposed operands and never packs, and spike orders memory functionally. The rank-2 path
cannot hit it either (one pack, one kernel, no rewrite mid-read). It was localised by elimination —
scalar stand-in bit-exact on spike, both legs sharing a `build_hash`, and a 7-case RTL certification of the
exact attention extents returning 0 mismatches — which exonerated routing, ABI and datapath and left only
what the batched form does differently.

### 4. Architectural state the ISA cannot see

The MRF is **64 KB per core** of state that no context switch saves. It is `regs` inside each
`OuterProductCell`, addressed by `mrf_idx` — microarchitectural, *not* the RVV register file — so Zephyr's
`z_riscv_vstate_save` (v0–v31) does not touch it. Harmless on a single-hart bare-metal image with no
preemption; on preemptive SMP a switch mid-GEMM corrupts it silently. `OPMVOUT`/`OPMVIN` *can* save and
restore it (~256 row ops) but nothing does. Extracted from RTL structure via mlc/CIRCT, and the same shape
as the `CONFIG_RISCV_ISA_EXT_V`/`mstatus.VS` hang: **no simulator we have reproduces either**.

## Tooling: what exists, what is missing, what is not needed

**Already present, needs feeding rather than building.** `opu_cert.solve_unit_rate` (pricing, given two K
points); `op_profile` + `kernels/census.py` (per-contraction attribution by `prov` key — note the
instrumented module is round-tripped through `mlir-opt`, so `func.call` records as `call` and matching only
the qualified spelling reads as "never profiled"); `dse_guidance` (ranks axes by measured gap closure);
`routing.MeasuredCost`, which already accepts `pack_cycles_per_element`, `requires_k_major`, `tile_edge`
and `tile_overhead_cycles` — the cost model can *express* the pack cost today.

**Genuinely missing: a layout-decision seam.** The cost model can price packing, but the decision to pack
is hardcoded in the emitted shim (`llvmlower/opu_shim.py`), so "consume an already-K-major A" is not
expressible as an alternative. That is why a cost worth 87% of the routed region has nowhere to enter a
decision, and why the structural fix (stop packing; `linalg.transpose` is 7.92% of the run) cannot currently
be chosen by the compiler rather than by an edit. This is the one piece of new machinery this analysis
justifies.

**Missing but not ours to write.** A hazard oracle for the decoupling and unsaved-state classes belongs in
`merlin.liveness` ("would it stall or fault on silicon?"), which currently lives on another branch and
arrives with that merge. Writing a second one would duplicate it.

**Not needed.** A new dialect. Every fact above is already derivable or already measurable; what is missing
is where the measured cost enters the choice, not a new representation.

## Consequences for corpus generation

1. An ISA-derived corpus proves a backend **correct** and can never say whether packing, transposing or
   batching was the right thing to do. Both kinds of evidence are needed, and they come from different
   places.
2. Any generated corpus must include **at least two K points** per shape family, or the unit cannot be
   priced from it.
3. Generation must not assume one tile per target (see the two-surfaces table).
4. Coverage that derives to an empty class list cannot fail. On the command-buffer endpoint that is honest —
   there is no decode — but it means a submission passes coverage vacuously there, and the surface with real
   classes is the one that has no bench corpus yet.
