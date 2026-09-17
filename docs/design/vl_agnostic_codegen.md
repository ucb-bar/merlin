---
title: "Design note: VL-agnostic (scalable) RVV codegen — dropping the VLEN pin"
kind: design
status: current
owner: core
last_verified: 2026-07-19
related: [expert_gap_attribution]
code_refs: [merlin/python/merlin/kernels/microkernel.py, merlin/python/merlin/mining/from_strategy.py, merlin/python/merlin/llvmlower/impr_features.py, merlin/python/merlin/kernels/decode/rvv.py, merlin/python/merlin/mining/k1.py, build_tools/scripts/k1_microkernel_ipc_sweep.py]
---

# VL-agnostic RVV codegen

## Why the VLEN pin had to go

Our codegen emitted **fixed-width** vectors (`vector<32xf32>`), but `-march=rv64gcv` promises only the
RVV **minimum** VLEN of 128 bits. The backend therefore sizes each register group for that worst case,
so on the VLEN=256 K1 a `vector<32xf32>` becomes `e32,m8` — **double** the LMUL the board needs — which
spills inside the K loop and idles half the datapath. Pinning `_zvl256b` fixes it (`codegen_march`), but
pinning is target-overfitting with a correctness edge: on a VLEN=128 part `_zvl256b` is a **miscompile**
(the backend sizes register groups for lanes that do not exist), and on a VLEN=512 part it silently
recreates the half-idle datapath. The expert kernels never hit this because they size to the vector
length they **query at run time** (XNNPACK calls `__riscv_vsetvl_e32m4`), which is why their edge read as
"lanes per issue".

The general fix is `MicrokernelSpec.vl_strategy = VL_DYNAMIC`: emit a scalable register block so the loop
sizes itself to the hardware VL via a runtime `vsetvli`, needing no `_zvl` pin on any part.

## What actually blocked scalable lowering — reproduced on LLVM/MLIR 23

The prior note (`from_strategy._recipe`) refused the axis with *"MLIR's scalable-vector → RVV lowering is
incomplete (ub.poison survives)"* and pointed at the `custom_isa` inline-asm hatch. **That is no longer
true.** Re-running the accumulator-resident v3 recipe with a scalable N tile (`vector_sizes [4, [4], 1]`)
on the repo's LLVM-23 build, the ordinary lowering succeeds end to end — but only once two real defects of
the *naive* scalable schedule are handled. Both are reproduced with the exact IR in
`out/artifacts/cache/vl_agnostic/`:

1. **Masked scalable transfers block the accumulator hoist and leave an unlowerable transpose.**
   A scalable N tile makes the N trip count a non-constant multiple of the tile, so `vectorize` MASKS
   every transfer (`vector.mask { vector.transfer_read ... }`). Two consequences:
   - `loop-invariant-subset-hoisting` — the pass that makes the C accumulator a register-resident
     `scf.for` iter_arg — does not fire through the mask, so the K loop round-trips C through memory (the
     exact defect v3 exists to remove).
   - the masked read lowers through a `vector.transpose` on `vector<4x1x[4]xf32>` that has **no** scalable
     lowering; it survives to the LLVM edge as `error: Dialect 'vector' not found for custom op
     'vector.transpose'` (captured in `stage3.mlir`). *This* is the real residue behind the old
     "incomplete lowering" note — not `ub.poison` (which is inert and disappears with the masks).

2. **The fix: peel the N loop, then assume the tile matches.** `transform.loop.peel` on the scalable N
   loop makes the main loop's trip count an exact multiple of `vscale * k`, and vectorizing it with
   `assume_dynamic_dims_match_vec_sizes` (which the peel is what makes *true*) drops all masks. The
   subset hoist then fires exactly as in the fixed-width recipe, the transpose never forms, and the loop
   lowers to LLVM cleanly. The peeled remainder keeps its `linalg.matmul` and falls through
   `convert-linalg-to-loops` to a scalar tail (correct for any N, zero-trip whenever the hardware VL
   divides N).

3. **Inherited precondition `MR | M`.** `assume_dynamic_dims_match_vec_sizes` asserts *every* dynamic
   tile dim equals its vector size, M as well as N. The static `MR` M-tile has a partial last iteration
   when `MR` does not divide M, and the flag then writes `MR` rows into a shorter tile — an
   out-of-bounds write (measured at 130³: `malloc(): corrupted top size`; 128³ and 100³, where `MR|M`,
   are bit-exact). This is the **same** "MR must divide M" constraint the fixed 2-D `vector<MRxNR>` v3
   recipe already carries. It is *not* fixed by peeling M as well: `transform.loop.peel` fails on a
   statically-divisible loop, so an M peel would break the common `MR|M` case (M=128) to protect the
   `MR∤M` one. The M-tail is an orthogonal axis (pad/peel, shared with the fixed recipe); until it
   lands, `MR∤M` is **fail-closed** — the harness records `not_run` on the crash, never a false timing.

Realized in `impr_features._accumulator_resident_v3_scalable_pre_schedule` /
`ensure_v3_scalable_microkernel`, wired to the declared axis in `from_strategy._recipe`. **No
llvm-project fork and no `custom_isa` inline-asm were needed** — it is the ordinary MLIR scalable path.

## The emitted code — register-form `vsetvli`, no pin, no spill

`NR` under `VL_DYNAMIC` is reinterpreted as **lanes at the minimum VLEN**: a scalable `vector<[k]xT>`
holds `2k` lanes at 128 bits, so `k = NR/2` and the LMUL is fixed by the type
(`LMUL = k·sizeof(T)/8`), never widened for a worst-case VLEN. `NR=16 → vector<[8]xf32> → e32,m4`.
Emitted micro-kernel loops (128³ f32, K1 objdump, decoded via `kernels/decode/rvv.py`):

| config | march | effective vtype | inner-loop insns | in-loop spills |
|---|---|---|---|---|
| fixed `NR=32`, **pinned** | `…_zvl256b` | `e32,m4` | 12 | 0 |
| fixed `NR=32`, **no pin** | `rv64gcv…` | **`e32,m8`** | **25** (`vl8r.v`/`vs8r.v`) | **4** |
| **VL-agnostic `NR=16`, no pin** | `rv64gcv…` | `e32,m4` | 12 | 0 |

The fixed path without the pin is the miscompile-scale waste (m8, spills). The VL-agnostic path, **with no
pin**, recovers the pinned baseline's exact 12-instruction `vfmacc.vf` loop. Its vtype is established by a
**register-form** `vsetvli a0, zero, e32, m4` (rs1=`x0`, rd≠`x0` → set `vl = VLMAX` at run time), not a
`vsetivli` immediate — this is the instruction that makes the loop size itself to whatever VLEN the part
reports. (Decoder note: `InsnStream.innermost_vector_loop` scopes to the FMA-bearing loop, because the
peel leaves a shorter remainder loop that clang may auto-vectorize into a `vfredosum` dot-product.)

## Board measurement vs the pinned baseline

K1 (VLEN=256), 128³ f32, kernel bracket, min of 3, correctness-gated
(`k1_microkernel_ipc_sweep.py`, tags `vl_`):

| config | ticks | instret | digest |
|---|---|---|---|
| fixed `NR=32` **pinned** (baseline) | 14,018 | 273,887 | `7bde3077` |
| **VL-agnostic `NR=16` no pin** | 13,584 | 275,899 | `e164855a` |
| VL-agnostic `NR=16` KC=64 (identical-code control) | 13,244 | 275,898 | `e164855a` |

`KC` does not tile the reduction in this recipe, so the two VL rows emit **byte-identical** code (same
digest, same instret) — a free identical-config noise control per the sweep's own rule. Their 13,584 vs
13,244 spread is **2.6%**, i.e. the measurement floor. The VL-agnostic mean (13,414) sits **within that
noise band of the pinned baseline (14,018)** — the same performance, reached **without any `_zvl` flag**.

**Verdict: we can stop pinning for the accumulator-resident f32 micro-kernel.** The VL-agnostic loop
matches the pinned baseline on this board and, unlike the pin, is correct and full-width on any RVV part.
This is reinforced by a cross-cutting finding that the whole-model pin may not even take effect at scale
(emitted code stayed `e32,m4 vl=16` despite `codegen_march()` returning `zvl256b`); a runtime `vsetvli`
loop sidesteps the pin entirely.

## Correctness, including the N tail

Bit-exact vs the scalar golden on the board (`errors=0`, VERIFY PASS): 128³ (N tail zero-trip at
VLEN=256, since 128 mod 32 = 0) and **100³, where the peeled scalar N tail runs (100 mod 32 = 4)**. The
N peel makes `assume_dynamic_dims_match_vec_sizes` sound on N, so the main loop is exact and the N
remainder is a plain scalar matmul — bit-exact by construction and confirmed. Both verified shapes have
`MR | M`; the `MR∤M` case (130³) is the inherited M-tail limitation above and is fail-closed (`not_run`).

## What remains

- The **M-tail** (`MR∤M`) — pad-M-to-`MR` before tiling, or an M peel guarded against the
  statically-divisible case — would lift the `MR|M` precondition. It is the same missing capability the
  fixed 2-D recipe needs, so it should land once for both. Today `MR∤M` is fail-closed, not wrong.
- A **masked scalable N tail** (instead of the scalar peel remainder) would recover vector width on the
  last `N mod VL` columns; it needs the scalable `vector.transpose` lowering that today's LLVM-23 build
  lacks — the one genuine toolchain gap this work found. The scalar tail is correct meanwhile.
- Composition with `unroll_m` / `pack` / `k_block` still raises `UnsupportedAxis` (each replaces the whole
  schedule); a single composed recipe is the follow-on.
- Extend the scalable route to int8 (`vwmacc`) and fp16 (`vfwmacc`), and let the beam trade
  `VL_FIXED`↔`VL_DYNAMIC` as an ordinary axis now that it resolves.
