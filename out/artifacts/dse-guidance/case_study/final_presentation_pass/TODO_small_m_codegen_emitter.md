# TODO: emit RVV-native tail directly when the generic vectorizer falls to scalar (small-M)

**Status:** open / Phase B of the approved methodology-fix plan
(`~/.claude/plans/sunny-scribbling-cerf.md`). Context doc: `WHY_WE_DONT_MATCH_XNNPACK.md`.

## The insight (why this is worth doing)
The openvla (M=17) / rdt2 (M=1) loss is **not** an ISA limit, **not** "LLVM can't emit a masked
store," and **not** a missing-instruction problem. It is an **abstraction mismatch in MLIR's generic
vectorizer**, and we are NOT bound by it:

- **RVV does tails natively.** It is vector-length-agnostic: `vsetvl` to the remaining length and the
  *same* store handles the tail (M=17, MR=4 → vl=1 on the last step). XNNPACK does exactly this.
  There is no "M not divisible by MR" problem at the instruction level.
- **What actually fails:** our pipeline models the M×N tile as a **fixed-shape** `vector<MR×NR×f32>`
  and then predicates the M dim for the tail → a masked `vector.transfer_write` on a **scalable**
  type. *That specific pattern* is what LLVM-23's RVV backend can't legalize → scalar fallback.
- **We already proved we can bypass it.** `accumulator_resident_microkernel_v3` + `accum_microkernel.py`
  (via the transform dialect, sidestepping the generic vectorizer) emits a register-resident
  `vfmacc.vf` kernel that **matches the hand ceiling on isolated cubes** (50,695 vs OpenBLAS 84,483 cyc).

## The TODO
Generalize the v3 register-resident emitter so it lowers on small-M **the RVV-native way** instead of
through the fixed-shape masked-transfer LLVM rejects:

1. **vl-clamped tail, not masked fixed-shape store.** In the v3 / `accum_microkernel` lowering, emit the
   M-tail by clamping `vl` (RVV-native) so M=17/M=1 lower to a tight `vfmacc.vf` + `vse32.v` loop, no
   scalar fallback. (M=1 is a GEMV — accept that MR-blocking degenerates, but it must still vectorize N,
   not go scalar.)
2. **Whole-model fidelity.** Verify (via Phase A intended-vs-achieved on the whole-model emit) the
   accumulator stays register-resident whole-model, not just on the isolated cube; fix the lowering so it
   does. lift_asm must report `accumulator_resident=True` whole-model.
3. **Route it as the real CODEGEN action.** Flip the `accumulator_resident` CODEGEN route
   (`action_catalog.py`, currently `pass:rvv-microkernel-emitter`, `forkable_now=False`) to
   `forkable_now=True` pointing at the generalized feature, so the (attainment-driven, Phase A) beam can
   select + measure it.

## Hard constraints (do NOT violate)
- **Emitting intrinsics / `llvm.inline_asm` / direct LLVM IR from a CODEGEN pass is allowed and IS the
  point** — that's still "the compiler." The line is *who authored it*: a pass parameterized on
  shape/dtype = compiler; a kernel hand-written once and called = a library.
- **The `ours_board` C shim stays a measurement instrument only.** Never ship a hand-written intrinsic
  kernel as the product — that rebuilds XNNPACK and voids the comparison.
- **Default-off + per-model `run_id` selection; cos-gated ≥ 0.9999; frozen baseline byte-identical.**
- **bitVLA regression gate:** re-run `scripts/k1_e2e_xnnpack.py`; fail if the bitVLA whole-model win
  (148 vs 167/180 ms) regresses.

## Why it matters / the payoff
This is the only kernel-level path to the experts' A-reuse on M=17/M=1, i.e. the only way our matmul
deficit on openvla/rdt2 shrinks from 7.9×/13.6× toward parity — the prerequisite for the ~30 ms
integration credit (inlined-vs-routed) to actually flip those whole-model results the way it does on
bitVLA. It also directly demonstrates the thesis: **a compiler emits the right instructions per-shape
where the generic path fails; a library can't fuse across its call boundary at all.**
