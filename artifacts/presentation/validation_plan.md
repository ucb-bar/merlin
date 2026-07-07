# Validation plan — what makes the result defensible

This is the test / benchmark / validate checklist for a strong, defensible result. It has
two axes: **(A) the runtime/execution result** (models lower + run correctly and measurably
on RISC-V hardware) and **(B) the abstraction research** (which HW/SW abstractions are worth
exposing — the actual thesis). Status legend: ✅ done · ◐ partial · ⏳ todo.

---

## Axis A — Execution: correct, general, real, measurable

### Claims
- **C1 Correct** — execution is numerically faithful to PyTorch across a broad matrix.
- **C2 General** — LLMs *and* diverse VLAs, fp32/int8/fp8, not one cherry-picked net.
- **C3 Real** — runs on cycle-accurate FPGA (FireSim), measured cycles.
- **C5 Performance** — measured scalar baseline → quantified RVV / accelerator speedup.

### Test (correctness)
Governing invariant — two independent checks: **host == torch** (faithfulness) and
**substrate == host** (codegen).
- [ ] Full matrix per (model × dtype × substrate): 11 models × {fp32,int8,fp8} ×
      {host, spike, FireSim}. Metrics per cell: **cos, rel, full-output SUM(rel),
      per-row ARGMAX token-exactness**. ◐ (FireSim: 7 strict-pass + bitvla×2 functional)
- [ ] cos alone is insufficient for >4096-elem outputs → SUM+ARGMAX gating ON for every
      large-output cell. ✅ (wired into `firesim_sweep.py`)
- [ ] Per-kernel bisection vs numpy (`kernel_backend.py`). ✅
- [ ] Outliner + DispatchProgram **bit-identical** to monolithic compile (regression gate). ✅
- [ ] Op-coverage fixtures (each broke once): bf16/`__truncsfbf2` ✅, int8/fp8 dequant ✅,
      torchao qinner ⏳ (xr0 int8 cos=0), bool→float cast ✅, over-rank matmul ✅,
      `roundeven` ✅, rotary/softmax/rmsnorm precision ⏳ (xr0 fp32 0.9992), multi-input
      `input_order.json` ✅.
- [ ] Determinism: same MLIR ⇒ same output bits; ledgers are the audit trail. ✅
- [ ] Envelope/negative tests with hardware evidence: pi05_int8 > 16 GB DRAM (physical);
      >4 GB blob needs 2-cell DT; RVV-on-Saturn hang. ◐ (documented, not all reproduced as tests)

### Benchmark (performance)
- [ ] Compute-only cycles (mcycle around `merlin_run`, excludes boot/load/setup) per
      model×dtype, reported as cycles + cycles/token (or /output-elem). ◐
- [ ] **Scalar baseline → RVV speedup**, quantified per model. ⏳ ← biggest result gap (Stage 3).
- [ ] Memory footprint: weights blob, activation-arena high-water, image size; fp32 vs
      int8 vs fp8 size reduction. ◐
- [ ] Quantization cost/benefit: int8/fp8 dequant cycle overhead vs footprint win. ◐
- [ ] Multicore scaling: dispatch-DAG schedule across harts (small_llama 1.66×@4; tiny_llama
      critical-path-bound). ◐
- [ ] **An external baseline** to compare against (llama.cpp on the same core / torch-CPU /
      published Gemmini-Saturn). ⏳ ← "strong" is comparative; without this the cycles are
      descriptive only.
- [ ] FASED wall-rate + setup-overhead breakdown (boot/infrasetup/load vs compute). ◐

### Current execution scoreboard (FireSim, merlin-oscar, cos>0.9999 + full-output)
- PASS (9): small fp32, small_llama int8, tiny_llama int8, openvla int8+fp32, rdt2 int8+fp32,
  **bitvla int8+fp32**. bitvla promoted: cos 0.999995 + 100% per-row ARGMAX; its `sum_rel` (0.44)
  is a catastrophic-cancellation false-fail — the golden logits sum to ~1e-4 of their |mass|, so
  the SUM gate is meaningless there. `firesim_sweep.py` now applies SUM only when |sum| ≥ 0.1·‖g‖₂
  (well-conditioned); otherwise ARGMAX governs the full output.
- xr0 int8 (qinner) — **Fix B DONE** (`build_tools/scripts/fix_qinner.py`): compiled-path 0.0 →
  cos 0.9993 (spike), ≈ xr0_fp32 0.9992 ceiling ⇒ residual is the shared xr0 capture gap (#60),
  not a quant bug. xr0_int8_fixed FireSim run in flight (ledger /tmp/fs_xr0fix.jsonl).
- Open (grinding, FPGA-time): smolvla ×2 (bf16 regime ~0.978 expected), rdt ×2, molmoact int8,
  groot int8, tiny_llama fp32; groot/molmoact fp32 (external-weights 2-cell DT, **HW-unvalidated**
  — 3.5–4.6 GB image load); pi05 fp32 (heavy; interpreter cos 0.998).
- Capture-capped (not codegen): xr0 fp32 + int8 (~0.999, #60).
- Hard limit: pi05 int8 (current capture is unquantized ~15.4 GB > 16 GB DRAM; needs #61 real-int8 re-capture).

---

## Axis B — Abstraction research: which abstractions are worth exposing (the thesis)

Four workstreams + the closed loop. The defensible core is **prediction == measurement,
with falsifiers** — not "we ran models fast."

### WS2 — Kernel abstraction extraction (`docs/kernel_mining.md`)
**Claim A1 (discovery is real, not corpus-bias) · A2 (decisions not constants, generalizes).**
- [ ] Extraction fidelity: `kernel-audit` marker precision (a marker means what it claims);
      determinism (same repo ⇒ identical records); content-hash dedup collapses verbatim copies.
- [ ] Cross-source **fractions** (not counts) — the corpus-size-bias control (`motif_source_heatmap`).
- [ ] Promotion ladder is the falsifiability spine: Observation → Motif → Policy (≥2 sources
      OR ≥N kernels) → **Validated** = fires on benchmark positives across the regime grid
      where reuse ≥ 2 AND **silent on negative controls** (mutable-RHS, no-reuse).
- [ ] Every promoted policy carries a **falsifier** + downstream consumer + Stage-D verdict
      (actionability scorecard); invariants "surprise list" catches over-firing markers.
- [ ] Hard line: no kernel is executed/timed in mining; plots are evidence-frequency, never speedup.

### WS3A — Design pressure (`docs/design_pressure.md`)
**Claim A3 (demand is measured at the right granularity).**
- [ ] Controlled examples hit expected recommendations incl. the negative:
      `repeated_rhs_matmul → resident_packed_tensor`, `matmul_bias_requant_relu →
      accumulator_commit`, `no_reuse_matmul → none`.
- [ ] Pressures (reuse/mutability/lifetime/pack/layout/intermediate-bytes/dispatch) stable
      across all 6 cut points (graph/linalg/loop/bufferized/dispatch/trace).
- [ ] WS2↔WS3A convergence: the workload that *demands* X is the regime where X was *mined*.

### WS3B — DSE / exploitability (`docs/dse.md`)
**Claim A4 (worth is quantified with a measurable, calibrated cost model).**
- [ ] Variant ordering sane: oracle ≥ software_visible ≥ {baseline, hardware_managed}.
- [ ] **Cost-model params calibrated from real spike/FireSim measurements** (dispatch_fixed,
      pack_startup, bytes/cycle, …) — NOT guessed. (Indefensible on guessed params.)
- [ ] `exploitability_report` over a parameter sweep; expose/don't-expose verdict robust,
      incl. a regime where it's **not** worth it (conditional > universal).

### WS1 — TargetGen / MLIR dialect generation (`docs/targetgen.md`, `adding_a_target.md`)
**Claim A5 (chosen abstractions representable, verifiable, lowerable; targets generatable).**
- [ ] Generated artifacts validate against schemas (target_contract, dialect_plan,
      dialect_requirement); generated dialect `verify()`s + round-trips; deterministic, no LLM.
- [ ] Every dialect op in `dialect_plan.yaml` has ≥1 lowering (interface→target table covers it).
- [ ] `toy_npu` fully functional; ≥1 **real** target (saturn/gemmini) past skeleton.
- [ ] Hard line: "structured human-reviewable scaffold with validation gates," NOT
      "RTL → correct dialect."

### Dialect-stack lowering on real models (`docs/lowering_pipeline.md`)
- [ ] Phases 1–4 (contract → schedule → interface → target) run on a **real** model, not just
      the synthetic workload. ⏳ ← the architectural proof; today only Phase 3 (outline) /
      Phase 5 (dispatch program) / 6–8 (backend) are real-model.

### The closed loop (the killer result) — **Claim A6: prediction ≈ measurement**
Take one abstraction (e.g. `resident_packed_tensor`) end-to-end:
```
mined (cross-source, deduped) → demanded (design pressure, silent on no_reuse)
  → worth it (exploitability on MEASURED params) → represented (interface op, verifies, lowers)
  → generated (toynpu via dialect_plan) → executed (spike/FireSim)
  → MEASURED speedup vs baseline ≈ predicted exploitability
```
- [ ] The last line — cost-model-predicted capturable benefit matches measured HW speedup —
      validates the whole methodology, not one number. ⏳

**Claim A7 (honest & falsifiable):** every candidate has a falsifier; negative controls stay
silent; the promotion ladder filters one-source flukes; dialect status reported truthfully.

---

## The gaps between "works" and "strong & defensible"
1. RVV codegen (Stage 3) — no speedup number without it (biggest gap, Axis A & B).
2. An external baseline to compare against (Axis A).
3. Calibrate DSE cost-model params from real measurements, then prediction-vs-measurement (Axis B).
4. xr0 correctness (qinner + precision) — the one not-clean model.
5. Phases 1–4 on a real model — the dialect architecture proof.
6. fp8 coverage on FireSim; finish the FireSim matrix (FPGA-time bound).

---

## Diagnostics (2026-06-08) — precise characterization of the open gaps

Findings from local analysis (no FPGA, no heavy execution), sharpening the ⏳ items above:

- **xr0 fp32 is a CAPTURE gap, not codegen.** The numpy interpreter (the host==torch
  reference) gives xr0_fp32 **cos 0.9998 / rel 0.023** — it already fails the gate *before any
  lowering*. So the FireSim 0.9992 is mostly inherited from the capture, not a compiled-path
  bug. Fix lives in model2MLIR/golden faithfulness (capture workstream), not the runtime.
  ⇒ xr0_fp32 should not be counted as a clean T1 model; re-audit its capture.

- **xr0 int8 (cos=0) needs qinner MATERIALIZATION.** Its 57 torchao subclass inner tensors are
  internal `tensor.empty`s the numpy interpreter overwrites at eval time from `extra.npz`
  `qinner::`; the compiled path can't touch internal allocs. Fix = lift the tagged empties to
  forward-function arguments before outlining, and embed them via `c_runtime` from `extra.npz`
  (reuse the existing buffer/lifted-constant embedding path). Bounded but intricate; isolated
  to torchao-subclass models (only xr0 today).

- **Phases 1–4 on a real model — the gap is the abstraction TRIGGER, not wiring.** Verified on
  small_llama: `run_dialect_plane` runs only quant→outline→dispatch (0/3/5); `lower_to_contract`
  and `lower_to_schedule` *run cleanly* but attach **0 ops**. Reason: `find_matmuls` works (15
  matmuls found in small_llama) but the resident-pack contract fires only on the synthetic
  **≥2 matmuls sharing a block-arg RHS** pattern — real single-forward captures have **15
  distinct RHS, 0 shared** (weight reuse is *batched* across the seq/contraction dim, not R
  separate matmuls). ⇒ To make Phase 1 meaningful on real models, extend contract inference to
  recognize an **immutable block-arg weight RHS** (read-only, reused across the batch dim) as a
  resident-pack candidate on a single batched matmul — then schedule/interface/target can fire.

### Update (2026-06-08, Fix A) — the resident-pack abstraction is a DECODE phenomenon
Extending contract inference + wiring contract→schedule into `run_dialect_plane` revealed a
sharp, defensible point: **a single-forward (prefill) capture has NO resident-pack reuse** —
each weight is used exactly once (small_llama: 15 matmuls, 15 distinct RHS, 0 shared even after
tracing through the `A@Wᵀ` transpose). The contract plane now runs on real models and correctly
reports **0** candidates there, matching `design_pressure`'s `no_reuse → none` negative control.
Resident-pack reuse is **cross-dispatch** (a weight used by ≥2 matmuls) — it appears in the
synthetic `repeated_rhs` and in **real autoregressive decode** (same weights across steps), NOT
in a one-shot forward. ⇒ To demonstrate the abstraction firing on a *real* model, capture a
**decode loop / multi-step** workload; a prefill capture is the wrong probe. (Transpose-tracing
is now in; the single-matmul "row reuse" idea was wrong — intra-matmul ≠ cross-dispatch — and
was reverted. Phase 3 `lower_to_interface` still rebuilds-from-scratch keeping only matmuls, so
real-model interface materialization needs an in-place rewrite.)

**Bottom line:** defensible when, for ≥1 abstraction, the four gates agree (mined / demanded /
exploitability-positive on measured params / representable+lowerable) **and** the measured HW
speedup matches the predicted exploitability — with negative controls proving the method
rejects non-abstractions.
