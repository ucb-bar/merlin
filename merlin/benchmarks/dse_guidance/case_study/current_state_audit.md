# Current-state audit (V0 freeze)

> **SUPERSEDED (P21–P23).** This is a historical V0 freeze (flat-capture corpus, pre-loop). The current
> state — loop-preserving corpus, structural role attribution (prefix/decode split), IR-recovered K,
> deployment magnitudes, native low-bit, residency-from-IR — is in `manual_validation/final_judgment.md`
> (P21/P22/P23 UPDATE sections) and `final_analysis.html`. Verifier is now 628/628 on the loop corpus.

**Date:** 2026-06-16 · **Branch:** `feature/kernel-policy-mining` · **Package commit:** `81ff6c9`

This is a freeze checkpoint. It answers, from this folder alone: can a reader understand and
reproduce the contribution, is every claim evidence-labeled, are caveats explicit, does the package
avoid speedup / quantitative-DSE claims, and does it regenerate byte-stably. **Verdict: PASS.**

Read this with `claim_evidence_matrix.csv` (claim → artifact), `known_limitations.md` (what is not
claimed and why), and `reproducibility_check.log` (the verification run). For an **independent
re-derivation** of every key number (recomputed from the raw captures and cross-checked against the
emitted artifacts), see `verification_report.md` — regenerate it with
`.venv/bin/python merlin/benchmarks/dse_guidance/verify_implementation.py` (37/37 checks pass).

---

## What this contribution is

Merlin is a **compiler-based workload-contract analysis** front-end for accelerator DSE. A flat
model capture (`model.mlir`, one unrolled forward) is not a DSE-ready workload description. From
provenance-rich captures Merlin recovers the temporal + numerical contract the flat capture erases
and hands a future DSE engine a package: region roles and facts, hardware-independent requirements,
the HW/SW abstractions the workload implies (with the compiler proof each needs), a DSE search-space
template, and a prioritized list of what is still unmeasured.

It does **not** run DSE, pick a design, or claim a speedup. That is the thesis, not a gap.

Workloads (real `prov.fqn` recaptures): **rdt, openvla, small_llama, tiny_llama, rdt2, groot_n1d7,
molmoact** (7 studyable). Deferred (captures exist via model2MLIR but not in the comparable-config
corpus yet): **xr0** (batched-attention DiT — only 2/19 matmuls are plain-2D; needs batch_matmul
support), **bitvla / smolvla** (parse-blocked on `tensor.collapse/expand_shape` typed-reassociation
syntax in the ingest xDSL), **pi05** (full-VLM, too large for the comparable corpus).

---

## What is real — no caveats

- **Role recovery from `prov.fqn`** — backbone/head/prefix-KV roles auto-recover with no operator
  map; OpenVLA's vision-backbone/decode-head split recovers automatically.
  → `<workload>/region_attribution.yaml`, `cross_workload_provenance.csv`.
- **The numerical-fidelity finding** — 20/20 quantized zoo captures store weights low-bit but run
  f32 matmuls; native low-bit compute + packed layout are absent from the capture.
  → `numerical_contract_fidelity_report.md`.
- **Measured int8 accuracy** — W8A8-vs-fp32, 5/5 models pass the band. → `accuracy_gate_report.md`.
- **Measured dispatch count** — matmul proxy under-counts real dispatch granularity ~12–14×.
  → `dispatch_coupling_report.md`.
- **Discipline** — no `gap_closure`; `speedup` appears only in disclaimers / `what_is_not_claimed` /
  `not_claimed` fields (honesty grep in `reproducibility_check.log`).

## What is real but caveated
Magnitudes are small random-init instances; K/H/deadline are `assumed_reference`; accuracy is
int8-only and host-interpreter; dispatch *cost* is not measured. See `known_limitations.md` §1–4.

## What is honestly not real yet (blocks quantitative DSE)
Per-component cycle calibration; real command/sync latency; real K; fp8/int4 accuracy; full-depth
captures; scale/sparsity metadata (erased by capture). See `known_limitations.md` §2–7.

---

## Readiness (from `dse_readiness_summary.csv`)

| workload | ready_structural_DSE | ready_quantitative_DSE | missing before quantitative ranking |
|----------|----------------------|------------------------|-------------------------------------|
| rdt | **True** | **False** | accuracy gates; target command/sync latency; real K |
| openvla | **True** | **False** | target command/sync latency; real K |
| small_llama | **True** | **False** | target command/sync latency; real K |
| tiny_llama | **True** | **False** | target command/sync latency; real K |

Structural DSE is ready now; quantitative DSE is explicitly **not**, with the missing measurements
named. The package is built to consume those measurements the moment they exist.

---

## Verification summary (full log: `reproducibility_check.log`)

1. **Reproducible** — `bash reproduce_case_study.sh` regenerates **81/81 generated artifacts
   byte-stably** (66 at the original V0 freeze + 15 from the P5a/P5b/P6 layer). The four V0 docs are
   hand-authored verification overlays (the only non-generated files here).
2. **Isolated** — no unrelated repo files modified; in-flight gemmini/llvmlower/runtime work untouched.
3. **Honest** — dangerous terms (`improvement`, `optimal`, `best design`, `predicted cycles`,
   `calibrated future`, `gap_closure`, `faster`) absent; `speedup` only in disclaimers/blocked fields.
4. **Readiness** — structural=True, quantitative=False for all four workloads.
5. **Tests** — 74 guidance tests pass; bounded suite 323 passed / 27 skipped (lone pre-existing,
   unrelated `test_precision` NaN deselected).

### P5a/P5b/P6 layer added on top of the freeze
`traffic_table.csv` (memory/reuse envelope), `dispatch_granularity_table.csv` (honest command-graph
view — syncs/dependencies `unavailable`, loop unrolled), `accuracy_gated_dtype_candidates.csv`
(int8 measured-pass; fp8/int4 blocked), plus per-workload `memory_envelope.yaml` /
`command_graph.yaml` / `numerical_candidate_certificates.yaml`. Includes a correctness fix:
`accuracy_gate._family` matched `w8a8` before `fp8`, so `fp8_w8a8` was misclassified as int8 and
would have falsely inherited the measured int8 pass — fixed (fp8 checked first), with a regression
test. See `known_limitations.md` §8–9 for what these artifacts honestly cannot show.

## Acceptance criteria
- [x] `current_state_audit.md` reads standalone
- [x] every key claim points to an artifact (`claim_evidence_matrix.csv`)
- [x] no unbuilt-hardware performance claim exists
- [x] reproduction passes (byte-stable)
- [x] limitations are explicit (`known_limitations.md`)

**This state is frozen. The next layer (P5/P6) builds on top without obscuring it.**
