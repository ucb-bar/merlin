# Gemmini compiler: 57/57 device ops proven (frozen)

`submission/` is the frozen, authored Gemmini OOT compiler package (fix-repo `int` tip 5cb502c, grader package-id
a0ea14e44c6ab2da). Under the release grader and the frozen 193-overload harness (`../h193/`), it proves all 57 device
core-ATen ops. It also passes the in-scope 10k spec-validation cases on Spike and on Verilator, and every public
non-model capsule including the L3 Verilator RTL tier.

## Verdict (stage m27b, corpus fin6w2-m27b, evaluated 2026-11-06)

| gate | result | evidence |
|---|---|---|
| Native regrade (Spike L0-L2) + 193 FULL | device **57/57 proven** (weak 57, full definition 57, corpus-clean 57); core 320/321, scalability 111/111 | `evidence/frozen-int-5cb502c/regrade/`, `h193/` |
| 10k spec-validation, int8/int32 device lane, Spike | **466/466** | `evidence/frozen-int-5cb502c/10k/RESULT-10k-int1-spike.md` |
| 10k spec-validation, int8/int32 device lane, Verilator | **466/466** | `evidence/frozen-int-5cb502c/10k/RESULT-10k-int1-verilator.md` |
| Release-grader-equivalent grade incl. L3 Verilator (320 non-model core capsules) | **320/320**: 318 pass outright; 2 exceed the grader's fixed 10M-cycle L3 cap, which the grader's own harness exceeds with an empty kernel, and their uncapped RTL output is bit-identical to Spike (`L3-CAP.md`) | `evidence/frozen-int-5cb502c/rtl-fullgrade/` |
| 193 programs on RTL (Verilator, GemminiRocketConfig) | **57/57** pass, RoCC retired on RTL == Spike, outputs identical; per-op cycle counts | `evidence/frozen-int-5cb502c/h193/RESULT-rtl193-m27-int-5cb502c.md` |

"Proven" for a device op means both:
- the 193-overload harness case passes the FULL definition: it lowers on the device with RoCC instructions retired and
  exact output, and passes the numerical, host_compute (grader census and audit) and trace checks;
- every graded corpus capsule credited to the op passes the release grader.

Scope decisions (Loren Hung, 2026-11-06):
- **10k gate**: the device-lane spec-validation cases of the 57 device ops with dtype int8 or int32 (466 cases), the
  dtypes the public corpus exercises. The full 2751-case device-lane set (f32/f16/bf16/complex forms) is report-only.
- **SY_micro_model**, the one whole-model capsule, is excluded from "all capsules". It is credited to none of the 57
  device ops, and grading it needs an operator-private full-model spec. It is the only core capsule not passing
  (`unavailable`, not run).
- **L3 cycle cap**: SY_conv_operand_capacity_boundary (tier 1 for convolution) and SY_multi_block_beyond (tier 1 for
  mm) are counted as L3-verified on uncapped RTL runs that match Spike bit-exact. The grader's 10M-cycle hang bound is
  unreachable for these two even with an empty kernel (`evidence/frozen-int-5cb502c/rtl-fullgrade/L3-CAP.md`).
- The host-136 column is outside this claim.

## Provenance

1. Clean run 1: an **unseeded** Codex Phase 1 run on stage m27b, p12-g-m27b-codex-clean-20261105T100940Z
   (`--seed none --fresh-backend`, 10 h wall clock, the general author brief as errata). Its final submission (oot
   commit 4688f48) scored device 56/57 proven, core 318/321: div.Scalar was declined, and two must-refuse capsules were
   not refused.
2. 16 targeted fixer commits on top of that final package (`evidence/frozen-int-5cb502c/fixer-commits.txt`), developed
   by Codex fixers against the public grader and the public 10k cases and merged on branch `int`:
   - exact quantized device lowering of unary floating islands, including div.Scalar;
   - typed wrapping integer maps, reductions and contractions;
   - affine movement and broadcast recognition;
   - a source-bound check before readout fusion, which fixes the two must-refuse capsules;
   - native scalar admission proofs.

   All fixes are general IR or semantics rules. None keys on capsule names, paths or shapes.
3. The previously pushed package (seeded lineage S1 plus the div port, 57/57 on the Spike regrade) is kept for
   reference under `evidence/prior-seeded/`.

`submission.MANIFEST.sha256` pins the package bytes.

## Reproduce

- Regrade + 193: point `tooling/eval/eval_candidate.sh` (lorenshung/aws) at `submission/` with the fin6w2-m27b corpus
  template and MODE=full. Step 2 runs `../h193/run_193.sh` with FULL=1.
- 10k: `run_10k.sh submission LABEL --engine spike|verilator --scope int` (the v10k instrument).
- Full grade with L3: `regrade_full_eval.sh --corpus fin6w2-m27b --cand submission --groups core` (the fullgrade
  instrument, which mirrors joint_final_coverage.validate with the release adapters and tier policy).
