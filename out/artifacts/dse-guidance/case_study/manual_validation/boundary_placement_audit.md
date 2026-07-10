# HW/SW boundary-placement audit (P19 Phase 4)

Validates that boundary-placement candidates are demanded-by-data vs merely plausible, and separates
blocked-by-capture from blocked-by-proof. Structural only; Merlin enumerates, it does not choose.

## Status semantics (confirmed categorical, not a score)
`boundary_placement.py` levels are categorical {strong, possible, weak, blocked, unavailable, n/a},
hand-assigned per abstraction (no numeric ranking) — so the matrix does NOT imply a quantitative ordering
(addresses the P16-critique Q43 concern). `compiler_proof_matrix.csv` carries per-abstraction proof status
{proven_for_workload×1, assumed×5, unknown×4} (sourced from `models.py` MODEL_ARCH + structural facts).

## Demanded-by-data vs plausible (from the source audits)
- **Demanded by recovered structure** (strong): `matmul`/dense-GEMM tiling, `skinny_gemm_or_gemv_engine`
  (skinny projections are pervasive), epilogue fusion (bias-add addmm generics are real), partial-sum /
  reduction (matmul reductions are real). These rest on IR facts, not config.
- **Demanded but blocked-by-capture** (the central result): `resident_weight_object` / `bounded_loop_command`
  / `loop_carried_state_handle` (the K-loop is unrolled by torch.export — confirmed: no scf.for except
  smolvla's gather artifact); `kv_cache_object` (use_cache=False / non-AR); `packed_lowbit_tensor` /
  `native_lowbit_matmul` / scale objects (bitvla source has them, export dequantizes). These are blocked by
  **capture fidelity**, and the capture-level ablation (P18-B) shows high-level/qdq recaptures unblock
  attention/quant — but loop/KV remain **torch.export-blocked** (the one frontier needing a new frontend).
- **Plausible but not demanded** (descriptive): the full multi-level boundary enumeration for command-ISA /
  microcode / datapath placements — these are search-space candidates the data permits but does not require.
  The adversarial audit already marks "boundary placement is a DSE axis" as *descriptive (enumeration, not a
  decision)*; this audit confirms that framing.

## blocked-by-capture vs blocked-by-proof (the distinction the critique asked for)
- **blocked-by-capture** (richer capture would unblock): low-bit (qdq recapture), attention-as-named-op
  (high-level recapture), K-loop/KV (loop-preserving capture — currently torch.export-blocked).
- **blocked-by-proof** (compiler must prove a property): residency requires "weights invariant across K"
  (assumed, not proven — the loop is erased); bounded_loop_command requires a bounded trip count + loop-
  body invariance (assumed). These are in `compiler_proof_matrix.csv` as `assumed`/`unknown` — correctly NOT
  claimed as proven.

## Verdict
- `boundary_placement_heatmap`: **backup** (P16 already demoted it) — too many abstractions; the categorical
  `boundary_necessity_matrix` (6–8 key abstractions) is the main-slide view.
- The strong takeaway is the **blocked-by-capture vs blocked-by-proof split itself** — present it as: "which
  HW/SW boundary placements the compiler can target today (proven/recoverable) vs which need a richer
  capture (low-bit, attention, loop/KV) vs a compiler proof (K-invariance)." That is demanded-by-data and is
  the methodology contribution.
