# Primitive-set frontier audit (P19 Phase 4)

Validates the P16/P17 primitive-set frontier against source-grounded shape semantics. Structural only.

## Shape-class faithfulness — the key caveat: M comes from the CAPTURE, not the deployment
The "GEMV-like" headline is **capture-shape-induced**, not a model property:
- **tiny_llama**: all 15 GEMMs are `gemv_like` only because **M=4 = the capture sequence length (`M2M_SEQ`),
  a small PREFILL** (loader fixes batch=1; `use_cache=False`). The real model declares `token_decode K=32`
  (MODEL_ARCH) — a true decode step is **M=1**, and a real prefill is **M=seq(hundreds)**. So M=4 is an
  arbitrary capture knob; the shapes are **skinny GEMM at M=4**, not intrinsic GEMV.
- **small_llama**: same — M=8 (T=8 prefill), `wide_skinny`, mac_fraction 1.0; **skinny GEMM, not true
  GEMV** (true decode M=1).
- The `gemv_lane_64` primitive therefore covers these because of a small-M capture; at deployment M shifts
  (decode M=1 → even more GEMV-like; prefill M=512 → square/skinny). ⇒ **the frontier is a function of the
  captured M**, which must be stated. It is still a valid *structural* statement ("at these captured
  shapes, a {gemv-lane + small-tile} set covers the corpus"), but not a deployment claim.

## true-GEMV vs skinny-GEMM vs projection
The P16 critique's concern stands: `gemv_lane_64` covers `gemv_like` + `wide_skinny` + `tall_skinny`. Source
check confirms most "gemv" coverage is **skinny GEMM / projection at small M**, not single-vector GEMV
(M or N ≤ 4 in the true sense). P17 already renamed the abstraction `skinny_gemm_or_gemv_engine` and exposes
the `true_gemv` vs `skinny_gemm` split in `predicate_audit_table.csv` — **this audit confirms that rename is
warranted** (the engine serves skinny projections, decode-like GEMV is the minority except at M=1).

## Frontier robustness (cross-check of P17 `primitive_frontier_robustness`)
- The claim **"a 2-primitive set suffices"** holds across pad-waste 5/10/20% and LOO — but the **specific
  pair is threshold/LOO-sensitive** (P17 found the best single tile becomes `tile_4x16` once finer tiles are
  added, and the pair flips on openvla). This audit confirms that is correct and not a bug: it reflects the
  small-M skinny shapes, which many fine tiles cover comparably.
- Attention ops were absent from the frontier (frontier keys off linear-GEMM `tile_waste`). With P19
  attention recovery, a future frontier *could* include attention bmm shapes — currently it does not, so the
  frontier is a **linear-GEMM-only** coverage statement (state this).

## Verdict
- `primitive_set_frontier` / `primitive_frontier_by_threshold`: **main-slide, with two caveats** — (1) it is
  linear-GEMM-only (attention excluded); (2) coverage is a function of the captured M (prefill/decode), so
  "GEMV-like" is capture-induced, not a deployment property.
- `shape_class_mac_share`: **backup** (P16 already demoted it) — too coarse; the M-source caveat makes raw
  shape-class shares misleading without the capture context.
