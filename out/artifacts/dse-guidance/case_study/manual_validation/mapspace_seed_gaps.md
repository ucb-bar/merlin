# Mapspace seed gaps (what a full Timeloop problem needs that the flat capture lacks)

- **K denoise/decode loop trip count**: unrolled by torch.export -> seeds cover ONE step; the
  outer K loop is config/assumed (MODEL_ARCH), not in the seed. Needs loop-preserving capture.
- **Attention inner dims (heads, seq, head_dim)**: the recovered attention contraction folds
  batch/heads into M; the per-head/seq mapspace axes need the SDPA op kept un-decomposed
  (high-level capture) or a sidecar with heads/seq/head_dim.
- **Conv spatial/channel loops**: conv ops are counted (n_conv) but MACs/loops not quantified
  here (would need the conv window dims). 12 conv ops corpus-wide (patch embed).
- **Operand bit-width / packed layout**: f32 in the flat capture; the qdq recapture exposes
  per-channel int8 (quant_ext.dequantize) for a true low-bit mapspace.
- **Fusion/epilogue**: bias-add (addmm) + activation are separate generics here; a fused-epilogue
  mapspace needs the epilogue grouped onto the matmul (epilogue_pattern_table has the grouping).
