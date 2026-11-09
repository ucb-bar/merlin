# Historical authored model diagnostics

These loaders are preserved for compatibility studies. The selected
[`recipe.yaml`](../recipe.yaml) has an empty authored capsule list; verified
Phase 0 derives its corpus from hardware facts, the software spec and the
independent workloads named in [the Phase 0 guide](../README.md). These files
are not selected by that derivation.

- [`microvit.py`](microvit.py) already expresses its integer GEMMs among floating
  host operations. Its historical seed declares `capture_quantization:
  already_materialized`; the capture checks that the resulting MLIR still has
  integer contractions.
- [`host_island_seam.py`](host_island_seam.py) already expresses two int8 GEMMs
  separated by a floating LayerNorm. It uses the same materialized-capture
  declaration, and the capture must retain both integer contractions and the
  int8 input/output ABI.

The old SmolVLA-derived denoising loader was removed from Phase 0 inputs: it
was based on a held-out validation capture and must not select Phase 0/1/2
capsules. Its [historical capsule](../../../../experiments/corpora/model/M4_smolvla_denoise_gemmini/README.md)
remains available for archival inspection only.

Historical capsule snapshots under `experiments/corpora/model/` remain
inspectable, but verified runs select only the newly generated release bundle.
Generated capsules should be read from the run artifact, not edited here.
