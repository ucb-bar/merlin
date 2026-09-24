# Authored model inputs for the Gemmini Phase 0 example

These loaders define small, deterministic model capsules selected by
[`recipe.yaml`](../recipe.yaml). They are example inputs: Phase 0 captures each
loader through model2MLIR and writes the generated MLIR, external weights,
golden output, and provenance into the selected run's capsule directory.

- [`microvit.py`](microvit.py) already expresses its integer GEMMs among floating
  host operations. Its recipe entry declares `capture_quantization:
  already_materialized`; the capture checks that the resulting MLIR still has
  integer contractions.
- [`host_island_seam.py`](host_island_seam.py) already expresses two int8 GEMMs
  separated by a floating LayerNorm. It uses the same materialized-capture
  declaration, and the capture must retain both integer contractions and the
  int8 input/output ABI.
- [`smolvla_denoise_step.py`](smolvla_denoise_step.py) represents one small VLA
  denoising step. Its recipe entry uses the target-derived quantization recipe;
  an opaque or non-integer capture remains a failed capsule.

Historical capsule snapshots under `merlin/contract/capsules/model/` retain
their original loader bytes for inspection and compatibility. New runs select
these example paths. Generated capsules should be read from the run artifact,
not edited here.
