# Historical authored model diagnostics

These loaders are preserved for compatibility studies. The selected
[`recipe.yaml`](../recipe.yaml) has an empty authored capsule list; verified
test generation derives its corpus from hardware facts, the software spec and the
independent workloads named in [the test generation guide](../README.md). These files
are not selected by that derivation.

- [`microvit.py`](microvit.py) already expresses its integer GEMMs among floating
  host operations. Its historical seed declares `capture_quantization:
  already_materialized`; the capture checks that the resulting MLIR still has
  integer contractions.
- [`host_island_seam.py`](host_island_seam.py) already expresses two int8 GEMMs
  separated by a floating LayerNorm. It uses the same materialized-capture
  declaration, and the capture must retain both integer contractions and the
  int8 input/output ABI.

The old SmolVLA denoising loader was based on a held-out validation capture and
must not select new compiler tests. Historical capsule snapshots are retained
under ignored `out/` with their original source identities; current verified
runs select only a freshly generated and reviewed release bundle. Read generated
capsules from that selected artifact, rather than editing a source-tree corpus.
