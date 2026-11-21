# Atlas RTL — curated subset

`rtl/atlas/` holds a **curated, representative subset** of the Atlas Chisel RTL — the load-bearing
modules for understanding the hardware a backend targets, not the whole tree:

- `Elaborate.scala` — the top-level elaboration entry point.
- `common/` — the frozen shape/param facts: `AtlasParams`, `MxuParams`, `SystolicArrayParams`,
  `InnerProductTreeParams`, `MregParams`, `VmemParams`, `DmaParams`, `VpuParams`.
- `mxu/` — the matrix unit: `MxuBundles`, `WeightBuffers`, `AccumulationBuffers`, plus the two fabrics
  `sa/` (systolic array: `SystolicArray`, `SystolicArrayTop`, `PE`, `PEMesh`) and `ipt/` (inner-product
  tree: `InnerProductTrees`, `InnerProductTreesTop`).
- `scalar/` — the self-hosted scalar core's ISA surface: `ScalarISA`, `ScalarDecoder`, `Instructions`,
  `PcControl`.

The **full buildable RTL tree** (the LSU, vector engine lane boxes, the full scalar core / IDecode, the
MXU sequencers, DMA, the mreg file, Chipyard wiring, `baremetal/`, and the `npu-model/` performance
model) lives in the external Atlas NPU repository:

    $MERLIN_EXT_ATLAS_NPU/src/main/scala/atlas          # e.g. .../atlas-npu/src/main/scala/atlas

The authoritative, cycle-accurate facts a backend depends on (opcode encodings, DIM, memory map) are
**derived from this RTL by mlc** (the arc model `libatlas_model.so` + `atlas_hw.mlir`), not from this
curated snapshot — the snapshot is orientation material.
