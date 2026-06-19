# Low-bit capture requirements (P20 Tool E)

> What the qdq recapture exposes (unblocks the low-bit abstractions) vs what a NATIVE low-bit capture would still need. The qdq MLIR keeps explicit `quant_ext.dequantize*` with storage dtype + scale granularity; the dequant sits before the GEMM (compute stays f32), so the packed-compute datapath is still not exercised. Structural; no perf claim.

- Quant metadata recovered for 4 workload(s) with a qdq capture: bitvla, openvla, pi05, rdt.
- **Native-scheme gaps** (qdq is torchao int8, not the model's native scheme): bitvla needs a packed-ternary (W1.58) capture; native int4/fp8 datapaths need a compute-in-low-bit capture (dequant-on-load fused), not dequant-before-GEMM.
- Per-workload detail: `quant_metadata_visibility.csv`.
