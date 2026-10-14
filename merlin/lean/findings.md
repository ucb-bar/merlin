# Findings ledger

Defects and discrepancies the verification work surfaced. Each entry states what was observed,
the evidence, whether it is on the paper path, and its status. (Design §8.5.)

## F1 — xDSL and MLIR build different bodies for custom-syntax named linalg ops

- **Observed (2026-10-13, round trip over the repo's 735 `.mlir` files):** for custom-syntax
  named ops (e.g. `linalg.matmul ins(...) outs(...)` with no explicit region), xDSL materializes
  the body as `arith.addf(%mul, %acc)` while MLIR materializes `arith.addf(%acc, %mul)`. 76 files
  differ only in this operand order.
- **Why it matters:** under the parametric float semantics these are different programs. IEEE
  addition is commutative (bit-exact on RISC-V, whose NaN results are canonical), so the compiled
  numbers agree, but a boundary that re-prints such a module through xDSL is unprovable without a
  named `addf` commutativity law, and the verified reading must be MLIR's (design D9).
- **Paper path:** captures and Merlin's stage prints are generic form, which carries the region
  explicitly, so both parsers read the same body. To be confirmed by the measured inventory.
- **Status:** open; exporter will use MLIR's own parser (`torch_mlir.ir`) as the authority.

## F2 — MLIR's Python bindings hide the properties of unregistered ops

- **Observed (2026-10-13):** `torch_mlir.ir` exposes only discardable attributes of unregistered ops
  such as `quant_ext.dequantize_per_tensor`; their properties (`quant_min`, `quant_max`, `axis`) are
  invisible, so a naive exporter silently drops them. The round trip flagged all 8 affected files.
- **Paper path:** yes — `quant_ext` ops are in the captures until `lower_quant_ext` (U15).
- **Status:** fixed in `mlir_export.py` (the property dictionary is recovered from the op's own
  generic print with a balanced, string-aware scan and parsed by MLIR; failure fails the export).

## F3 — xDSL accepts IR that MLIR's verifier rejects

- **Observed:** a fixture with an ill-typed `linalg.transpose` (result 16x16 for an 8x4 permutation)
  parsed and passed through Merlin's xDSL preprocessing; MLIR rejects it
  (`dim(result, 0) = 16 doesn't match dim(input, permutation[0]) = 8`).
- **Why it matters:** Merlin's Python passes can operate on, and emit, IR the compiling MLIR would
  reject or read differently. Exporting every boundary with MLIR's parser (design D9) surfaces it at
  the boundary where it appears.
- **Status:** fixture corrected; the general risk is covered by MLIR-parser export of every stage.
