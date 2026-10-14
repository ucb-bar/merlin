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
