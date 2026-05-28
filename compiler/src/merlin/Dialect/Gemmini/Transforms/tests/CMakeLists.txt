iree_lit_test_suite(
  NAME
  lit
  SRCS
  "convert-to-gemmini.mlir"
  "fp8-no-convert.mlir"
  "fuse-matmul-rescale.mlir"
  "preprocess-pool-qdq-cancel.mlir"
  "legalize-for-llvm-export.mlir"
  "lower-gemmini-to-iree.mlir"
  "lower-intr-to-mmio.mlir"
  "lower-tile-to-isa.mlir"
  "lower-tile-to-loop-ws.mlir"
  "matmul-lower-to-isa.mlir"
  "post-global-opt-hook.mlir"
  TOOLS
  FileCheck
  iree-compile
  iree-opt)
# NOTE: `translate-to-llvmir.mlir` exists alongside this list as a doc fixture.
# It would exercise gemmini.intr.* -> llvm.riscv.* via `mlir-translate
# --mlir-to-llvmir`, but the upstream `mlir-translate` binary doesn't have the
# Gemmini dialect linked in, and a custom merlin-translate binary is out of
# scope for this delivery. See
# docs/dev_blog/2026-03-11-gemmini-workstream-log.md.
