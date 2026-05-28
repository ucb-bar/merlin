#ifndef IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_
#define IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def GemminiConvertToGemminiPass :
    InterfacePass<"gemmini-convert-to-gemmini", "mlir::FunctionOpInterface"> {
  let summary = "Recover Gemmini ops from normalized linalg.generic IR";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def GemminiPreprocessPass :
    InterfacePass<"gemmini-preprocess", "mlir::FunctionOpInterface"> {
  let summary = "Linalg-level rewrites that benefit Gemmini regardless of "
                "whether tensor-domain Gemmini ops are recovered (e.g. "
                "QDQ-cancellation around max-pool to elide f32 round-trips).";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def GemminiCanonicalizePass :
    InterfacePass<"gemmini-canonicalize", "mlir::FunctionOpInterface"> {
  let summary = "Canonicalize Gemmini ops";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect"
  ];
}

def GemminiLowerToISAPass :
    InterfacePass<"gemmini-lower-to-isa", "mlir::FunctionOpInterface"> {
  let summary = "Lower high-level Gemmini ops to staged ISA-like forms";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect"
  ];
}

def GemminiLowerToIREEPass :
    InterfacePass<"gemmini-lower-gemmini-to-iree", "mlir::FunctionOpInterface"> {
  let summary = "Lower Gemmini ops back to ordinary IREE/MLIR IR";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def GemminiLowerTileToISAPass :
    InterfacePass<"gemmini-lower-tile-to-isa", "mlir::FunctionOpInterface"> {
  let summary = "Bridge gemmini.matmul_tile/conv2d (tensor-like) onto "
                "gemmini.tile_matmul/tile_conv (memref-like ISA-tier)";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def GemminiLegalizeForLLVMExportPass :
    InterfacePass<"merlin-gemmini-legalize-for-llvm-export",
                  "mlir::FunctionOpInterface"> {
  let summary = "Lower gemmini.tile_matmul / mvin / preload / etc. into "
                "gemmini.intr.* RoCC intrinsic ops ready for translate.";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];

  // GemminiTargetConfig as tablegen pass options. These reach the pass via
  // the textual pipeline string (`...{dim=16 addr-len=32 ...}`) so the
  // descriptor flows into the dispatch-side codegen pipeline that
  // PipelineAttrInterface::buildPipeline parses. Defaults match Spike
  // libgemmini.so (DIM=16, ADDR_LEN=32, ACC_ROWS=1024, BANK_ROWS=4096,
  // elem_t=int8_t i.e. 8 bits, acc_t=int32_t i.e. 32 bits).
  let options = [
    Option<"dim", "dim", "int64_t", /*default=*/"16",
           "Systolic array side length (DIM).">,
    Option<"addrLen", "addr-len", "int64_t", /*default=*/"32",
           "Address-space bit-width (ADDR_LEN). MUST match libgemmini.">,
    Option<"accRows", "acc-rows", "int64_t", /*default=*/"1024",
           "Accumulator row count (ACC_ROWS).">,
    Option<"bankRows", "bank-rows", "int64_t", /*default=*/"4096",
           "Scratchpad rows per bank (BANK_ROWS).">,
    Option<"elemBits", "elem-bits", "int64_t", /*default=*/"8",
           "Element bit width (lhs/rhs of matmul).">,
    Option<"accBits", "acc-bits", "int64_t", /*default=*/"32",
           "Accumulator element bit width.">,
    // Phase-5 mxGemmini: MX format encoding for CONFIG_EX rs1 [11:10] /
    // [13:12] / [15:14] / [5]. -1 = Disabled (no MX bits). 0=Fp4 (E2M2),
    // 1=Fp6_0 (E2M4), 2=Fp8_0 (E4M4), 3=Fp6_1 (E3M3), 4=Fp8_1 (E5M3).
    Option<"mxFormat", "mx-format", "int64_t", /*default=*/"-1",
           "MX (microscaling) format encoding.">,
    // Phase-6 follow-up: when "mmio", the tile-matmul lowering clamps
    // every B/D MVIN's `cols` to `dim` (one block per j-tile) so the
    // packed value fits into mxGemmini's 6-bit MvinRs2.num_cols field
    // (RadianceGemminiOnlyConfig). RoCC-attached gemmini variants
    // (Spike, FireSim Lean/MxGemminiRocketConfig) keep the wider default
    // — those configs allocate at least 7 bits for num_cols.
    Option<"commandIssue", "command-issue", "std::string", /*default=*/"\"rocc\"",
           "Command issue path: 'rocc' or 'mmio'.">,
    // Phase-8 LOOP_WS lowering. When true, gemmini.tile_matmul lowers to
    // a single ~12-command LOOP_WS sequence (5 configs + 6 LOOP_WS_* +
    // FLUSH) instead of the per-tile MVIN/PRELOAD/COMPUTE/MVOUT
    // expansion (~56 commands per 16x64x64 matmul). The hardware loops
    // over I/J/K internally with proper backpressure, avoiding the
    // GemminiTile.scala:446 MMIO command-queue assertion. Default false
    // preserves Phase 1-7 byte-identical behavior on RoCC/Spike.
    Option<"loopWs", "loop-ws", "bool", /*default=*/"false",
           "Emit LOOP_WS sequence instead of per-tile MVIN/PRELOAD/COMPUTE/MVOUT.">,
    // Opt-in dispatch-debug instrumentation. When true, the pass emits
    // volatile stores of binding pointers + matmul-operand pointers to
    // fixed DRAM trace regions (MERLIN_DEBUG_BINDING_TRACE_ADDR /
    // MERLIN_DEBUG_MATMUL_TRACE_ADDR) so the runtime can read them
    // back. Pair with the runtime side's -DMERLIN_DISPATCH_DEBUG=ON
    // build. Default false; adds a few stores per dispatch when on.
    Option<"dispatchDebug", "dispatch-debug", "bool", /*default=*/"false",
           "Emit DRAM trace stores for the matching runtime debug probes.">,
  ];
}

def GemminiLowerIntrToMmioPass :
    InterfacePass<"gemmini-lower-intr-to-mmio",
                  "mlir::FunctionOpInterface"> {
  let summary = "Replace gemmini.intr.* RoCC intrinsic ops with volatile "
                "stores to mmioBase + 0x10/0x18/0x00 (MMIO-attached "
                "Gemmini, e.g. RadianceGemminiOnlyConfig).";
  let dependentDialects = [
    "mlir::iree_compiler::Gemmini::GemminiDialect",
    "mlir::arith::ArithDialect",
    "mlir::LLVM::LLVMDialect"
  ];

  let options = [
    Option<"mmioBase", "mmio-base", "int64_t", /*default=*/"0x40084000",
           "Gemmini control window base address.">,
  ];
}

def GemminiAttachCompilationInfoPass :
    InterfacePass<"gemmini-attach-compilation-info",
                  "mlir::FunctionOpInterface"> {
  let summary = "Attach iree_codegen.compilation_info on linalg.matmul "
                "ops, pinning a textual pass_pipeline that runs the gemmini "
                "recovery + ISA-tier lowering inside the dispatch.";
  let dependentDialects = [
    "mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "mlir::iree_compiler::IREE::CPU::IREECPUDialect"
  ];

  // Hardware descriptor — formatted into the textual pipeline string this
  // pass attaches, so the inside-dispatch
  // `merlin-gemmini-legalize-for-llvm-export` instance receives matching
  // values when PipelineAttrInterface::buildPipeline parses the string.
  // Defaults match Spike libgemmini.so. See LegalizeForLLVMExport.cpp for
  // why each parameter matters.
  let options = [
    Option<"dim", "dim", "int64_t", /*default=*/"16",
           "Systolic array side length (DIM).">,
    Option<"addrLen", "addr-len", "int64_t", /*default=*/"32",
           "Address-space bit-width (ADDR_LEN).">,
    Option<"accRows", "acc-rows", "int64_t", /*default=*/"1024",
           "Accumulator row count (ACC_ROWS).">,
    Option<"bankRows", "bank-rows", "int64_t", /*default=*/"4096",
           "Scratchpad rows per bank (BANK_ROWS).">,
    Option<"elemBits", "elem-bits", "int64_t", /*default=*/"8",
           "Element bit width.">,
    Option<"accBits", "acc-bits", "int64_t", /*default=*/"32",
           "Accumulator element bit width.">,
    // Phase-5 mxGemmini fields. Formatted into the textual pipeline:
    //   mx-format=<n>          → forwarded to legalize-for-llvm-export
    //   command-issue=<rocc|mmio> → toggles whether
    //                              gemmini-lower-intr-to-mmio is appended
    //   mmio-base=<addr>       → forwarded to gemmini-lower-intr-to-mmio
    Option<"mxFormat", "mx-format", "int64_t", /*default=*/"-1",
           "MX format encoding (-1=Disabled).">,
    Option<"commandIssue", "command-issue", "std::string", /*default=*/"\"rocc\"",
           "Command issue path: 'rocc' or 'mmio'.">,
    Option<"mmioBase", "mmio-base", "int64_t", /*default=*/"0x40084000",
           "Gemmini control window base address (MMIO mode).">,
    // Phase-8 LOOP_WS lowering. Forwarded into the textual pipeline as
    // `merlin-gemmini-legalize-for-llvm-export{... loop-ws=true}`.
    Option<"loopWs", "loop-ws", "bool", /*default=*/"false",
           "Emit LOOP_WS sequence (single hardware-loop matmul) instead of "
           "per-tile MVIN/PRELOAD/COMPUTE/MVOUT.">,
    // Forwarded into the textual pipeline as
    // `merlin-gemmini-legalize-for-llvm-export{... dispatch-debug=true}`.
    Option<"dispatchDebug", "dispatch-debug", "bool", /*default=*/"false",
           "Emit DRAM trace stores for the matching runtime debug probes.">,
  ];
}

#endif  // IREE_GEMMINI_COMPILER_DIALECT_GEMMINI_TRANSFORMS_PASSES_TD_
