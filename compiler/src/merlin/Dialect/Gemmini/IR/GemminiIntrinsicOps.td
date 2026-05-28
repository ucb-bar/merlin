#ifndef IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIINTRINSICOPS_TD_
#define IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIINTRINSICOPS_TD_

include "GemminiDialect.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Gemmini intrinsic op base. Mirrors the LLVM RoCC intrinsics declared in
// llvm/include/llvm/IR/IntrinsicsRISCVXUCBBAR.td: every op takes (i64, i64)
// and produces no result, so the auto-generated GemminiConversions.inc will
// rewrite the MLIR op into a `call void @llvm.riscv.<mnem>(i64, i64)`.
//
// IMPORTANT: TableGen strips the dialect-prefix on the def name before
// generating the C++ class (e.g. `def Gemmini_FlushIntrOp` -> `class
// FlushIntrOp`). Putting the underscore *between* `Flush` and `IntrOp`
// would make all variants collide on the bare `IntrOp` class name. So we
// use the long-form names (FlushIntrOp, MvinIntrOp, ...) here and rely
// on `using namespace mlir::iree_compiler::Gemmini;` in the consuming
// LegalizeForLLVMExport.cpp; the historical tags (`Flush_IntrOp`, ...)
// referenced in that file are restored via the `using` aliases below.
//===----------------------------------------------------------------------===//

class Gemmini_IntrinsicOpBase<string mnemonic, string llvmEnum>
    : LLVM_IntrOpBase</*dialect=*/Gemmini_Dialect,
                      /*opName=*/"intr." # mnemonic,
                      /*enumName=*/llvmEnum,
                      /*overloadedResults=*/[],
                      /*overloadedOperands=*/[],
                      /*traits=*/[],
                      /*numResults=*/0> {
  let arguments = (ins I64:$rs1, I64:$rs2);
  let assemblyFormat = "$rs1 `,` $rs2 attr-dict";
}

def Gemmini_FlushIntrOp                  : Gemmini_IntrinsicOpBase<"flush",                  "riscv_flush">;
def Gemmini_ConfigIntrOp                 : Gemmini_IntrinsicOpBase<"config",                 "riscv_config">;
def Gemmini_MvinIntrOp                   : Gemmini_IntrinsicOpBase<"mvin",                   "riscv_mvin">;
def Gemmini_Mvin2IntrOp                  : Gemmini_IntrinsicOpBase<"mvin2",                  "riscv_mvin2">;
def Gemmini_Mvin3IntrOp                  : Gemmini_IntrinsicOpBase<"mvin3",                  "riscv_mvin3">;
def Gemmini_MvoutIntrOp                  : Gemmini_IntrinsicOpBase<"mvout",                  "riscv_mvout">;
def Gemmini_PreloadIntrOp                : Gemmini_IntrinsicOpBase<"preload",                "riscv_preload">;
def Gemmini_ComputePreloadedIntrOp       : Gemmini_IntrinsicOpBase<"compute.preloaded",      "riscv_compute_preloaded">;
def Gemmini_ComputeAccumulatedIntrOp     : Gemmini_IntrinsicOpBase<"compute.accumulated",    "riscv_compute_accumulated">;
def Gemmini_LoopWsConfigBoundsIntrOp     : Gemmini_IntrinsicOpBase<"loop_ws.config_bounds",  "riscv_loop_ws_config_bounds">;
def Gemmini_LoopWsConfigAddrsABIntrOp    : Gemmini_IntrinsicOpBase<"loop_ws.config_addrs_ab","riscv_loop_ws_config_addrs_ab">;
def Gemmini_LoopWsConfigAddrsDCIntrOp    : Gemmini_IntrinsicOpBase<"loop_ws.config_addrs_dc","riscv_loop_ws_config_addrs_dc">;
def Gemmini_LoopWsConfigStridesABIntrOp  : Gemmini_IntrinsicOpBase<"loop_ws.config_strides_ab","riscv_loop_ws_config_strides_ab">;
def Gemmini_LoopWsConfigStridesDCIntrOp  : Gemmini_IntrinsicOpBase<"loop_ws.config_strides_dc","riscv_loop_ws_config_strides_dc">;
def Gemmini_LoopWsIntrOp                 : Gemmini_IntrinsicOpBase<"loop_ws",                "riscv_loop_ws">;
// Spad-source LOOP_WS preconfig (funct=24). rs1=A_spad_addr,
// rs2=B_spad_end_addr. Used by mxGemmini's MX matmul flow where A/B
// are pre-MVINed to scratchpad before LOOP_WS fires; LOOP_WS rs2[9]=
// spad_only=1 directs the loop FSM to fetch from spad rather than
// DRAM. See gemmini.h:gemmini_loop_ws_spad and matmul_ws_mx_generic.c.
def Gemmini_LoopWsConfigSpadABIntrOp     : Gemmini_IntrinsicOpBase<"loop_ws.config_spad_ab", "riscv_loop_ws_config_spad_ab">;
// Scratchpad-source MVOUT (funct=23 = k_MVOUT_SPAD per gemmini.h).
// Copies bf16 data from spad to DRAM. rs1 = (dst_stride<<32)|dst_addr,
// rs2 = (rows<<48)|(cols<<32)|src_addr. mxGemmini's WS-MX path writes
// its bf16 output into spad (at the c_spad_addr set in LOOP_WS rs2);
// we use MVOUT_SPAD after LOOP_WS to copy the bf16 tile into the
// IREE-allocated DRAM result buffer.
def Gemmini_MvoutSpadIntrOp              : Gemmini_IntrinsicOpBase<"mvout_spad",            "riscv_mvout_spad">;
// CONFIG_SCALE_MEM (funct=26 = k_CONFIG_SCALE_MEM = "mxquant_config_mvout"
// in gemmini.h:211). Configures the per-block scale-factor write path
// for MX-quantized matmul outputs. rs1 packs (lut_update_granularity[15:0]
// | scale_w_sel[16] | scale_act_sel[17] | reserved | CONFIG_SCALE_MEM_op).
// rs2 packs the bounds (i_loop[15:0] | j_loop[31:16] | k_loop[47:32]) and
// DRAM addr in upper bits — consult gemmini.h's mxquant_config_mvout
// macro for exact bit positions.
def Gemmini_MxQuantConfigIntrOp          : Gemmini_IntrinsicOpBase<"mxquant_config",         "riscv_mxquant_config">;
def Gemmini_LoopConvWsConfig1IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config1",   "riscv_loop_conv_ws_config1">;
def Gemmini_LoopConvWsConfig2IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config2",   "riscv_loop_conv_ws_config2">;
def Gemmini_LoopConvWsConfig3IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config3",   "riscv_loop_conv_ws_config3">;
def Gemmini_LoopConvWsConfig4IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config4",   "riscv_loop_conv_ws_config4">;
def Gemmini_LoopConvWsConfig5IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config5",   "riscv_loop_conv_ws_config5">;
def Gemmini_LoopConvWsConfig6IntrOp      : Gemmini_IntrinsicOpBase<"loop_conv_ws.config6",   "riscv_loop_conv_ws_config6">;
def Gemmini_LoopConvWsIntrOp             : Gemmini_IntrinsicOpBase<"loop_conv_ws",           "riscv_loop_conv_ws">;

//===----------------------------------------------------------------------===//
// Gemmini ISA-tier ops. These exist in the IR only between the
// `gemmini-lower-tile-to-isa` and `merlin-gemmini-legalize-for-llvm-export`
// passes. They are constructed by the new lower-tile-to-isa pass, then
// expanded into the *_IntrOp set above by LegalizeForLLVMExport.cpp.
//
// Operand orderings are derived from `rewriter.create<...>` call sites in
// LegalizeForLLVMExport.cpp; the Op accessor names (e.g. getStride, getAct)
// must match the GetXxx() lookups in that file verbatim.
//===----------------------------------------------------------------------===//

class Gemmini_ISAOpBase<string mnemonic, list<Trait> traits = []>
    : Op<Gemmini_Dialect, mnemonic, traits>;

// gemmini.flush  — ConfigStOp epilogue. rs2 is hard-coded to 0 by the
// lowering, so only `skip` (rs1) is plumbed.
def Gemmini_FlushOp : Gemmini_ISAOpBase<"flush"> {
  let summary = "Gemmini accelerator pipeline flush";
  let arguments = (ins I64:$skip);
  let assemblyFormat = "$skip attr-dict";
}

// gemmini.config_st — store config (stride + activation + scale).
def Gemmini_ConfigStOp : Gemmini_ISAOpBase<"config_st"> {
  let summary = "Configure Gemmini store unit (stride/activation/scale)";
  let arguments = (ins I64:$stride, I64Attr:$activation, F32Attr:$scale);
  let assemblyFormat = "$stride attr-dict";
}

// gemmini.config_ld — load config (stride + scale + optional flags).
def Gemmini_ConfigLdOp : Gemmini_ISAOpBase<"config_ld"> {
  let summary = "Configure Gemmini load unit (stride/scale/id/etc.)";
  let arguments = (ins
      I64:$stride,
      F32Attr:$scale,
      DefaultValuedAttr<I1Attr, "false">:$shrunk,
      DefaultValuedAttr<I64Attr, "0">:$id,
      DefaultValuedAttr<I64Attr, "(uint64_t)-1">:$blockMvinStride,
      // 2026-05-11: gemmini-mx's CONFIG_LD macro hardcodes
      // pixel_repeats=1 (gemmini.h's gemmini_extended5_config_ld
      // call sites). Setting it to 0 here leaves the field unset in
      // hardware and an MX matmul silently produces all-zero output.
      DefaultValuedAttr<I64Attr, "1">:$pixelRepeats);
  let assemblyFormat = "$stride attr-dict";
}

// gemmini.config_ex — execute config (dataflow/activation/strides/...).
def Gemmini_ConfigExOp : Gemmini_ISAOpBase<"config_ex"> {
  let summary = "Configure Gemmini execute unit";
  let arguments = (ins
      I64Attr:$dataflow,
      I64Attr:$sysAct,
      I64Attr:$sysShift,
      F32Attr:$sysAccScale,
      DefaultValuedAttr<I64Attr, "0">:$cStride,
      DefaultValuedAttr<I64Attr, "1">:$aStride,
      DefaultValuedAttr<I1Attr, "false">:$aTranspose,
      DefaultValuedAttr<I1Attr, "false">:$bTranspose,
      DefaultValuedAttr<I1Attr, "false">:$setOnlyStrides);
  let assemblyFormat = "attr-dict";
}

// gemmini.config_norm — BERT-norm config used by IGELU/SOFTMAX paths.
def Gemmini_ConfigNormOp : Gemmini_ISAOpBase<"config_norm"> {
  let summary = "Configure Gemmini normalization unit (BERT)";
  let arguments = (ins
      I64Attr:$qConst,
      I64Attr:$qConstType,
      I64Attr:$setStatsIdOnly,
      I64Attr:$actMsb,
      I64Attr:$statsId,
      I64Attr:$igeluQb,
      I64Attr:$igeluQc);
  let assemblyFormat = "attr-dict";
}

// gemmini.mvin / mvin2 / mvin3 — DMA from `input` to scratchpad row `addr`.
class Gemmini_MvinOpBase<string mnemonic> : Gemmini_ISAOpBase<mnemonic> {
  let arguments = (ins AnyMemRef:$input, I64:$addr);
  let assemblyFormat = "$input `,` $addr attr-dict `:` type($input)";
}

def Gemmini_MvinOp  : Gemmini_MvinOpBase<"mvin">  { let summary = "Gemmini mvin (DMA in, port 0)";  }
def Gemmini_Mvin2Op : Gemmini_MvinOpBase<"mvin2"> { let summary = "Gemmini mvin (DMA in, port 1)";  }
def Gemmini_Mvin3Op : Gemmini_MvinOpBase<"mvin3"> { let summary = "Gemmini mvin (DMA in, port 2)";  }

// gemmini.mvout — DMA from scratchpad row `addr` to memref `output`.
def Gemmini_MvoutOp : Gemmini_ISAOpBase<"mvout"> {
  let summary = "Gemmini mvout (DMA out)";
  let arguments = (ins AnyMemRef:$output, I64:$addr);
  let assemblyFormat = "$output `,` $addr attr-dict `:` type($output)";
}

// gemmini.preload — issue preload of B/D tile + C output address.
def Gemmini_PreloadOp : Gemmini_ISAOpBase<"preload"> {
  let summary = "Gemmini preload";
  let arguments = (ins
      I64:$bdAddr, I64:$cAddr,
      I64:$bdRows, I64:$bdCols,
      I64:$cRows,  I64:$cCols);
  let assemblyFormat = [{
    $bdAddr `,` $cAddr `,` $bdRows `,` $bdCols `,` $cRows `,` $cCols attr-dict
  }];
}

// gemmini.preload_zeros — preload zero-tile to scratchpad `addr`.
def Gemmini_PreloadZerosOp : Gemmini_ISAOpBase<"preload_zeros"> {
  let summary = "Gemmini preload-zeros";
  let arguments = (ins I64:$addr, I64:$cRows, I64:$cCols);
  let assemblyFormat = "$addr `,` $cRows `,` $cCols attr-dict";
}

// gemmini.compute.preloaded / .accumulated — issue compute on preloaded tiles.
class Gemmini_ComputeOpBase<string mnemonic> : Gemmini_ISAOpBase<mnemonic> {
  let arguments = (ins
      I64:$aAddr, I64:$bdAddr,
      I64:$aRows, I64:$aCols,
      I64:$bdRows, I64:$bdCols);
  let assemblyFormat = [{
    $aAddr `,` $bdAddr `,` $aRows `,` $aCols `,` $bdRows `,` $bdCols attr-dict
  }];
}

def Gemmini_ComputePreloadedOp   : Gemmini_ComputeOpBase<"compute.preloaded"> {
  let summary = "Gemmini compute.preloaded";
}
def Gemmini_ComputeAccumulatedOp : Gemmini_ComputeOpBase<"compute.accumulated"> {
  let summary = "Gemmini compute.accumulated";
}

// gemmini.tile_matmul — high-level 2D matmul over scratchpad tiles. Inputs
// are flat memrefs (A,B,C,D), shape provides DimI/DimJ/DimK; the
// LegalizeForLLVMExport pass walks the tile loop nest.
def Gemmini_TileMatMulOp : Gemmini_ISAOpBase<"tile_matmul"> {
  let summary = "Gemmini tile-matmul on memrefs";
  let arguments = (ins
      AnyMemRef:$aArray, AnyMemRef:$bArray,
      AnyMemRef:$cArray, AnyMemRef:$dArray,
      F32Attr:$aScaleFactor, F32Attr:$bScaleFactor, F32Attr:$dScaleFactor,
      I64Attr:$act,
      F32Attr:$accScale, F32Attr:$bertScale,
      DefaultValuedAttr<I1Attr, "false">:$repeatingBias,
      DefaultValuedAttr<I1Attr, "false">:$aTranspose,
      DefaultValuedAttr<I1Attr, "false">:$bTranspose,
      DefaultValuedAttr<I1Attr, "false">:$fullC,
      DefaultValuedAttr<I1Attr, "false">:$lowD,
      DefaultValuedAttr<I64Attr, "0">:$weightA,
      DefaultValuedAttr<I64Attr, "0">:$dataflow);
  let assemblyFormat = [{
    $aArray `,` $bArray `,` $cArray `,` $dArray attr-dict `:`
      type($aArray) `,` type($bArray) `,` type($cArray) `,` type($dArray)
  }];
}

// gemmini.tile_conv — high-level conv2d over scratchpad tiles.
def Gemmini_TileConvOp : Gemmini_ISAOpBase<"tile_conv"> {
  let summary = "Gemmini tile-conv on memrefs";
  let arguments = (ins
      AnyMemRef:$input, AnyMemRef:$output,
      AnyMemRef:$weights, AnyMemRef:$bias,
      I64:$outRowDim, I64:$outColDim, I64:$kernelDim,
      I64Attr:$stride, I64Attr:$inputDilation, I64Attr:$kernelDilation,
      I64Attr:$padding, I64Attr:$act, F32Attr:$scale,
      I64Attr:$poolSize, I64Attr:$poolStride, I64Attr:$poolPadding,
      DefaultValuedAttr<I1Attr, "false">:$wrot180,
      DefaultValuedAttr<I1Attr, "false">:$transOutput1203,
      DefaultValuedAttr<I1Attr, "false">:$transInput3120,
      DefaultValuedAttr<I1Attr, "false">:$transWeight1203,
      DefaultValuedAttr<I1Attr, "false">:$transWeight0132);
  let assemblyFormat = [{
    $input `,` $output `,` $weights `,` $bias `,`
      $outRowDim `,` $outColDim `,` $kernelDim attr-dict `:`
      type($input) `,` type($output) `,` type($weights) `,` type($bias)
  }];
}

// gemmini.print — debug printer. LegalizeForLLVMExport.cpp marks this
// addIllegalOp so we declare it for completeness; no lowering is provided
// today and any pipeline that emits it is expected to remove it before
// reaching legalize.
def Gemmini_PrintOp : Gemmini_ISAOpBase<"print"> {
  let summary = "Gemmini debug print";
  let arguments = (ins AnyMemRef:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif  // IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIINTRINSICOPS_TD_
