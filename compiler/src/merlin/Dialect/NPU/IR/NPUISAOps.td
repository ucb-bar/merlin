#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISAOPS_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISAOPS_TD_

include "mlir/IR/OpBase.td"
include "NPUISADialect.td"

class NPUISA_Op<string mnemonic, list<Trait> traits = []> :
    Op<NPUISA_Dialect, mnemonic, traits>;

def NPUISA_DmaLoadOp : NPUISA_Op<"dma_load"> {
  let summary = "Near-ISA DMA load to matrix register file";
  let arguments = (ins I64Attr:$rd, I64Attr:$base, I64Attr:$size, I64Attr:$flag);
  let assemblyFormat = [{
    `rd` `=` $rd `,` `base` `=` $base `,` `size` `=` $size `,` `flag` `=` $flag attr-dict
  }];
}

def NPUISA_DmaLoadMxu0Op : NPUISA_Op<"dma_load_mxu0"> {
  let summary = "Near-ISA DMA load to MXU0 weight buffer";
  let arguments = (ins I64Attr:$rd, I64Attr:$base, I64Attr:$size, I64Attr:$flag);
  let assemblyFormat = [{
    `rd` `=` $rd `,` `base` `=` $base `,` `size` `=` $size `,` `flag` `=` $flag attr-dict
  }];
}

def NPUISA_DmaLoadMxu1Op : NPUISA_Op<"dma_load_mxu1"> {
  let summary = "Near-ISA DMA load to MXU1 weight buffer";
  let arguments = (ins I64Attr:$rd, I64Attr:$base, I64Attr:$size, I64Attr:$flag);
  let assemblyFormat = [{
    `rd` `=` $rd `,` `base` `=` $base `,` `size` `=` $size `,` `flag` `=` $flag attr-dict
  }];
}

def NPUISA_DmaStoreOp : NPUISA_Op<"dma_store"> {
  let summary = "Near-ISA DMA store from matrix/vector register file";
  let arguments = (ins AnyType:$src, I64Attr:$rs1, I64Attr:$base, I64Attr:$size, I64Attr:$flag);
  let assemblyFormat = [{
    $src `rs1` `=` $rs1 `,` `base` `=` $base `,` `size` `=` $size `,` `flag` `=` $flag attr-dict
    `:` type($src)
  }];
}

def NPUISA_DmaWaitOp : NPUISA_Op<"dma_wait"> {
  let summary = "Near-ISA DMA wait on flag";
  let arguments = (ins I64Attr:$flag);
  let assemblyFormat = "`flag` `=` $flag attr-dict";
}

def NPUISA_VMulOp : NPUISA_Op<"vmul"> {
  let summary = "Near-ISA vector multiply";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$vrd, I64Attr:$vs1, I64Attr:$vs2);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `regs` `=` `(` $vrd `,` $vs1 `,` $vs2 `)` attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def NPUISA_VExpOp : NPUISA_Op<"vexp"> {
  let summary = "Near-ISA vector exp";
  let arguments = (ins AnyType:$src, I64Attr:$vrd, I64Attr:$vs1);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $src `regs` `=` `(` $vrd `,` $vs1 `)` attr-dict
    `:` type($src) `->` type($result)
  }];
}

def NPUISA_VReduceSumOp : NPUISA_Op<"vreduce_sum"> {
  let summary = "Near-ISA vector reduce sum";
  let arguments = (ins AnyType:$src, I64Attr:$vrd, I64Attr:$vs1);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $src `regs` `=` `(` $vrd `,` $vs1 `)` attr-dict
    `:` type($src) `->` type($result)
  }];
}

def NPUISA_VRcpOp : NPUISA_Op<"vrcp"> {
  let summary = "Near-ISA vector reciprocal";
  let arguments = (ins AnyType:$src, I64Attr:$vrd, I64Attr:$vs1);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $src `regs` `=` `(` $vrd `,` $vs1 `)` attr-dict
    `:` type($src) `->` type($result)
  }];
}

def NPUISA_MatmulMxu0Op : NPUISA_Op<"matmul_mxu0"> {
  let summary = "Near-ISA MXU0 matmul instruction";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$rd, I64Attr:$rs1, I64Attr:$rs2);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `regs` `=` `(` $rd `,` $rs1 `,` $rs2 `)` attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def NPUISA_NativeInstOp : NPUISA_Op<"native_inst"> {
  let summary = "Native npu_model instruction";
  let description = [{
    Carries a manifest-backed simulator instruction exactly as consumed by
    npu_model. The compiler uses this as the bridge from scheduled kernel
    launches to the already-verified native SaturnNPU kernel library.
  }];
  let arguments = (ins StrAttr:$mnemonic, DictionaryAttr:$args);
  let assemblyFormat = "$mnemonic $args attr-dict";
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUISAOPS_TD_
