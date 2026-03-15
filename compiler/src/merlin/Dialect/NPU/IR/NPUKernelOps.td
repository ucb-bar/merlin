#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELOPS_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELOPS_TD_

include "mlir/IR/OpBase.td"
include "NPUKernelDialect.td"

class NPUKernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<NPUKernel_Dialect, mnemonic, traits>;

def NPUKernel_UKernelGenericOp : NPUKernel_Op<"ukernel_generic"> {
  let summary = "Kernel-level generic ukernel call";
  let arguments = (ins StrAttr:$symbol, Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $symbol `(` $inputs `)` attr-dict `:` type($inputs) `->` type($result)
  }];
}

def NPUKernel_MatmulOp : NPUKernel_Op<"matmul"> {
  let summary = "NPU kernel-level matmul intent";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def NPUKernel_DmaStartOp : NPUKernel_Op<"dma_start"> {
  let summary = "Async DMA launch returning an async token";
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, I64Attr:$size);
  let results = (outs AnyType:$token);
  let assemblyFormat = [{
    $src `,` $dst `size` `=` $size attr-dict `:` type($src) `to` type($dst) `->` type($token)
  }];
}

def NPUKernel_DmaAwaitOp : NPUKernel_Op<"dma_await"> {
  let summary = "Wait on async DMA token";
  let arguments = (ins AnyType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUKERNELOPS_TD_
