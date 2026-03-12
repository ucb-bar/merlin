#ifndef IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEOPS_TD_
#define IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEOPS_TD_

include "mlir/IR/OpBase.td"
include "NPUScheduleDialect.td"

class NPUSchedule_Op<string mnemonic, list<Trait> traits = []> :
    Op<NPUSchedule_Dialect, mnemonic, traits>;

def NPUSchedule_SoftmaxFragmentOp : NPUSchedule_Op<"softmax_fragment"> {
  let summary = "Scheduled softmax vector fragment";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def NPUSchedule_UKernelLaunchOp : NPUSchedule_Op<"ukernel_launch"> {
  let summary = "Scheduled ukernel launch";
  let arguments = (ins StrAttr:$symbol, Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $symbol `(` $inputs `)` attr-dict `:` type($inputs) `->` type($result)
  }];
}

def NPUSchedule_MatmulTileOp : NPUSchedule_Op<"matmul_tile"> {
  let summary = "Scheduled matmul tile";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def NPUSchedule_DmaStartOp : NPUSchedule_Op<"dma_start"> {
  let summary = "Scheduled async DMA launch";
  let arguments = (ins
    AnyMemRef:$src,
    AnyMemRef:$dst,
    I64Attr:$size,
    I64Attr:$src_space,
    I64Attr:$dst_space
  );
  let results = (outs AnyType:$token);
  let assemblyFormat = [{
    $src `,` $dst `size` `=` $size `src_space` `=` $src_space `dst_space` `=` $dst_space
    attr-dict `:` type($src) `to` type($dst) `->` type($token)
  }];
}

def NPUSchedule_DmaWaitOp : NPUSchedule_Op<"dma_wait"> {
  let summary = "Scheduled wait on DMA token";
  let arguments = (ins AnyType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

#endif // IREE_NPU_COMPILER_DIALECT_NPU_IR_NPUSCHEDULEOPS_TD_
