#ifndef IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIOPS_TD_
#define IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIOPS_TD_

include "GemminiDialect.td"
include "GemminiAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gemmini_Dialect, mnemonic, traits>;

def Gemmini_MatmulOp : Gemmini_Op<"matmul", [Pure]> {
  let summary = "Recovered int8 x int8 -> i32 matmul";
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    I64Attr:$lhsZeroPoint,
    I64Attr:$rhsZeroPoint,
    Gemmini_Dataflow:$dataflow
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Gemmini_MatmulTileOp : Gemmini_Op<"matmul_tile", [Pure]> {
  let summary = "Recovered matmul with explicit tile metadata";
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    I64Attr:$lhsZeroPoint,
    I64Attr:$rhsZeroPoint,
    Gemmini_Dataflow:$dataflow,
    I64Attr:$tileM,
    I64Attr:$tileN,
    I64Attr:$tileK
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Gemmini_Conv2DOp : Gemmini_Op<"conv2d", [Pure]> {
  let summary = "Recovered batchless CHW/FCHW int8 conv2d";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$filter,
    I64Attr:$strideH,
    I64Attr:$strideW,
    I64Attr:$dilationH,
    I64Attr:$dilationW,
    I64Attr:$inputZeroPoint,
    I64Attr:$filterZeroPoint,
    Gemmini_Dataflow:$dataflow
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($result)
  }];
}

def Gemmini_RequantizeOp : Gemmini_Op<"requantize", [Pure]> {
  let summary = "Recovered f32 -> i8 requantization";
  let arguments = (ins
    AnyRankedTensor:$input,
    F32Attr:$scale,
    I64Attr:$zeroPoint,
    I64Attr:$qmin,
    I64Attr:$qmax
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

def Gemmini_ClampOp : Gemmini_Op<"clamp", [Pure]> {
  let summary = "Recovered f32 clamp";
  let arguments = (ins
    AnyRankedTensor:$input,
    F32Attr:$minValue,
    F32Attr:$maxValue
  );
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

#endif  // IREE_GEMMINI_DIALECT_GEMMINI_IR_GEMMINIOPS_TD_
