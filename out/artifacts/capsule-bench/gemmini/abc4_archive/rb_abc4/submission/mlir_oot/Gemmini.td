#ifndef GEMMINI_TD
#define GEMMINI_TD

include "mlir/IR/OpBase.td"

def Gemmini_Dialect : Dialect {
  let name = "gemmini";
  let cppNamespace = "::gem";
}

class Gemmini_Op<string mnemonic, list<Trait> traits = []>
    : Op<Gemmini_Dialect, mnemonic, traits>;

def Gemmini_TensorOp : Gemmini_Op<"tensor"> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Gemmini_ResidentPackOp : Gemmini_Op<"resident_pack"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type($src, $result)";
}

def Gemmini_MatmulOp : Gemmini_Op<"matmul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

def Gemmini_Conv2dOp : Gemmini_Op<"conv2d"> {
  let arguments = (ins AnyType:$ifm, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ifm `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

def Gemmini_MovementOp : Gemmini_Op<"movement"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type($src, $result)";
}

def Gemmini_CommitOp : Gemmini_Op<"commit"> {
  let arguments = (ins AnyType:$acc);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$acc attr-dict `:` functional-type($acc, $result)";
}

def Gemmini_EvictOp : Gemmini_Op<"evict"> {
  let arguments = (ins AnyType:$handle);
  let results = (outs);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, results)";
}

#endif // GEMMINI_TD
