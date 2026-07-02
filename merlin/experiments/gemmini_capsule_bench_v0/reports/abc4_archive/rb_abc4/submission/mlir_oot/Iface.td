#ifndef IFACE_TD
#define IFACE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Iface_Dialect : Dialect {
  let name = "merlin_iface";
  let cppNamespace = "::iface";
  let useDefaultTypePrinterParser = 1;
}

class Iface_Type<string name, string mnem> : TypeDef<Iface_Dialect, name> {
  let mnemonic = mnem;
}

def Iface_ResidentType : Iface_Type<"Resident", "resident">;

def Iface_AccType : Iface_Type<"Acc", "acc"> {
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

class Iface_Op<string mnemonic, list<Trait> traits = []>
    : Op<Iface_Dialect, mnemonic, traits>;

// Leaf input/weight/bias tensor declaration.
def Iface_TensorOp : Iface_Op<"tensor"> {
  let arguments = (ins);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// Make a weight resident (packed/stationary).
def Iface_ResidentPackOp : Iface_Op<"resident_pack"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type($src, $result)";
}

// Matmul against a resident weight: lhs @ resolve(rhs) -> acc.
def Iface_MatmulOp : Iface_Op<"matmul"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

// Conv2d (im2col) against a resident weight.
def Iface_Conv2dOp : Iface_Op<"conv2d"> {
  let arguments = (ins AnyType:$ifm, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ifm `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

// Pure data movement (mvin/mvout copy).
def Iface_MovementOp : Iface_Op<"movement"> {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$src attr-dict `:` functional-type($src, $result)";
}

// Apply epilogue, commit accumulator to an output tensor.
def Iface_CommitOp : Iface_Op<"commit"> {
  let arguments = (ins AnyType:$acc);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$acc attr-dict `:` functional-type($acc, $result)";
}

// Release a resident weight.
def Iface_EvictOp : Iface_Op<"evict"> {
  let arguments = (ins AnyType:$handle);
  let results = (outs);
  let assemblyFormat = "$handle attr-dict `:` functional-type($handle, results)";
}

#endif // IFACE_TD
