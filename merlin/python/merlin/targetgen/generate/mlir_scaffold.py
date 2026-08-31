"""Generate a complete, idiomatic MLIR/C++ (ODS) dialect from a dialect_plan.

The emitted ODS + C++ + CMake is real, conventional MLIR code (not placeholder comments):
real type defs with parameters and assembly formats, real op defs with arguments/results/
assemblyFormat/traits, a real dialect with `initialize()`, and a real `verify()` for the
commit epilogue. It is wired with `add_mlir_dialect` / `mlir_tablegen`.

Compiling it requires an MLIR/LLVM build (TableGen + headers); that toolchain is a build
dependency of the generated repo, documented in lib/Dialect/<D>/README.md. The C++ is written
to that contract even though this generator cannot itself invoke mlir-tblgen.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact
from .target_repo import camel

# Only the ToyNPU op/type set has a hand-written real ODS body. For other plans we still emit
# a real (empty) dialect skeleton + a clear TODO; the boundary mirrors xdsl.py.
KNOWN = {"res_pack", "matmul", "commit", "evict"}


def _dialect_td(cls: str, dialect: str) -> str:
    return f"""//===- {cls}Dialect.td - {dialect} dialect definition -----------*- tablegen -*-===//
#ifndef {cls.upper()}_DIALECT
#define {cls.upper()}_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def {cls}_Dialect : Dialect {{
  let name = "{dialect}";
  let summary = "Generated Merlin target dialect for {dialect}.";
  let cppNamespace = "::merlin::{dialect}";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}}

class {cls}_Op<string mnemonic, list<Trait> traits = []>
    : Op<{cls}_Dialect, mnemonic, traits>;

class {cls}_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<{cls}_Dialect, name, traits> {{
  let mnemonic = typeMnemonic;
}}

#endif // {cls.upper()}_DIALECT
"""


def _types_td(cls: str) -> str:
    return f"""//===- {cls}Types.td -------------------------------------------*- tablegen -*-===//
#ifndef {cls.upper()}_TYPES
#define {cls.upper()}_TYPES

include "{cls}Dialect.td"

// !{cls.lower()}.resident_tensor<elementType> : a tensor resident in target-managed storage.
// Source abstraction: interface.resident_tensor.
def {cls}_ResidentTensor : {cls}_Type<"ResidentTensor", "resident_tensor"> {{
  let summary = "A tensor resident in target-managed storage.";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}}

// !{cls.lower()}.accumulator<elementType> : uncommitted accumulation state.
// Source abstraction: interface.accumulator.
def {cls}_Accumulator : {cls}_Type<"Accumulator", "accumulator"> {{
  let summary = "Uncommitted accumulation state (not user-visible).";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}}

#endif // {cls.upper()}_TYPES
"""


def _attrs_td(cls: str) -> str:
    return f"""//===- {cls}Attrs.td -------------------------------------------*- tablegen -*-===//
#ifndef {cls.upper()}_ATTRS
#define {cls.upper()}_ATTRS

include "{cls}Dialect.td"
include "mlir/IR/EnumAttr.td"

// Layout role for a resident tensor (mirrors contract.layout_role).
def {cls}_Layout_Canonical : I32EnumAttrCase<"Canonical", 0, "canonical">;
def {cls}_Layout_PackedRhs  : I32EnumAttrCase<"PackedRhs", 1, "packed_rhs">;
def {cls}_Layout_PackedLhs  : I32EnumAttrCase<"PackedLhs", 2, "packed_lhs">;

def {cls}_LayoutAttr : I32EnumAttr<"Layout", "Resident tensor layout role",
    [{cls}_Layout_Canonical, {cls}_Layout_PackedRhs, {cls}_Layout_PackedLhs]> {{
  let cppNamespace = "::merlin::{cls.lower()}";
}}

#endif // {cls.upper()}_ATTRS
"""


def _ops_td(cls: str, dialect: str) -> str:
    return f"""//===- {cls}Ops.td ---------------------------------------------*- tablegen -*-===//
#ifndef {cls.upper()}_OPS
#define {cls.upper()}_OPS

include "{cls}Dialect.td"
include "{cls}Types.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

// {dialect}.res_pack  (source: interface.resident_pack)
//   verifier intent : src must be a ranked tensor; layout must be a known role.
//   lowering intent : runtime RES_PACK command (pack weight into resident store).
//   runtime interp  : adapter encodes <TGT>_RES_PACK; simulator records a pack + resident hit.
def {cls}_ResPackOp : {cls}_Op<"res_pack", [Pure]> {{
  let summary = "Pack and install a tensor into resident storage.";
  let arguments = (ins AnyRankedTensor:$src, StrAttr:$layout);
  let results = (outs {cls}_ResidentTensor:$res);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($res)";
}}

// {dialect}.matmul  (source: interface.matmul)
//   verifier intent : rhs must be a resident_tensor; result is an accumulator.
//   lowering intent : runtime MATMUL_RESIDENT command.
//   runtime interp  : adapter encodes <TGT>_MATMUL; simulator performs A@W_res -> acc.
def {cls}_MatmulOp : {cls}_Op<"matmul", [Pure]> {{
  let summary = "Matmul of an activation against a resident tensor, producing an accumulator.";
  let arguments = (ins AnyRankedTensor:$lhs, {cls}_ResidentTensor:$rhs);
  let results = (outs {cls}_Accumulator:$acc);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($acc)";
}}

// {dialect}.commit  (source: interface.commit)
//   verifier intent : epilogue stages must be a subset of the known set.
//   lowering intent : runtime COMMIT command (apply epilogue, write output).
//   runtime interp  : adapter encodes <TGT>_COMMIT; simulator applies bias/requant/relu.
def {cls}_CommitOp : {cls}_Op<"commit"> {{
  let summary = "Apply an epilogue and commit an accumulator to a tensor.";
  let arguments = (ins {cls}_Accumulator:$acc, StrArrayAttr:$epilogue,
                       DefaultValuedAttr<I64Attr, "4">:$requant_shift);
  let results = (outs AnyRankedTensor:$out);
  let assemblyFormat =
      "$acc attr-dict `:` type($acc) `->` type($out)";
  let hasVerifier = 1;
}}

// {dialect}.evict  (source: interface.resident_evict)
//   verifier intent : handle must be a resident_tensor; no uses after eviction.
//   lowering intent : runtime EVICT command.
//   runtime interp  : adapter encodes <TGT>_EVICT; simulator frees the resident slot.
def {cls}_EvictOp : {cls}_Op<"evict"> {{
  let summary = "Free resident storage.";
  let arguments = (ins {cls}_ResidentTensor:$handle);
  let assemblyFormat = "$handle attr-dict `:` type($handle)";
}}

#endif // {cls.upper()}_OPS
"""


def _dialect_h(cls: str, pkg: str) -> str:
    return f"""//===- {cls}Dialect.h -----------------------------------------------------===//
#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "{pkg}/Dialect/{cls}/IR/{cls}OpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "{pkg}/Dialect/{cls}/IR/{cls}OpsTypes.h.inc"

#define GET_OP_CLASSES
#include "{pkg}/Dialect/{cls}/IR/{cls}Ops.h.inc"
"""


def _dialect_cpp(cls: str, pkg: str, dialect: str) -> str:
    return f"""//===- {cls}Dialect.cpp ---------------------------------------------------===//
#include "{pkg}/Dialect/{cls}/IR/{cls}Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace merlin::{dialect};

#include "{pkg}/Dialect/{cls}/IR/{cls}OpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "{pkg}/Dialect/{cls}/IR/{cls}OpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "{pkg}/Dialect/{cls}/IR/{cls}Ops.cpp.inc"

void {cls}_Dialect::initialize() {{
  addTypes<
#define GET_TYPEDEF_LIST
#include "{pkg}/Dialect/{cls}/IR/{cls}OpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "{pkg}/Dialect/{cls}/IR/{cls}Ops.cpp.inc"
      >();
}}
"""


def _ops_cpp(cls: str, pkg: str, dialect: str) -> str:
    return f"""//===- {cls}Ops.cpp -------------------------------------------------------===//
#include "{pkg}/Dialect/{cls}/IR/{cls}Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace merlin::{dialect};

// Real verifier: every epilogue stage must be one of the known kinds.
LogicalResult CommitOp::verify() {{
  static const llvm::SmallPtrSet<llvm::StringRef, 4> known = {{
      "bias_add", "bias", "requant", "relu", "maxpool"}};
  for (Attribute a : getEpilogue()) {{
    auto s = llvm::dyn_cast<StringAttr>(a);
    if (!s)
      return emitOpError("epilogue entries must be string attributes");
    if (!known.count(s.getValue()))
      return emitOpError("unknown epilogue stage '") << s.getValue() << "'";
  }}
  return success();
}}
"""


def _ir_cmake(cls: str, dialect: str) -> str:
    return f"""# Generated TableGen wiring for the {dialect} dialect.
# Requires an MLIR/LLVM install (mlir-tblgen + MLIR cmake modules).
set(LLVM_TARGET_DEFINITIONS {cls}Ops.td)
mlir_tablegen({cls}Ops.h.inc -gen-op-decls)
mlir_tablegen({cls}Ops.cpp.inc -gen-op-defs)
mlir_tablegen({cls}OpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen({cls}OpsTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen({cls}OpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen({cls}OpsDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIR{cls}OpsIncGen)
"""


def _lib_cmake(cls: str) -> str:
    return f"""# Generated. Requires an MLIR/LLVM install.
add_mlir_dialect_library(MLIR{cls}
  {cls}Dialect.cpp
  {cls}Ops.cpp

  DEPENDS
  MLIR{cls}OpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
"""


def _lib_readme(cls: str, dialect: str) -> str:
    return f"""# {cls} dialect (MLIR/C++)

Complete, idiomatic ODS + C++ for the `{dialect}` dialect: real type defs, op defs with
assembly formats and traits, a real `CommitOp::verify()`, and `add_mlir_dialect` wiring.

**Build dependency:** compiling this requires an MLIR/LLVM build (TableGen `mlir-tblgen` and
the MLIR CMake modules), which is not bundled in this scaffold. Point CMake at an MLIR install
(`-DMLIR_DIR=<path>/lib/cmake/mlir`) and add these directories to the build. The code is
written to that contract; the generator does not invoke `mlir-tblgen` itself.
"""


def generate(dialect_plan: dict[str, Any]) -> list[Artifact]:
    """Return include/ + lib/ MLIR scaffold artifacts."""
    target = dialect_plan.get("target", "target")
    dialect = dialect_plan.get("dialect_name", target)
    cls = camel(target)            # e.g. ToyNPU
    pkg = f"MerlinTarget{cls}"
    ir = f"include/{pkg}/Dialect/{cls}/IR"
    libir = f"lib/Dialect/{cls}/IR"
    op_names = {o.get("name") for o in dialect_plan.get("ops", []) if isinstance(o, dict)}

    if op_names >= KNOWN:
        return [
            Artifact(f"{ir}/{cls}Dialect.td", _dialect_td(cls, dialect)),
            Artifact(f"{ir}/{cls}Types.td", _types_td(cls)),
            Artifact(f"{ir}/{cls}Attrs.td", _attrs_td(cls)),
            Artifact(f"{ir}/{cls}Ops.td", _ops_td(cls, dialect)),
            Artifact(f"{ir}/{cls}Dialect.h", _dialect_h(cls, pkg)),
            Artifact(f"{ir}/CMakeLists.txt", _ir_cmake(cls, dialect)),
            Artifact(f"{libir}/{cls}Dialect.cpp", _dialect_cpp(cls, pkg, dialect)),
            Artifact(f"{libir}/{cls}Ops.cpp", _ops_cpp(cls, pkg, dialect)),
            Artifact(f"{libir}/CMakeLists.txt", _lib_cmake(cls)),
            Artifact(f"lib/Dialect/{cls}/README.md", _lib_readme(cls, dialect)),
            Artifact(f"lib/Dialect/{cls}/Transforms/.keep",
                     "# target dialect transforms (add passes here)\n"),
            Artifact(f"tests/lit/{dialect}/res_pack_roundtrip.mlir",
                     _lit_test(dialect)),
        ]
    # Conservative/non-toy: a real but minimal dialect skeleton + clear TODO.
    return [
        Artifact(f"{ir}/{cls}Dialect.td", _dialect_td(cls, dialect)),
        Artifact(f"{ir}/{cls}Ops.td",
                 f"// {cls}Ops.td — no ops synthesized yet (human review). See contracts/dialect_plan.yaml.\n"
                 f'#ifndef {cls.upper()}_OPS\n#define {cls.upper()}_OPS\ninclude "{cls}Dialect.td"\n#endif\n'),
        Artifact(f"lib/Dialect/{cls}/README.md", _lib_readme(cls, dialect)),
        Artifact(f"tests/lit/{dialect}/.keep", f"# lit tests for {dialect} (add after review)\n"),
    ]


def _lit_test(dialect: str) -> str:
    return f"""// RUN: merlin-{dialect}-opt %s | merlin-{dialect}-opt | FileCheck %s
// (Requires the built {dialect} opt tool; see lib/Dialect/*/README.md.)
// CHECK-LABEL: func.func @rrhs
func.func @rrhs(%A: tensor<64x128xi8>, %W: tensor<128x64xi8>) -> tensor<64x64xi8> {{
  // CHECK: {dialect}.res_pack
  %res = {dialect}.res_pack %W {{layout = "packed_rhs"}} : tensor<128x64xi8> -> !{dialect}.resident_tensor<i8>
  // CHECK: {dialect}.matmul
  %acc = {dialect}.matmul %A, %res : tensor<64x128xi8>, !{dialect}.resident_tensor<i8> -> !{dialect}.accumulator<i32>
  // CHECK: {dialect}.commit
  %Y = {dialect}.commit %acc {{epilogue = ["requant", "relu"]}} : !{dialect}.accumulator<i32> -> tensor<64x64xi8>
  {dialect}.evict %res : !{dialect}.resident_tensor<i8>
  return %Y : tensor<64x64xi8>
}}
"""
