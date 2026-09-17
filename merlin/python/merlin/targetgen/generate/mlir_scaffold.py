"""Generate a complete, idiomatic MLIR/C++ (ODS) dialect from a dialect_plan.

The emitted ODS + C++ + CMake is real, conventional MLIR code (not placeholder comments):
type and operation definitions for every declaration, a registered dialect with `initialize()`,
and a name-based lowering pass for every reviewed mapping. The reference ToyNPU signatures add
assembly formats, traits, and a commit verifier. Everything is wired through
`add_mlir_dialect` / `mlir_tablegen`.

Compiling it requires an MLIR/LLVM build (TableGen + headers); that toolchain is a build
dependency of the generated repo, documented in lib/Dialect/<D>/README.md. The C++ is written
to that contract even though this generator cannot itself invoke mlir-tblgen.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact
from .target_repo import camel

# The ToyNPU op/type set keeps its stronger hand-written signatures and verifier. Every other
# reviewed plan is still concrete: its own declarations become generic, variadic ODS operations
# and element-typed ODS types. That is deliberately less semantic than the ToyNPU reference, but it
# is manipulable IR rather than a comment-only placeholder.
KNOWN = {"res_pack", "matmul", "commit", "evict"}


def _definition_token(name: str) -> str:
    """Return a deterministic TableGen/C++ class-name component for a plan declaration."""
    words: list[str] = []
    current = ""
    for char in name:
        if char.isalnum():
            current += char
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    token = "".join(word[:1].upper() + word[1:] for word in words) or "Declared"
    return f"N{token}" if token[0].isdigit() else token


def _plan_entries(plan: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return validated named declarations while preserving plan order."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in plan.get(key, []):
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            continue
        name = raw["name"].strip()
        if not name or name in seen:
            continue
        entries.append({**raw, "name": name})
        seen.add(name)
    return entries


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


def _declared_types_td(cls: str, types: list[dict[str, Any]]) -> str:
    declarations: list[str] = []
    for entry in types:
        name = entry["name"]
        token = _definition_token(name)
        summary = str(entry.get("summary") or f"Declared {name} target type.").replace('"', "'")
        declarations.append(
            f'''def {cls}_{token} : {cls}_Type<"{token}", "{name}"> {{
  let summary = "{summary}";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}}'''
        )
    body = "\n\n".join(declarations)
    return f"""//===- {cls}Types.td - declarations from dialect_plan --------*- tablegen -*-===//
#ifndef {cls.upper()}_TYPES
#define {cls.upper()}_TYPES

include "{cls}Dialect.td"

{body}

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


def _ops_td(cls: str, dialect: str, extra_ops: list[dict[str, Any]] | None = None) -> str:
    extra = "\n\n".join(
        _declared_op_definition(cls, dialect, entry) for entry in (extra_ops or [])
    )
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

{extra}

#endif // {cls.upper()}_OPS
"""


def _declared_op_definition(cls: str, dialect: str, entry: dict[str, Any]) -> str:
    name = entry["name"]
    summary = str(entry.get("summary") or f"Declared {name} target operation.").replace('"', "'")
    return f'''// {dialect}.{name} is declared by contracts/dialect_plan.yaml.
def {cls}_{_definition_token(name)}Op : {cls}_Op<"{name}"> {{
  let summary = "{summary}";
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
}}'''


def _declared_ops_td(cls: str, dialect: str, ops: list[dict[str, Any]]) -> str:
    body = "\n\n".join(_declared_op_definition(cls, dialect, entry) for entry in ops)
    return f"""//===- {cls}Ops.td - declarations from dialect_plan ----------*- tablegen -*-===//
#ifndef {cls.upper()}_OPS
#define {cls.upper()}_OPS

include "{cls}Dialect.td"
include "{cls}Types.td"
include "mlir/IR/BuiltinTypes.td"

{body}

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


def _declared_ops_cpp(cls: str, pkg: str, dialect: str) -> str:
    return f"""//===- {cls}Ops.cpp - generated operation definitions ------------------===//
#include "{pkg}/Dialect/{cls}/IR/{cls}Dialect.h"

using namespace mlir;
using namespace merlin::{dialect};

// Operation definitions are generated by mlir-tblgen from the reviewed dialect plan.
// This translation unit intentionally contains no target semantics: those belong in
// explicit verifiers and lowering patterns added during human review.
"""


def _passes_h(cls: str, pkg: str, dialect: str) -> str:
    return f"""//===- Passes.h - {dialect} target transformation passes ----------------===//
#pragma once

#include <memory>

namespace mlir {{
class Pass;
}}

namespace merlin::{dialect} {{

/// Create the reviewed, name-based interface-to-{dialect} lowering pass.
std::unique_ptr<::mlir::Pass> createLowerInterfacePass();

/// Register all generated {dialect} passes with MLIR's pass registry.
void registerPasses();

}} // namespace merlin::{dialect}
"""


def _lowering_cpp(
    cls: str,
    pkg: str,
    dialect: str,
    lowering: list[dict[str, str]],
) -> str:
    pattern_lines: list[str] = []
    remaining_lines: list[str] = []
    mapping_comments: list[str] = []
    for mapping in lowering:
        source = mapping["from"]
        target = mapping["to"]
        mapping_comments.append(f"//   {source} -> {target}")
        if source != target:
            pattern_lines.append(
                f'    patterns.add<RenameByNamePattern>(context, "{source}", "{target}");'
            )
        remaining_lines.append(
            f'      if (name == "{source}") {{\n'
            f'        op->emitError("generated lowering did not convert {source} to {target}");\n'
            "        conversionFailed = true;\n"
            "      }"
        )
    patterns = "\n".join(pattern_lines) or "    // No non-identity mappings were declared."
    remaining = "\n".join(remaining_lines) or "      (void)name;"
    mapping_table = "\n".join(mapping_comments) or "//   (no mappings declared)"
    return f"""//===- LowerInterface.cpp - generated interface lowering ---------------===//
#include "{pkg}/Dialect/{cls}/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <string>

using namespace mlir;

namespace merlin::{dialect} {{
namespace {{

// Exact mappings reviewed in contracts/dialect_plan.yaml:
{mapping_table}
//
// This generic conversion preserves locations, operands, result types, and attributes. It
// rejects region- or successor-bearing operations because silently cloning their control-flow
// semantics would exceed a name-based scaffold's contract.
class RenameByNamePattern final : public RewritePattern {{
public:
  RenameByNamePattern(MLIRContext *context, StringRef sourceName,
                      StringRef targetName)
      : RewritePattern(sourceName, PatternBenefit(1), context),
        targetName(targetName.str()) {{}}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {{
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
      return rewriter.notifyMatchFailure(
          op, "generic target lowering only accepts regionless, successor-free operations");

    OperationState state(op->getLoc(), targetName);
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    Operation *lowered = rewriter.create(state);
    rewriter.replaceOp(op, lowered->getResults());
    return success();
  }}

private:
  std::string targetName;
}};

class LowerInterfacePass final
    : public PassWrapper<LowerInterfacePass, OperationPass<ModuleOp>> {{
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerInterfacePass)

  StringRef getArgument() const final {{ return "lower-interface-to-{dialect}"; }}
  StringRef getDescription() const final {{
    return "Lower reviewed interface operation names to the {dialect} target dialect";
  }}

  void runOnOperation() override {{
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
{patterns}
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {{
      signalPassFailure();
      return;
    }}

    bool conversionFailed = false;
    getOperation().walk([&](Operation *op) {{
      StringRef name = op->getName().getStringRef();
{remaining}
    }});
    if (conversionFailed)
      signalPassFailure();
  }}
}};

}} // namespace

std::unique_ptr<Pass> createLowerInterfacePass() {{
  return std::make_unique<LowerInterfacePass>();
}}

void registerPasses() {{ PassRegistration<LowerInterfacePass>(); }}

}} // namespace merlin::{dialect}
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


def _transforms_cmake(cls: str) -> str:
    return f"""# Generated name-based interface lowering for the reviewed dialect plan.
add_mlir_library(MLIR{cls}Transforms
  LowerInterface.cpp

  LINK_LIBS PUBLIC
  MLIR{cls}
  MLIRIR
  MLIRPass
  MLIRTransforms
)
"""


def _dialect_cmake() -> str:
    return """# Generated target dialect and its reviewed interface lowering.
add_subdirectory(IR)
add_subdirectory(Transforms)
"""


def _root_cmake(cls: str, pkg: str) -> str:
    return f"""cmake_minimum_required(VERSION 3.20)
project({pkg} LANGUAGES CXX C)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${{MLIR_DIR}}")

list(APPEND CMAKE_MODULE_PATH "${{MLIR_CMAKE_DIR}}")
list(APPEND CMAKE_MODULE_PATH "${{LLVM_CMAKE_DIR}}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories("${{MLIR_INCLUDE_DIRS}}")
include_directories("${{CMAKE_CURRENT_SOURCE_DIR}}/include")
include_directories("${{CMAKE_CURRENT_BINARY_DIR}}/include")
add_definitions(${{LLVM_DEFINITIONS}})

add_subdirectory(include/{pkg}/Dialect/{cls}/IR)
add_subdirectory(lib/Dialect/{cls})
"""


def _lib_readme(
    cls: str,
    dialect: str,
    *,
    op_count: int = 4,
    type_count: int = 2,
    lowering_count: int = 0,
    specialized: bool = True,
) -> str:
    detail = (
        "typed operation definitions and a real `CommitOp::verify()`"
        if specialized
        else f"{op_count} reviewed operation definitions, {type_count} reviewed type definitions, "
             f"and {lowering_count} exact name-based interface lowering patterns"
    )
    return f"""# {cls} dialect (MLIR/C++)

Complete ODS + C++ for the `{dialect}` dialect: {detail}, registered through
`add_mlir_dialect` / `add_mlir_library` wiring. The generic lowering refuses operations with
regions or successors and verifies that no declared source operation remains.

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
    ops = _plan_entries(dialect_plan, "ops")
    types = _plan_entries(dialect_plan, "types")
    op_names = {o["name"] for o in ops}
    lowering = [
        {"from": row["from"].strip(), "to": row["to"].strip()}
        for row in dialect_plan.get("lowering", [])
        if isinstance(row, dict)
        and isinstance(row.get("from"), str)
        and isinstance(row.get("to"), str)
        and row["from"].strip()
        and row["to"].strip()
    ]
    transforms = f"include/{pkg}/Dialect/{cls}/Transforms"
    libtransforms = f"lib/Dialect/{cls}/Transforms"

    if op_names >= KNOWN:
        return [
            Artifact("CMakeLists.txt", _root_cmake(cls, pkg)),
            Artifact(f"{ir}/{cls}Dialect.td", _dialect_td(cls, dialect)),
            Artifact(f"{ir}/{cls}Types.td", _types_td(cls)),
            Artifact(f"{ir}/{cls}Attrs.td", _attrs_td(cls)),
            Artifact(f"{ir}/{cls}Ops.td",
                     _ops_td(cls, dialect, [entry for entry in ops if entry["name"] not in KNOWN])),
            Artifact(f"{ir}/{cls}Dialect.h", _dialect_h(cls, pkg)),
            Artifact(f"{ir}/CMakeLists.txt", _ir_cmake(cls, dialect)),
            Artifact(f"{libir}/{cls}Dialect.cpp", _dialect_cpp(cls, pkg, dialect)),
            Artifact(f"{libir}/{cls}Ops.cpp", _ops_cpp(cls, pkg, dialect)),
            Artifact(f"{libir}/CMakeLists.txt", _lib_cmake(cls)),
            Artifact(f"{transforms}/Passes.h", _passes_h(cls, pkg, dialect)),
            Artifact(f"{libtransforms}/LowerInterface.cpp",
                     _lowering_cpp(cls, pkg, dialect, lowering)),
            Artifact(f"{libtransforms}/CMakeLists.txt", _transforms_cmake(cls)),
            Artifact(f"lib/Dialect/{cls}/CMakeLists.txt", _dialect_cmake()),
            Artifact(f"lib/Dialect/{cls}/README.md",
                     _lib_readme(cls, dialect, lowering_count=len(lowering))),
            Artifact(f"tests/lit/{dialect}/res_pack_roundtrip.mlir",
                     _lit_test(dialect)),
            Artifact(f"tests/lit/{dialect}/interface_lowering.mlir",
                     _lowering_lit_test(dialect, lowering)),
        ]
    # Reviewed non-reference plans get one concrete ODS declaration per plan declaration.
    return [
        Artifact("CMakeLists.txt", _root_cmake(cls, pkg)),
        Artifact(f"{ir}/{cls}Dialect.td", _dialect_td(cls, dialect)),
        Artifact(f"{ir}/{cls}Types.td", _declared_types_td(cls, types)),
        Artifact(f"{ir}/{cls}Ops.td", _declared_ops_td(cls, dialect, ops)),
        Artifact(f"{ir}/{cls}Dialect.h", _dialect_h(cls, pkg)),
        Artifact(f"{ir}/CMakeLists.txt", _ir_cmake(cls, dialect)),
        Artifact(f"{libir}/{cls}Dialect.cpp", _dialect_cpp(cls, pkg, dialect)),
        Artifact(f"{libir}/{cls}Ops.cpp", _declared_ops_cpp(cls, pkg, dialect)),
        Artifact(f"{libir}/CMakeLists.txt", _lib_cmake(cls)),
        Artifact(f"{transforms}/Passes.h", _passes_h(cls, pkg, dialect)),
        Artifact(f"{libtransforms}/LowerInterface.cpp",
                 _lowering_cpp(cls, pkg, dialect, lowering)),
        Artifact(f"{libtransforms}/CMakeLists.txt", _transforms_cmake(cls)),
        Artifact(f"lib/Dialect/{cls}/CMakeLists.txt", _dialect_cmake()),
        Artifact(f"lib/Dialect/{cls}/README.md",
                 _lib_readme(cls, dialect, op_count=len(ops), type_count=len(types),
                             lowering_count=len(lowering), specialized=False)),
        Artifact(f"tests/lit/{dialect}/interface_lowering.mlir",
                 _lowering_lit_test(dialect, lowering)),
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


def _lowering_lit_test(dialect: str, lowering: list[dict[str, str]]) -> str:
    checks = "\n".join(f'  // CHECK: "{row["to"]}"' for row in lowering)
    operations = "\n".join(
        f'  "{row["from"]}"() : () -> ()' for row in lowering
    )
    return f"""// RUN: merlin-{dialect}-opt --allow-unregistered-dialect \\
// RUN:   --lower-interface-to-{dialect} %s | FileCheck %s
// Exact name-based smoke test generated from contracts/dialect_plan.yaml.
func.func @lower_all_declared_interface_ops() {{
{checks}
{operations}
  return
}}
"""
