"""Conservative host admission for graph proofs, not a proof of the parser.

Raw file metadata is outside xDSL's ModuleOp and must be checked BEFORE parsing.
The structural check is also required: callers may supply already parsed IR or
resource handles without a file dictionary. Unknown attribute representations
are refused, rather than searched as strings. Shape's direct-call mode permits
only the callee field handled by its separate finite-depth call proof.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

from .lean_graph_generic_affine import affine_payload
from .lean_graph_input_admission import ModelRefusal


class InputAdmissionError(ModelRefusal):
    """Unsupported proof domain; parser/IR-verifier errors remain distinct."""

    def __init__(self, code: str, path: str):
        self.code, self.path = code, path
        super().__init__(f"unsupported graph input [{code}] at {path}")


def admit_text(raw: str) -> None:
    """Lex ORIGINAL text; quoted strings/comments are not metadata delimiters.

    This uses the selected parser's real lexer, not a substring/line heuristic.
    Lexical errors propagate as ParseError; finding metadata is a domain refusal
    and does not assert that the rest of the input is well formed.
    """
    from xdsl.utils.lexer import Input
    from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

    lexer = MLIRLexer(Input(raw, "graph-original-input"))
    while True:
        token = lexer.lex()
        if token.kind == MLIRTokenKind.FILE_METADATA_BEGIN:
            raise InputAdmissionError("file-metadata", f"byte/character-offset:{token.span.start}")
        if token.kind == MLIRTokenKind.EOF:
            return


def _attributes(value, path):
    """Yield every structural attribute, refusing uninspectable payloads."""
    from xdsl.dialects.builtin import OpaqueAttr, UnregisteredAttr
    from xdsl.ir import Attribute, Data, ParametrizedAttribute

    if isinstance(value, (OpaqueAttr, UnregisteredAttr)):
        raise InputAdmissionError("opaque-attribute", path)
    if isinstance(value, Attribute):
        yield value, path
        if isinstance(value, ParametrizedAttribute):
            yield from _attributes(value.parameters, path + ".parameters")
        elif isinstance(value, Data):
            yield from _attributes(value.data, path + ".data")
        else:
            raise InputAdmissionError("unknown-attribute-representation", path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise InputAdmissionError("unknown-metadata-key", path)
            yield from _attributes(item, path + "." + key)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            yield from _attributes(item, f"{path}[{index}]")
    elif isinstance(value, Enum):
        yield from _attributes(value.value, path + ".enum")
    elif (affine := affine_payload(value)) is not None:
        yield from _attributes(affine, path + ".affine-data")
    elif value is not None and type(value) not in (str, int, bool, float, bytes):
        raise InputAdmissionError("unknown-attribute-data", path)


def _fields(operation):
    yield operation.attributes, "attributes"
    yield operation.properties, "properties"
    yield operation.location, "location"
    for index, value in enumerate((*operation.operands, *operation.results)):
        yield value.type, f"value-type[{index}]"
    for ri, region in enumerate(operation.regions):
        for bi, block in enumerate(region.blocks):
            for ai, arg in enumerate(block.args):
                yield arg.type, f"region[{ri}].block[{bi}].argument[{ai}]"


def admit_resources(operation) -> None:
    """Resource-only producer guard; this does NOT grant a call semantics claim."""
    from xdsl.dialects.builtin import DenseResourceAttr

    for index, op in enumerate(operation.walk()):
        for value, field in _fields(op):
            for attribute, path in _attributes(value, f"operation[{index}].{field}"):
                if isinstance(attribute, DenseResourceAttr):
                    raise InputAdmissionError("dense-resource", path)


def admit_module(module, *, calls: str = "external-only") -> None:
    """Admit resources and call dependencies in a complete parsed module.

    `shape-direct` is deliberately not an arbitrary symbol-reference exception:
    only a registered direct call's flat, signature-checked callee is allowed.
    The downstream shape checker must still prove body/depth correspondence.
    All other SymbolRefAttr uses are currently unsupported (including external
    symbolic metadata), avoiding implicit symbol lookup in opaque primitives.
    """
    from xdsl.dialects import builtin, func

    if calls not in ("external-only", "shape-direct"):
        raise ValueError("unknown graph call admission mode")
    if type(module) is not builtin.ModuleOp or len(module.body.blocks) != 1:
        raise InputAdmissionError("complete-module-required", "module")
    admit_resources(module)
    functions = {}
    for op in module.body.block.ops:
        if type(op) is func.FuncOp:
            if op.sym_name.data in functions:
                raise InputAdmissionError("duplicate-function", op.sym_name.data)
            functions[op.sym_name.data] = op
    for index, op in enumerate(module.walk()):
        path = f"operation[{index}]"
        name = op.op_name.data if isinstance(op, builtin.UnregisteredOp) else op.name
        if name.startswith("linalg.") and ("library_call" in op.properties or "library_call" in op.attributes):
            raise InputAdmissionError("opaque-library-call-unbound-symbol", path)
        if name in ("func.constant", "func.call_indirect"):
            raise InputAdmissionError("function-value-route", path)
        if name in ("func.func", "func.call") and type(op) not in (func.FuncOp, func.CallOp):
            raise InputAdmissionError("unregistered-function-operation", path)
        if type(op) is func.FuncOp and op not in functions.values():
            raise InputAdmissionError("nested-function", path)
        if type(op) is func.CallOp:
            callee = op.callee
            target = functions.get(callee.root_reference.data)
            if callee.nested_references or target is None:
                raise InputAdmissionError("unresolved-or-nested-callee", path)
            if target.body.blocks and calls != "shape-direct":
                raise InputAdmissionError("defined-callee", path)
            if tuple(v.type for v in op.operands) != tuple(target.function_type.inputs) or tuple(
                v.type for v in op.results
            ) != tuple(target.function_type.outputs):
                raise InputAdmissionError("call-signature", path)
        for value, field in _fields(op):
            for attribute, subpath in _attributes(value, field):
                if isinstance(attribute, builtin.SymbolRefAttr):
                    if not (type(op) is func.CallOp and subpath == "properties.callee"):
                        raise InputAdmissionError("unmodeled-symbol-reference", path + "." + subpath)
                if isinstance(attribute, builtin.FunctionType):
                    if not (type(op) is func.FuncOp and subpath == "properties.function_type"):
                        raise InputAdmissionError("function-value-type", path + "." + subpath)


def parse_admitted(raw: str, parser, *, calls: str = "external-only"):
    """Guard raw bytes' decoded text, parse/verify, then inspect actual IR."""
    admit_text(raw)
    module = parser(raw)
    module.verify()
    admit_module(module, calls=calls)
    return module


def source_reference():
    """Bind the NEW admission implementation in capture/source closures."""
    path = Path(__file__).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
