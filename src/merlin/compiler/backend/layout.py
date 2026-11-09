"""Compile one proved static tensor layout operation to portable scalar C.

This provider preserves storage bits using the shared layout emitter. It is not
a general scalar compiler and never selects an accelerator or builds an image.
"""

from __future__ import annotations

import json

from xdsl.dialects import builtin, func, tensor

from merlin.compiler.artifact import Artifact, Compilation
from merlin.compiler.backend.cabi import CRegionABI, TensorPort
from merlin.compiler.lower.c.layout import emit_layout_c
from merlin.compiler.request import CompileOptions
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

name = "merlin.layout"


def _port(value) -> TensorPort:
    typ = value.type
    if type(typ) is not builtin.TensorType or type(typ.encoding) is not builtin.NoneAttr:
        raise ValueError("layout compilation requires ordinary unencoded tensor ports")
    return TensorPort(tuple(typ.get_shape()), str(typ.element_type))


def _prepare(request):
    if request.source.kind != "mlir" or request.options != CompileOptions():
        raise ValueError("layout compilation accepts MLIR without deployment or backend options")
    module = parse_mlir_text(request.source.text)
    module.verify()
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not func.FuncOp:
        raise ValueError("layout compilation needs exactly one function")
    function = functions[0]
    if (
        module.properties
        or any(not key.startswith("prov.") for key in module.attributes)
        or any(not key.startswith("prov.") for key in function.attributes)
        or set(function.properties) - {"sym_name", "function_type", "sym_visibility"}
        or len(function.body.blocks) != 1
    ):
        raise ValueError("layout function has unsupported properties, metadata or control")
    block = function.body.block
    operations = tuple(block.ops)
    if len(block.args) != 1 or len(operations) != 2 or type(operations[-1]) is not func.ReturnOp:
        raise ValueError("layout compilation needs one input, one layout operation and a direct return")
    operation, returned = operations
    if (
        type(operation) not in {tensor.CastOp, tensor.CollapseShapeOp, tensor.ExpandShapeOp}
        or operation.regions
        or operation.successors
        or len(operation.results) != 1
        or tuple(operation.operands) != (block.args[0],)
        or tuple(returned.operands) != tuple(operation.results)
        or returned.attributes
        or returned.properties
        or any(not key.startswith("prov.") for key in operation.attributes)
    ):
        raise ValueError("layout compilation admits only a pure single cast/collapse/expand and its exact result")
    input_port, output_port = _port(block.args[0]), _port(operation.results[0])
    if type(operation) is tensor.CastOp:
        if operation.properties or block.args[0].type != operation.results[0].type:
            raise ValueError("static tensor.cast must preserve the exact input type")
        typ = TensorType(input_port.shape, input_port.dtype, "bit-preserving")
        kernel = KernelRequest(
            nodes=(SemanticNode("input", "input", (), typ, effect="input"),
                   SemanticNode("output", "identity", ("input",), typ)),
            outputs=("output",), output_storages=("external",),
            input_storages=(("input", "external"),), target_identity=name,
            source_identity=request.source.sha256,
        )
    else:
        # The existing semantic bridge checks complete static reassociation,
        # shape products, element identity, properties and initialized input.
        kernel = translate_linalg_text(
            request.source.text, entry=function.sym_name.data, target_identity=name,
        ).request
    # 192 digest bits retain deterministic identity within the C packager's
    # 64-character identifier limit. The full source hash remains in records.
    symbol = "merlin_layout_" + request.source.sha256[:48]
    description, files = emit_layout_c(kernel, symbol)
    abi = CRegionABI(symbol, (input_port,), (output_port,), 1,
                     tuple(path for path in files if path.endswith(".c")), resource_kind="cpu")
    return kernel, description, files, abi


def validate(request) -> None:
    """Admit source and generated C before creating output files."""
    _prepare(request)


def compile(request) -> Compilation:
    kernel, description, files, abi = _prepare(request)
    if request.output.exists():
        raise FileExistsError(f"layout compilation requires fresh output: {request.output}")
    request.output.mkdir(parents=True)
    artifacts = []
    for relative, text in files.items():
        path = request.output / relative
        path.write_text(text, encoding="utf-8")
        artifacts.append(Artifact.from_file(path, kind="c-source" if path.suffix == ".c" else "c-header",
                                            role="region"))
    record = request.output / "layout.json"
    record.write_text(json.dumps({
        "schema": "merlin.layout.v1", "source_sha256": request.source.sha256,
        "semantic_sha256": kernel.digest(), "implementation": description,
        "scope": "single-static-byte-preserving-layout", "execution_observed": False,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts.append(Artifact.from_file(record, kind="json", role="provider-record"))
    result = Compilation(request.output, name, request.source.sha256, tuple(artifacts),
                         metadata={"c_call": abi.record(), "execution_observed": False})
    CRegionABI.from_compilation(result)
    return result
