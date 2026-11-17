"""Consumer-owned correspondence fixtures; no producer or evaluator import."""

import hashlib
from copy import deepcopy
from io import StringIO

import pytest
from xdsl.dialects import arith, builtin, func
from xdsl.printer import Printer

from merlin.frontends.execution_model_correspondence import checked_execution_model
from merlin.frontends.linalg_mlir import parse_mlir_text


def printed(module, *, locations=False):
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True, print_debuginfo=locations).print_op(module)
    return stream.getvalue() + "\n"


@pytest.fixture
def selection():
    parent = parse_mlir_text("""module attributes {prov.weights_file="weights.safetensors"} {
      func.func @forward(%x:tensor<4xi32>)->tensor<4xi32> {
        %unused=arith.constant 7:i32
        %small=arith.trunci %unused:i32 to i16
        %r=func.call @original(%x) {prov.kernel="fixture::identity",
          prov.kernel_operands=["x"],prov.kernel_args={}}:(tensor<4xi32>)->tensor<4xi32>
        return %r:tensor<4xi32>
      }
      func.func private @original(tensor<4xi32>)->tensor<4xi32>
        attributes {prov.kernel="fixture::identity"}
    }""")
    # A handwritten expected expansion: preserve even the unused source
    # arithmetic and replace the actual identity call result by its operand.
    bound = parse_mlir_text("""module attributes {prov.weights_file="weights.safetensors",
        prov.kernel_binding="t2m.model-kernel-binding.v4"} {
      func.func @forward(%x:tensor<4xi32>)->tensor<4xi32> {
        %unused=arith.constant 7:i32
        %small=arith.trunci %unused:i32 to i16
        return %x:tensor<4xi32>
      }
    }""")
    kernel = parse_mlir_text("""module {
      func.func @identity(%x:tensor<4xi32>)->tensor<4xi32> {return %x:tensor<4xi32>}
    }""")
    for module in (parent, bound, kernel):
        module.verify()
    source = printed(kernel, locations=True)
    digest = hashlib.sha256(printed(kernel).encode()).hexdigest()
    spec = {
        "schema": "t2m.shared-tensor-abi.v2",
        "entry": "identity",
        "buffers": [
            {"name": "x", "shape": [4], "effect": "read", "dtype": "i32"},
            {"name": "out", "shape": [4], "effect": "write", "dtype": "i32"},
        ],
        "grid": [1],
        "scalar_constants": [],
        "intent": "",
    }
    original = next(op for op in parent.walk() if isinstance(op, func.CallOp))
    record = {
        "id": "kernel_call.0000",
        "parent_call_ordinal": 0,
        "callee": "original",
        "kernel": "fixture::identity",
        "schema_operands": ["x"],
        "model_call_attributes": {name: str(value) for name, value in original.attributes.items()},
        "source_operand_bindings": {"x": "x"},
        "source_result_order": ["out"],
        "captured_scalar_arguments": {},
        "runtime_scalar_arguments": {},
        "runtime_operand_arguments": {},
        "source_layout_arguments": {},
        "implementation_sha256": digest,
        "source_sha256": "a" * 64,
    }
    trace = {
        "schema": "t2m.model-kernel-binding.v4",
        "binding_profile": "t2m.execution-checked-calls.v1",
        "result_mode": "execution-owned",
        "resource_scope": "execution-allocator-accounted",
        "calls": [record],
        "kernels": {digest: {"spec": spec, "source_sha256": "a" * 64}},
        "located_implementations": {
            digest: {"sha256": hashlib.sha256(source.encode()).hexdigest(), "bytes": len(source.encode())}
        },
    }
    return parent, bound, trace, {digest: (spec, source)}


def test_handwritten_complete_expansion_needs_no_producer_analysis(selection):
    report = checked_execution_model(*selection)
    assert report["schema"] == "merlin.executed-model-correspondence.v1"
    assert report["calls"] == 1


def test_retained_kernel_dependency_cannot_survive_as_an_opaque_call(selection):
    parent, bound, trace, interfaces = selection
    previous = next(iter(interfaces))
    spec = interfaces[previous][0]
    kernel = parse_mlir_text("""module {
      func.func @identity(%x:tensor<4xi32>)->tensor<4xi32> {
        %next=func.call @identity(%x):(tensor<4xi32>)->tensor<4xi32>
        return %next:tensor<4xi32>
      }
    }""")
    kernel.verify()
    source = printed(kernel, locations=True)
    digest = hashlib.sha256(printed(kernel).encode()).hexdigest()
    trace = deepcopy(trace)
    trace["kernels"] = {digest: trace["kernels"][previous]}
    trace["calls"][0]["implementation_sha256"] = digest
    trace["located_implementations"] = {
        digest: {"sha256": hashlib.sha256(source.encode()).hexdigest(), "bytes": len(source.encode())}
    }
    with pytest.raises(ValueError, match="unresolved source dependency"):
        checked_execution_model(parent, bound, trace, {digest: (spec, source)})


@pytest.mark.parametrize(
    "damage",
    [
        "ordinary-op",
        "result-type",
        "extra-op",
        "ordinal",
        "operand",
        "result",
        "extension",
        "located",
        "program",
        "inventory",
    ],
)
def test_receipt_rehashing_cannot_substitute_complete_source_correspondence(selection, damage):
    parent, bound, trace, interfaces = selection
    trace = deepcopy(trace)
    if damage == "ordinary-op":
        value = next(op for op in bound.walk() if isinstance(op, arith.ConstantOp))
        value.properties["value"] = builtin.IntegerAttr(8, builtin.i32)
    elif damage == "result-type":
        old = next(op for op in bound.walk() if isinstance(op, arith.TruncIOp))
        new = arith.TruncIOp(old.operands[0], builtin.i8)
        new.attributes.update(old.attributes)
        new.location = old.location
        old.parent.insert_op_before(new, old)
        old.parent.erase_op(old)
    elif damage == "extra-op":
        block = next(op for op in bound.ops if isinstance(op, func.FuncOp)).body.block
        block.insert_op_before(arith.ConstantOp.from_int_and_width(0, builtin.i32), block.last_op)
    elif damage == "ordinal":
        trace["calls"][0]["parent_call_ordinal"] = 1
    elif damage == "operand":
        trace["calls"][0]["source_operand_bindings"] = {"x": "other"}
    elif damage == "result":
        trace["calls"][0]["source_result_order"] = []
    elif damage == "extension":
        trace["calls"][0]["unknown_shape_permission"] = True
    elif damage == "located":
        trace["located_implementations"] = {}
    elif damage == "program":
        trace["runtime_shape_contract"] = {"outputs": []}
    else:
        interfaces = {}
    bound.verify()
    trace["mlir_sha256"] = hashlib.sha256(printed(bound, locations=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        checked_execution_model(parent, bound, trace, interfaces)
