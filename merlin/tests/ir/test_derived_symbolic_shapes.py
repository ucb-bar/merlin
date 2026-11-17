"""Independent symbolic staging proves source axes and executable destinations."""

from pathlib import Path

import pytest
from test_derived_compile_inputs import capture_fixture, encoded, record
from xdsl.dialects.func import CallOp

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.derived_symbolic_shapes import checked_model_shapes
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.frontends.model_shape_facts import PROFILE


def example():
    parent = parse_mlir_text("""module attributes {prov.weights_file="weights.safetensors"} {
      func.func @forward(%x:tensor<?xi32>)->(tensor<?xi32>,tensor<?xi32>,tensor<?xi32>) {
        %zero=arith.constant 0:index
        %n=tensor.dim %x,%zero:tensor<?xi32>
        %word=arith.index_castui %n:index to i64
        %back=arith.index_castui %word:i64 to index
        %same=arith.cmpi eq,%n,%back:index
        cf.assert %same,"round trip"
        %low=arith.constant 0:i64
        %high=arith.constant 31:i64
        %minimum=arith.cmpi uge,%word,%low:i64
        %maximum=arith.cmpi ule,%word,%high:i64
        %valid=arith.andi %minimum,%maximum:i1
        cf.assert %valid,"source domain"
        %first=func.call @kernel(%x,%n) {prov.kernel="test::dynamic",prov.kernel_operands=["input","n"]}:
          (tensor<?xi32>,index)->tensor<?xi32>
        %second=func.call @kernel(%first,%n) {prov.kernel="test::dynamic",prov.kernel_operands=["input","n"]}:
          (tensor<?xi32>,index)->tensor<?xi32>
        return %second,%first,%x:tensor<?xi32>,tensor<?xi32>,tensor<?xi32>
      }
      func.func private @kernel(tensor<?xi32>,index)->tensor<?xi32> attributes {prov.kernel="test::dynamic"}
    }""")
    kernel = parse_mlir_text("""module {func.func @source(%x:tensor<?xi32>,%n:i32)->tensor<?xi32> {
      %size=arith.index_cast %n:i32 to index
      %out=tensor.empty(%size):tensor<?xi32>
      return %out:tensor<?xi32>
    }}""")
    spec = {
        "schema": "t2m.shared-tensor-abi.v5",
        "entry": "source",
        "buffers": [
            {"name": "x", "shape": ["n"], "dtype": "i32", "effect": "read"},
            {"name": "out", "shape": ["n"], "dtype": "i32", "effect": "write"},
        ],
        "runtime_scalars": [{"name": "n", "dtype": "i32", "source_arg": 1, "bounds": [0, 31]}],
    }
    calls = [
        {
            "id": f"call.{i}",
            "implementation_sha256": "kernel",
            "parent_call_ordinal": i,
            "kernel": "test::dynamic",
            "callee": "kernel",
            "schema_operands": ["input", "n"],
            "source_operand_bindings": {"x": "input"},
            "source_result_order": ["out"],
            "runtime_scalar_bindings": {},
            "runtime_operand_bindings": {
                "n": {
                    "schema_operand": "n",
                    "caller_type": "index",
                    "kernel_type": "i32",
                    "adapter": "checked-index-castui",
                    "unsigned_integer_range": [0, 31],
                }
            },
            "symbolic_tensor_shapes": {"x": ["d0_0"], "out": ["d0_0"]},
            "model_call_attributes": {key: str(value) for key, value in op.attributes.items()},
        }
        for i, op in enumerate(op for op in parent.walk() if isinstance(op, CallOp))
    ]
    bindings = {
        "model_shape_profile": PROFILE,
        "argument_binding_profile": "t2m.checked-index-runtime-arguments.v1",
        "calls": calls,
        "kernels": {"kernel": {}},
        "runtime_shape_contract": {
            "schema": "t2m.runtime-shapes.v2",
            "inputs": [{"shape": ["d0_0"], "dtype": "i32"}],
            "outputs": [{"shape": ["d0_0"], "dtype": "i32"}] * 3,
            "dimension_bindings": {"d0_0": {"argument": 0, "axis": 0}},
        },
    }
    bound = parent.clone()
    for op in list(bound.walk()):
        if isinstance(op, CallOp):
            op.results[0].replace_all_uses_with(op.operands[0])
            op.parent.erase_op(op)
    bound.body.block.erase_op(list(bound.ops)[1])
    for module in (parent, bound, kernel):
        module.verify()
    return parent, bound, bindings, {"kernel": (spec, next(iter(kernel.ops)))}


def test_dependent_shapes_and_maximum_scratch_sizes_follow_original_source():
    parent, bound, bindings, interfaces = example()
    actual = checked_model_shapes(parent, bound, bindings, interfaces)
    assert actual["contract"] == bindings["runtime_shape_contract"]
    assert actual["call_shapes"] == {i: {"x": ["d0_0"], "out": ["d0_0"]} for i in (0, 1)}
    assert actual["maximum_shapes"] == {i: {"x": [31], "out": [31]} for i in (0, 1)}


@pytest.mark.parametrize("damage", ["profile", "downgrade", "order", "range", "axis", "result", "scope", "body"])
def test_claims_cannot_replace_original_axes_guards_or_complete_body(damage):
    parent, bound, bindings, interfaces = example()
    if damage == "profile":
        bindings.pop("model_shape_profile")
    elif damage == "downgrade":
        bindings["argument_binding_profile"] = "t2m.live-runtime-arguments.v1"
    elif damage == "order":
        bindings["calls"].reverse()
    elif damage == "range":
        bindings["calls"][0]["runtime_operand_bindings"]["n"]["unsigned_integer_range"] = [0, 30]
    elif damage == "axis":
        bindings["calls"][1]["symbolic_tensor_shapes"]["x"] = [13]
    elif damage == "result":
        bindings["runtime_shape_contract"]["outputs"] = [{"shape": [13], "dtype": "i32"}] * 3
    elif damage == "scope":
        bindings.update(guard_storage_bytes=32, guard_storage_scope="sampled-shape")
    else:
        spec, _function = interfaces["kernel"]
        wrong = parse_mlir_text("""module {func.func @source(%x:tensor<?xi32>,%n:i32)->tensor<?xi32> {
          %one=arith.constant 1:index
          %out=tensor.empty(%one):tensor<?xi32>
          return %out:tensor<?xi32>
        }}""")
        interfaces["kernel"] = spec, next(iter(wrong.ops))
    with pytest.raises(ValueError):
        checked_model_shapes(parent, bound, bindings, interfaces)


def test_symbolic_staging_reads_compiler_members_only(tmp_path, monkeypatch):
    import json
    from io import StringIO

    from xdsl.printer import Printer

    def serialize(module):
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        return stream.getvalue().encode()

    source, target = tmp_path / "capture", tmp_path / "compiler"
    receipt = capture_fixture(source)
    parent, bound, bindings, interfaces = example()
    parent_raw, bound_raw = serialize(parent), serialize(bound)
    spec, function = interfaces["kernel"]
    kernel_raw = serialize(function.parent_op())
    digest = record(kernel_raw)["sha256"]
    ttir = (source / "kernels/0/source.ttir").read_bytes()
    provenance = {
        "schema": "t2m.shared-tensor-provenance.v1",
        "spec": spec,
        "mlir_sha256": digest,
        "source_sha256": record(ttir)["sha256"],
    }
    bindings.update(
        schema="t2m.model-kernel-binding.v1",
        mlir_sha256=record(bound_raw)["sha256"],
        input_model_sha256=record(parent_raw)["sha256"],
        kernels={digest: provenance},
    )
    for call in bindings["calls"]:
        call.update(implementation_sha256=digest, source_sha256=provenance["source_sha256"])
    parent_trace = encoded({"status": "complete", "mlir": record(parent_raw)})
    parent_receipt = json.loads((source / "parent/capture_receipt.json").read_text())
    parent_receipt["artifacts"].update({"model.mlir": record(parent_raw), "frontend-trace.json": record(parent_trace)})
    files = {
        "model.mlir": bound_raw,
        "parent/model.mlir": parent_raw,
        "parent/frontend-trace.json": parent_trace,
        "parent/capture_receipt.json": encoded(parent_receipt),
        "kernel-bindings.json": encoded(bindings),
        "kernels/0/model.mlir": kernel_raw,
        "kernels/0/spec.json": encoded(spec),
        "kernels/0/provenance.json": encoded(provenance),
    }
    for name, raw in files.items():
        (source / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt.update(parent_receipt=record(encoded(parent_receipt)), kernel_artifacts={digest: "kernels/0"})
    (source / "capture_receipt.json").write_bytes(encoded(receipt))
    read = Path.read_bytes

    def compiler_only(path):
        if path.name in ("inputs.npz", "extra.npz", "golden.npy", "goldens.npz", "tensor_words.npz", "execution.json"):
            raise AssertionError("symbolic compiler staging accessed evaluator data")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", compiler_only)
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature["schema"] == "merlin.compile-signature.v3"
    assert signature["dimension_bindings"] == bindings["runtime_shape_contract"]["dimension_bindings"]
    assert signature["outputs"] == bindings["runtime_shape_contract"]["outputs"]
    assert "EVALUATOR" not in b"".join(p.read_bytes() for p in target.rglob("*") if p.is_file()).decode()
