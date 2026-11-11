"""Independent dense-view declaration checks, with no numerical admission."""

import json
import math
from pathlib import Path

import pytest
from test_derived_compile_inputs import encoded, record, runtime_guarded_fixture
from xdsl.dialects.func import CallOp

from merlin.frontends.compile_inputs import stage_compile_inputs
from merlin.frontends.linalg_mlir import parse_mlir_text


def fixture(root, shape=(1, 2), *, static=False, change=None):
    count = math.prod(shape)
    runtime_guarded_fixture(root, sizes=(count,))
    receipt = json.loads((root / "capture_receipt.json").read_text())
    bindings = json.loads((root / "kernel-bindings.json").read_text())
    spec = json.loads((root / "kernels/0/spec.json").read_text())
    kernel = (root / "kernels/0/model.mlir").read_bytes()
    if static:
        spec.update(schema="t2m.shared-tensor-abi.v2")
        spec.pop("runtime_scalars")
        for buffer in spec["buffers"]:
            buffer["shape"] = [count]
        kernel = (
            f"""module {{func.func @copy(%x:tensor<{count}xf32>)->tensor<{count}xf32>
          attributes {{prov.memory_guard_bytes_per_result_element=16:i64}} {{return %x:tensor<{count}xf32>}}}}"""
        ).encode()
        bindings.pop("argument_binding_profile")
    if change == "count":
        shape = (count + 1,)
    dtype = "f64" if change == "dtype" else "f32"
    type_ = "tensor<" + "".join(str(size) + "x" for size in shape) + dtype + ">"
    if change == "dynamic":
        type_ = "tensor<?xf32>"
    attributes = "" if static else f",prov.kernel_args={{extent={count}:i64}}"
    parent = (
        f"""module attributes {{prov.weights_file="weights.safetensors"}} {{
      func.func @forward(%x:{type_})->{type_} {{
        %r=func.call @kernel0(%x) {{prov.kernel="test::runtime",
          prov.kernel_operands=["input"]{attributes}}}:({type_})->{type_}
        return %r:{type_}
      }}
      func.func private @kernel0({type_})->{type_} attributes {{prov.kernel="test::runtime"}}
    }}"""
    ).encode()
    cost = 16 * count
    scratch = f"attributes {{prov.memory_guard_scratch_bytes={cost}:i64}}" if cost else ""
    model = (
        f"""module attributes {{prov.weights_file="weights.safetensors",
          prov.kernel_binding="t2m.model-kernel-binding.v2"}} {{
      func.func @forward(%x:{type_})->{type_} {scratch} {{
        return %x:{type_}
      }}
    }}"""
    ).encode()
    call = next(op for op in parse_mlir_text(parent.decode()).walk() if isinstance(op, CallOp))
    provenance = json.loads((root / "kernels/0/provenance.json").read_text())
    admission = json.loads((root / "kernels/0/memory-admission.json").read_text())
    admission["spec"] = spec
    digest = record(kernel)["sha256"]
    provenance.update(spec=spec, mlir_sha256=digest)
    provenance["memory"]["admission_sha256"] = record(encoded(admission))["sha256"]
    row = bindings["calls"][0]
    row.update(
        implementation_sha256=digest,
        tensor_views="dense-row-major-v1",
        model_call_attributes={key: str(value) for key, value in call.attributes.items()},
        argument_adaptations=[
            {
                "kind": kind,
                "name": name,
                "caller_type": type_,
                "concrete_kernel_type": f"tensor<{count}xf32>",
                "policy": "dense-row-major-v1",
                **({"argument": 0} if name == "x" else {}),
            }
            for kind, name in (("dense-tensor-input-view", "x"), ("dense-tensor-result-view", "out"))
        ],
    )
    if shape == (count,) and change is None:
        row["argument_adaptations"] = []
    if static:
        row["runtime_scalar_bindings"] = {}
    bindings.update(
        schema="t2m.model-kernel-binding.v2",
        tensor_view_binding_profile="t2m.captured-dense-tensor-views.v1",
        mlir_sha256=record(model)["sha256"],
        input_model_sha256=record(parent)["sha256"],
        kernels={digest: provenance},
    )
    if change == "schema":
        bindings["schema"] = "t2m.model-kernel-binding.v1"
    elif change == "profile":
        bindings.pop("tensor_view_binding_profile")
    elif change == "policy":
        row["tensor_views"] = "unknown"
    elif change == "mapping":
        row.pop("tensor_views")
    elif change == "adaptation":
        row["argument_adaptations"][0]["caller_type"] = "tensor<2xf32>"
    elif change == "missing":
        row["argument_adaptations"].pop()
    parent_trace = encoded({"status": "complete", "mlir": record(parent)})
    parent_receipt = json.loads((root / "parent/capture_receipt.json").read_text())
    parent_receipt["artifacts"].update({"model.mlir": record(parent), "frontend-trace.json": record(parent_trace)})
    parent_raw = encoded(parent_receipt)
    files = {
        "model.mlir": model,
        "parent/model.mlir": parent,
        "parent/frontend-trace.json": parent_trace,
        "parent/capture_receipt.json": parent_raw,
        "kernel-bindings.json": encoded(bindings),
        "kernels/0/model.mlir": kernel,
        "kernels/0/spec.json": encoded(spec),
        "kernels/0/provenance.json": encoded(provenance),
        "kernels/0/memory-admission.json": encoded(admission),
    }
    for name, raw in files.items():
        (root / name).write_bytes(raw)
        receipt["artifacts"][name] = record(raw)
    receipt.update(parent_receipt=record(parent_raw), kernel_artifacts={digest: "kernels/0"})
    (root / "capture_receipt.json").write_bytes(encoded(receipt))


@pytest.mark.parametrize(
    "shape,static",
    [((1, 2), False), ((2, 1), False), ((), False), ((0, 3), False), ((1, 2), True), ((2, 1), True), ((), True)],
)
def test_dense_view_shapes_and_scratch_come_from_typed_parent_without_evaluator(tmp_path, monkeypatch, shape, static):
    source, target = tmp_path / "source", tmp_path / "compiler"
    fixture(source, shape, static=static)
    read = Path.read_bytes

    def checked(path):
        if path.name in ("inputs.npz", "golden.npy", "goldens.npz", "execution.json", "tensor_words.npz"):
            raise AssertionError("dense-view staging accessed evaluator members")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", checked)
    stage_compile_inputs(source, target)
    signature = json.loads((target / "signature.json").read_text())
    assert signature.get("guard_storage_bytes", 0) == 16 * math.prod(shape)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize(
    "change", ["schema", "profile", "policy", "mapping", "count", "dtype", "dynamic", "adaptation", "missing"]
)
def test_rehashed_dense_view_claims_cannot_replace_original_parent_types(tmp_path, static, change):
    source, target = tmp_path / "source", tmp_path / "compiler"
    fixture(source, static=static, change=change)
    with pytest.raises(ValueError):
        stage_compile_inputs(source, target)
    assert not target.exists()
