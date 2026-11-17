"""Live runtime custody follows original scalar edges, not producer values."""

from copy import deepcopy

import pytest
from xdsl.dialects.func import CallOp

from merlin.frontends.derived_runtime_arguments import checked_runtime_call_shapes
from merlin.frontends.linalg_mlir import parse_mlir_text


def example(dtype="i32", shape="", *, shape_value=False):
    control = "index" if shape_value else f"tensor<{shape}{dtype}>"
    payload = f'tensor<7x{"i32" if dtype == "i1" else dtype}>'
    argument = "" if shape_value else f",%control:{control}"
    definitions = '%zero=arith.constant 0:index\n %control=tensor.dim %x,%zero:' + payload if shape_value else ""
    module = parse_mlir_text(f'''module {{
      func.func @forward(%x:{payload}{argument})->{payload} {{
        {definitions}
        %out=func.call @kernel(%x,%control) {{prov.kernel="test::live",prov.kernel_operands=["input","control"]}}:
          ({payload},{control})->{payload}
        return %out:{payload}
      }}
      func.func private @kernel({payload},{control})->{payload} attributes {{prov.kernel="test::live"}}
    }}''')
    module.verify()
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    spec = {"schema": "t2m.shared-tensor-abi.v3", "entry": "source", "buffers": [
        {"name": "x", "shape": [7], "dtype": "i32" if dtype == "i1" else dtype, "effect": "read"},
        {"name": "out", "shape": [7], "dtype": "i32" if dtype == "i1" else dtype, "effect": "write"}],
        "runtime_scalars": [{"name": "control", "dtype": dtype, "source_arg": 1}]}
    binding = {"schema_operand": "control", "caller_type": control, "kernel_type": dtype,
               "adapter": "proved-index-cast" if shape_value else "singleton-tensor-extract"}
    if shape_value:
        binding["constant_integer"] = 7
    record = {"parent_call_ordinal": 0, "kernel": "test::live", "callee": "kernel",
        "schema_operands": ["input", "control"], "source_operand_bindings": {"x": "input"},
        "source_result_order": ["out"], "runtime_scalar_bindings": {},
        "runtime_operand_bindings": {"control": binding},
        "model_call_attributes": {key: str(value) for key, value in call.attributes.items()}}
    return module, spec, record


@pytest.mark.parametrize("dtype", ["i1", "i8", "i16", "i32", "i64", "f32", "f64"])
@pytest.mark.parametrize("shape", ["", "1x", "1x1x"])
def test_exact_singleton_runtime_values_reopen_all_scalar_types(dtype, shape):
    module, spec, record = example(dtype, shape)
    assert checked_runtime_call_shapes(module, spec, record, live_operands=True) == {"x": [7], "out": [7]}


def test_live_shape_fact_comes_from_actual_tensor_dim_not_a_source_value_archive():
    module, spec, record = example(shape_value=True)
    for buffer in spec["buffers"]:
        buffer["shape"] = ["control"]
    spec["schema"] = "t2m.shared-tensor-abi.v5"
    spec["runtime_scalars"][0]["bounds"] = [0, 15]
    assert checked_runtime_call_shapes(module, spec, record, live_operands=True) == {"x": [7], "out": [7]}
    broken = deepcopy(record)
    broken["runtime_operand_bindings"]["control"]["constant_integer"] = 8
    with pytest.raises(ValueError):
        checked_runtime_call_shapes(module, spec, broken, live_operands=True)


@pytest.mark.parametrize("damage", ["profile", "missing", "fake-fact", "dtype", "adapter", "unknown-field",
                                    "operand", "both", "dimension"])
def test_live_runtime_claims_cannot_override_source_scalar_shape_or_profile(damage):
    module, spec, record = example()
    live = record["runtime_operand_bindings"]["control"]
    enabled = True
    if damage == "profile":
        enabled = False
    elif damage == "missing":
        record["runtime_operand_bindings"].clear()
    elif damage == "fake-fact":
        live["constant_integer"] = 7
    elif damage == "dtype":
        live["kernel_type"] = "i64"
    elif damage == "adapter":
        live["adapter"] = "exact-scalar"
    elif damage == "unknown-field":
        live["bounds_permission"] = True
    elif damage == "operand":
        live["schema_operand"] = "absent"
    elif damage == "both":
        record["runtime_scalar_bindings"]["control"] = {"schema_argument": "control", "attribute": "7:i64"}
    else:
        spec["schema"] = "t2m.shared-tensor-abi.v5"
        spec["runtime_scalars"][0]["bounds"] = [0, 15]
        for buffer in spec["buffers"]:
            buffer["shape"] = ["control"]
    before = str(module)
    with pytest.raises(ValueError):
        checked_runtime_call_shapes(module, spec, record, live_operands=enabled)
    assert str(module) == before
