"""Runtime shape custody comes from parent IR, never producer/evaluator claims."""

import pytest
from xdsl.dialects.func import CallOp

from merlin.frontends.derived_runtime_arguments import checked_runtime_call_shapes
from merlin.frontends.linalg_mlir import parse_mlir_text


def example(m=3, n=2, *, rows=None, columns=None, dtype="f32", result_n=None):
    rows = f"{m}:i64" if rows is None else rows
    columns = f"{n}:i64" if columns is None else columns
    tensor = f"tensor<{m}x{n}x{dtype}>"
    output = f"tensor<{m}x{n if result_n is None else result_n}x{dtype}>"
    module = parse_mlir_text(f'''module {{
      func.func @forward(%x:{tensor})->{output} {{
        %r=func.call @kernel(%x) {{prov.kernel="test::runtime",prov.kernel_operands=["input"],
          prov.kernel_args={{rows={rows},columns={columns}}}}}:({tensor})->{output}
        return %r:{output}
      }}
      func.func private @kernel({tensor})->{output} attributes {{prov.kernel="test::runtime"}}
    }}''')
    module.verify()
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    attributes = call.attributes["prov.kernel_args"].data
    spec = {"schema": "t2m.shared-tensor-abi.v5", "entry": "source", "buffers": [
        {"name": "x", "dtype": "f32", "shape": ["m", "n"], "effect": "read"},
        {"name": "out", "dtype": "f32", "shape": ["m", "n"], "effect": "write"}],
        "grid": [{"dimension": "m", "divisor": 4}, {"dimension": "n", "divisor": 8}],
        "runtime_scalars": [{"name": "m", "dtype": "i32", "source_arg": 2, "bounds": [0, 8]},
                            {"name": "n", "dtype": "i32", "source_arg": 3, "bounds": [0, 16]}]}
    record = {"parent_call_ordinal": 0, "kernel": "test::runtime", "callee": "kernel",
        "schema_operands": ["input"], "source_operand_bindings": {"x": "input"},
        "source_result_order": ["out"],
        "model_call_attributes": {key: str(value) for key, value in call.attributes.items()},
        "runtime_scalar_bindings": {"m": {"schema_argument": "rows", "attribute": str(attributes["rows"])},
                                    "n": {"schema_argument": "columns", "attribute": str(attributes["columns"])}}}
    return module, spec, record


@pytest.mark.parametrize("m,n", [(0, 2), (3, 2), (8, 16)])
def test_runtime_extents_reopen_parent_call_and_allow_empty_values(m, n):
    module, spec, record = example(m, n)
    assert checked_runtime_call_shapes(module, spec, record) == {"x": [m, n], "out": [m, n]}


@pytest.mark.parametrize("change", ["ordinal", "identity", "callee", "attributes", "scalar-value", "missing-scalar",
    "schema-scalar", "operand", "result", "bounds", "range", "position", "dtype", "shape", "boolean", "negative-bounds"])
def test_runtime_claims_cannot_replace_actual_parent_types_and_attributes(change):
    module, spec, record = example()
    if change == "ordinal":
        record["parent_call_ordinal"] = 1
    elif change == "identity":
        record["kernel"] = "other::kernel"
    elif change == "callee":
        record["callee"] = "other"
    elif change == "attributes":
        record["model_call_attributes"]["prov.kernel_args"] = "producer-only shape claim"
    elif change == "scalar-value":
        record["runtime_scalar_bindings"]["m"]["attribute"] = "4 : i64"
    elif change == "missing-scalar":
        record["runtime_scalar_bindings"].pop("n")
    elif change == "schema-scalar":
        record["runtime_scalar_bindings"]["m"]["schema_argument"] = "not-present"
    elif change == "operand":
        record["source_operand_bindings"]["x"] = "other"
    elif change == "result":
        record["source_result_order"] = ["other"]
    elif change == "bounds":
        spec["runtime_scalars"][0]["bounds"] = [0, 2]
    elif change == "range":
        spec["runtime_scalars"][0]["bounds"] = [0, 2**32]
    elif change == "position":
        spec["runtime_scalars"][0]["source_arg"] = 3
    elif change == "dtype":
        module, spec, record = example(dtype="f64")
    elif change == "shape":
        module, spec, record = example(result_n=3)
    elif change == "boolean":
        module, spec, record = example(rows="true")
    else:
        spec["runtime_scalars"][0]["bounds"] = [-1, 8]
    before = str(module)
    with pytest.raises(ValueError):
        checked_runtime_call_shapes(module, spec, record)
    assert str(module) == before
