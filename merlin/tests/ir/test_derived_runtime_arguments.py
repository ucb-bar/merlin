"""Runtime shape custody comes from parent IR, never producer/evaluator claims."""

import pytest
from xdsl.dialects.func import CallOp

from merlin.frontends.derived_runtime_arguments import checked_runtime_call_shapes
from merlin.frontends.linalg_mlir import parse_mlir_text


@pytest.mark.parametrize("damage", [None, "profile", "selection", "missing", "name", "attribute", "type",
    "identity", "ordinal", "call-attributes", "boolean", "unknown", "extra", "declaration"])
def test_source_grouping_selection_is_bound_to_original_typed_call(damage):
    from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, StringAttr, i32, i64
    from merlin.frontends.derived_runtime_arguments import checked_source_layout_call
    module, _, record = example()
    call = next(op for op in module.walk() if isinstance(op, CallOp))
    selection = {"target": "source:architecture", "num_warps": 4, "threads_per_warp": 32, "num_ctas": 1}
    values = {"source_" + key: StringAttr(value) if isinstance(value, str) else IntegerAttr(value, i64)
              for key, value in selection.items()}
    call.attributes["prov.kernel_args"] = DictionaryAttr(values)
    record["model_call_attributes"] = {k: str(v) for k, v in call.attributes.items()}
    record["source_layout_bindings"] = {key: {"schema_argument": "source_" + key, "attribute": str(values["source_" + key])}
                                        for key in selection}
    provenance = {"source_layout_specialization": {"schema": "t2m.source-map-layout.v1", "source_options": selection}}
    profile = "t2m.captured-source-layout.v1"
    if damage == "profile": profile = None
    elif damage == "selection": selection["num_warps"] = 8
    elif damage == "missing": record["source_layout_bindings"].pop("target")
    elif damage == "name": record["source_layout_bindings"]["target"]["schema_argument"] = "absent"
    elif damage == "attribute": record["source_layout_bindings"]["target"]["attribute"] = "another source"
    elif damage == "type":
        call.attributes["prov.kernel_args"] = DictionaryAttr({**values, "source_num_warps": IntegerAttr(4, i32)})
        record["model_call_attributes"] = {k: str(v) for k, v in call.attributes.items()}
    elif damage == "identity": record["kernel"] = "another::kernel"
    elif damage == "ordinal": record["parent_call_ordinal"] = 1
    elif damage == "call-attributes": record["model_call_attributes"] = {}
    elif damage == "boolean": selection["num_warps"] = True
    elif damage == "unknown": provenance["source_layout_specialization"]["schema"] = "unknown"
    elif damage == "extra": record["source_layout_bindings"]["target"]["grant"] = True
    elif damage == "declaration":
        declaration = list(module.ops)[1]
        declaration.attributes["prov.kernel"] = StringAttr("another::kernel")
    if damage is None:
        checked_source_layout_call(module, provenance, record, profile)
        checked_source_layout_call(module, {}, {}, None)
        with pytest.raises(ValueError, match="lack a source specialization"):
            checked_source_layout_call(module, {}, record, profile)
    else:
        with pytest.raises(ValueError, match="source layout disagrees"):
            checked_source_layout_call(module, provenance, record, profile)


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
