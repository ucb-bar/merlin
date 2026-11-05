"""Static tensor source views preserve original bytes and independently checked products."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from merlin_experiments.phase0 import component_integer_bounds as B
from merlin_experiments.phase0 import component_numerics as N
from merlin_experiments.phase0.component_execution_budget import source_for_capsule
from xdsl.dialects import func, tensor
from xdsl.dialects.linalg.ops import GenericOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen import component_program as P
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.input_palette import SCHEMA as PALETTE_SCHEMA


def capsule(program, *, palette=None, overflow="modular_wrap"):
    typed = P.analyze(program, operand_dtype="i8", accumulator_dtype="i32")
    cap = {"name": "static_views", "inputs": typed["inputs"], "component_program": typed,
           "operation": {"op": "component_program", "attributes": {"program": program}},
           "numerical_semantics": {"model": {"engine": "integer_reference"}, "overflow": overflow}}
    if palette is not None:
        cap["input_palette"] = {"schema": PALETTE_SCHEMA, "inputs": [
            {"name": name, "axis": "linear", "values": values, "offset": 0}
            for name, values in palette.items()]}
    cap["integer_partial_sum_bound"] = B.derive(source_for_capsule(cap), cap["numerical_semantics"])
    return cap


def views(axis=0):
    return {"schema": P.SCHEMA_V3, "inputs": [
        {"name": "X", "role": "input", "shape": [5, 7], "dtype": "i8"}], "nodes": [
        {"name": "A", "op": "extract_slice", "inputs": ["X"],
         "offsets": [0, 1], "sizes": [2, 3], "strides": [2, 2]},
        {"name": "B", "op": "extract_slice", "inputs": ["X"],
         "offsets": [1, 0], "sizes": [2, 3], "strides": [2, 2]},
        {"name": "Joined", "op": "concat", "inputs": ["A", "B"], "axis": axis}],
        "outputs": [{"name": "Y", "value": "Joined"}]}


@pytest.mark.parametrize("axis", [0, 1])
def test_nonunit_slices_and_both_concat_axes_use_original_rows(axis):
    program = views(axis)
    _, source = P.render(program, operand_dtype="i8", accumulator_dtype="i32")
    module = parse_mlir_text(source)
    module.verify()
    assert len([op for op in module.walk() if isinstance(op, tensor.ExtractSliceOp)]) == 2
    assert len([op for op in module.walk() if isinstance(op, tensor.ConcatOp)]) == 1
    cap = capsule(program)
    original = np.asarray(materialize_capsule_leaves(cap)["X"].data).reshape(5, 7)
    expected = np.concatenate((original[0:4:2, 1:7:2], original[1:5:2, 0:6:2]), axis=axis)
    np.testing.assert_array_equal(np.asarray(N.evaluate(cap)["Y"]), expected)


@pytest.mark.parametrize("field,value", [
    ("offsets", [-1, 0]), ("offsets", [True, 0]), ("offsets", [0]),
    ("sizes", [0, 1]), ("sizes", [3, 4]), ("strides", [1, 0]), ("strides", [1, 3]),
])
def test_slice_refuses_nonstatic_or_out_of_bounds_geometry(field, value):
    program = views()
    program["nodes"][0][field] = value
    with pytest.raises(ValueError, match="slice"):
        P.analyze(program, operand_dtype="i8", accumulator_dtype="i32")


@pytest.mark.parametrize("schema", [None, P.SCHEMA_V2])
def test_old_program_versions_do_not_acquire_new_operations(schema):
    program = views()
    if schema is None:
        del program["schema"]
    else:
        program["schema"] = schema
    with pytest.raises(ValueError, match="unsupported component node"):
        P.analyze(program, operand_dtype="i8", accumulator_dtype="i32")


@pytest.mark.parametrize("change", ["axis", "shape", "dtype", "arity"])
def test_concat_refuses_mixed_storage_or_axes(change):
    program = views()
    if change == "axis":
        program["nodes"][-1]["axis"] = True
    elif change == "shape":
        program["nodes"][1]["sizes"] = [2, 2]
    elif change == "dtype":
        program["nodes"].insert(2, {"name": "Wide", "op": "cast", "inputs": ["B"], "dtype": "i32"})
        program["nodes"][-1]["inputs"][1] = "Wide"
    else:
        program["nodes"][-1]["inputs"] = ["A"]
    with pytest.raises(ValueError, match="concatenation|arity"):
        P.analyze(program, operand_dtype="i8", accumulator_dtype="i32")


def test_concat_interval_includes_every_input_before_allocation():
    program = {"schema": P.SCHEMA_V3, "inputs": [
        {"name": n, "role": "input", "shape": [1, 2], "dtype": "i8"} for n in ("A", "B")],
        "nodes": [{"name": "C", "op": "concat", "inputs": ["A", "B"], "axis": 1}],
        "outputs": [{"name": "Y", "value": "C"}]}
    cap = capsule(program, palette={"A": [0], "B": [-127]}, overflow="bounded_exact")
    assert cap["integer_partial_sum_bound"]["nodes"][0]["interval"] == [-127, 0]
    assert N.evaluate(cap)["Y"] == [[0, 0, -127, -127]]
    changed = copy.deepcopy(cap)
    changed["operation"]["attributes"]["program"]["nodes"][0]["inputs"].reverse()
    with pytest.raises(ValueError, match="differs"):
        N.evaluate(changed)


def qk_program(groups=4, heads=8, queries=16, depth=64, context=16):
    """Original physical Q/query-major and transposed K storage, with all head products."""
    nodes, outputs = [], []
    for group in range(groups):
        key = f"Keys{group}"
        nodes.append({"name": key, "op": "extract_slice", "inputs": ["K"],
                      "offsets": [group * depth, 0], "sizes": [depth, context], "strides": [1, 1]})
        products = []
        for head in range(heads):
            query, product = f"Query{group}h{head}", f"Product{group}h{head}"
            nodes.append({"name": query, "op": "extract_slice", "inputs": ["Q"],
                          "offsets": [0, (group * heads + head) * depth],
                          "sizes": [queries, depth], "strides": [1, 1]})
            nodes.append({"name": product, "op": "matmul", "inputs": [query, key]})
            products.append(product)
        joined = f"Group{group}"
        nodes.append({"name": joined, "op": "concat", "inputs": products, "axis": 0})
        outputs.append({"name": f"Y{group}", "value": joined})
    return {"schema": P.SCHEMA_V3, "logical_io": P.LOGICAL_IO_V1,
            "inputs": [{"name": "Q", "role": "input", "shape": [queries, groups * heads * depth], "dtype": "i8"},
                       {"name": "K", "role": "weight", "shape": [groups * depth, context], "dtype": "i8"}],
            "nodes": nodes, "outputs": outputs}


@pytest.mark.parametrize("queries,context", [(16, 16), (32, 16), (17, 19)])
def test_original_strided_qk_has_six_pointer_names_and_all_independent_products(queries, context):
    program = qk_program(queries=queries, context=context)
    typed, source = P.render(program, operand_dtype="i8", accumulator_dtype="i32")
    module = parse_mlir_text(source)
    module.verify()
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    assert len(function.function_type.inputs) == 2 and len(function.function_type.outputs) == 4
    assert [x.data for x in function.attributes["merlin.logical_inputs"]] == ["Q", "K"]
    assert [x.data for x in function.attributes["merlin.logical_outputs"]] == [f"Y{g}" for g in range(4)]
    contractions = [op for op in module.walk() if isinstance(op, GenericOp)]
    assert len(contractions) == 32  # No pre-batched answer kernel in the source.
    assert [x["shape"] for x in typed["outputs"]] == [[8 * queries, context]] * 4
    cap = capsule(program)
    env = materialize_capsule_leaves(cap)
    q_flat = np.asarray(env["Q"].data, dtype=np.int8).reshape(queries, 2048)
    keys_flat = np.asarray(env["K"].data, dtype=np.int8).reshape(256, context)
    q_original = q_flat.reshape(queries, 4, 8, 64)
    keys_original = keys_flat.reshape(4, 64, context)
    assert q_original.tobytes() == q_flat.tobytes()
    assert keys_original.tobytes() == keys_flat.tobytes()
    observed = N.evaluate(cap)
    assert set(observed) == {f"Y{g}" for g in range(4)}
    for group in range(4):
        expected = np.stack([q_original[:, group, head, :].astype(np.int64)
                             @ keys_original[group].astype(np.int64) for head in range(8)])
        expected = expected.astype(np.int32).reshape(8 * queries, context)
        np.testing.assert_array_equal(np.asarray(observed[f"Y{group}"]), expected)
