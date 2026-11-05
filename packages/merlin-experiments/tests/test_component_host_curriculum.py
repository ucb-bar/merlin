"""Closed host arithmetic sources retain exact outputs and allocation bounds.

These tests exercise construction and independent numerical references only.
They issue no host execution, physical lifetime or performance qualification.
"""

from copy import deepcopy

import pytest
from merlin_experiments.phase0 import component_execution_budget as budget
from merlin_experiments.phase0 import component_integer_bounds as bounds
from merlin_experiments.phase0 import component_numerics

from merlin.targetgen import component_program, corpus_spec, input_palette
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir


def program(axis=1):
    return {
        "schema": component_program.SCHEMA_V2,
        "inputs": [
            {"name": "X", "role": "input", "shape": [2, 3], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [3, 2], "dtype": "operand"},
        ],
        "nodes": [
            {"name": "moved", "op": "copy", "inputs": ["X"]},
            {"name": "wide", "op": "cast", "inputs": ["moved"], "dtype": "accumulator"},
            {"name": "sum", "op": "reduce_sum", "inputs": ["wide"], "axis": axis, "dtype": "accumulator"},
            {"name": "product", "op": "matmul", "inputs": ["moved", "W"]},
            {"name": "narrow", "op": "cast", "inputs": ["product"], "dtype": "operand"},
        ],
        "outputs": [{"name": name, "value": name} for name in ("wide", "sum", "product", "narrow")],
    }


def capsule(original, *, overflow="modular_wrap", values=None):
    binding = corpus_spec.CorpusBinding(
        target="independent",
        tile_dim=1,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=[],
        compare="exact_int",
    )
    entry = {
        "name": "closed_host",
        "kind": "model_slice",
        "source_role": "derived_sweep",
        "source_reference": "independent host semantics",
        "program": original,
    }
    cap, source = component_program.build(entry, binding)
    cap["numerical_semantics"] = {"model": {"engine": "integer_reference"}, "overflow": overflow}
    if values is not None:
        cap["input_palette"] = {
            "schema": input_palette.SCHEMA,
            "inputs": [{"name": name, "axis": "linear", "values": data, "offset": 0} for name, data in values.items()],
        }
    cap["integer_partial_sum_bound"] = bounds.derive(budget.source_for_capsule(cap), cap["numerical_semantics"])
    return cap, source


@pytest.mark.parametrize("axis,expected", [(0, [[125, 3, -1]]), (1, [[128], [-1]])])
def test_real_reduction_cast_and_closed_mixed_graph_complete_reference(axis, expected):
    cap, source = capsule(program(axis), values={"X": [127, -2, 3, -2, 5, -4], "W": [1, 2, 3, 4, 5, 6]})
    observed = component_numerics.evaluate(cap)
    assert observed == {
        "wide": [[127, -2, 3], [-2, 5, -4]],
        "sum": expected,
        "product": [[136, 264], [-7, -8]],
        "narrow": [[-120, 8], [-7, -8]],
    }
    parsed = parse_linalg_mlir(source)
    assert parsed["results"] == [
        {"shape": [2, 3], "dtype": "i32"},
        {"shape": [1, 3] if axis == 0 else [2, 1], "dtype": "i32"},
        {"shape": [2, 2], "dtype": "i32"},
        {"shape": [2, 2], "dtype": "i8"},
    ]
    assert source.count("arith.muli") == 1
    assert 'prov.op = "reduce_sum"' in source and "arith.trunci" in source and "arith.extsi" in source
    assert {"shared_producer", "multiple_consumers", "escaped_use"} <= set(cap["component_program"]["effects"])


def test_reduction_budget_charges_full_input_even_for_single_output():
    original = program()
    original["inputs"] = original["inputs"][:1]
    original["inputs"][0]["shape"] = [1, 10**9]
    original["nodes"] = [{"name": "sum", "op": "reduce_sum", "inputs": ["X"], "axis": 1, "dtype": "accumulator"}]
    original["outputs"] = [{"name": "Y", "value": "sum"}]
    typed = component_program.analyze(original, operand_dtype="i8", accumulator_dtype="i32")
    cost = budget.measure({"kind": "component_program", "program": typed, "input_palette": None})
    assert cost["reference_work"] > 3 * 10**9
    assert cost["tensor_payload_bytes"] >= 10**9
    assert typed["nodes"][0]["shape"] == [1, 1]


def test_bounded_exact_sum_checks_partial_overflow_before_reference_allocation():
    original = program()
    original["inputs"] = [{"name": "X", "role": "input", "shape": [1, 4], "dtype": "i32"}]
    original["nodes"] = [{"name": "sum", "op": "reduce_sum", "inputs": ["X"], "axis": 1, "dtype": "i32"}]
    original["outputs"] = [{"name": "Y", "value": "sum"}]
    maximum = (1 << 31) - 1
    with pytest.raises(ValueError, match="partial sum may overflow"):
        capsule(original, overflow="bounded_exact", values={"X": [maximum, maximum, -maximum, -maximum]})


@pytest.mark.parametrize("mutation", ["legacy", "axis", "dtype", "future", "extra"])
def test_unknown_source_forms_and_versions_refuse(mutation):
    original = deepcopy(program())
    if mutation == "legacy":
        original.pop("schema")
    elif mutation == "axis":
        original["nodes"][2]["axis"] = True
    elif mutation == "dtype":
        original["nodes"][1]["dtype"] = "f32"
    elif mutation == "future":
        original["schema"] = "merlin.component_program.v999"
    else:
        original["nodes"][2]["schedule"] = "preferred"
    with pytest.raises(ValueError):
        component_program.analyze(original, operand_dtype="i8", accumulator_dtype="i32")


def test_changed_source_cannot_reuse_the_original_typed_reference():
    cap, _ = capsule(program())
    cap["operation"]["attributes"]["program"]["nodes"][2]["axis"] = 0
    with pytest.raises(ValueError, match="differs"):
        component_numerics.evaluate(cap)
