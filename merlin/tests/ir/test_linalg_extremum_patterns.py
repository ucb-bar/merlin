"""Neutral two-result integer-minimum source contracts, not generated kernels."""

from hashlib import sha256
from itertools import permutations

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_extremum_patterns import (
    recognize_static_i64_argmin,
    screen_static_i64_argmin_source,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def _program():
    return """builtin.module {
  func.func @pair(%x: tensor<2x5xi64>) -> (tensor<2xi64>, tensor<2xi64>) {
    %seed = arith.constant 9223372036854775807 : i64
    %vinit = tensor.splat %seed : tensor<2xi64>
    %iinit = tensor.splat %seed : tensor<2xi64>
    %value, %index = "linalg.generic"(%x, %vinit, %iinit) <{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      operandSegmentSizes = array<i32: 1, 2>}> ({
      ^bb0(%input: i64, %old: i64, %old_index: i64):
        %coordinate = "linalg.index"() <{dim = 1 : i64}> : () -> index
        %position = arith.index_cast %coordinate : index to i64
        %better = arith.cmpi slt, %input, %old : i64
        %same = arith.cmpi eq, %input, %old : i64
        %earlier = arith.cmpi ult, %position, %old_index : i64
        %tie = arith.andi %same, %earlier : i1
        %choose = arith.ori %better, %tie : i1
        %new_value = arith.select %choose, %input, %old : i64
        %new_index = arith.select %choose, %position, %old_index : i64
        "linalg.yield"(%new_value, %new_index) : (i64, i64) -> ()
    }) : (tensor<2x5xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>)
    func.return %value, %index : tensor<2xi64>, tensor<2xi64>
  }
}"""


def _generic(text):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


@pytest.mark.parametrize("swapped", (False, True))
def test_semantic_translation_preserves_both_minimum_results_and_return_order(swapped):
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
    from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

    text = _program()
    if swapped:
        text = text.replace("func.return %value, %index", "func.return %index, %value")
    translated = translate_linalg_text(text, entry="pair", target_identity="synthetic")
    graph = translated.request
    limit = (1 << 63) - 1
    # Ties at the largest representable value must still select the first index.
    values = (limit,) * 5 + (3, -7, -7, 2, 0)
    inputs = {"arg0": TensorValue(graph.node("arg0").type, values)}
    outputs = evaluate_graph(graph, inputs, constants=translated.constants)
    expected = [(limit, -7), (0, 1)]
    assert [result.elements for result in outputs] == (expected[::-1] if swapped else expected)
    assert [n.op for n in graph.nodes[-2:]] == ["i64_min_value", "i64_first_min_index"]
    assert graph.nodes[-1].inputs == graph.nodes[-2].inputs == ("arg0",)
    assert type(graph).from_record(graph.record()).digest() == graph.digest()


@pytest.mark.parametrize(
    "old,new",
    (
        ("9223372036854775807", "0"),
        ("arith.cmpi slt", "arith.cmpi ult"),
        ("arith.cmpi ult", "arith.cmpi ule"),
        ('"linalg.yield"(%new_value, %new_index)', '"linalg.yield"(%new_index, %new_value)'),
    ),
)
def test_semantic_translation_rejects_changed_pair_seeds_signedness_and_projection_order(old, new):
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

    with pytest.raises(ValueError):
        translate_linalg_text(_program().replace(old, new), entry="pair", target_identity="synthetic")


def test_paired_minimum_host_screen_binds_both_results_and_source_context():
    from xdsl.dialects.builtin import StringAttr

    from merlin.targetgen.application_inventory import operation_structure
    from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

    op = _generic(_program())
    op.attributes["prov.aten"] = StringAttr("aten.min.dim")
    structure = operation_structure(op)
    row = {"mlir_operation": "linalg.generic", "frontend_op": "aten.min.dim", "count": 1, **structure}
    observed = {
        "family": "reduction",
        "ordered_operand_dtypes": ["i64", "i64", "i64"],
        "ordered_result_dtypes": ["i64", "i64"],
        "rank": 1,
    }
    body = {"schema": "merlin.static_integer_reduction_source_body.v1", "operation": "i64_min_first_index"}
    declaration = {
        "id": "neutral_minimum",
        "ops": ["aten.min.dim"],
        "families": ["reduction"],
        "placement": "host",
        "signature": {
            "family": "reduction",
            "ordered_operand_dtypes": observed["ordered_operand_dtypes"],
            "ordered_result_dtypes": observed["ordered_result_dtypes"],
        },
        "source_body": body,
    }
    selected = {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral structural test, not numerical qualification"},
            },
        }
    }
    context = {
        "selected_index_observation": {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "neutral-clang",
            "compiler_resolved": "/neutral/clang",
            "compiler_sha256": "c" * 64,
            "cross_flags": ["--target=neutral"],
            "data_layout": "e-p:64:64",
            "index_bits": 64,
            "scope": "neutral selected-width premise only",
        }
    }
    validate_host_capabilities(selected["neutral"]["capability_spec"])
    assert admit_host_operation(selected, row, observed, source_operations=(op,))["status"] == "unknown"
    result = admit_host_operation(selected, row, observed, source_operations=(op,), source_context=context)
    assert result["status"] == "admitted"
    assert result["source_body_proof"]["patterns"][0]["ordered_types"] == ["i64"] * 5
    assert (
        admit_host_operation(
            selected,
            row,
            {**observed, "ordered_result_dtypes": ["i64"]},
            source_operations=(op,),
            source_context=context,
        )["status"]
        == "unsupported"
    )
    assert (
        admit_host_operation(
            selected,
            {**row, "ordered_result_types": structure["ordered_result_types"][:1]},
            observed,
            source_operations=(op,),
            source_context=context,
        )["status"]
        == "unsupported"
    )
    wrong = _generic(_program().replace("cmpi ult", "cmpi slt"))
    wrong.attributes["prov.aten"] = StringAttr("aten.min.dim")
    assert (
        admit_host_operation(selected, row, observed, source_operations=(wrong,), source_context=context)["status"]
        == "unsupported"
    )
    wrong_pair = _generic(
        _program().replace('linalg.yield"(%new_value, %new_index)', 'linalg.yield"(%new_index, %new_value)')
    )
    wrong_pair.attributes["prov.aten"] = StringAttr("aten.min.dim")
    assert (
        admit_host_operation(selected, row, observed, source_operations=(wrong_pair,), source_context=context)["status"]
        == "unsupported"
    )


def test_minimum_first_index_source_and_exact_module_binding(tmp_path):
    text = _program()
    module = mq.parse(text)
    module.verify()
    generic = next(mq.walk(module, "linalg.generic"))
    pattern = recognize_static_i64_argmin(generic, index_bits=64)
    assert (pattern.operation, pattern.axis) == ("i64_min_first_index", 1)
    assert pattern.input_shape == (2, 5) and pattern.output_shape == (2,)
    assert pattern.index_bits == 64 and pattern.ordered_types == ("i64",) * 5
    path = tmp_path / "neutral.mlir"
    path.write_text(text)
    ordinal = tuple(mq.walk(module)).index(generic)
    witness = screen_static_i64_argmin_source(path, (ordinal,), index_bits=64)
    assert witness.raw_sha256 == sha256(text.encode()).hexdigest()
    assert witness.ordinals == ((ordinal, pattern),)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("9223372036854775807", "9223372036854775806"),
        ("cmpi slt", "cmpi sgt"),
        ("cmpi ult", "cmpi slt"),
        ("cmpi eq", "cmpi ne"),
        ("andi %same, %earlier", "andi %same, %same"),
        ("select %choose, %position, %old_index", "select %choose, %old_index, %position"),
        ('linalg.yield"(%new_value, %new_index)', 'linalg.yield"(%new_index, %new_value)'),
        ("dim = 1", "dim = 0"),
        ("tensor<2x5xi64>", "tensor<2x0xi64>"),
        ("#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"),
    ],
)
def test_minimum_rejects_wrong_seeds_signedness_ties_results_and_axis(old, new):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_i64_argmin(_generic(_program().replace(old, new)), index_bits=64)


def test_minimum_requires_representable_loop_extent_and_explicit_index_premise():
    generic = _generic(_program())
    for bits in (None, True, 1, 2, 3, 129):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_i64_argmin(generic, index_bits=bits)
    # A reduced tensor may be scalar; that must not turn into an unqualified ABI claim.
    scalar_text = _program().replace("2x5", "5").replace("2xi64", "i64")
    scalar_text = scalar_text.replace("(d0, d1) -> (d0, d1)", "(d0) -> (d0)")
    scalar_text = scalar_text.replace("(d0, d1) -> (d0)", "(d0) -> ()")
    scalar_text = scalar_text.replace("#linalg.iterator_type<parallel>, ", "").replace("dim = 1", "dim = 0")
    assert recognize_static_i64_argmin(_generic(scalar_text), index_bits=64).output_shape == ()


def test_integer_pair_minimum_ties_are_independent_of_visit_order():
    # Finite algebra regression; this is not a theorem about compiler output.
    limit = (1 << 63) - 1
    for values in ((limit, limit, limit), (-5, 7, -5), (-(1 << 63), 0, -(1 << 63))):
        for order in permutations(range(len(values))):
            best = (limit, limit)
            for index in order:
                candidate = (values[index], index)
                if candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                    best = candidate
            assert best == min((value, index) for index, value in enumerate(values))
