"""Exact constant specialization is checked against the independent evaluator."""

import copy
import itertools
import random
from dataclasses import replace

import pytest

from merlin.semantic_compiler.constant_folding import fold_source_constants, verify_constant_folding
from merlin.semantic_compiler.linalg_constants import read_tensor_bytes
from merlin.semantic_compiler.model import (
    ConstantBinding,
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reference import evaluate_graph


def request(operation, source_dtype, dtype, policy, arrays):
    shape = (len(arrays[0]),)
    source_type = TensorType(shape, source_dtype, "signed-i64" if source_dtype == "i64" else "raw-bit-pattern")
    output_type = TensorType(shape, dtype, "signed-i64" if dtype == "i64" else "raw-bit-pattern")
    operands = tuple(SemanticNode(f"c{i}", "constant", (), source_type, effect="constant") for i in range(len(arrays)))
    width = {"bf16": 2, "f32": 4, "f64": 8, "i64": 8}[source_dtype]
    encoding = "i64-le" if source_dtype == "i64" else source_dtype + "-le-raw"
    bindings = tuple(
        ConstantBinding(
            node.id, encoding, b"".join((n % (1 << (width * 8))).to_bytes(width, "little") for n in values).hex()
        )
        for node, values in zip(operands, arrays, strict=True)
    )
    node = SemanticNode(
        "result",
        operation,
        tuple(n.id for n in operands),
        output_type,
        (("numeric_semantics", policy),),
        index_maps=pointwise_identity_maps(1, len(operands)),
    )
    return KernelRequest((*operands, node), ("result",), ("external",), (), "synthetic", constants=bindings)


def evaluate(source, arguments=None):
    constants = {
        c.node_id: read_tensor_bytes(source.node(c.node_id).type, bytes.fromhex(c.data_hex))[1]
        for c in source.constants
    }
    return evaluate_graph(source, arguments or {}, constants=constants)


@pytest.mark.parametrize("dtype,width,fraction", [("bf16", 16, 7), ("f32", 32, 23), ("f64", 64, 52)])
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_ieee_arithmetic_edges_and_seeded_pairs_agree_with_independent_rationals(dtype, width, fraction, operation):
    sign = 1 << (width - 1)
    infinity = sign - (1 << fraction)
    one = (1023 if dtype == "f64" else 127) << fraction
    positive = [0, 1, (1 << fraction) - 1, 1 << fraction, one, one + 1, infinity - 1, infinity]
    edges = [value | s for s in (0, sign) for value in positive]
    rng = random.Random(width * 20 + len(operation))
    pairs = [
        *itertools.product(edges, repeat=2),
        *((rng.getrandbits(width), rng.getrandbits(width)) for _ in range(128)),
    ]
    folded_count = 0
    for a, b in pairs:
        source = request(dtype + "_" + operation, dtype, dtype, f"ieee754-{dtype}-{operation}-rne-v1", [[a], [b]])
        folded, receipt = fold_source_constants(source)
        verify_constant_folding(folded, receipt)
        try:
            expected = evaluate(source)
        except ValueError as error:
            assert "NaN" in str(error)
            assert folded.digest() == source.digest()
            assert receipt["deferred"]
        else:
            assert len(receipt["folds"]) == 1
            assert evaluate(folded) == expected
            folded_count += 1
    assert folded_count > 250


@pytest.mark.parametrize("eliminate_dead", [False, True])
def test_f32_folding_keeps_legacy_receipts_replayable(eliminate_dead):
    source = request("f32_mul", "f32", "f32", "ieee754-f32-mul-rne-v1", [[0x3F800001], [0x3F7FFFFF]])
    legacy, receipt = fold_source_constants(source, eliminate_dead=eliminate_dead, include_f32=False)
    assert not receipt["folds"] and legacy.digest() == source.digest()
    assert receipt["schema"] == f"merlin.semantic_constant_folding.v{2 if eliminate_dead else 1}"
    verify_constant_folding(legacy, receipt)
    folded, current = fold_source_constants(source, eliminate_dead=eliminate_dead)
    assert len(current["folds"]) == 1
    assert current["schema"] == f"merlin.semantic_constant_folding.v{4 if eliminate_dead else 3}"
    assert evaluate(folded) == evaluate(source)
    verify_constant_folding(folded, current)


@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_integer_constants_wrap_at_signed_endpoints(operation):
    source = request(
        "i64_" + operation,
        "i64",
        "i64",
        f"signed-i64-{operation}-wrap-v1",
        [[-(1 << 63), (1 << 63) - 1, -1, 0, 1], [-1, 1, (1 << 63) - 1, -(1 << 63), -1]],
    )
    folded, receipt = fold_source_constants(source)
    assert len(receipt["folds"]) == 1
    assert evaluate(folded) == evaluate(source)


@pytest.mark.parametrize("dtype,width", [("i64", 64), ("f32", 32), ("f64", 64)])
@pytest.mark.parametrize("comparison", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_constant_comparisons_preserve_ieee_ordering_and_signed_integer_endpoints(dtype, width, comparison):
    sign = 1 << (width - 1)
    if dtype == "i64":
        values = [0, 1, -1, -(1 << 63), (1 << 63) - 1]
        predicate = ("s" if comparison not in {"eq", "ne"} else "") + comparison
        operation = "arith.cmpi"
    else:
        fraction = 23 if width == 32 else 52
        infinity = sign - (1 << fraction)
        positive = [0, 1, (1 << fraction) - 1, 1 << fraction, infinity - 1, infinity, infinity + 1]
        values = [code | s for s in (0, sign) for code in positive]
        predicate, operation = "o" + comparison, "arith.cmpf"
    pairs = tuple(itertools.product(values, repeat=2))
    source = request(dtype + "_add", dtype, dtype, "unused", list(zip(*pairs, strict=True)))
    result = replace(
        source.nodes[-1], op="mask", type=TensorType(source.nodes[-1].type.shape, "i1", "boolean"),
        attrs=(("operation", operation), ("predicate", predicate)),
    )
    source = replace(source, nodes=(*source.nodes[:-1], result))
    folded, receipt = fold_source_constants(source)
    verify_constant_folding(folded, receipt)
    assert len(receipt["folds"]) == 1
    assert evaluate(folded) == evaluate(source)
    assert folded.constants[-1].encoding == "i1-raw"


@pytest.mark.parametrize("dtype,width", [("f32", 32), ("f64", 64)])
def test_constant_select_keeps_nan_payloads_and_signed_zero_bits(dtype, width):
    sign = 1 << (width - 1)
    fraction = 23 if width == 32 else 52
    infinity = sign - (1 << fraction)
    source = request(dtype + "_add", dtype, dtype, "unused", [[0, sign, infinity + 1], [sign, 0, infinity + 37]])
    condition = SemanticNode("condition", "constant", (), TensorType((3,), "i1", "boolean"), effect="constant")
    result = replace(source.nodes[-1], op="select", inputs=(condition.id, *source.nodes[-1].inputs), attrs=(),
                     index_maps=pointwise_identity_maps(1, 3))
    source = replace(source, nodes=(condition, *source.nodes[:-1], result),
                     constants=(ConstantBinding(condition.id, "i1-raw", "010001"), *source.constants))
    folded, receipt = fold_source_constants(source)
    verify_constant_folding(folded, receipt)
    assert len(receipt["folds"]) == 1
    assert evaluate(folded) == evaluate(source)
    assert bytes.fromhex(folded.constants[-1].data_hex) == b"".join(
        code.to_bytes(width // 8, "little") for code in (0, 0, infinity + 1)
    )
    wrong = replace(source, nodes=(*source.nodes[:-1], replace(result, attrs=(("unknown", True),))))
    refused, record = fold_source_constants(wrong)
    assert refused.digest() == wrong.digest() and not record["folds"] and record["deferred"]


@pytest.mark.parametrize("operation", ["arith.andi", "arith.xori", "arith.extui"])
def test_constant_boolean_operations_and_widening(operation):
    boolean = TensorType((4,), "i1", "boolean")
    inputs = tuple(SemanticNode(name, "constant", (), boolean, effect="constant") for name in ("a", "b"))
    arity = 1 if operation == "arith.extui" else 2
    inputs = inputs[:arity]
    result = SemanticNode(
        "result", "mask", tuple(n.id for n in inputs),
        TensorType((4,), "i64", "signed-i64") if arity == 1 else boolean,
        (("operation", operation), ("predicate", None)), index_maps=pointwise_identity_maps(1, arity),
    )
    source = KernelRequest(
        (*inputs, result), ("result",), ("external",), (), "synthetic",
        constants=tuple(ConstantBinding(n.id, "i1-raw", data)
                        for n, data in zip(inputs, ("00000101", "00010001"))),
    )
    folded, receipt = fold_source_constants(source)
    verify_constant_folding(folded, receipt)
    assert len(receipt["folds"]) == 1 and evaluate(folded) == evaluate(source)


@pytest.mark.parametrize(
    "operation,source_dtype,dtype,policy,values",
    [
        (
            "signed_i64_to_f32",
            "i64",
            "f32",
            "signed-i64-to-f32-rne-v1",
            [-(1 << 63), (1 << 63) - 1, 0, (1 << 24) + 1, -(1 << 53) - 1],
        ),
        (
            "signed_i64_to_f64",
            "i64",
            "f64",
            "signed-i64-to-f64-rne-v1",
            [-(1 << 63), (1 << 63) - 1, 0, (1 << 53) + 1, -(1 << 53) - 1],
        ),
        ("f32_to_f64", "f32", "f64", "f32-to-f64-exact-v1", [0, 0x80000000, 1, 0x7F800000, 0xFF7FFFFF]),
        ("f64_to_f32", "f64", "f32", "f64-to-f32-rne-v1", [0, 1 << 63, 1, 0x7FEFFFFFFFFFFFFF, 0x3FF0000010000000]),
        ("bf16_to_f32", "bf16", "f32", "bf16-to-f32-exact-v1", [0, 0x8000, 1, 0x7F80, 0xFF7F]),
        ("f32_to_bf16", "f32", "bf16", "f32-to-bf16-rne-v1", [0, 0x80000000, 1, 0x7F7FFFFF, 0x3F808000]),
    ],
)
def test_conversions_preserve_rne_signed_zero_subnormals_and_overflow(operation, source_dtype, dtype, policy, values):
    source = request(operation, source_dtype, dtype, policy, [values])
    folded, receipt = fold_source_constants(source)
    assert len(receipt["folds"]) == 1
    assert evaluate(folded) == evaluate(source)


def chain():
    source = request("signed_i64_to_f64", "i64", "f64", "signed-i64-to-f64-rne-v1", [[-2, -1, 0, 1, 2]])
    converted = replace(source.nodes[-1], id="converted")
    type_ = converted.type
    square = SemanticNode(
        "square",
        "f64_mul",
        ("converted", "converted"),
        type_,
        (("numeric_semantics", "ieee754-f64-mul-rne-v1"),),
        index_maps=pointwise_identity_maps(1, 2),
    )
    runtime = SemanticNode("runtime", "input", (), type_, effect="input")
    final = replace(
        square,
        id="dynamic",
        op="f64_add",
        inputs=("runtime", "square"),
        attrs=(("numeric_semantics", "ieee754-f64-add-rne-v1"),),
    )
    return replace(
        source,
        nodes=(*source.nodes[:-1], converted, square, runtime, final),
        outputs=("dynamic", "square", "converted", "square"),
        output_storages=("external",) * 4,
        input_storages=(("runtime", "external"),),
    )


def test_chains_share_values_preserve_outputs_and_do_not_fold_runtime_arguments():
    source = chain()
    folded, receipt = fold_source_constants(source)
    assert [row["original"]["id"] for row in receipt["folds"]] == ["converted", "square"]
    assert folded.node("dynamic").effect == "pure"
    assert folded.outputs == source.outputs and folded.input_storages == source.input_storages
    assert type(folded).from_record(folded.record()).digest() == folded.digest()
    verify_constant_folding(folded, receipt)
    from merlin.semantic_compiler.reference import TensorValue

    arguments = {"runtime": TensorValue(source.node("runtime").type, (0x3FF0000000000000,) * 5)}
    assert evaluate(folded, arguments) == evaluate(source, arguments)
    limited, record = fold_source_constants(source, element_budget=10)
    assert len(record["folds"]) == 1 and limited.node("square").effect == "pure"
    assert record["deferred"] == [{"node": "square", "reason": "element_budget"}]
    untouched, _ = fold_source_constants(source, element_budget=0)
    assert untouched.digest() == source.digest()


def test_replay_rejects_mutated_bytes_policies_substitutions_and_budgets():
    folded, receipt = fold_source_constants(chain())
    changes = []
    for key in ("before_digest", "after_digest"):
        item = copy.deepcopy(receipt)
        item[key] = "0" * 64
        changes.append(item)
    item = copy.deepcopy(receipt)
    item["schema"] = "unknown"
    changes.append(item)
    item = copy.deepcopy(receipt)
    item["element_budget"] = 0
    changes.append(item)
    item = copy.deepcopy(receipt)
    item["folds"][1]["original"]["inputs"] = ["converted", "c0"]
    changes.append(item)
    item = copy.deepcopy(receipt)
    item["folds"][1]["original"]["attrs"]["numeric_semantics"] = "reassociated"
    changes.append(item)
    for altered in changes:
        with pytest.raises(ValueError):
            verify_constant_folding(folded, altered)
    binding = folded.constants[-1]
    changed = replace(binding, data_hex="00" * (len(binding.data_hex) // 2))
    with pytest.raises(ValueError):
        verify_constant_folding(replace(folded, constants=(*folded.constants[:-1], changed)), receipt)


def test_unknown_policies_maps_intrinsics_and_runtime_values_remain_unfolded():
    source = chain()
    node = source.node("converted")
    for altered in [
        replace(node, attrs=(("numeric_semantics", "different-rounding"),)),
        replace(node, index_maps=(IndexMap(1, ((0,),), (0,)),) * 2),
        replace(node, op="source_f64_pow_unqualified"),
    ]:
        changed = replace(source, nodes=(source.nodes[0], altered, *source.nodes[2:]))
        folded, receipt = fold_source_constants(changed)
        assert folded.digest() == changed.digest() and not receipt["folds"]


def test_dead_pure_chain_is_removed_with_replay_and_unchanged_boundary_outputs():
    source = chain()
    first = replace(source.node("dynamic"), id="unused_first", inputs=("runtime", "runtime"))
    second = replace(first, id="unused_second", inputs=(first.id, "runtime"))
    boundary = replace(source.node("runtime"), id="unused_boundary")
    source = replace(
        source,
        nodes=(*source.nodes, first, second, boundary),
        input_storages=(*source.input_storages, (boundary.id, "external")),
    )
    folded, receipt = fold_source_constants(source, eliminate_dead=True)
    assert receipt["schema"] == "merlin.semantic_constant_folding.v4"
    assert [row["node"]["id"] for row in receipt["dead_computations"]] == [first.id, second.id]
    assert folded.outputs == source.outputs and folded.output_storages == source.output_storages
    assert folded.input_storages == source.input_storages
    assert folded.node(boundary.id) == boundary
    assert {c.node_id for c in folded.constants}.issuperset(c.node_id for c in source.constants)
    verify_constant_folding(folded, receipt)
    from merlin.semantic_compiler.reference import TensorValue

    arguments = {n.id: TensorValue(n.type, (0x3FF0000000000000,) * 5) for n in (source.node("runtime"), boundary)}
    assert evaluate(folded, arguments) == evaluate(source, arguments)
    kept, old = fold_source_constants(source)
    assert kept.node(second.id) == second and old["schema"] == "merlin.semantic_constant_folding.v3"
    live_source = replace(source, outputs=(second.id, first.id, second.id), output_storages=("external",) * 3)
    live, live_receipt = fold_source_constants(live_source, eliminate_dead=True)
    assert live.node(second.id) == second and live.node(first.id) == first
    assert live.outputs == live_source.outputs
    verify_constant_folding(live, live_receipt)

    for mutation in ("effect", "position", "node", "missing", "duplicate"):
        altered = copy.deepcopy(receipt)
        row = altered["dead_computations"][0]
        if mutation == "effect":
            row["node"].update(op="input", inputs=[], index_maps=[], attrs={}, effect="input")
        elif mutation == "position":
            row["position"] = len(source.nodes)
        elif mutation == "node":
            row["node"]["inputs"] = ["square", "runtime"]
        elif mutation == "missing":
            altered["dead_computations"].pop()
        else:
            altered["dead_computations"].append(copy.deepcopy(row))
        with pytest.raises(ValueError):
            verify_constant_folding(folded, altered)


def test_dead_elimination_never_evaluates_unqualified_computations_or_changes_inputs():
    source = chain()
    dead = replace(source.node("dynamic"), id="dead_unqualified", op="source_f64_powf_unqualified")
    source = replace(source, nodes=(*source.nodes, dead))
    pruned, receipt = fold_source_constants(source, element_budget=0, eliminate_dead=True)
    assert len(receipt["dead_computations"]) == 1 and not receipt["folds"]
    assert pruned.input_storages == source.input_storages and pruned.constants == source.constants
    verify_constant_folding(pruned, receipt)
    with pytest.raises(ValueError):
        fold_source_constants(source, eliminate_dead=1)
