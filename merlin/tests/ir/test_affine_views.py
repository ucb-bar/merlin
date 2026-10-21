"""Exact broadcast fusion preserves source bits, escaping values and replay."""

from copy import deepcopy
from dataclasses import replace

import pytest

from merlin.semantic_compiler.affine_views import fuse_affine_views, verify_affine_views
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.source_preparation import preparation_counts, prepare_source, verify_source_preparation


def source(operation="mul", *, transpose=False, shape=(3, 7)):
    small = (shape[0],) if transpose else (shape[1],)
    bf16 = TensorType(shape, "bf16", "raw-bit-pattern")
    f32 = replace(bf16, dtype="f32")
    weight = SemanticNode("weight", "input", (), replace(bf16, shape=small), effect="input")
    x = SemanticNode("x", "input", (), f32, effect="input")
    read = IndexMap(2, ((1, 0),) if transpose else ((0, 1),), (0,))
    view = SemanticNode("view", "gather", (weight.id,), bf16, index_maps=(read, pointwise_identity_maps(2, 1)[-1]))
    cast = SemanticNode(
        "cast",
        "bf16_to_f32",
        (view.id,),
        f32,
        (("numeric_semantics", "bf16-to-f32-exact-v1"),),
        index_maps=pointwise_identity_maps(2, 1),
    )
    out = SemanticNode(
        "out",
        "f32_" + operation,
        (cast.id, x.id),
        f32,
        (("numeric_semantics", "ieee754-f32-" + operation + "-rne-v1"),),
        index_maps=pointwise_identity_maps(2, 2),
    )
    return KernelRequest(
        (weight, x, view, cast, out),
        (out.id,),
        ("external",),
        ((weight.id, "external"), (x.id, "external")),
        "synthetic",
    )


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("transpose", [False, True])
def test_exact_cast_moves_before_complete_projection_and_pointwise_reads_base(operation, transpose):
    original = source(operation, transpose=transpose)
    result, receipt = prepare_source(original)
    assert verify_source_preparation(result, receipt).digest() == original.digest()
    assert preparation_counts(receipt)["source_affine_view_rewrites"] == 2
    assert result.node("out").inputs == ("cast_affine_source", "x")
    assert result.node("cast_affine_source").type.shape == original.node("weight").type.shape
    assert {node.id for node in result.nodes} == {"weight", "x", "cast_affine_source", "out"}
    assert result.input_storages == original.input_storages and result.outputs == original.outputs
    # Compare separate logical interpreters of the original and transformed
    # graphs, before any target precision selection or physical lowering.
    bits = (0x0000, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x4001, 0x0080)
    weights = bits[:3] if transpose else bits
    inputs = {
        "weight": TensorValue(original.node("weight").type, weights),
        "x": TensorValue(original.node("x").type, (0x3F800001, 0xBF000000, 0x40000000) * 7),
    }
    assert evaluate_graph(original, inputs) == evaluate_graph(result, inputs)


def test_escaping_broadcast_and_cast_remain_explicit_results():
    original = source()
    original = replace(original, outputs=("out", "cast", "view"), output_storages=("external",) * 3)
    result, receipt = prepare_source(original)
    assert result.outputs == original.outputs and result.node("cast").op == "gather"
    assert result.node("view").op == "gather"
    assert verify_source_preparation(result, receipt).digest() == original.digest()


def test_rank_three_projection_preserves_permuted_matrix_indices():
    original = source()
    weight, x, view, cast, output = original.nodes
    shape = (2, 3, 4)
    weight = replace(weight, type=replace(weight.type, shape=(4, 3)))
    x = replace(x, type=replace(x.type, shape=shape))
    view = replace(
        view,
        type=replace(view.type, shape=shape),
        index_maps=(
            IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
            pointwise_identity_maps(3, 1)[-1],
        ),
    )
    cast = replace(cast, type=x.type, index_maps=pointwise_identity_maps(3, 1))
    output = replace(output, type=x.type, index_maps=pointwise_identity_maps(3, 2))
    original = replace(original, nodes=(weight, x, view, cast, output))
    prepared, record = prepare_source(original)
    assert prepared.node("out").index_maps[0] == view.index_maps[0]
    inputs = {
        "weight": TensorValue(weight.type, tuple(0x3F80 + i for i in range(12))),
        "x": TensorValue(x.type, tuple(0xBF800000 + i for i in range(24))),
    }
    # The existing f32 reference accepts increasing singleton projections.
    # Expand the permuted operand through its separately interpreted gather;
    # do not widen that reference's admitted arithmetic/indexing policy here.
    result = prepared.node("out")
    expanded = SemanticNode(
        "expanded",
        "gather",
        (result.inputs[0],),
        result.type,
        index_maps=(result.index_maps[0], pointwise_identity_maps(3, 1)[-1]),
    )
    reference = replace(
        prepared,
        nodes=(
            *prepared.nodes[:-1],
            expanded,
            replace(result, inputs=(expanded.id, result.inputs[1]), index_maps=pointwise_identity_maps(3, 2)),
        ),
    )
    assert evaluate_graph(original, inputs) == evaluate_graph(reference, inputs)
    assert verify_source_preparation(prepared, record).digest() == original.digest()


def test_wrapper_rejects_changed_preparation_order_and_empty_indexing_stage():
    original = source()
    prepared, record = prepare_source(original)
    for fault in ("nested", "empty"):
        changed = deepcopy(record)
        if fault == "nested":
            changed["numerical"] = deepcopy(record)
        else:
            changed["views"]["rewrites"] = []
        with pytest.raises(ValueError):
            verify_source_preparation(prepared, changed)


@pytest.mark.parametrize("mutation", ["offset", "partial", "stride", "cast_policy", "consumer_policy"])
def test_partial_views_and_changed_numerical_policies_are_not_absorbed(mutation):
    original = source()
    nodes = list(original.nodes)
    if mutation == "offset":
        nodes[2] = replace(nodes[2], index_maps=(IndexMap(2, ((0, 1),), (1,)), nodes[2].index_maps[-1]))
    elif mutation == "partial":
        nodes[0] = replace(nodes[0], type=replace(nodes[0].type, shape=(8,)))
    elif mutation == "stride":
        nodes[0] = replace(nodes[0], type=replace(nodes[0].type, shape=(14,)))
        nodes[2] = replace(nodes[2], index_maps=(IndexMap(2, ((0, 2),), (0,)), nodes[2].index_maps[-1]))
    elif mutation == "cast_policy":
        nodes[3] = replace(nodes[3], attrs=(("numeric_semantics", "different"),))
    else:
        nodes[4] = replace(nodes[4], attrs=(("numeric_semantics", "different"),))
    changed = replace(original, nodes=tuple(nodes))
    result, receipt = fuse_affine_views(changed)
    assert result.node("out").inputs == changed.node("out").inputs
    assert verify_affine_views(result, receipt).digest() == changed.digest()


@pytest.mark.parametrize("fault", ["before", "after", "map", "generated", "source"])
def test_replay_rejects_changed_source_maps_and_generated_nodes(fault):
    original = source()
    result, receipt = fuse_affine_views(original)
    record = deepcopy(receipt)
    if fault in {"before", "after"}:
        record[fault + "_digest"] = "0" * 64
    elif fault == "map":
        record["rewrites"][-1]["original"]["index_maps"][0]["offsets"][0] = 1
    elif fault == "generated":
        record["rewrites"][0]["generated"] = ["weight"]
    else:
        nodes = list(result.nodes)
        nodes[0] = replace(nodes[0], type=replace(nodes[0].type, shape=(8,)))
        result = replace(result, nodes=tuple(nodes))
    with pytest.raises((ValueError, KeyError)):
        verify_affine_views(result, record)
