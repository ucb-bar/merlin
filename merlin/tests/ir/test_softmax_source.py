"""Closed source softmax recognition, including uses outside a fusion region."""

from dataclasses import replace
from math import prod

import pytest

from merlin.semantic_compiler.model import (
    ConstantBinding,
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.softmax_source import recognize_row_softmax


def source_graph(shape):
    rank = len(shape)
    full, row, column = (
        TensorType(extent, "f32", "raw-bit-pattern") for extent in (shape, shape[:-1], (*shape[:-1], 1))
    )
    identity = pointwise_identity_maps(rank, 1)[0]
    prefix = IndexMap(rank, identity.coefficients[:-1], (0,) * (rank - 1))
    reduce_maps = (identity, prefix, prefix)
    broadcast = (identity, IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank), identity)
    max_attrs = (
        ("axis", rank - 1),
        ("reduction_order", "unresolved"),
        ("numeric_semantics", "source-f32-maximum-unqualified-v1"),
        ("scalar_operand_order", "input-accumulator"),
    )
    sum_attrs = (
        ("axis", rank - 1),
        ("reduction_order", "unresolved"),
        ("numeric_semantics", "source-f32-add-unqualified-v1"),
        ("scalar_add_order", "input-accumulator"),
    )
    nodes = (
        SemanticNode("scores", "input", (), full, effect="input"),
        SemanticNode("min_seed", "constant", (), row, effect="constant"),
        SemanticNode(
            "maximum",
            "source_f32_reduce_max_unqualified",
            ("scores", "min_seed"),
            row,
            max_attrs,
            index_maps=reduce_maps,
        ),
        SemanticNode("max_view", "reshape", ("maximum",), column),
        SemanticNode(
            "shifted",
            "f32_sub",
            ("scores", "max_view"),
            full,
            (("numeric_semantics", "ieee754-f32-sub-rne-v1"),),
            index_maps=broadcast,
        ),
        SemanticNode(
            "exponential",
            "source_f32_exp_unqualified",
            ("shifted",),
            full,
            (("numeric_semantics", "source-f32-exp-unqualified-v1"),),
            index_maps=pointwise_identity_maps(rank, 1),
        ),
        SemanticNode("zero_seed", "constant", (), row, effect="constant"),
        SemanticNode(
            "total",
            "source_f32_reduce_add_unqualified",
            ("exponential", "zero_seed"),
            row,
            sum_attrs,
            index_maps=reduce_maps,
        ),
        SemanticNode("sum_view", "reshape", ("total",), column),
        SemanticNode(
            "result",
            "f32_div",
            ("exponential", "sum_view"),
            full,
            (("numeric_semantics", "ieee754-f32-div-rne-v1"),),
            index_maps=broadcast,
        ),
    )
    return KernelRequest(
        nodes,
        ("result",),
        ("external",),
        (("scores", "external"),),
        "synthetic",
        constants=(
            ConstantBinding("min_seed", "f32-le-raw", "000080ff" * prod(shape[:-1])),
            ConstantBinding("zero_seed", "f32-le-raw", "00000000" * prod(shape[:-1])),
        ),
    )


@pytest.mark.parametrize("shape", [(1, 7), (33, 65), (7, 96), (2, 3, 65), (2, 3, 2, 35)])
def test_source_geometry_and_a_shared_result_are_preserved(shape):
    graph = source_graph(shape)
    found = recognize_row_softmax(graph, "result")
    assert found.shape == shape and len(found.computations) == 7
    assert found.source_digest == graph.digest()
    assert found.matrix_shape == (prod(shape[:-1]), shape[-1])
    if len(shape) > 2:
        assert found.record()["dense_row_view"]["element_order"] == "row-major-unchanged"
    consumer = SemanticNode("consumer", "copy", ("result",), graph.node("result").type)
    extended = replace(graph, nodes=(*graph.nodes, consumer), outputs=("consumer",))
    assert recognize_row_softmax(extended, "result").result == "result"


@pytest.mark.parametrize("fault", ["seed", "axis", "broadcast", "order", "outside_use", "extra_output"])
def test_wrong_source_or_escaping_intermediates_are_rejected(fault):
    graph = source_graph((3, 5))
    nodes = list(graph.nodes)
    if fault == "seed":
        graph = replace(graph, constants=(graph.constants[0], replace(graph.constants[1], data_hex="0000803f" * 3)))
    elif fault in {"axis", "order"}:
        index = next(i for i, node in enumerate(nodes) if node.id == "total")
        attrs = dict(nodes[index].attrs)
        attrs["axis" if fault == "axis" else "reduction_order"] = 0 if fault == "axis" else "axis-ascending"
        nodes[index] = replace(nodes[index], attrs=tuple(attrs.items()))
        graph = replace(graph, nodes=tuple(nodes))
    elif fault == "broadcast":
        index = next(i for i, node in enumerate(nodes) if node.id == "shifted")
        maps = list(nodes[index].index_maps)
        maps[1] = IndexMap(2, ((0, 1), (0, 0)), (0, 0))
        nodes[index] = replace(nodes[index], index_maps=tuple(maps))
        graph = replace(graph, nodes=tuple(nodes))
    elif fault == "outside_use":
        extra = SemanticNode("escaped", "copy", ("exponential",), graph.node("exponential").type)
        graph = replace(graph, nodes=(*nodes, extra))
    else:
        graph = replace(graph, outputs=("result", "exponential"), output_storages=("external", "external"))
    with pytest.raises(ValueError, match="source row softmax"):
        recognize_row_softmax(graph, "result")


def test_batched_rows_cannot_be_mixed_or_reduced_together():
    source = source_graph((2, 3, 35))
    for key in ("total", "shifted"):
        node = source.node(key)
        maps = list(node.index_maps)
        # Swapping the two prefix dimensions changes which rows interact.
        mapping = maps[1]
        maps[1] = replace(
            mapping, coefficients=(mapping.coefficients[1], mapping.coefficients[0], *mapping.coefficients[2:])
        )
        wrong = replace(node, index_maps=tuple(maps))
        changed = replace(source, nodes=tuple(wrong if item.id == key else item for item in source.nodes))
        with pytest.raises(ValueError, match="index maps"):
            recognize_row_softmax(changed, "result")
    node = source.node("sum_view")
    wrong = replace(node, type=replace(node.type, shape=(3, 2, 1)))
    with pytest.raises(ValueError, match="broadcast reshape"):
        recognize_row_softmax(
            replace(source, nodes=tuple(wrong if item.id == node.id else item for item in source.nodes)), "result"
        )


def test_dense_reshape_chains_preserve_rows_and_cannot_escape():
    source = source_graph((2, 3, 35))
    nodes = []
    for node in source.nodes:
        if node.id in {"max_view", "sum_view"}:
            intermediate = SemanticNode(
                node.id + "_flat", "reshape", node.inputs, TensorType((6,), "f32", "raw-bit-pattern")
            )
            nodes.append(intermediate)
            node = replace(node, inputs=(intermediate.id,))
        nodes.append(node)
    graph = replace(source, nodes=tuple(nodes))
    found = recognize_row_softmax(graph, "result")
    assert found.matrix_shape == (6, 35) and len(found.computations) == 9
    escaped = SemanticNode("escape", "copy", ("max_view_flat",), graph.node("max_view_flat").type)
    with pytest.raises(ValueError, match="outside consumer"):
        recognize_row_softmax(replace(graph, nodes=(*graph.nodes, escaped)), "result")


@pytest.mark.parametrize("shape", [(2, 3, 35), (2, 4, 3, 35)])
def test_masked_source_keeps_affine_mask_and_rejects_escaping_add(shape):
    from merlin.semantic_compiler.softmax_source import recognize_masked_row_softmax

    source = source_graph(shape)
    rank = len(shape)
    score = source.node("scores")
    mask = SemanticNode("mask", "input", (), TensorType(shape[-2:], "f32", "raw-bit-pattern"), effect="input")
    identity = pointwise_identity_maps(rank, 1)[0]
    mapping = IndexMap(rank, identity.coefficients[-2:], (0, 0))
    add = SemanticNode(
        "masked",
        "f32_add",
        (score.id, mask.id),
        score.type,
        (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
        index_maps=(identity, mapping, identity),
    )
    nodes = []
    for node in source.nodes:
        nodes.append(replace(node, inputs=tuple(add.id if key == score.id else key for key in node.inputs)))
        if node.id == score.id:
            nodes.extend((mask, add))
    graph = replace(source, nodes=tuple(nodes), input_storages=(*source.input_storages, (mask.id, "external")))
    found = recognize_masked_row_softmax(graph, "result")
    assert found.scores == score.id and found.mask == mask.id and found.mask_map == mapping
    assert add.id in found.computations and found.matrix_shape == (prod(shape[:-1]), shape[-1])
    assert found.record()["mask_shape"] == list(shape[-2:])
    escaped = SemanticNode("escape", "copy", (add.id,), add.type)
    with pytest.raises(ValueError, match="outside consumer"):
        recognize_masked_row_softmax(replace(graph, nodes=(*graph.nodes, escaped)), "result")
    bad_map = replace(mapping, offsets=(0, 1))
    with pytest.raises(ValueError, match="leaves its source"):
        recognize_masked_row_softmax(
            replace(
                graph,
                nodes=tuple(
                    replace(node, index_maps=(identity, bad_map, identity)) if node.id == add.id else node
                    for node in graph.nodes
                ),
            ),
            "result",
        )
