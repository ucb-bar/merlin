"""The guarded softmax fusion accounts for masks, constants and every consumer."""

from dataclasses import replace
from math import prod

import pytest

from merlin.semantic_compiler.guarded_softmax_source import recognize_guarded_masked_row_softmax
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from merlin.tests.ir.test_softmax_source import source_graph


def guarded_graph(shape=(2, 3, 5), *, shared_maximum=False):
    source = source_graph(shape)
    rank = len(shape)
    full = source.node("scores").type
    column = TensorType((*shape[:-1], 1), "f32", "raw-bit-pattern")
    identity = pointwise_identity_maps(rank, 1)[0]
    scalar_map = IndexMap(rank, (), ())
    row_map = IndexMap(rank, (*identity.coefficients[:-1], (0,) * rank), (0,) * rank)
    mask = SemanticNode("excluded", "input", (), TensorType(shape[-2:], "i1", "boolean"), effect="input")
    negative = SemanticNode("negative_inf", "constant", (), TensorType((), "f32", "raw-bit-pattern"), effect="constant")
    zero = replace(negative, id="zero")
    guard_limit = replace(negative, id="guard_limit", type=column)
    mask_view = SemanticNode(
        "mask_view",
        "gather",
        (mask.id,),
        TensorType(shape, "i1", "boolean"),
        index_maps=(IndexMap(rank, identity.coefficients[-2:], (0, 0)), identity),
    )
    fill = SemanticNode("fill", "gather", (negative.id,), full, index_maps=(scalar_map, identity))
    masked = SemanticNode("masked", "select", (mask_view.id, fill.id, "scores"), full, index_maps=(identity,) * 4)
    core = tuple(replace(n, inputs=tuple("masked" if k == "scores" else k for k in n.inputs)) for n in source.nodes[1:])
    maximum = next(n for n in core if n.id == "maximum")
    duplicate = replace(maximum, id="guard_maximum")
    parent = maximum.id if shared_maximum else duplicate.id
    view = SemanticNode("guard_view", "reshape", (parent,), column)
    compare = SemanticNode(
        "guard_equal",
        "mask",
        (view.id, guard_limit.id),
        TensorType(column.shape, "i1", "boolean"),
        (("operation", "arith.cmpf"), ("predicate", "oeq")),
        index_maps=pointwise_identity_maps(rank, 2),
    )
    condition = SemanticNode(
        "guard_condition", "gather", (compare.id,), TensorType(shape, "i1", "boolean"), index_maps=(row_map, identity)
    )
    zero_view = SemanticNode("zero_view", "gather", (zero.id,), full, index_maps=(scalar_map, identity))
    result = SemanticNode("guarded", "select", (condition.id, zero_view.id, "result"), full, index_maps=(identity,) * 4)
    return replace(
        source,
        nodes=(
            source.nodes[0],
            mask,
            negative,
            mask_view,
            fill,
            masked,
            *core,
            zero,
            guard_limit,
            *((duplicate,) if not shared_maximum else ()),
            view,
            compare,
            condition,
            zero_view,
            result,
        ),
        outputs=(result.id,),
        input_storages=((*source.input_storages, (mask.id, "external"))),
        constants=(
            *source.constants,
            ConstantBinding(negative.id, "f32-le-raw", "000080ff"),
            ConstantBinding(zero.id, "f32-le-raw", "00000000"),
            ConstantBinding(guard_limit.id, "f32-le-raw", "000080ff" * prod(shape[:-1])),
        ),
    )


@pytest.mark.parametrize("shape", [(3, 7), (2, 3, 5), (1, 2, 3, 33)])
@pytest.mark.parametrize("shared", [False, True])
def test_guarded_source_preserves_rows_and_all_computation(shape, shared):
    source = guarded_graph(shape, shared_maximum=shared)
    region = recognize_guarded_masked_row_softmax(source, "guarded")
    assert region.scores == "scores" and region.mask == "excluded" and region.result == "guarded"
    assert region.matrix_shape == (prod(shape[:-1]), shape[-1])
    assert set(region.computations) == {n.id for n in source.nodes if n.effect == "pure"}
    assert set(region.initializers) == {n.id for n in source.nodes if n.effect == "constant"}
    assert region.record()["all_masked_rows"] == "positive-zero"
    assert region.mask_map.apply((0,) * len(shape)) == (0, 0)
    assert type(source).from_record(source.record()).digest() == source.digest()


@pytest.mark.parametrize(
    "fault", ["predicate", "zero", "limit", "fill", "maximum", "source", "map", "escape", "output"]
)
def test_guarded_source_rejects_non_equivalent_or_escaping_forms(fault):
    source = guarded_graph()
    node = None
    if fault == "predicate":
        node = replace(source.node("guard_equal"), attrs=(("operation", "arith.cmpf"), ("predicate", "one")))
    elif fault == "maximum":
        node = replace(source.node("guard_maximum"), attrs=((*source.node("guard_maximum").attrs, ("changed", True))))
    elif fault == "source":
        node = replace(source.node("guard_maximum"), inputs=("scores", "min_seed"))
    elif fault == "map":
        view = source.node("mask_view")
        node = replace(view, index_maps=(replace(view.index_maps[0], offsets=(1, 0)), view.index_maps[1]))
    elif fault in {"zero", "limit", "fill"}:
        key = {"zero": "zero", "limit": "guard_limit", "fill": "negative_inf"}[fault]
        binding = next(c for c in source.constants if c.node_id == key)
        source = replace(
            source,
            constants=tuple(
                replace(c, data_hex="0000803f" * (len(binding.data_hex) // 8)) if c is binding else c
                for c in source.constants
            ),
        )
    elif fault == "escape":
        source = replace(
            source,
            nodes=(*source.nodes, SemanticNode("escape", "copy", ("guard_equal",), source.node("guard_equal").type)),
        )
    else:
        source = replace(source, outputs=("guarded", "guard_equal"), output_storages=("external",) * 2)
    if node is not None:
        source = replace(source, nodes=tuple(node if n.id == node.id else n for n in source.nodes))
    with pytest.raises(ValueError):
        recognize_guarded_masked_row_softmax(source, "guarded")
