"""Private probabilities may change representation only inside the fused region."""

from dataclasses import replace
from math import prod

import pytest

from merlin.semantic_compiler.model import ConstantBinding, IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.softmax_value_source import recognize_softmax_value
from merlin.tests.ir.test_softmax_source import source_graph


def graph(shape, channels=64, views=False):
    source = source_graph(shape)
    batch = shape[:-2]
    matrix_shape = (*batch, shape[-1], channels)
    output_type = TensorType((*batch, shape[-2], channels), "f32", "raw-bit-pattern")
    value = SemanticNode("values", "input", (), TensorType(matrix_shape, "f32", "raw-bit-pattern"), effect="input")
    seed = SemanticNode("value_seed", "constant", (), output_type, effect="constant")
    lhs, extra = "result", []
    if views:
        extra.append(SemanticNode("copy_probability", "gather", (lhs,), source.node(lhs).type,
                                  index_maps=pointwise_identity_maps(len(shape), 1)))
        extra.append(SemanticNode("flat_probability", "reshape", ("copy_probability",),
                                  TensorType((prod(shape),), "f32", "raw-bit-pattern")))
        extra.append(SemanticNode("matrix_probability", "reshape", ("flat_probability",), source.node(lhs).type))
        lhs = "matrix_probability"
    rank = len(shape)
    axes = tuple(tuple(int(i == j) for j in range(rank + 1)) for i in range(rank + 1))
    maps = tuple(IndexMap(rank + 1, (*axes[:rank - 2], *pair), (0,) * rank) for pair in (
        (axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    result = SemanticNode("context", "source_f32_matmul_unqualified", (lhs, "values", "value_seed"), output_type,
                          (("reduction_order", "unresolved"), ("numeric_semantics", "source-f32-mul-add-unqualified-v1")),
                          index_maps=(*maps, maps[-1]))
    return replace(source, nodes=(*source.nodes, value, seed, *extra, result), outputs=(result.id,),
                   input_storages=(*source.input_storages, (value.id, "external")),
                   constants=(*source.constants, ConstantBinding(seed.id, "f32-le-raw", "00000000" * prod(output_type.shape))))


@pytest.mark.parametrize("shape", [(1, 129), (128, 128), (32, 1, 129), (1, 32, 128, 128)])
@pytest.mark.parametrize("views", [False, True])
def test_dense_batches_and_identity_views_preserve_private_probabilities(shape, views):
    source = graph(shape, views=views)
    found = recognize_softmax_value(source, "context")
    assert (found.query_rows, found.keys, found.channels) == (*shape[-2:], 64)
    assert found.batch_shape == shape[:-2] and found.probabilities == "result"
    assert found.source_digest == source.digest() and found.record()["probability_boundary"] == "private"
    assert "values" not in found.computations and "context" in found.computations


@pytest.mark.parametrize("fault", ["escape", "output", "view_escape", "seed", "maps", "transpose"])
def test_escaping_probabilities_wrong_seeds_and_crossed_axes_refuse(fault):
    source = graph((2, 3, 5), views=True)
    if fault in {"escape", "view_escape"}:
        key = "result" if fault == "escape" else "flat_probability"
        source = replace(source, nodes=(*source.nodes, SemanticNode("escape", "copy", (key,), source.node(key).type)))
    elif fault == "output":
        source = replace(source, outputs=("context", "result"), output_storages=("external", "external"))
    elif fault == "seed":
        source = replace(source, constants=tuple(replace(c, data_hex="0000803f" * (2 * 3 * 64)) if c.node_id == "value_seed" else c for c in source.constants))
    else:
        key = "context" if fault == "maps" else "copy_probability"
        node = source.node(key)
        wrong = replace(node, index_maps=tuple(replace(m, coefficients=tuple(reversed(m.coefficients))) for m in node.index_maps))
        source = replace(source, nodes=tuple(wrong if n.id == key else n for n in source.nodes))
    with pytest.raises(ValueError):
        recognize_softmax_value(source, "context")
