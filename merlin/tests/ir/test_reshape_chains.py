"""Dense reshape composition preserves bits, public values and live branches."""

from copy import deepcopy
from dataclasses import replace

import pytest

from merlin.semantic_compiler.model import (
    ConstantBinding,
    IndexMap,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.reshape_chains import compose_reshape_chains, verify_reshape_chains
from merlin.semantic_compiler.source_preparation import (
    append_reshape_preparation,
    preparation_counts,
    preparation_filename,
    prepare_source,
    verify_source_preparation,
)


def graph(shapes=((4, 4), (16,), (2, 8), (1, 16)), dtype="bf16"):
    x = SemanticNode("x", "input", (), TensorType(shapes[0], dtype, "raw-bit-pattern"), effect="input")
    nodes = [x]
    for key, shape in zip(("a", "b", "c"), shapes[1:], strict=True):
        nodes.append(SemanticNode(key, "reshape", (nodes[-1].id,), replace(x.type, shape=shape)))
    return KernelRequest(tuple(nodes), ("c",), ("external",), (("x", "external"),), "synthetic")


def test_transitive_composition_matches_every_bf16_encoding():
    source = graph(((256, 256), (65536,), (64, 1024), (1, 65536)))
    result, receipt = compose_reshape_chains(source)
    assert [n.id for n in result.nodes] == ["x", "c"]
    assert result.node("c").inputs == ("x",)
    inputs = {"x": TensorValue(source.node("x").type, tuple(range(65536)))}
    # Includes infinities, signed zeros, subnormals and every NaN payload.
    assert evaluate_graph(source, inputs) == evaluate_graph(result, inputs)
    assert verify_reshape_chains(result, receipt) == source
    assert compose_reshape_chains(result)[0] == result
    assert preparation_counts(receipt) == {"source_reshape_rewrites": 1, "source_removed_reshapes": 2}


def test_other_consumers_and_repeated_public_outputs_are_preserved():
    source = graph()
    side = SemanticNode("side", "identity", ("a",), source.node("a").type)
    source = replace(
        source, nodes=(*source.nodes, side), outputs=("c", "side", "b", "b", "x"), output_storages=("external",) * 5
    )
    result, receipt = compose_reshape_chains(source)
    assert {n.id for n in result.nodes} == {n.id for n in source.nodes}
    assert result.outputs == source.outputs and result.input_storages == source.input_storages
    assert result.node("side").inputs == ("a",)
    assert result.node("b").inputs == result.node("c").inputs == ("x",)
    inputs = {"x": TensorValue(source.node("x").type, tuple(range(16)))}
    assert evaluate_graph(source, inputs) == evaluate_graph(result, inputs)
    assert verify_reshape_chains(result, receipt) == source


@pytest.mark.parametrize("change", ["attribute", "map", "type", "policy", "count", "transpose", "gather", "copy"])
def test_nontransparent_parent_is_not_bypassed(change):
    source = graph()
    parent = source.node("a")
    if change == "attribute":
        parent = replace(parent, attrs=(("prov.layout", "physical"),))
    elif change == "map":
        parent = replace(parent, index_maps=(IndexMap(1, ((0,), (0,)), (0, 0)), pointwise_identity_maps(1, 1)[0]))
    elif change in {"type", "policy", "count"}:
        field, value = {
            "type": ("dtype", "f32"),
            "policy": ("numerical_policy", "different"),
            "count": ("shape", (32,)),
        }[change]
        parent = replace(parent, type=replace(parent.type, **{field: value}))
    else:
        parent = replace(parent, op=change)
    source = replace(source, nodes=tuple(parent if n.id == "a" else n for n in source.nodes))
    result, receipt = compose_reshape_chains(source)
    assert result.node("a") == parent
    assert result.node("c").inputs != ("x",)
    if "b" in {n.id for n in result.nodes}:
        assert result.node("b").inputs == ("a",)
    assert verify_reshape_chains(result, receipt) == source


def test_scalar_constants_and_unused_declared_inputs_keep_exact_bindings():
    source = graph(((), (1,), (1, 1), (1, 1, 1)))
    x = replace(source.node("x"), op="constant", effect="constant")
    spare = SemanticNode("spare", "input", (), TensorType((2,), "i8", "raw-byte-copy"), effect="input")
    source = replace(
        source,
        nodes=(spare, x, *source.nodes[1:]),
        input_storages=(("spare", "external"),),
        constants=(ConstantBinding("x", "bf16-le-raw", "0180"),),
    )
    result, receipt = compose_reshape_chains(source)
    assert result.constants == source.constants and result.node("spare") == spare
    assert result.node("c").inputs == ("x",)
    inputs = {"spare": TensorValue(spare.type, (0, 255))}
    constants = {"x": TensorValue(x.type, (0x8001,))}
    assert evaluate_graph(source, inputs, constants=constants) == evaluate_graph(result, inputs, constants=constants)
    assert verify_reshape_chains(result, receipt) == source


def test_allocated_sources_are_refused():
    with pytest.raises(ValueError, match="unallocated"):
        compose_reshape_chains(replace(graph(), physical_extents=(("x", 16),)))


@pytest.mark.parametrize("tamper", ["before", "after", "node", "index", "duplicate", "removed", "extra"])
def test_rewrite_receipts_refuse_changes(tamper):
    result, record = compose_reshape_chains(graph())
    record = deepcopy(record)
    if tamper in {"before", "after"}:
        record[tamper + "_digest"] = "0" * 64
    elif tamper == "node":
        record["changes"][0]["original"]["attrs"]["layout"] = "changed"
    elif tamper == "index":
        record["changes"][0]["index"] = True
    elif tamper == "duplicate":
        record["changes"].append(record["changes"][0])
    elif tamper == "removed":
        record["changes"][0]["removed"] = 1
    else:
        record["extra"] = True
    with pytest.raises(ValueError):
        verify_reshape_chains(result, record)


def test_both_opt_in_passes_replay_in_order_and_legacy_default_stays_unchanged():
    source = graph()
    source = replace(
        source,
        nodes=(
            source.nodes[0],
            source.nodes[1],
            replace(source.nodes[1], id="a2"),
            replace(source.nodes[2], inputs=("a2",)),
            source.nodes[3],
        ),
        outputs=("a", "c"),
        output_storages=("external",) * 2,
    )
    default, old = prepare_source(source)
    assert default == source
    result, record = prepare_source(source, common_values=True, compose_reshapes=True)
    assert record["schema"] == "merlin.source_preparation.v5"
    assert record["prepared"]["schema"] == "merlin.source_preparation.v4"
    assert preparation_counts(record)["source_common_values"] == 1
    assert preparation_counts(record)["source_removed_reshapes"] == 1
    assert preparation_filename(record) == "source-preparation.json"
    assert verify_source_preparation(result, record) == source
    assert verify_source_preparation(default, old) == source
    for field, value in [
        ("before_digest", "0" * 64),
        ("limits", ["changed"]),
        ("prepared", {"schema": "merlin.source_preparation.v5"}),
    ]:
        with pytest.raises(ValueError):
            verify_source_preparation(result, {**record, field: value})
    with pytest.raises(ValueError, match="Boolean"):
        prepare_source(source, compose_reshapes=1)
    with pytest.raises(ValueError, match="supplied source"):
        append_reshape_preparation(result, old)


def test_standalone_receipt_is_admitted_by_preparation_verifier():
    source = graph()
    result, record = compose_reshape_chains(source)
    assert verify_source_preparation(result, record) == source
    assert preparation_filename(record) == "source-preparation.json"
