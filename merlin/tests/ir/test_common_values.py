"""Pure-value commoning preserves exact semantics, public ports and receipts."""

from copy import deepcopy
from dataclasses import replace

import pytest

from merlin.semantic_compiler.common_values import common_pure_values, verify_common_pure_values
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
from merlin.semantic_compiler.source_preparation import preparation_counts, prepare_source, verify_source_preparation


def graph():
    typ = TensorType((4,), "f32", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), typ, effect="input")
    y = replace(x, id="y")
    a = SemanticNode(
        "a",
        "f32_add",
        ("x", "y"),
        typ,
        (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
        index_maps=pointwise_identity_maps(1, 2),
    )
    b = replace(a, id="b")
    c = replace(a, id="c", inputs=("a", "b"))
    d = replace(c, id="d", inputs=("b", "a"))
    out = replace(a, id="out", inputs=("c", "d"))
    return KernelRequest(
        (x, y, a, b, c, d, out), ("out",), ("external",), (("x", "external"), ("y", "external")), "synthetic"
    )


def test_transitive_commoning_matches_independent_float_reference():
    source = graph()
    result, receipt = common_pure_values(source)
    assert [node.id for node in result.nodes] == ["x", "y", "a", "c", "out"]
    assert result.node("out").inputs == ("c", "c")
    assert verify_common_pure_values(result, receipt) == source
    for bits in [(0, 0x80000000, 1, 0x007FFFFF), (0x3F800001, 0xBF800000, 0x7F7FFFFF, 0x00800000)]:
        inputs = {name: TensorValue(source.node(name).type, bits) for name in ("x", "y")}
        assert evaluate_graph(source, inputs) == evaluate_graph(result, inputs)
    assert common_pure_values(result)[0] == result


def test_provenance_is_preserved_in_replay_without_granting_numeric_equivalence():
    source = graph()
    nodes = tuple(replace(n, attrs=(*n.attrs, ("prov.fqn", n.id))) if n.effect == "pure" else n for n in source.nodes)
    source = replace(source, nodes=nodes)
    result, receipt = common_pure_values(source)
    assert "b" not in {n.id for n in result.nodes}
    assert dict(result.node("a").attrs)["prov.fqn"] == "a"
    assert verify_common_pure_values(result, receipt) == source


@pytest.mark.parametrize("change", ["attribute", "policy", "order", "map", "scalar_type", "signed_zero"])
def test_semantic_differences_refuse_commoning(change):
    source = graph()
    b = source.node("b")
    if change == "attribute":
        b = replace(b, attrs=(*b.attrs, ("scale", -8)))
    elif change == "policy":
        b = replace(b, type=replace(b.type, numerical_policy="different"))
    elif change == "order":
        b = replace(b, inputs=tuple(reversed(b.inputs)))
    elif change == "map":
        b = replace(b, index_maps=(replace(b.index_maps[0], offsets=(1,)), *b.index_maps[1:]))
    else:
        first, second = (True, 1) if change == "scalar_type" else (0.0, -0.0)
        source = replace(
            source,
            nodes=tuple(replace(n, attrs=(*n.attrs, ("option", first))) if n.id == "a" else n for n in source.nodes),
        )
        b = replace(b, attrs=(*b.attrs, ("option", second)))
    source = replace(source, nodes=tuple(b if n.id == "b" else n for n in source.nodes))
    result, receipt = common_pure_values(source)
    assert "b" in {n.id for n in result.nodes}
    assert verify_common_pure_values(result, receipt) == source


def test_public_results_and_input_ports_keep_identity_and_order():
    source = replace(graph(), outputs=("b", "a", "b", "x", "out"), output_storages=("external",) * 5)
    result, receipt = common_pure_values(source)
    assert result.outputs == source.outputs and result.input_storages == source.input_storages
    assert result.node("b").op == "f32_add"
    assert result.node("x") == source.node("x")
    assert verify_common_pure_values(result, receipt) == source


def test_allocated_sources_and_mutable_input_values_are_not_merged():
    source = graph()
    with pytest.raises(ValueError, match="unallocated"):
        common_pure_values(replace(source, physical_extents=(("x", 4), ("y", 4))))
    result, _ = common_pure_values(source)
    assert result.node("a").inputs == ("x", "y")


@pytest.mark.parametrize("tamper", ["before", "after", "node", "index", "duplicate", "replacement", "extra"])
def test_receipt_changes_are_rejected(tamper):
    result, receipt = common_pure_values(graph())
    record = deepcopy(receipt)
    if tamper in {"before", "after"}:
        record[tamper + "_digest"] = "0" * 64
    elif tamper == "node":
        record["changes"][0]["original"]["attrs"]["different"] = True
    elif tamper == "index":
        record["changes"][0]["index"] = -1
    elif tamper == "duplicate":
        record["changes"].append(record["changes"][0])
    elif tamper == "replacement":
        record["changes"][0]["replacement"] = "y"
    else:
        record["extra"] = True
    with pytest.raises(ValueError):
        verify_common_pure_values(result, record)


def test_opt_in_preparation_replays_without_changing_legacy_default():
    source = graph()
    default, old = prepare_source(source)
    assert "b" in {n.id for n in default.nodes}
    result, receipt = prepare_source(source, common_values=True)
    assert receipt["schema"] == "merlin.source_preparation.v4"
    assert preparation_counts(receipt)["source_common_values"] == 2
    assert verify_source_preparation(result, receipt) == source
    assert verify_source_preparation(default, old) == source
    with pytest.raises(ValueError, match="Boolean"):
        prepare_source(source, common_values=1)


def test_standalone_receipt_and_composite_tampering():
    source = graph()
    result, receipt = common_pure_values(source)
    assert verify_source_preparation(result, receipt) == source
    assert preparation_counts(receipt) == {"source_common_values": 2}
    result, receipt = prepare_source(source, common_values=True)
    for field, value in [
        ("before_digest", "0" * 64),
        ("limits", ["unknown"]),
        ("prepared", {"schema": "merlin.source_preparation.v4"}),
    ]:
        changed = {**receipt, field: value}
        with pytest.raises(ValueError):
            verify_source_preparation(result, changed)


def sibling_graph():
    source = graph()
    a = source.node("a")
    ae = replace(a, id="ae", op="f32_mul", attrs=(("numeric_semantics", "ieee754-f32-mul-rne-v1"),))
    b, be = replace(a, id="b"), replace(ae, id="be")
    out = replace(source.node("out"), inputs=("b", "be"))
    return replace(source, nodes=(*source.nodes[:2], a, ae, b, be, out)), (("a", "ae"), ("b", "be"))


def test_sibling_regions_are_shared_atomically_and_replay():
    source, groups = sibling_graph()
    result, receipt = common_pure_values(source, groups=groups)
    assert result.node("out").inputs == ("a", "ae")
    assert {n.id for n in result.nodes} == {"x", "y", "a", "ae", "out"}
    assert verify_common_pure_values(result, receipt) == source
    inputs = {key: TensorValue(source.node(key).type, (0x3F800001, 0xBF800000, 0, 0x80000000)) for key in ("x", "y")}
    assert evaluate_graph(source, inputs) == evaluate_graph(result, inputs)


@pytest.mark.parametrize("case", ["public", "different_result", "crossing", "dependent"])
def test_groups_cannot_be_partially_commoned_or_introduce_forward_references(case):
    source, groups = sibling_graph()
    if case == "public":
        source = replace(source, outputs=("out", "b"), output_storages=("external",) * 2)
    elif case == "different_result":
        source = replace(
            source,
            nodes=tuple(
                replace(n, type=replace(n.type, numerical_policy="different")) if n.id == "be" else n
                for n in source.nodes
            ),
        )
    elif case == "crossing":
        source = replace(source, nodes=tuple(source.node(key) for key in ("x", "y", "a", "b", "be", "out", "ae")))
    else:
        source = replace(
            source,
            nodes=tuple(
                replace(n, inputs=("a", "a")) if n.id == "ae" else replace(n, inputs=("b", "b")) if n.id == "be" else n
                for n in source.nodes
            ),
        )
    result, receipt = common_pure_values(source, groups=groups)
    assert {"b", "be"} <= {n.id for n in result.nodes}
    assert verify_common_pure_values(result, receipt) == source


@pytest.mark.parametrize("groups", [(("a", "a"),), (("missing",),), (("x",),), (("a",), ("a",)), ((),)])
def test_malformed_atomic_groups_are_rejected(groups):
    with pytest.raises(ValueError, match="atomic"):
        common_pure_values(graph(), groups=groups)
