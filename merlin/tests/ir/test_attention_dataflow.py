"""Source attention relationships belong to the generic frontend inventory."""

from __future__ import annotations

import copy

import pytest

from merlin.frontends.attention_dataflow import find_attention_pairs


def operation(index, role, region, parents=(), shape=(3, 5, 7)):
    return {
        "id": index,
        "kind": "linalg.generic",
        "prov": {
            "region_id": region,
            "family": "contraction" if role == "batch_matmul" else "normalization" if role == "softmax" else "layout",
            "op": role,
            "aten": "aten.bmm.default"
            if role == "batch_matmul"
            else "aten._softmax.default"
            if role == "softmax"
            else "",
        },
        "ins": [{"source": ("op", parent), "shape": list(shape)} for parent in parents],
        "results": [{"shape": list(shape)}],
    }


def graph(labels=("first", "normalization", "last")):
    first = operation(0, "batch_matmul", labels[0])
    first["ins"] = [{"source": ("arg", 0)}, {"source": ("arg", 1)}]
    view = operation(1, "view", "view", (0,))
    normalization = operation(2, "softmax", labels[1], (1,))
    cast = operation(3, "cast", "cast", (2,))
    last = operation(4, "batch_matmul", labels[2], (3,))
    last["ins"].append({"source": ("arg", 2)})
    return {"ops": [first, view, normalization, cast, last]}


@pytest.mark.parametrize(
    "labels",
    (
        ("first", "normalization", "last"),
        ("looks_like_pv", "unrelated_label", "looks_like_qk"),
        ("opaque#add", "opaque#mul", "opaque#softmax"),
    ),
)
def test_relationship_uses_ssa_and_semantics_without_label_or_module_names(labels):
    inventory = graph(labels)
    original = copy.deepcopy(inventory)
    assert find_attention_pairs(inventory) == [
        {"qk_region_id": labels[0], "pv_region_id": labels[2], "softmax_region_id": labels[1]}
    ]
    assert inventory == original
    for op in inventory["ops"]:
        op["prov"]["fqn"] = "arbitrary.unrelated_module"
    assert find_attention_pairs(inventory) == find_attention_pairs(original)


@pytest.mark.parametrize(
    "change", ("missing_softmax", "wrong_softmax_tag", "wrong_contraction_tag", "shape_mismatch", "probability_on_rhs")
)
def test_source_relationship_refuses_unestablished_path(change):
    inventory = graph()
    ops = inventory["ops"]
    if change == "missing_softmax":
        ops[2]["prov"]["op"] = "view"
    elif change == "wrong_softmax_tag":
        ops[2]["prov"]["aten"] = "aten.other.default"
    elif change == "wrong_contraction_tag":
        ops[0]["prov"]["aten"] = "aten.other.default"
    elif change == "shape_mismatch":
        ops[4]["ins"][0]["shape"] = [3, 7, 5]
    else:
        ops[4]["ins"].reverse()
    assert find_attention_pairs(inventory) == []


def test_equal_shapes_and_names_do_not_invent_dataflow():
    inventory = graph()
    inventory["ops"][4]["ins"][0]["source"] = ("arg", 3)
    for op in inventory["ops"]:
        op["prov"]["fqn"] = "arbitrary.self_attn"
    assert find_attention_pairs(inventory) == []


def test_two_upstream_contractions_are_ambiguous():
    inventory = graph()
    alternative = copy.deepcopy(inventory["ops"][0])
    alternative["id"] = 1
    alternative["prov"]["region_id"] = "alternative"
    inventory["ops"][1] = alternative
    inventory["ops"][2]["ins"].append({"source": ("op", 0), "shape": [3, 5, 7]})
    assert find_attention_pairs(inventory) == []


def test_multiple_softmax_regions_are_ambiguous():
    inventory = graph()
    extra = operation(3, "softmax", "other_normalization", (0,))
    merged = operation(4, "merge", "merge", (2, 3))
    last = operation(5, "batch_matmul", "last", (4,))
    last["ins"].append({"source": ("arg", 2)})
    inventory["ops"] = inventory["ops"][:3] + [extra, merged, last]
    assert find_attention_pairs(inventory) == []


def test_same_softmax_region_can_have_multiple_decomposed_nodes():
    inventory = graph()
    inventory["ops"][3] = operation(3, "softmax", "normalization", (2,))
    assert find_attention_pairs(inventory) == [
        {"qk_region_id": "first", "pv_region_id": "last", "softmax_region_id": "normalization"}
    ]


def test_upstream_contraction_is_not_paired_twice():
    inventory = graph()
    other = operation(5, "batch_matmul", "another_consumer", (3,))
    other["ins"].append({"source": ("arg", 4)})
    inventory["ops"].append(other)
    assert find_attention_pairs(inventory) == [
        {"qk_region_id": "first", "pv_region_id": "last", "softmax_region_id": "normalization"}
    ]
