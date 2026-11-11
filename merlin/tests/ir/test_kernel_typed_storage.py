"""Public typed results and private byte scratch are independently derived."""

import copy
import json

import pytest
from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, StringAttr, i64
from xdsl.dialects.func import FuncOp

from merlin.frontends.kernel_typed_storage import checked_guard_costs, storage_spec


def fixture():
    public = {
        "buffers": [
            {"name": "root", "dtype": "bf16", "shape": [2, "n"], "effect": "readwrite"},
            {"name": "out", "dtype": "i64", "shape": ["n"], "effect": "write"},
        ]
    }
    profile = {
        "schema": "t2m.typed-storage-transport.v1",
        "byte_order": "little",
        "grants_semantics": False,
        "roots": [
            {
                "name": "root",
                "logical_dtype": "bf16",
                "logical_shape": [2, "n"],
                "storage_dtype": "i8",
                "storage_shape": [2, "n", 2],
                "element_bytes": 2,
                "effect": "readwrite",
            }
        ],
    }
    provenance = {"spec": public, "profile": "shared-source-word-buffer", "typed_storage_transport": profile}
    entry = FuncOp("test", ([], []))
    entry.attributes.update(
        {
            "prov.typed_storage_transport": StringAttr(json.dumps(profile)),
            "prov.memory_guard_bytes_per_result_element": IntegerAttr(48, i64),
            "prov.memory_guard_bytes_by_result_element": DenseArrayBase.from_list(i64, [48, 24]),
        }
    )
    return provenance, entry


def test_public_results_retain_their_types_and_distinct_guard_costs():
    provenance, entry = fixture()
    before = copy.deepcopy(provenance)
    storage = storage_spec(provenance)
    assert storage["buffers"][0]["shape"] == [2, "n", 2]
    assert storage["buffers"][0]["dtype"] == "i8"
    assert storage["buffers"][1] == provenance["spec"]["buffers"][1]
    assert checked_guard_costs(entry, provenance, 24) == [48, 24]
    assert provenance == before


@pytest.mark.parametrize(
    "change", ("width", "shape", "dtype", "order", "grant", "extra-root", "maximum", "vector", "marker", "missing")
)
def test_rehashed_storage_declarations_cannot_change_abi_or_scratch(change):
    provenance, entry = fixture()
    profile = provenance["typed_storage_transport"]
    root = profile["roots"][0]
    if change == "width":
        root["element_bytes"] = 4
    if change == "shape":
        root["storage_shape"] = [2, "n", 4]
    if change == "dtype":
        root["logical_dtype"] = "i16"
    if change == "order":
        profile["byte_order"] = "native"
    if change == "grant":
        profile["grants_semantics"] = True
    if change == "extra-root":
        profile["roots"].append(dict(root, name="absent"))
    if change == "maximum":
        entry.attributes["prov.memory_guard_bytes_per_result_element"] = IntegerAttr(24, i64)
    if change == "vector":
        entry.attributes["prov.memory_guard_bytes_by_result_element"] = DenseArrayBase.from_list(i64, [24, 24])
    if change == "marker":
        entry.attributes["prov.typed_storage_transport"] = StringAttr("{}")
    if change == "missing":
        provenance.pop("typed_storage_transport")
    with pytest.raises(ValueError, match="typed storage|scratch storage"):
        checked_guard_costs(entry, provenance, 24)
