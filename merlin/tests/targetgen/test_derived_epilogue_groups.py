"""Phase 0 demands the fused epilogue a target's derived output path performs, capture or no capture."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import conformance as CF
from merlin.targetgen import corpus_synth as CS


def _oracle(derived=("acc_scale", "relu", "maxpool"), *, routed=True, dtypes=("int8",)):
    facet = SimpleNamespace(derived_stages=tuple(derived))
    routes = (SimpleNamespace(stage="bias_add", composed_with="contraction"),) if routed else ()
    return SimpleNamespace(
        readout=SimpleNamespace(facets=(facet,), scale_granularities=("tensor",)),
        routes=routes,
        cap_map={"contraction": SimpleNamespace(dtypes=dtypes)},
    )


def _writable(monkeypatch, stages=("bias_add", "acc_scale", "relu", "maxpool")):
    monkeypatch.setattr(CF, "_epilogue_axis", lambda _t: {"required": [{"stage": s} for s in stages]})


def test_the_derived_chain_is_demanded_per_contraction_format(monkeypatch):
    _writable(monkeypatch)
    rows = CF._derived_epilogue_groups("synthetic", _oracle(dtypes=("int8", "uint8")))
    chains = {(tuple(r["epilogue"]), r["dtype"]) for r in rows}
    assert chains == {
        (("bias_add", "acc_scale", "relu"), "int8"),
        (("bias_add", "acc_scale", "relu", "maxpool"), "int8"),
        (("bias_add", "acc_scale", "relu"), "uint8"),
        (("bias_add", "acc_scale", "relu", "maxpool"), "uint8"),
    }
    assert all(r["groups"] == 0 and r["observed_in"] == [] for r in rows)
    assert all(r["evidenced_by"] == ["derived_output_epilogue"] for r in rows)
    assert all(r["scale_granularities"] == ["tensor"] for r in rows)


def test_only_stages_a_builder_writes_and_a_route_licenses_are_chained(monkeypatch):
    _writable(monkeypatch, stages=("acc_scale", "relu"))
    rows = CF._derived_epilogue_groups("synthetic", _oracle())
    assert [r["epilogue"] for r in rows] == [["acc_scale", "relu"]]
    _writable(monkeypatch)
    assert [r["epilogue"] for r in CF._derived_epilogue_groups("synthetic", _oracle(routed=False))] == [
        ["acc_scale", "relu"],
        ["acc_scale", "relu", "maxpool"],
    ]


def test_nothing_is_demanded_without_a_derived_scale(monkeypatch):
    _writable(monkeypatch)
    assert CF._derived_epilogue_groups("synthetic", _oracle(derived=())) == []
    # A chain the output path cannot requantize is not a readout this axis demands.
    assert CF._derived_epilogue_groups("synthetic", _oracle(derived=("relu",))) == []


def test_a_derived_group_is_written_as_one_accelerator_member():
    root = merlin_dir() / "contract/capsules/conformance"
    doc = yaml.safe_load(next(iter(sorted(root.glob("*.yaml")))).read_text(encoding="utf-8"))
    dtype = next(str(c["dtype"]) for c in doc["cells"] if c.get("family") == "contraction")
    row = {
        "signature": f"matmul|acc_scale+relu|{dtype}",
        "op": "matmul",
        "epilogue": ["acc_scale", "relu"],
        "dtype": dtype,
        "scale_granularities": ["tensor"],
        "groups": 0,
        "observed_in": [],
        "evidenced_by": ["derived_output_epilogue"],
    }
    doc["groups"] = {"required": [row]}
    members = [
        e
        for e in CS.synthesize(doc)["capsules"]
        if (e.get("generalization") or {}).get("generalization_axis") == "groups"
    ]
    assert [m["epilogue"] for m in members] == [["acc_scale", "relu"]]
    assert members[0]["op"] == "matmul" and "acc_scale" in members[0]
    assert "derived output path" in members[0]["source_reference"]
