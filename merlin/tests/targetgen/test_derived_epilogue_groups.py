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


def _output_path(*, preload=True, activation=True, pool=True, accumulate=True):
    readout = [
        *([{"kind": "activation", "function": "relu", "domain": "accumulator"}] if activation else []),
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {
            "kind": "scale",
            "dtype": "f32",
            "rounding": "half_even",
            "granularity": "per_command",
            "columns_per_scale": 4,
        },
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        *([{"kind": "pool", "reductions": ["max"], "padding": "unknown"}] if pool else []),
    ]
    accumulator = [
        *([{"kind": "preload", "dtype": "i32", "range": [-8, 7], "rounding": "none"}] if preload else []),
        *([{"kind": "accumulate", "dtype": "i32", "operand_scale": "none"}] if accumulate else []),
    ]
    return {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": accumulator,
        "readout": readout,
        "exits": {"accumulator": True, "requant": True},
    }


def test_the_declared_output_path_alone_decides_the_demanded_chains(monkeypatch):
    _writable(monkeypatch, stages=("bias_add", "acc_scale", "relu", "maxpool"))
    no_facets = _oracle(derived=(), routed=False)
    monkeypatch.setattr(CF, "_declared_output_path", lambda _t: _output_path())
    assert [r["epilogue"] for r in CF._derived_epilogue_groups("synthetic", no_facets)] == [
        ["bias_add", "acc_scale", "relu"],
        ["bias_add", "acc_scale", "relu", "maxpool"],
    ]
    monkeypatch.setattr(CF, "_declared_output_path", lambda _t: _output_path(preload=False, pool=False))
    assert [r["epilogue"] for r in CF._derived_epilogue_groups("synthetic", no_facets)] == [["acc_scale", "relu"]]
    monkeypatch.setattr(CF, "_declared_output_path", lambda _t: _output_path(activation=False, pool=False))
    assert [r["epilogue"] for r in CF._derived_epilogue_groups("synthetic", no_facets)] == [["bias_add", "acc_scale"]]


def test_a_capability_no_capsule_can_state_is_reported_unstated_not_dropped():
    unstated = CF._unstatable_path_stages(_output_path())
    assert set(unstated) == {"accumulate"}
    assert CF._unstatable_path_stages(_output_path(accumulate=False)) == {}
    unsupported = _output_path()
    unsupported["readout"][2]["dtype"] = "f16"
    assert "column_group_scale" in CF._unstatable_path_stages(unsupported)
    assert CF._unstatable_path_stages(None) == {}


def test_the_epilogue_axis_requires_what_the_output_path_performs(monkeypatch):
    from merlin.targetgen import eligibility as E
    from merlin.targetgen import isa_taxonomy as IT
    from merlin.targetgen import readout_facet as RF

    monkeypatch.setattr(E, "capability_map_for_target", lambda _t: {})
    monkeypatch.setattr(IT, "taxonomy_for_target", lambda _t: IT._unknown("target 'synthetic'", "no descriptor"))
    monkeypatch.setattr(RF, "epilogue_readouts", lambda _t: None)
    monkeypatch.setattr(RF, "epilogue_stage_routes", lambda _t: ())
    monkeypatch.setattr(CF, "_declared_output_path", lambda _t: _output_path())
    required = {row["stage"]: row["evidenced_by"] for row in CF._epilogue_axis("synthetic")["required"]}
    for stage in ("bias_add", "acc_scale", "relu", "maxpool"):
        assert required[stage] == ["derived_output_path"]
    assert "requant" not in required  # an integer shift is not on this path
