"""Derive hands scheme resolution the output path and composition facts the selected inputs establish.

WHY THIS EXISTS. A data-only selection's prepared contract carries no ``output_epilogue`` and sits in a run tree
beside no semantic probes, and ``fact_capabilities`` reports only the output path and producer placement for a
composition. Derive therefore resolved no scheme field that reads the operand width, the K-extension bound, the
weight-panel width or a column-block scale geometry, and every such field became an UNKNOWN row. The first real
verified derive on a spec with quantization schemes refused for that reason alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from merlin.targetgen import composition_facts, target_registry
from merlin.targetgen.rtl import output_epilogue, semantic_facts
from merlin.targetgen.spec_fact_drift import FactField
from merlin_experiments.phase0 import evidence


def _stub_derivation(monkeypatch, facts, proof=None, unknown=None):
    monkeypatch.setattr(
        composition_facts,
        "derive",
        lambda **_: {"facts": facts, "evidence": proof or {k: f"proof {k}" for k in facts}, "unknown": unknown or {}},
    )


def test_established_composition_facts_reach_the_fact_view(monkeypatch):
    _stub_derivation(
        monkeypatch,
        {"weight_panel_width": 64, "k_extension_within": {"max_k": 33, "partial_sum_bits": 32}, "operand_signed_bits": 8},
    )
    view = {"composition": {"output_path": {"readout": []}}}
    evidence._add_composition_facts(view, "t", {}, {}, {"compositions": [], "quantization": {}})
    assert view["composition"]["weight_panel_width"] == FactField((64,), True, ('"proof weight_panel_width"',))
    assert view["composition"]["max_k_extension"].values == (33,)
    assert view["composition"]["dtype"].values == ("i8",)


def test_a_fact_the_inputs_do_not_establish_is_left_absent(monkeypatch):
    _stub_derivation(monkeypatch, {}, unknown={"weight_panel_width": "no payload"})
    view = {"composition": {"output_path": None}}
    evidence._add_composition_facts(view, "t", {}, {}, {})
    assert set(view["composition"]) == {"output_path"}


def test_an_existing_composition_fact_is_never_replaced(monkeypatch):
    _stub_derivation(monkeypatch, {"weight_panel_width": 64})
    kept = FactField((16,), True, ("authored by the fact view",))
    view = {"composition": {"weight_panel_width": kept}}
    evidence._add_composition_facts(view, "t", {}, {}, {})
    assert view["composition"]["weight_panel_width"] is kept


def _provider(tmp_path):
    return SimpleNamespace(base=tmp_path)


def test_output_epilogue_reads_probes_beside_the_target_declarations(monkeypatch, tmp_path):
    (tmp_path / "contracts").mkdir()
    probes = tmp_path / "contracts" / semantic_facts.PROBES_FILE
    probes.write_text("schema: semantic_probes_v1\n")
    seen = {}

    def fake_semantic(target, *, probes_file, only):
        seen.update(target=target, probes_file=probes_file, only=tuple(only))
        return {"facts": {}}

    monkeypatch.setattr(target_registry, "resolve", lambda target: _provider(tmp_path))
    monkeypatch.setattr(semantic_facts, "derive", fake_semantic)
    monkeypatch.setattr(output_epilogue, "derive", lambda facts, *, contract, semantic_facts: {"semantic": semantic_facts})
    assert evidence._derived_output_epilogue("t", {}, {}) == {"semantic": {"facts": {}}}
    assert seen == {"target": "t", "probes_file": probes, "only": ("accumulate_on_load",)}


def test_absent_probes_file_is_not_invented(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(target_registry, "resolve", lambda target: _provider(tmp_path))
    monkeypatch.setattr(
        semantic_facts, "derive", lambda target, *, probes_file, only: seen.setdefault("probes_file", probes_file) or {}
    )
    monkeypatch.setattr(output_epilogue, "derive", lambda facts, *, contract, semantic_facts: {})
    evidence._derived_output_epilogue("t", {}, {})
    assert seen["probes_file"] is None


def test_a_failed_epilogue_derivation_is_an_underived_record_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(target_registry, "resolve", lambda target: _provider(tmp_path))
    monkeypatch.setattr(semantic_facts, "derive", lambda target, *, probes_file, only: {})

    def boom(facts, *, contract, semantic_facts):
        raise ValueError("bad facts")

    monkeypatch.setattr(output_epilogue, "derive", boom)
    record = evidence._derived_output_epilogue("t", {}, {})
    assert record == output_epilogue.underived("ValueError: bad facts")
