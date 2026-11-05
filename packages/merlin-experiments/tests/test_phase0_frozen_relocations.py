"""Explicit byte-identical RTL relocation stays bound across ordinary freezing."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import freeze
from merlin_experiments.phase0.evidence_rtl_sources import observe_sources
from test_phase0_evidence_relocations import _fixture


def plan_for(tmp_path, selected):
    path = tmp_path / "relocations.json"
    path.write_text(json.dumps(selected))
    key = "phase0:rtl-source-relocations-selection"
    return {
        "input_paths": {key: str(path)},
        "inputs": {key: {"path": str(path), "sha256": freeze.fingerprint(path)}},
        "phases": {
            "0": {
                "inputs": {"software_spec": "selected.yaml", "descriptor": "descriptor.yaml"},
                "phase0_evidence": {"status": "verified"},
                "phase0_rtl_source_relocations": copy.deepcopy(selected),
            }
        },
    }


def test_selection_forwards_the_actual_explicit_mapping(tmp_path, monkeypatch):
    _, selected, *_ = _fixture(tmp_path)
    plan = plan_for(tmp_path, selected)
    observed = {}

    def capture(_target, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(freeze, "select_evidence", capture)
    freeze.selection(plan["phases"]["0"], "fixture")
    assert observed["rtl_source_relocations"] == selected
    assert freeze._rtl_source_relocations(plan, plan["phases"]["0"]) == selected


@pytest.mark.parametrize("change", ["file", "mapping", "pin", "path"])
def test_changed_relocation_selection_refuses_before_any_source_staging(tmp_path, monkeypatch, change):
    _, selected, *_ = _fixture(tmp_path)
    plan = plan_for(tmp_path, selected)
    key = "phase0:rtl-source-relocations-selection"
    if change == "file":
        (tmp_path / "relocations.json").write_text("{}")
    elif change == "mapping":
        plan["phases"]["0"]["phase0_rtl_source_relocations"] = {}
    elif change == "pin":
        plan["inputs"][key]["sha256"] = "0" * 64
    else:
        plan["input_paths"][key] = str(tmp_path / "other.json")
    monkeypatch.setattr(freeze, "selection", lambda *_: pytest.fail("must refuse before selecting evidence"))
    with pytest.raises(ValueError, match="pinned input"):
        freeze.stage(plan)
    with pytest.raises(ValueError, match="pinned input"):
        freeze._verify_frozen_sources(plan)


@pytest.mark.parametrize("change", [None, "digest", "facts", "omitted", "unselected", "source"])
def test_frozen_evidence_must_bind_exact_mapping_and_facts(tmp_path, change):
    facts, selected, _, observe = _fixture(tmp_path)
    raw = json.dumps(facts).encode()
    facts_sha = hashlib.sha256(raw).hexdigest()
    actual = observe_sources(facts, observe, [], relocations=selected, facts_sha256=facts_sha)
    evidence = SimpleNamespace(views_json=json.dumps({"rtl_source_relocations": actual}), raw_facts_sha256=facts_sha)
    if change == "digest":
        next(iter(selected.values()))["sha256"] = "0" * 64
    elif change == "facts":
        evidence.raw_facts_sha256 = "0" * 64
    elif change == "omitted":
        evidence.views_json = b"{}"
    elif change == "unselected":
        selected = None
    elif change == "source":
        actual["sources"][0]["replacement_path"] = str(tmp_path / "foreign.py")
        evidence.views_json = json.dumps({"rtl_source_relocations": actual})
    if change is None:
        freeze._verify_relocations(evidence, selected)
        freeze._verify_relocations(SimpleNamespace(views_json=b"{}"), None)
    else:
        with pytest.raises(ValueError, match="relocation"):
            freeze._verify_relocations(evidence, selected)
