"""Independent V3 source checks do not consult private shapes or implementations."""

import copy

import pytest
from merlin_experiments.phase0 import public_boundaries as B
from merlin_experiments.phase0.operator_exclusion import PublicationRefusal
from merlin_experiments.phase0.public_source_eligibility import require
from test_phase0_capture_runtime_reuse import functional_profile


def selected(tmp_path):
    path, entries = functional_profile(tmp_path)
    actual = copy.deepcopy(entries)
    for entry in actual:
        entry["loader"] = str(tmp_path / entry["loader"])
    return path, actual


def test_complete_original_public_construction_is_eligible_without_private_selection(tmp_path):
    path, entries = selected(tmp_path)
    record = require(entries, {"_synth_verification": {"status": "verified"}}, path, None)
    assert [row["name"] for row in record["members"]] == [entry["name"] for entry in entries]
    assert record["private_shape_disjointness"] == "not_claimed"
    assert record["execution"] == "not_established" and record["admission"].startswith("pending_")


@pytest.mark.parametrize("change", ["tag_only", "loader", "recipe", "context", "basis", "unverified"])
def test_tag_or_mutated_original_proof_never_grants_eligibility(tmp_path, change):
    path, entries = selected(tmp_path)
    profile = {"_synth_verification": {"status": "verified"}}
    if change == "tag_only":
        entries[0] = {"name": "tag", "public_boundary": {"schema": B.SCHEMA_V3}}
    elif change == "loader":
        from pathlib import Path

        Path(entries[0]["loader"]).write_text("# unrelated public source\n")
    elif change == "recipe":
        entries[0]["quant_recipe"] = {"substituted": True}
    elif change == "context":
        context = entries[0]["public_boundary"]["source_context"]
        (tmp_path / context["path"]).write_bytes(b"{}")
    elif change == "basis":
        entries[0]["public_boundary"]["source_execution_basis"]["operations"] = []
    else:
        profile["_synth_verification"]["status"] = "incomplete_diagnostic"
    with pytest.raises(ValueError):
        require(entries, profile, path, None)


@pytest.mark.parametrize("schema", [B.SCHEMA, B.SCHEMA_V2, "unrecognized"])
def test_mixed_or_historical_boundaries_still_require_private_exclusion(tmp_path, schema):
    path, entries = selected(tmp_path)
    entries.append({"name": "historical", "public_boundary": {"schema": schema}})
    with pytest.raises(PublicationRefusal):
        require(entries, {"_synth_verification": {"status": "verified"}}, path, None)
    legacy = entries[-1:]
    assert require(legacy, {}, None, object()) is None  # caller's ordinary live-owner check remains upstream
    assert require([], {}, None, None) is None


def test_actual_generation_dispatches_verified_independent_sources(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import generation
    from test_phase0_m2m_preflight import _generation_with_entries

    path, entries = selected(tmp_path)
    run, calls, _ = _generation_with_entries(monkeypatch, tmp_path, entries)
    profile = generation.load_profile(None)
    profile.update(
        _synth_verification={"status": "verified"},
        _claim_model_evaluation={
            "schema": "claim_model_evaluation_v1",
            "model_count": 0,
            "visibility": "owner_only_after_phase1_freeze",
            "public_capsules_emitted": 0,
        },
    )
    monkeypatch.setattr(generation, "load_profile", lambda *_a, **_k: profile)
    monkeypatch.setattr(generation, "_prepare_model_capture_entry", lambda entry, **_k: entry)
    assert run(synth_profile=path) == []
    assert calls == [entry["name"] for entry in entries]
