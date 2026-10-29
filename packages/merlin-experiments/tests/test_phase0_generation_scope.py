"""Functional selection preserves mandatory members and complete deferred claims."""

import pytest
from merlin_experiments.phase0.generation import _select_generation_scope


def test_functional_scope_preserves_all_nonperformance_members_and_claims():
    members = [
        {"cat": category, "name": str(index), "required": True}
        for index, category in enumerate(["isa", "layers", "model", "model_slices", "hidden", "_diagnostic"])
    ]
    performance = {
        "cat": "_perf",
        "name": "measurement",
        "performance": {"claim": "PREDICTS", "acceptance": {"replicates": 2}},
    }
    selected, deferred = _select_generation_scope(members + [performance], "phase1-functional")
    assert selected == members
    assert deferred == [performance]
    deferred[0]["performance"]["acceptance"]["replicates"] = 99
    assert performance["performance"]["acceptance"]["replicates"] == 2


def test_full_scope_preserves_historical_roster():
    entries = [{"cat": "_perf"}, {"cat": "hidden"}]
    selected, deferred = _select_generation_scope(entries, "full")
    assert selected is entries
    assert deferred == []


@pytest.mark.parametrize("scope", ["phase1", None, "", True])
def test_unknown_scope_refuses(scope):
    with pytest.raises(ValueError):
        _select_generation_scope([], scope)


def test_performance_without_claim_cannot_be_silently_deferred():
    with pytest.raises(ValueError, match="explicit claim"):
        _select_generation_scope([{"cat": "_perf"}], "phase1-functional")


def test_prepared_release_retains_and_rejects_changed_deferred_claims(tmp_path):
    import hashlib
    import json

    from merlin_experiments.corpus.preparation import retain_deferred_performance
    from merlin_experiments.spec import SpecError

    generated, prepared = tmp_path / "generated", tmp_path / "prepared"
    generated.mkdir()
    prepared.mkdir()
    ledger = {
        "generation_scope": "phase1-functional",
        "status": "deferred_not_prepared",
        "phase": "phase2",
        "entries": [{"cat": "_perf", "performance": {"claim": "PREDICTS"}}],
    }
    raw = json.dumps(ledger).encode()
    source = generated / "phase2-deferred.json"
    source.write_bytes(raw)
    provenance = {
        "performance_generation": {
            "fixture": {
                "preparation": {
                    "scope": "phase1-functional",
                    "status": "deferred_not_prepared",
                    "deferred_members": 1,
                    "ledger": source.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            }
        }
    }
    retain_deferred_performance(generated, prepared, provenance, target="fixture")
    assert (prepared / source.name).read_bytes() == raw
    source.write_bytes(raw + b" ")
    with pytest.raises(SpecError, match="changed"):
        retain_deferred_performance(generated, prepared, provenance, target="fixture")
