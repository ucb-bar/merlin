"""Original facts parser with neutral live-owner transport controls only."""

import hashlib
import json
from dataclasses import replace

import pytest
from merlin_experiments.phase1.feedback import instruction_admission_facts as F


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path):
    fir = tmp_path / "selected.fir"
    fir.write_text("circuit Neutral :\n")
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {
                    "target": "neutral",
                    "fir_sha256": _sha(fir),
                    "firrtl_inputs": [{"path": str(fir), "sha256": _sha(fir)}],
                },
                "facts": {"source": {"config": "SelectedConfig"}},
            }
        )
    )
    return F.InstructionFactsSelection(facts, _sha(facts), facts, _sha(facts), "SelectedConfig")


def _owner(monkeypatch):
    owner = object.__new__(F.C.PreparedInstructionSelection)
    object.__setattr__(owner, "service_selection", ("neutral",))
    calls = []
    monkeypatch.setattr(F.C.PreparedInstructionSelection, "require_facts", lambda self, binding: calls.append(binding))
    return owner, calls


def test_original_facts_are_reopened_before_and_after_same_live_owner(selected, monkeypatch):
    owner, calls = _owner(monkeypatch)
    with F.selected_facts(owner, selected) as current:
        assert current is owner and len(calls) == 1
        assert calls[0]["private_input"]["firrtl_sha256"] == _sha(selected.input_facts.parent / "selected.fir")
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize("change", ["facts", "firrtl", "selection", "symlink", "config"])
def test_drift_during_use_refuses(selected, monkeypatch, change):
    owner, calls = _owner(monkeypatch)
    with pytest.raises(ValueError):
        with F.selected_facts(owner, selected):
            if change == "facts":
                selected.input_facts.write_text("{}")
            elif change == "firrtl":
                (selected.input_facts.parent / "selected.fir").write_text("changed")
            elif change == "selection":
                object.__setattr__(selected, "input_sha256", "0" * 64)
            elif change == "symlink":
                original = selected.input_facts.read_bytes()
                selected.input_facts.unlink()
                other = selected.input_facts.with_name("other.json")
                other.write_bytes(original)
                selected.input_facts.symlink_to(other)
            else:
                object.__setattr__(selected, "config", "ForeignConfig")
    assert len(calls) == 1


def test_exception_still_runs_original_final_join(selected, monkeypatch):
    owner, calls = _owner(monkeypatch)
    with pytest.raises(RuntimeError, match="body failed"):
        with F.selected_facts(owner, selected):
            raise RuntimeError("body failed")
    assert len(calls) == 2


def test_original_post_join_refusal_is_not_ignored(selected, monkeypatch):
    owner, _ = _owner(monkeypatch)
    count = 0

    def join(self, binding):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("original live source changed")

    monkeypatch.setattr(F.C.PreparedInstructionSelection, "require_facts", join)
    with pytest.raises(ValueError, match="original live source changed"):
        with F.selected_facts(owner, selected):
            pass


def test_saved_receipts_and_foreign_typed_selections_cannot_issue_join(selected, monkeypatch):
    owner, _ = _owner(monkeypatch)
    with pytest.raises(ValueError, match="original live"):
        with F.selected_facts({"status": "PREPARED"}, selected):
            pytest.fail("receipt reached current work")
    with pytest.raises(ValueError, match="typed facts"):
        with F.selected_facts(owner, {"facts": str(selected.input_facts)}):
            pytest.fail("untyped selection reached current work")
    with pytest.raises(ValueError):
        replace(selected, public_sha256="0" * 64).verify(target="neutral")


def test_invalid_facts_refuse_before_prepare(selected, monkeypatch):
    selection = F.InstructionAdmissionSelection(
        "neutral",
        selected.input_facts,
        _sha(selected.input_facts),
        selected.input_facts.parent / "fresh",
        selected.input_facts.parent,
    )
    monkeypatch.setattr(F, "prepare_live", lambda _: pytest.fail("invalid facts launched host issuers"))
    with pytest.raises(ValueError):
        with F.prepare_for_facts(selection, replace(selected, input_sha256="0" * 64)):
            pass


def test_prepare_preserves_actual_same_owner(selected, monkeypatch):
    selection = F.InstructionAdmissionSelection(
        "neutral",
        selected.input_facts,
        _sha(selected.input_facts),
        selected.input_facts.parent / "fresh",
        selected.input_facts.parent,
    )
    owner, calls = _owner(monkeypatch)
    monkeypatch.setattr(F, "prepare_live", lambda _: owner)
    with F.prepare_for_facts(selection, selected) as current:
        assert current is owner
    assert len(calls) == 2
