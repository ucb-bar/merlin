"""Must-refuse capsules: a stated decline passes, an emitted program fails.

The operation such a capsule declares needs a stage the target's own evidence says no readout or
route performs. The fixture target below declares a readout that applies scale and activation and
no integer shift, so a ``requant`` member is exactly that case.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth as CSY
from merlin.targetgen import expected_refusal as ER
from merlin.targetgen import readout_facet as RF
from merlin.verify.epilogue_applicability import ReadoutCapability

_TARGET = "readout_without_shift_fixture"


@pytest.fixture(autouse=True)
def _no_shift_readout(monkeypatch):
    declared = RF.epilogue_readouts
    readouts = [ReadoutCapability(selector="i8", applies=frozenset({"acc_scale", "relu"}), evidence="fixture")]
    monkeypatch.setattr(RF, "epilogue_readouts", lambda target: readouts if target == _TARGET else declared(target))
    monkeypatch.setattr(RF, "epilogue_stage_routes", lambda target: ())


def _binding() -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target=_TARGET,
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1", "L2", "L3"],
        compare="exact_int",
        requant_output_dtype="i8",
        requant_shift=3,
        classes_for=lambda **_: [],
    )


def _entry(**over) -> dict:
    entry = {
        "name": "RF0",
        "kind": "layer",
        "op": "matmul",
        "M": 16,
        "K": 32,
        "N": 16,
        "epilogue": ["requant"],
        "source_role": "derived_sweep",
        "source_reference": "must-refuse test",
    }
    entry.update(over)
    return entry


_REFUSE = {
    "outcome": "refuse",
    "refusal": {"stage": "requant", "reason": "no readout applies the shift", "evidence": "fixture readout"},
}


def test_an_unperformable_stage_is_still_refused_without_the_declaration():
    with pytest.raises(ValueError, match="declares no readout"):
        CS.build(_entry(), _binding())


def test_a_must_refuse_entry_builds_and_says_why_on_the_capsule():
    capsule, interface = CS.build(_entry(**_REFUSE), _binding())
    assert ER.expects_refusal(capsule)
    assert capsule["expected"]["refusal"] == _REFUSE["refusal"]
    assert capsule["operation"]["attributes"]["output_dtype"] in ("i8", "int8")
    assert "requant" in interface
    schema = json.loads((merlin_dir() / "contract/schemas/capsule.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(capsule["expected"], schema["properties"]["expected"])


def test_a_must_refuse_entry_needs_its_reason_and_evidence():
    with pytest.raises(ValueError, match="evidence"):
        CS.build(_entry(outcome="refuse", refusal={"stage": "requant"}), _binding())
    with pytest.raises(ValueError, match="outcome"):
        CS.build(_entry(outcome="maybe", epilogue=["relu"]), _binding())


@pytest.mark.parametrize(
    ("status", "declined", "want"),
    [
        ("declined", {"reason": "integer shift not on this readout"}, "pass"),
        ("pass", None, "fail"),
        ("fail", None, "fail"),
        ("infrastructure_fault", None, "infrastructure_fault"),
        ("error", None, "error"),
    ],
)
def test_the_grade_inverts_only_for_a_must_refuse_capsule(status, declined, want):
    capsule = {"expected": {"instruction_classes": [], **_REFUSE}}
    got, failure, record = ER.verdict(capsule, status, {"category": "X"} if status != "declined" else None, declined)
    assert got == want
    if want == "fail":
        assert failure["category"] == ER.VIOLATED and record["status"] == "violated"
    if want == "pass":
        assert failure is None and record["status"] == "met" and record["declined"] == declined
    plain = {"expected": {"instruction_classes": []}}
    assert ER.verdict(plain, status, None, declined)[:2] == (status, None)


def test_a_rejected_stage_becomes_a_must_refuse_member_and_an_unresolved_one_does_not():
    spec = {
        "target": _TARGET,
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8", "alignment": "aligned"}],
        "boundaries": {"extent_probes": [{"boundary": "tile_edge", "edge": 16, "points": [1, 15, 16, 17, 32]}]},
        "epilogue": {
            "required": [],
            "rejected": [{"stage": "requant", "family": "elementwise_map", "why": "no readout applies it"}],
            "unresolved": [{"stage": "relu", "family": "elementwise_map", "why": "unread"}],
        },
    }
    members = [e for e in CSY.synthesize(spec)["capsules"] if e.get("outcome") == "refuse"]
    assert [e["name"] for e in members] == ["SY_refuse_requant"]
    (member,) = members
    assert member["refusal"]["stage"] == "requant" and member["refusal"]["evidence"] == "no readout applies it"
    assert member["stimulus_range"][0] < 0


def test_a_must_refuse_entry_is_never_demoted_by_its_expected_refusal():
    from merlin_experiments.phase0.program_admission import entry_refusal_is_final

    decision = {"status": "unsupported"}
    assert entry_refusal_is_final(_entry(), decision)
    assert not entry_refusal_is_final(_entry(**_REFUSE), decision)
