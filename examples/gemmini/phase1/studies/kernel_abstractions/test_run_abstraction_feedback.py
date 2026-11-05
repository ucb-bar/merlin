"""Study wiring for advisory admission/seed-fallback fields and author-named shape sweeps."""
import importlib.util
from pathlib import Path

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location("test_af_" + name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


R = module("run")
P = module("prepare_runs")
SELECTED = {"schema": "private.kernel_abstractions.abstraction_feedback.v1", "seed_fallback": True,
            "coverage_sweep": {"max_shapes": 3, "numerical": False}}


def request(**fields):
    return {"suite": "scalability", "sim": "spike", "capsules": "PF2_projection_m16k2048n2048",
            "coverage_sweep": "m=24,m=64:k=1024", **fields}


def test_sweep_requests_need_the_prepared_selection_and_one_training_template(monkeypatch):
    monkeypatch.setattr(R, "CASES", {"PF2_projection_m16k2048n2048"}, raising=False)
    monkeypatch.setattr(R, "C", {}, raising=False)
    with pytest.raises(ValueError, match="not enabled"):
        R.sweep_request(request())
    monkeypatch.setattr(R, "C", {"abstraction_feedback": SELECTED})
    assert R.sweep_request(request()) == ("PF2_projection_m16k2048n2048", [{"m": 24}, {"m": 64, "k": 1024}], False)
    for bad, match in ((request(capsules="PF2_projection_m24k2048n2048"), "public training case"),
                       (request(capsules="PF2_projection_m16k2048n2048,PF2_projection_m16k1024n2048"), "exactly one"),
                       (request(coverage_sweep="m=1,m=2,m=3,m=4"), "1 to 3"),
                       (request(coverage_sweep_numerical=True), "functional checks"),
                       (request(sim="gsim"), "spike")):
        with pytest.raises(ValueError, match=match):
            R.sweep_request(bad)
    monkeypatch.setattr(R, "C", {"abstraction_feedback": {"schema": "other"}})
    with pytest.raises(ValueError, match="unknown"):
        R.sweep_request(request())


def test_seed_fallback_is_absent_for_configurations_without_the_selection(monkeypatch):
    monkeypatch.setattr(R, "C", {}, raising=False)
    assert R.seed_selection() is None


def test_resize_uses_the_study_dense_resizer_for_named_extents():
    cb = {
        "tensors": {"A": {"shape": [16, 64]}, "W": {"shape": [64, 32]}, "Y": {"shape": [16, 32]}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"dst": "w", "src": "W"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "w", "dst": "p"}},
            {"opcode": "COMMIT", "operands": {"src": "p", "dst": "Y"}},
        ],
    }
    resized = R.resize_case(cb, {"m": 24, "n": 48})
    assert resized["tensors"]["A"]["shape"] == [24, 64] and resized["tensors"]["W"]["shape"] == [64, 48]
    assert resized["tensors"]["Y"]["shape"] == [24, 48] and cb["tensors"]["A"]["shape"] == [16, 64]
    with pytest.raises(ValueError, match="accepts"):
        R.resize_case(cb, {"batch": 2})


def test_preparation_selection_and_task_note_are_neutral():
    assert P.ABSTRACTION_FEEDBACK["schema"] == SELECTED["schema"]
    note = P.ABSTRACTION_NOTE
    assert "abstraction_admission" in note and "fallback_to_seed" in note and "--coverage-sweep" in note
    assert "schedule_dense" not in note and "must" not in note
