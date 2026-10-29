"""Native product byte bounds follow the case's declared operands, with the fixed bound as floor."""

import json
from types import SimpleNamespace

from merlin_experiments.phase1.feedback import declared_bounds as B
from merlin_experiments.phase1.feedback import rolling_scalability as R

FLOOR = 32 << 20
LM_HEAD = {"inputs": [{"name": "W", "shape": [2048, 32000], "dtype": "i8"},
                      {"name": "A0", "shape": [1, 2048], "dtype": "i8"}]}


def test_small_or_undeclared_cases_keep_the_fixed_floor():
    assert B.result_bytes({}, FLOOR) == FLOOR
    assert B.result_bytes({"inputs": [{"shape": [4, "n"], "dtype": "i8"}]}, FLOOR) == FLOOR
    assert B.binary_bytes(None, 64 << 20) == 64 << 20
    assert B.result_bytes({"inputs": [{"shape": [16, 16], "dtype": "i8"}]}, FLOOR) > FLOOR


def test_large_declared_operands_raise_the_bound_proportionally():
    elements = 2048 * 32000 + 2048
    # The observed LM-head result (163,862,061 bytes) and harness object (65,548,144 bytes) now fit.
    assert B.result_bytes(LM_HEAD, FLOOR) >= FLOOR + elements * len("-128,") > 163_862_061
    assert B.binary_bytes(LM_HEAD, 64 << 20) >= (64 << 20) + elements > 65_548_144
    wide = {"inputs": [{"shape": [1000], "dtype": "i32"}]}
    assert B.result_bytes(wide, 0) == B.MARGIN * 1000 * (len(str(-(1 << 31))) + 2)
    assert B._bits("bf16") == 16 and B._bits("f32") == 32 and B._bits("x8") is None


def test_rolling_observation_pins_a_result_larger_than_the_floor(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common, capsule_runner

    case = tmp_path / "case"
    case.mkdir()
    runs = tmp_path / "runs"
    result = runs / "runs/suite/big/capsule_result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"capsule": "big", "pad": "x" * 64}))
    monkeypatch.setattr(R, "RESULT_MAX_BYTES", 16)  # a floor the result exceeds
    monkeypatch.setattr(capsule_common, "load_capsule", lambda d: {"inputs": [{"shape": [64], "dtype": "i8"}]})
    monkeypatch.setattr(capsule_runner, "suite_for", lambda target: "suite")
    from merlin_experiments.phase1.feedback import joint_performance

    monkeypatch.setattr(joint_performance, "attach", lambda projection, *a, **k: None)
    feedback = R.PerformanceFeedback.__new__(R.PerformanceFeedback)
    feedback.selection = SimpleNamespace(cases={"big": case}, context=SimpleNamespace(target="t"))
    feedback.compiler, feedback.runs_root, feedback.environment = tmp_path, runs, {}
    feedback.timeout, feedback.records, feedback.observed, feedback.bounds = 10.0, {}, {}, {}
    feedback.started, feedback.pool, feedback.pending = set(), None, {}
    import threading

    feedback.lock = threading.Lock()
    monkeypatch.setattr(joint_performance, "environment_scope", lambda env: _null())
    feedback.completed({"capsule": "big"})  # no projection: returns without refusing the size
    assert feedback.bounds["big"] > 16


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
