"""Framework feedback selftest; synthetic observations claim no timing qualification."""

import json
from types import SimpleNamespace

import pytest

from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.experiments.phase1.feedback import joint_performance as P
from merlin.experiments.phase1.feedback import rolling_scalability as R
from merlin.targetgen import capsule_common


@pytest.fixture
def rolling_scene(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr(P, "deadline", lambda timeout: clock[0] + min(timeout, 10))
    monkeypatch.setattr(capsule_eligibility, "suite_for", lambda target: "suite")
    monkeypatch.setattr(capsule_common, "load_capsule", lambda p: {"name": p.name})
    cases = {name: tmp_path / "frozen" / name for name in ("small", "large", "tail")}
    selection = SimpleNamespace(cases=cases, context=SimpleNamespace(target="test"), verify=lambda: None)
    runs = tmp_path / "runs"
    receipts = {}
    for name in cases:
        path = runs / "runs" / "suite" / name / "capsule_result.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "capsule": name,
                    "numeric": {"status": "pass"},
                    "tiers": {"L2": {"status": "pass", "engine": "spike"}},
                    "status": "pass",
                }
            )
        )
        receipts[name] = path
    calls = []

    def attach(row, root, **kw):
        calls.append((root, kw))
        clock[0] += 3
        row["performance"] = {"status": "PROVISIONAL", "estimated_whole_soc_cycles": 123}
        row["compiler_diagnostics"] = {"status": "UNKNOWN"}
        row["private_should_not_project"] = "fixture"

    monkeypatch.setattr(P, "attach", attach)
    feedback = R.PerformanceFeedback(selection, compiler=tmp_path / "candidate", runs_root=runs, timeout=60)
    return SimpleNamespace(
        feedback=feedback, selection=selection, receipts=receipts, calls=calls, clock=clock, runs=runs
    )


def test_a_multi_mebibyte_result_is_observed_like_a_small_one(rolling_scene):
    s = rolling_scene
    path = s.receipts["large"]
    stored = json.loads(path.read_text())
    stored["padding"] = "x" * (3 << 20)
    path.write_text(json.dumps(stored))
    s.feedback.completed({"capsule": "large"})
    verdict = {"per_capsule": [{"capsule": "large", "status": "pass"}]}
    s.feedback.project(verdict)
    assert verdict["per_capsule"][0]["performance"]["status"] == "PROVISIONAL"
