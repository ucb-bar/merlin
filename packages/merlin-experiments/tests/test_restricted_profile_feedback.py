"""Restricted author feedback retains diagnostics without withheld modern recipes."""
from __future__ import annotations

import copy
import json

import composition_fixtures as F
import pytest
from merlin_experiments.phase1.feedback import brief
from merlin_experiments.phase1.feedback import idle_tuning as U
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback.loop_grading import write_verdict
from test_idle_tuning import _performance
from test_performance_headline import _background, _workspace

PROFILES = ("xs-el-v1/el1", "xs-el-v1/el2", "xs-el-v1/el3")
FORBIDDEN = ("heavy-pass-recipes", "apply_library_pass", "parallel_schedule_recipe",
             "merlin.compile.scheduling", "roofline-and-utilization", "ab-library-pass",
             "pass-library", "<catalogue id>")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("passes", [False, True])
def test_actual_selfcheck_annotation_cannot_supply_modern_composition_routes(
    tmp_path, monkeypatch, profile, passes
):
    monkeypatch.setenv("MERLIN_AUTHOR_PROFILE", profile)
    F.declare(tmp_path, "layers", F.conv_pool("chain"))
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission/backend.py").write_text("import json\n")
    (tmp_path / "PUBLIC_CAPABILITIES.json").write_text(json.dumps({
        "schema": "merlin.public_capability_inventory.v1",
        "members": {"chain": {"workload_role": "connected"}},
    }))
    rows = [{"capsule": "chain", "pass": passes, "performance": _performance(400),
             **({} if passes else {"failure": {"category": "host_compute_in_accelerator_group"}})}]
    document = {"per_capsule": rows}
    original = copy.deepcopy(document)
    result = PB.attach(document, rows, workspace=tmp_path)
    assert result == original
    assert all(token not in json.dumps(result) for token in FORBIDDEN)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("passes", [False, True])
def test_actual_grade_publication_refreshes_brief_without_modern_guidance(
    tmp_path, monkeypatch, profile, passes
):
    monkeypatch.setenv("MERLIN_AUTHOR_PROFILE", profile)
    ws, verdict = _workspace(tmp_path)
    if not passes:
        for row in verdict["per_capsule"]:
            row["tiers"]["L2"] = "fail"
            row["status"] = "fail"
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    brief.refresh_before_launch(run, ws, 0)
    before = copy.deepcopy(verdict["per_capsule"])
    _background(ws, run, write_verdict)(verdict, 1793181600000000000)
    for relative in ("qa/latest_completed_grade.json", "qa/round_brief.md", "qa/latest_grade_summary.txt"):
        text = (ws / relative).read_text()
        assert all(token not in text for token in FORBIDDEN), relative
    stored = json.loads((ws / "qa/latest_completed_grade.json").read_text())
    assert stored["performance_headline"]["numerics"]["passing"] == (5 if passes else 0)
    for initial in before:
        after = next(row for row in stored["per_capsule"] if row["capsule"] == initial["capsule"])
        assert after["tiers"] == initial["tiers"]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("passes", [False, True])
def test_actual_idle_ledger_has_only_generic_alternative_instructions(
    tmp_path, monkeypatch, profile, passes
):
    monkeypatch.setenv("MERLIN_AUTHOR_PROFILE", profile)
    ws = tmp_path / "workspace"
    channel = ws / ".qa_channel"
    channel.mkdir(parents=True)
    (ws / "submission").mkdir()
    (ws / "submission/backend.py").write_text("print('authored fixture')\n")
    digest = "a" * 64
    known = {"case": [(1, digest, _performance(400), True)]} if passes else {}
    tuner = U.IdleTuner(ws)
    tuner.refresh(channel, digest, known, cohort=["case"], l2={"case": "pass" if passes else "fail"})
    ledger = json.loads((ws / U.LEDGER).read_text())
    assert ledger["progress"]["ready_baselines"] == int(passes)
    assert ledger["progress"]["declared_candidates"] == 0
    assert ledger["progress"]["construction_task"]["kind"] == "declare_alternatives"
    assert "<descriptive label>" in ledger["offer"]
    assert all(token not in json.dumps(ledger) for token in FORBIDDEN)
    assert not list(channel.glob("req_*.json")), "publication must not launch measurements"


def test_legacy_idle_guidance_remains_available(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_AUTHOR_PROFILE", raising=False)
    assert U.offer() == U.OFFER and "<catalogue id>" in U.offer()
    assert "parallel_schedule_recipe.py" in U.construction_task(tmp_path)["command"]
