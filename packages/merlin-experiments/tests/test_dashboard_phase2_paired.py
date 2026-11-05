"""The paired Phase 2 view reads the real checkpoint controller's output.

The experiment comes from ``test_checkpoint_lifecycle.build_lifecycle`` (the controller, the paired
measurement and the statistics run for real with an injected engine), plus a trial's authoring records:
broker receipts, a tuning-feedback document with a roofline and executed commands, and an event stream
naming broker actions.  Held-out member names and operator-only reference ratios stay hidden by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dashboard_fixtures import T0, codex_stream, command, write
from merlin_experiments.tracking import phase2_paired, records, write_dashboard
from merlin_experiments.tracking.tail import Tails
from test_checkpoint_lifecycle import build_lifecycle


def _feedback(round_number: int) -> dict:
    return {
        "schema_version": 1,
        "kind": "host_owned_tuning_gsim_feedback",
        "round": round_number,
        "invocation": 1,
        "engine": "gsim",
        "cells": [
            {
                "family": "PK",
                "capsule": "case",
                "baseline_gsim_cycles": 120,
                "candidate_gsim_cycles": 100,
                "baseline_over_candidate": 1.2,
                "verdict": "improved",
                "measured": True,
                "roofline": {
                    "status": "derived",
                    "limiter": "movement",
                    "baseline_over_roofline": 1.875,
                    "candidate_over_roofline": 1.5625,
                },
                "executed_commands": {
                    "baseline": {
                        "status": "measured",
                        "accelerator_commands": 3,
                        "retired_instructions": 6,
                        "by_class": {"MOVE": 2, "WORK": 1},
                        "local_memory": {
                            "scratchpad_rows_high_water": 40,
                            "scratchpad_rows_capacity": 16384,
                            "accumulator_rows_high_water": 16,
                            "accumulator_rows_capacity": 1024,
                        },
                    },
                    "candidate": {"status": "unknown", "why": "no executed profile was recorded for this arm"},
                },
            }
        ],
    }


@pytest.fixture()
def experiment(tmp_path, monkeypatch):
    lifecycle = build_lifecycle(tmp_path, monkeypatch, form_holdout=True)
    lifecycle.run()
    stage = lifecycle.config.context.stage_root / "lifecycle__trial_00"
    control = stage / "control" / "round_00"
    doc = _feedback(0)
    write(control / "feedback" / "sha256" / ("a" * 64 + ".json"), doc)
    receipts = [
        {"index": 0, "action": "tuning-gsim-feedback", "state": "complete", "returncode": 0, "elapsed_s": 41.0},
        {
            "index": 1,
            "action": "profile-tuning-member",
            "state": "complete",
            "returncode": 0,
            "elapsed_s": 92.4,
            "bindings": {"member": "PK/case"},
        },
        {
            "index": 2,
            "action": "profile-whole-model",
            "state": "rejected",
            "returncode": 126,
            "elapsed_s": 0.0,
            "rejection_reason": "no whole-model inputs",
        },
    ]
    write(control / "receipts.jsonl", "".join(json.dumps(r) + "\n" for r in receipts))
    codex_stream(
        stage,
        0,
        T0,
        [
            {"type": "turn.started"},
            *command("p1", "python3 perf_tool.py profile-tuning-member --member PK/case", exit_code=0),
            *command(
                "p2",
                "python3 perf_tool.py analyze-command-buffers --baseline-json b.json --candidate-json c.json",
                exit_code=0,
            ),
            {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2}},
        ],
    )
    write(stage / "cost_time_toolcalls.yaml", "tokens_total: 123456\ntokens_output: 4567\ntool_calls: 12\n")
    cell = lifecycle.config.context.measurement_root / "lifecycle__trial_00__held_out_form"
    write(
        cell / "reference_comparison.json",
        {
            "schema": "merlin_perf_reference_comparison_v1",
            "visibility": "operator_only",
            "reference": {"label": "hand-written"},
            "rows": [
                {
                    "family": "PW",
                    "capsule": "form_case",
                    "replicate": "r000",
                    "candidate_cycles": 80,
                    "reference_cycles": 64,
                    "candidate_over_reference": 1.25,
                    "state": "compared",
                }
            ],
            "summary": {"cells": 1, "compared": 1, "geomean_candidate_over_reference": 1.25},
        },
    )
    return lifecycle


def test_the_cells_grid_carries_the_recorded_ratio_and_marks(experiment):
    paired = phase2_paired.summary(experiment.config.root, records.Inventory(), Tails())
    assert paired["form"] and paired["labels"] == ["tuning", "held_out", "held_out_form"]
    assert paired["done"] == paired["expected"] and paired["sealed"]
    cell = paired["cells"]["trial_00:tuning"]
    member = cell["members"]["PK/case"]
    assert [p["baseline_over_candidate"] for p in member["pairs"]] == [1.25, 1.25]
    assert cell["completion"]["complete"] and paired["statistics"]["aggregate"]["median_speedup"] == 1.25
    stage = paired["stages"]["trial_00"]
    assert stage["feedback"][0]["candidate_over_roofline"] == 1.5625
    assert [a["action"] for a in stage["actions"]] == ["profile-tuning-member", "analyze-command-buffers"]
    assert stage["tokens"]["tokens_total"] == 123456
    assert paired["cells"]["trial_00:held_out_form"]["reference"] == {"present": True, "cells": 1}


def test_public_page_hides_held_out_names_and_reference_ratios(experiment, tmp_path):
    page = Path(write_dashboard(run_dir=experiment.config.root, out=tmp_path / "p2.html")["dashboard"]).read_text(
        encoding="utf-8"
    )
    for expected in (
        "1.250x",
        "0.800</b> cand/base",
        "trial_00",
        "held_out_form",
        "2 reps",
        "Position against the roofline",
        "profile-tuning-member",
        "Holdout commit and reveal",
        "Measurement slots (reconstructed)",
        "MOVE 2",
        "40/16384",
        "123,456",
        "operator-only ratios are hidden",
        "Checkpoint chain",
        "held-out #1",
    ):
        assert expected in page, expected
    assert "form_case" not in page and "candidate/reference" not in page
    assert "http://" not in page and "<link" not in page


def test_operator_private_page_names_members_and_shows_reference_ratios(experiment, tmp_path):
    out = tmp_path / "p2-private.html"
    write_dashboard(run_dir=experiment.config.root, out=out, operator_private=True)
    page = out.read_text(encoding="utf-8")
    assert "form_case" in page and "candidate/reference" in page and "1.25" in page


def test_an_unstarted_experiment_reads_as_not_started(tmp_path):
    root = tmp_path / "experiment"
    (root / "state").mkdir(parents=True)
    write(
        root / "state" / ("checkpoint.0000." + "0" * 64 + ".json"), {"index": 0, "stage": "predeclared", "evidence": {}}
    )
    paired = phase2_paired.summary(root, records.Inventory(), Tails())
    assert paired["done"] == ["predeclared"] and paired["cells"] == {} and not paired["sealed"]
    page = Path(write_dashboard(run_dir=root, out=tmp_path / "p.html")["dashboard"]).read_text(encoding="utf-8")
    assert "not started" in page and "not recorded" in page
