"""The shape-schedule-selection author skill: the choice table is honest and the worked example verifies.

``shape_choice_table.py`` runs on a small synthetic knob file, case list, facts body and results file:
classes are derived structurally, each class is held out against the others, unknown or failing
results stay unranked, aliases merge, and the capacity predicate that decides the synthetic winners is
reported. ``capacity_pipeline_example.py`` must build its three plans from capacity facts and reproduce
the source product from every emitted kernel; a smaller accumulator must shrink the capacity-limited
blocks.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest
from author_skill_fixtures import assert_generic_skill, front_matter, load_script, skill_dir

SKILL = "shape-schedule-selection"
TABLE = skill_dir(SKILL) / "scripts" / "shape_choice_table.py"


def _facts(operand_rows=256, accumulator_rows=64, edge=4):
    """A bare facts body: one array, an i8 operand store and an i32 accumulator store."""
    return {"arrays": [{"name": "array", "rows": edge, "cols": edge}],
            "memories": [{"name": "operand_sram", "banks": 1, "depth": operand_rows, "bytes": operand_rows * edge},
                         {"name": "acc_sram", "banks": 1, "depth": accumulator_rows,
                          "bytes": accumulator_rows * edge * 4}],
            "datapaths": [{"name": "operand_sram", "dtype": "i8"}, {"name": "acc_sram", "dtype": "i32"}]}


CASES = [{"case": "sq16", "m": 16, "k": 16, "n": 16}, {"case": "sq32", "m": 32, "k": 32, "n": 32},
         {"case": "longm", "m": 128, "k": 16, "n": 8}, {"case": "longn", "m": 8, "k": 16, "n": 128},
         {"case": "m1", "m": 1, "k": 64, "n": 64}, {"case": "rag", "m": 13, "k": 20, "n": 30},
         {"case": "layer", "m": 64, "k": 32, "n": 16, "model": True},
         {"case": "sq16_alias", "operand_shapes": [[16, 16], [16, 16]]}]


def _cycles(case, block, transfer):
    """Synthetic costs: wide panels win while the output fits the accumulator, retained RHS once it does not."""
    m, k, n = case["m"], case["k"], case["n"]
    fits = m * math.ceil(n / 4) <= 64  # rows of the synthetic accumulator
    factor = {"baseline": 1.0, "retain-rhs": 1.1 if fits else 0.8, "wide-panels": 0.9 if fits else 1.2}[block]
    return round((m * k * n / 16 + 100) * factor * (0.97 if transfer == "coalesced" and n % 32 == 0 else 1.0))


def _workspace(tmp_path):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "TUNING_KNOBS.json").write_text(json.dumps({
        "schema": "merlin.tuning_knobs.v1", "settings_file": "tuning_settings.json",
        "knobs": {"block_policy": {"values": ["baseline", "retain-rhs", "wide-panels"]},
                  "transfer_policy": {"values": ["baseline", "coalesced"]}}}))
    (tmp_path / "submission" / "tuning_settings.json").write_text(
        json.dumps({"block_policy": "baseline", "transfer_policy": "baseline"}))
    (tmp_path / "facts.json").write_text(json.dumps(_facts()))
    (tmp_path / "cases.json").write_text(json.dumps(CASES))
    rows = []
    for case in CASES[:-1]:
        for block in ("baseline", "retain-rhs", "wide-panels"):
            for transfer in ("baseline", "coalesced"):
                unknown = case["case"] == "rag" and block == "wide-panels"
                rows.append({"case": case["case"], "settings": {"block_policy": block, "transfer_policy": transfer},
                             "cycles": None if unknown else _cycles(case, block, transfer),
                             "basis": "modeled_roi", "domain": "synthetic-model-v1",
                             "numerics": not (case["case"] == "m1" and (block, transfer) == ("wide-panels",
                                                                                               "coalesced"))})
    # One row of another knob family and one of another timing domain are set aside, not mixed in.
    rows.append({"case": "sq16", "settings": {"block_policy": "baseline", "unknown_knob": 1}, "cycles": 5})
    rows.append({"case": "sq16", "settings": {"block_policy": "baseline", "transfer_policy": "baseline"},
                 "cycles": 1, "basis": "measured_roi", "domain": "other-sim"})
    (tmp_path / "results.json").write_text(json.dumps({"schema": "merlin.author_shape_results.v1", "rows": rows}))
    return tmp_path


def _run(ws, *extra):
    return subprocess.run([sys.executable, "-B", str(TABLE), "--knobs", "submission/TUNING_KNOBS.json", "--cases",
                           "cases.json", "--results", "results.json", "--facts", "facts.json", *extra],
                          cwd=ws, capture_output=True, text=True, timeout=300)


def test_skill_is_generic_bounded_and_points_at_its_files():
    assert_generic_skill(SKILL, allowed_imports=frozenset({"merlin"}))
    fields, body = front_matter(SKILL)
    assert "shape_choice_table.py" in fields["description"] and "capacity_pipeline_example.py" in body
    for reference in ("reference/choice-table.md", "reference/capacity-pipeline.md", "reference/shape-policy.md"):
        assert reference in body and (skill_dir(SKILL) / reference).is_file()
    for symbol in ("ExplorationGeometry", "ValidationPartition", "explore_candidates", "rank_observed_metrics",
                   "evaluate_variants", "schedule_ir", "loop_schedule", "dense_panel_pipeline",
                   "dense_array_retention", "dense_readout_overlap", "merlin.compile.capacity",
                   "merlin.transforms.shape_policy", "ResourcePredicate", "assess_shape_policy"):
        assert symbol in body, symbol


def test_the_table_ranks_each_class_against_the_others_and_names_the_capacity_cutoff(tmp_path):
    ws = _workspace(tmp_path)
    proc = _run(ws, "--json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["basis"] == "modeled_roi" and report["domain"] == "synthetic-model-v1" and report["edge"] == 4
    classes = {row["class"]: row for row in report["classes"]}
    assert set(classes) == {"square", "long-m", "long-n", "m1", "ragged", "model-layer"}
    assert classes["square"]["cases"] == ["sq16", "sq32"]
    for row in classes.values():
        transfer = row["trained_on_other_classes"]
        assert len(transfer["partition_sha256"]) == 64 and transfer["winner"]
    # Long N overruns the accumulator: retained RHS wins there, as on most other classes. One LHS row
    # fits it: wide panels win, which the other classes would not have picked.
    assert classes["long-n"]["best"].startswith("block_policy=retain-rhs")
    assert classes["long-n"]["trained_on_other_classes"]["holds"] is True
    assert classes["m1"]["best"] == "block_policy=wide-panels,transfer_policy=baseline"
    assert classes["m1"]["trained_on_other_classes"]["holds"] is False
    # Unknown and failing cells keep a candidate unranked instead of counting as zero.
    assert any(name.startswith("block_policy=wide-panels") for name in classes["ragged"]["unknown"])
    assert "block_policy=wide-panels,transfer_policy=coalesced" in classes["m1"]["unknown"]
    assert ("m1", "block_policy=wide-panels,transfer_policy=coalesced") in [
        tuple(x) for x in report["set_aside"]["failing_numerics"]]
    cases = {row["case"]: row for row in report["cases"]}
    assert cases["sq16"]["aliases"] == ["sq16_alias"] and "sq16_alias" not in cases
    assert cases["longn"]["predicates"]["output_fits"] is False and cases["sq16"]["predicates"]["output_fits"] is True
    assert report["set_aside"]["other_knob_settings"] == 1 and report["set_aside"]["other_domains"] == 1
    separating = {(s["knob"], s["predicate"]): s for s in report["separating_predicates"]}
    cutoff = separating[("block_policy", "output_fits")]
    assert (cutoff["when_true"], cutoff["when_false"]) == ("wide-panels", "retain-rhs")
    # The cutoff is stated as the ResourcePredicate guards shape_policy consumes, with boundary coverage.
    assert cutoff["guards"]["when_true"]["predicate"] == {"resource": "accumulator_output_copies", "at_least": 1}
    assert cutoff["guards"]["when_false"]["predicate"] == {"resource": "accumulator_output_stores_needed",
                                                           "at_least": 2}
    assert all(not side["missing_boundary_values"] for side in cutoff["guards"].values())
    assert cases["sq16"]["resources"]["accumulator_output_copies"] == 1  # 16 rows x 4 tiles = the 64-row store
    assert cases["longn"]["resources"]["accumulator_output_stores_needed"] == 4
    assert classes["square"]["trained_on_other_classes"]["held_out_max_ratio"] > 1  # regresses on sq16
    assert report["performance_qualification"] is False and report["unmeasured_candidates"] == []


def test_a_panel_without_the_exact_boundary_case_is_reported(tmp_path):
    ws = _workspace(tmp_path)
    (ws / "cases.json").write_text(json.dumps([c for c in CASES if not c["case"].startswith("sq16")]))
    report = json.loads(_run(ws, "--json").stdout)
    cutoff = {(s["knob"], s["predicate"]): s for s in report["separating_predicates"]}[("block_policy",
                                                                                          "output_fits")]
    assert cutoff["guards"]["when_true"]["missing_boundary_values"] == [1]  # no case fills the store exactly
    assert cutoff["guards"]["when_false"]["missing_boundary_values"] == []
    assert report["set_aside"]["unmatched_cases"] == ["sq16"]
    assert "needs a case with accumulator_output_copies = 1" in _run(ws).stdout


def test_the_guards_drive_assess_shape_policy_and_its_report_reads_back(tmp_path):
    """Table resources become PolicyFacts.resources, its guard a ShapeScheduleClass; the report reads back."""
    from dataclasses import replace

    from merlin.transforms.certificates import parse_ir
    from merlin.transforms.evaluated_exploration import (
        CompletedObservation,
        ExplorationIdentity,
        exploration_context_identity,
    )
    from merlin.transforms.pass_exploration import ExplorationGeometry, ValidationPartition
    from merlin.transforms.pipeline_policy import PipelineVariant, PolicyFacts, ResourcePredicate
    from merlin.transforms.pipeline_state import PipelineContext, PipelineSession
    from merlin.transforms.shape_policy import ShapeCaseFeedback, ShapeScheduleClass, assess_shape_policy

    table = load_script(SKILL, "shape_choice_table")
    ws = _workspace(tmp_path)
    report = json.loads(_run(ws, "--json").stdout)
    cutoff = {(s["knob"], s["predicate"]): s for s in report["separating_predicates"]}[("block_policy",
                                                                                          "output_fits")]
    guard = ResourcePredicate(**cutoff["guards"]["when_false"]["predicate"])
    context = PipelineContext(hardware="facts", resources="derived-stores", library="selected")
    session = PipelineSession(parse_ir("builtin.module { func.func @f() { func.return } }"), context=context)
    identity = replace(ExplorationIdentity(*("1" * 64 for _ in range(9)), exploration_context_identity(context),
                                           "xdsl", "modeled_roi"), source_sha256=session.stamp().source_sha256)
    feedback, classes = {}, {}
    for row in report["cases"]:
        m, k, n = row["gemm"]
        facts = PolicyFacts(session.stamp(), shapes=((m, k), (k, n)),
                            resources=tuple(sorted(row["resources"].items())))
        control = CompletedObservation(identity, "2" * 64, "3" * 64, "pass", "complete", True, 1000)
        fits = row["predicates"]["output_fits"]
        retained = replace(control, program_sha256="4" * 64, cycles=1100 if fits else 800)
        geometry = ExplorationGeometry("dense", facts.shapes)
        feedback[row["case"]] = ShapeCaseFeedback(geometry, facts, (("original", control), ("retain", retained)),
                                                  "5" * 64)
        classes.setdefault(row["class"], []).append(geometry)
    held = tuple(classes["long-n"])
    training = tuple(g for name, gs in classes.items() if name != "long-n" for g in gs)
    group = ShapeScheduleClass("overflows_accumulator", (guard,), "original", ValidationPartition(training, held))
    paths = (PipelineVariant("original", ("emit",)), PipelineVariant("retain", ("normalize", "emit")))
    assessed = assess_shape_policy((group,), paths, tuple(feedback.values()), fallback="original")
    assert assessed.selected[0][1] == "retain"  # boundary values 2 and 1 are both in the panel
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(assessed.report, default=list))
    text = table.render_shape_policy(json.loads(path.read_text()))
    assert "overflows_accumulator: empirically_supported_declared_panel; selected retain" in text
    proc = subprocess.run([sys.executable, "-B", str(TABLE), "--shape-policy", str(path)], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0 and "retain" in proc.stdout and "supported" in proc.stdout, proc.stderr


def test_completed_observation_rows_follow_its_eligibility():
    table = load_script(SKILL, "shape_choice_table")
    identity = {"basis": "measured_roi", "provider_sha256": "a" * 64, "facts_sha256": "b" * 64,
                "library_sha256": "c" * 64, "scope_sha256": "d" * 64}
    base = {"identity": identity, "program_sha256": "e" * 64, "numerical_status": "pass",
            "timing_status": "complete", "complete_roi": True, "cycles": 500}
    complete = table._row({"case": "A", "observation": base, "implementation_sha256": "f" * 64}, "x")
    assert complete["cycles"] == 500 and complete["basis"] == "measured_roi" and complete["numerics"] is True
    assert "a" * 64 in complete["domain"] and "f" * 64 in complete["domain"]
    for change in ({"timing_status": "unknown"}, {"complete_roi": False}, {"numerical_status": "fail"}):
        row = table._row({"case": "A", "observation": {**base, **change}}, "x")
        assert row["cycles"] is None or row["numerics"] is False


def test_the_text_table_prints_classes_cases_and_the_cutoff(tmp_path):
    proc = _run(_workspace(tmp_path))
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    assert "class" in text and "holds-out" in text and "long-n" in text and "LROAS=" in text
    assert "block_policy: wide-panels where output_fits (2 cases), retain-rhs where not (5 cases)" in text


def test_self_check_responses_carry_their_knob_overrides(tmp_path):
    table = load_script(SKILL, "shape_choice_table")
    response = {"variant": {"kind": "knob_overrides", "tuning_overrides": {"block_policy": "retain-rhs"}},
                "per_capsule": [
                    {"capsule": "A", "tiers": {"L2": "pass"}, "barrier_tier": "L3", "pass": True,
                     "barrier_cycles": 1200, "barrier_timing_identity": {"engine": "sim", "simulator_sha256": "s"}},
                    {"capsule": "B", "tiers": {"L2": "pass"},
                     "performance": {"estimated_whole_soc_cycles": 900, "model_sha256": "m"}},
                    {"capsule": "C", "tiers": {"L2": "fail"}}]}
    (tmp_path / "resp.json").write_text(json.dumps(response))
    (tmp_path / "results.json").write_text(json.dumps({"responses": ["resp.json", {
        "response": {"per_capsule": [{"capsule": "A", "pass": True}]}, "settings": {"block_policy": "baseline"}}]}))
    rows = table.load_results(tmp_path / "results.json")
    by_case = {(r["case"], r["settings"].get("block_policy")): r for r in rows}
    assert by_case["A", "retain-rhs"]["basis"] == "measured_roi" and by_case["A", "retain-rhs"]["cycles"] == 1200
    assert by_case["B", "retain-rhs"]["basis"] == "modeled_roi" and by_case["B", "retain-rhs"]["cycles"] == 900
    assert by_case["C", "retain-rhs"]["numerics"] is False and by_case["C", "retain-rhs"]["cycles"] is None
    assert by_case["A", "baseline"]["cycles"] is None
    with pytest.raises(SystemExit, match="explicit settings"):
        table._response_rows({"variant": {"path": "variants/x"}, "per_capsule": []}, None)


def test_classes_are_structural_and_conv_geometry_keeps_stride():
    table = load_script(SKILL, "shape_choice_table")
    cases = [{"case": c, "class": None, "model": False, "gemm": g} for c, g in
             (("a", (1, 64, 64)), ("b", (30, 64, 64)), ("c", (128, 64, 32)), ("d", (32, 64, 128)), ("e", (64,) * 3))]
    assert [table.shape_class(c, 16) for c in cases] == ["m1", "ragged", "long-m", "long-n", "square"]
    one, _ = table._conv({"c": 3, "h": 8, "w": 8, "k": 16, "r": 3, "stride": 1, "pad": 1})
    two, gemm = table._conv({"c": 3, "h": 8, "w": 8, "k": 16, "r": 3, "stride": 2, "pad": 1})
    assert one[:2] == two[:2] and one != two and gemm == (16, 27, 16)


@pytest.fixture(scope="module")
def example():
    return load_script(SKILL, "capacity_pipeline_example")


def test_the_worked_example_builds_every_plan_from_capacity_and_reproduces_the_product(example):
    out = example.example()
    assert out["facts"].startswith("synthetic") and out["tile"] == (4, 4, 4)
    compact, retained, segmented = (out["pipelines"][k] for k in ("compact", "retained", "segmented"))
    # Same products everywhere; array retention replaces fresh pairs by STAY; readout overlap drops drains.
    assert compact["calls"]["compute"] == retained["panel_calls"]["compute"] == segmented["calls"]["compute"]
    assert retained["retained_pairs"] > 0 and retained["calls"]["stay"] == retained["retained_pairs"]
    assert retained["block"][1] == out["tiles"][1] and "reorder" in retained["loop_transforms"]
    assert {"schedule.plan", "schedule.block_contraction", "schedule.loop_transform"} <= set(
        retained["schedule_ir_ops"])
    assert segmented["block"][1] < out["tiles"][1] and segmented["overlapped_compute_pairs"] > 0
    assert segmented["disjoint_readout_epochs"] > 0 and segmented["epoch_drains"][1] < segmented["epoch_drains"][0]


def test_a_smaller_accumulator_shrinks_the_capacity_limited_blocks(example, tmp_path):
    shape = (31, 16, 63)
    blocks = {}
    for rows in (64, 32):
        path = tmp_path / f"facts{rows}.json"
        path.write_text(json.dumps(_facts(operand_rows=512, accumulator_rows=rows)))
        out = example.example(str(path), shape, ("retained", "segmented"))
        assert out["facts"] == str(path)
        blocks[rows] = {name: tuple(row["block"]) for name, row in out["pipelines"].items()}
        for name, block in blocks[rows].items():
            output_rows = block[0] * block[2] * out["tile"][0]
            assert output_rows <= (rows if name == "retained" else rows // 2), (rows, name, block)
    assert blocks[32] != blocks[64]


def test_both_scripts_answer_help_with_the_standard_library_only():
    for script in ("shape_choice_table.py", "capacity_pipeline_example.py"):
        proc = subprocess.run([sys.executable, "-B", "-I", str(skill_dir(SKILL) / "scripts" / script), "--help"],
                              capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0 and "usage" in proc.stdout.lower(), proc.stderr
