"""Failing test cases that share one redacted failure reach the author as one cluster with a reproduce command.

The details below are excerpts of real redacted verdict rows from a live joint run (operator paths and
the target's tool name anonymized): one elaboration error whose generated loop name varied, one link
error whose per-case object path and code offset varied (one long-named case's detail clipped in the
middle), and one host-compute veto whose block repeat count varied.
"""

from __future__ import annotations

import json

from merlin_experiments.phase1.feedback import brief, completed_grade, failure_clusters as FC
from merlin_experiments.phase1.feedback.qa import DETAIL_LIMIT
from merlin_experiments.phase1.feedback.visible_model_gate import publish

from merlin.targetgen.capsule_diagnostics import ELISION


def _elaboration(loop: str) -> str:
    return ("lower_interface_to_target rc=1: bwrap: Only the last --chdir option will take effect\n"
            f"target-opt: PrimitiveError: # sibling loops at this level are named '{loop}', so the cursor names "
            "no single one of them. Rename one, or address them by a path that distinguishes them -- guessing "
            "which was meant would rewrite the wrong loop, or every one of them.\n")


_LINK_HEAD = ("the emitted package did not LINK, so nothing could be run. This is a defect in what your compiler "
              "emitted (a missing or duplicate symbol, or an object built against the wrong ABI), not a simulator "
              "fault: link failed:\n/opt/toolchain/bin/../lib/gcc/riscv64-unknown-elf/#.#/../../../../"
              "riscv64-unknown-elf/bin/ld: /work/run/_qa_work/fruns_#/runs/t-capsule-bench/")
_LINK_TAIL = ("harness.c:(.text.startup.main+{offset}): undefined reference to `kernel_complete'\n"
              "collect2: error: ld returned # exit status\n")


def _link(name: str, offset: str) -> str:
    return f"{_LINK_HEAD}{name}/generated/harness.o: in function `main':\n" + _LINK_TAIL.format(offset=offset)


def _clipped_link(name: str, offset: str) -> str:
    """The diagnostic clipper's form: head and tail of the message around the elision marker."""
    return (f"{_LINK_HEAD}{name[:20]} {ELISION} {name[-12:]}/generated/harness.o: in function `main':\n"
            + _LINK_TAIL.format(offset=offset))


def _host(repeat: int) -> str:
    return ("the program passes numerics and spends its work on the host: placed on accelerator and the host still "
            "works at element scale: # host arithmetic operations per element (budget #). Dominant block: "
            f"'(add) x2 (udiv) x2 (urem) x2 mul sub mul' x{repeat} (function 'kernel_entry').")


def _row(name, plane, category, detail, status="fail", **extra):
    return {"capsule": name, "status": status, "failure_plane": plane, "failure_category": category,
            "failure_detail": detail, "stderr": "RAW_OUTPUT_NOT_COPIED", **extra}


def _rows():
    rows = [_row(f"G_conv_{i}", "interface_to_target", "elaboration_error", _elaboration(loop))
            for i, loop in enumerate(["tile5"] * 4 + ["tile3"] * 2 + ["tile2"])]
    rows += [_row(f"G_matmul_m{i}k8n8_raw", "spike", "functional_mismatch", _link(f"G_matmul_m{i}k8n8_raw", off))
             for i, off in enumerate(("0x7a", "0x66", "0x4c"))]
    long_name = "MF_multimodal_policy_matmul_activation_stationary_m1_acc_scale_gain_lt_1_max"
    rows.append(_row(long_name, "spike", "functional_mismatch", _clipped_link(long_name, "0x4c")))
    rows += [_row(f"ZA_conv_{i}", "host_compute", "host_compute_in_accelerator_group", _host(n))
             for i, n in enumerate((225792, 451584))]
    rows += [
        _row("SY_crash", "spike", "tool_crash", "spike invocation failed: 'utf-8' codec can't decode byte"),
        _row("P_pass", None, None, None, status="pass"),
        _row("P_gated", "gate", "ADMISSION_DEFERRED", "background source superseded", status="gated"),
        _row("P_not_run", "oracle_unavailable", "NOT_RUN_IS_NOT_PASS", "no adapter", status="incomplete"),
    ]
    return rows


def test_cases_failing_the_same_way_form_one_cluster_each():
    section = FC.build(_rows())
    assert section["schema"] == FC.SCHEMA and section["failing_test_cases"] == 14
    by_category = {c["failure_category"]: c for c in section["clusters"]}
    assert [c["count"] for c in section["clusters"]] == [7, 4, 2, 1]
    elaboration = by_category["elaboration_error"]
    assert elaboration["examples"] == ["G_conv_0", "G_conv_1", "G_conv_2"]
    assert len(elaboration["test_cases"]) == 7 and "tile5" in elaboration["example_failure_detail"]
    assert elaboration["reproduce"] == (
        "python agent_selfcheck.py --submission submission --capsules G_conv_0 --wait-budget 5")
    # The clipped long-named case joins the link cluster it was clipped from.
    link = by_category["functional_mismatch"]
    assert link["count"] == 4 and any(name.startswith("MF_") for name in link["test_cases"])
    assert by_category["host_compute_in_accelerator_group"]["count"] == 2
    # Rows that did not fail (pass, gated, not run) are never clustered; raw fields are never copied.
    text = json.dumps(section)
    assert "P_pass" not in text and "P_gated" not in text and "P_not_run" not in text
    assert "RAW_OUTPUT_NOT_COPIED" not in text


def test_normalization_is_structural():
    a, _ = FC.normalize(_link("G_a1_raw", "0x7a"), "G_a1_raw")
    b, _ = FC.normalize(_link("G_bb22_raw", "0x66"), "G_bb22_raw")
    assert a == b and "harness.o:" in a and "/work/" not in a and "G_a1_raw" not in a
    assert FC.normalize("named 'tile5'")[0] == FC.normalize("named 'tile12'")[0]
    assert FC.normalize("x225792 at 0x4c")[0] == "x# at #"
    # A redacted case name (digits scrubbed by the grader) is still recognized as the case's own name.
    assert FC.normalize("case G_lt_#_max failed", "G_lt_1_max")[0] == "case <case> failed"
    assert FC.normalize("missing   output\n Y0")[0] == "missing output Y#"
    assert FC.normalize("one two")[0] != FC.normalize("one three")[0]


def test_a_detail_cut_at_the_redaction_cap_joins_the_message_it_was_cut_from():
    body = " ".join(["advice"] * ((DETAIL_LIMIT - 60) // 7))

    def detail(name):  # the grader's redaction keeps the first DETAIL_LIMIT characters
        return f"error in {name}: {body}"[:DETAIL_LIMIT]

    short, long = "A_short", "B_" + "long" * 20
    assert len(detail(short)) < DETAIL_LIMIT == len(detail(long))
    rows = [_row(short, "emit", "error", detail(short)), _row(long, "emit", "error", detail(long)),
            _row("C_other", "emit", "error", "error in C_other: unrelated")]
    clusters = FC.build(rows)["clusters"]
    assert [c["count"] for c in clusters] == [2, 1]
    assert clusters[0]["test_cases"] == [short, long]


def test_different_planes_or_categories_stay_apart():
    rows = [_row("A", "spike", "tool_crash", "same text"), _row("B", "emit", "tool_crash", "same text"),
            _row("C", "spike", "elaboration_error", "same text")]
    assert FC.build(rows)["clusters_total"] == 3
    assert FC.build([_row("P", None, None, None, status="pass")]) is None


def test_public_inventory_scopes_names_and_selects_the_suite_and_engine(tmp_path):
    from merlin_experiments.phase1.capability_inventory import FILENAME, SCALABILITY_COHORT, SCHEMA

    members = {
        "core_l3": {"cohort": "original_correctness", "development_engines": ["spike", "gsim"]},
        "spike_only": {"cohort": "calibration", "development_engines": ["spike"]},
        "scale": {"cohort": SCALABILITY_COHORT, "development_engines": ["spike"]},
    }
    (tmp_path / FILENAME).write_text(json.dumps({"schema": SCHEMA, "members": members}))
    l3_only = {"tiers": {"L2": "pass", "L3": {"status": "fail"}}}
    rows = [_row("core_l3", "rtl", "mismatch", "a", **l3_only), _row("spike_only", "rtl", "mismatch", "b", **l3_only),
            _row("scale", "spike", "mismatch", "c"), _row("HIDDEN_PRIVATE_CASE", "spike", "mismatch", "c")]
    section = FC.build(rows, workspace=tmp_path)
    commands = {c["examples"][0]: c["reproduce"] for c in section["clusters"]}
    assert "HIDDEN_PRIVATE_CASE" not in json.dumps(section) and section["failing_test_cases"] == 3
    assert commands["core_l3"].endswith("--capsules core_l3 --sim gsim --wait-budget 5")
    assert "--sim" not in commands["spike_only"]
    assert commands["scale"] == ("python agent_selfcheck.py --submission submission --suite scalability "
                                 "--capsules scale --wait-budget 5")


def test_listing_is_bounded_and_counts_what_it_leaves_out(tmp_path):
    rows = [_row(f"case_{i}", "emit", f"category_{i}", "x") for i in range(FC.LIMIT + 3)]
    (tmp_path / "selfcheck_out" / "case_0").mkdir(parents=True)
    (tmp_path / "selfcheck_out" / "case_0" / "command_buffer.json").write_text("{}")
    section = FC.build(rows, workspace=tmp_path)
    assert len(section["clusters"]) == FC.LIMIT and section["clusters_not_listed"] == 3
    inspect = [c["inspect"] for c in section["clusters"] if "inspect" in c]
    assert inspect == ["python isa_tools.py debug selfcheck_out/case_0/command_buffer.json --capsule case_0"]


def test_attach_sets_and_clears_the_section():
    verdict = FC.attach(None, {"per_capsule": _rows()})
    assert verdict[FC.KEY]["clusters"]
    assert FC.KEY not in FC.attach(None, {"per_capsule": [], FC.KEY: {"stale": True}})


def test_the_live_verdict_carries_the_clusters(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback.loop_grading import write_verdict

    monkeypatch.setattr(joint_control, "stage_evidence", lambda row: row)
    path = tmp_path / "ws" / "qa" / "verdict.json"
    path.parent.mkdir(parents=True)
    written = write_verdict(path, {"per_capsule": _rows(), "all_pass": False})
    assert json.loads(path.read_text())[FC.KEY] == written[FC.KEY]
    assert written[FC.KEY]["clusters"][0]["failure_category"] == "elaboration_error"


def test_latest_completed_grade_clusters_its_merged_rows(tmp_path):
    ws = tmp_path / "ws"
    (ws / "qa").mkdir(parents=True)
    (ws / "qa" / "verdict.json").write_text(json.dumps({"model_gate": {"source_sha256": None}, "all_pass": None}))
    older = {"per_capsule": [_row("G_conv_old", "interface_to_target", "elaboration_error", _elaboration("tile3"))],
             "graded_source_sha256": "old", "graded_at": "2026-10-25T19:00:00+00:00", "snapshot_current": False}
    newer = {"per_capsule": [_row("G_conv_new", "interface_to_target", "elaboration_error", _elaboration("tile5"))],
             "graded_source_sha256": "new", "graded_at": "2026-10-25T19:30:00+00:00", "snapshot_current": False}
    completed_grade.publish(ws, newer, 200, publish)
    completed_grade.publish(ws, older, 150, publish)
    latest = json.loads((ws / "qa" / completed_grade.FILENAME).read_text())
    (cluster,) = latest[FC.KEY]["clusters"]
    assert cluster["test_cases"] == ["G_conv_new", "G_conv_old"]
    assert cluster["newest_graded_at"] == "2026-10-25T19:30:00+00:00"


def test_the_round_briefs_show_or_point_to_the_clusters(tmp_path):
    run, ws = tmp_path / "run", tmp_path / "ws"
    (run / "qa_history").mkdir(parents=True)
    ws.mkdir()
    (run / "qa_history" / "verdict_round_1.json").write_text(json.dumps({"per_capsule": _rows()}))
    text = brief.build(run, ws, 1)
    section = text[text.index("## Failure clusters"):]
    assert "7 case(s), `interface_to_target` / `elaboration_error`" in section
    assert "Reproduce: `python agent_selfcheck.py --submission submission --capsules G_conv_0" in section
    assert "a helper" in section and "must" not in section.lower()

    from merlin_experiments.phase1.feedback.capability_brief import render

    inventory = {"schema": "merlin.public_capability_inventory.v1",
                 "members": {"G_conv_0": {"cohort": "original_correctness", "family": "conv"}}}
    joint = render(inventory, {"per_capsule": _rows()})
    assert "`failure_clusters` in both verdict files" in joint
