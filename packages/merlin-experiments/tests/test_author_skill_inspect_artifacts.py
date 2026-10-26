"""The inspect-artifacts skill: its documented layout matches the self-check that writes it, and its script works."""

from __future__ import annotations

import ast
import json
import os
import time

from inspect_skill_support import check_skill_layout, load_script, skill_dir, source

SKILL = "inspect-artifacts"
SELFCHECK = "packages/merlin-experiments/src/merlin_experiments/phase1/feedback/selfcheck.py"
BROKER = "packages/merlin-experiments/src/merlin_experiments/phase1/brokers/selfcheck.py"
RUNNER = "packages/merlin-experiments/src/merlin/targetgen/capsule_runner.py"


def test_layout_rules(tmp_path):
    check_skill_layout(SKILL, tmp_path)


def test_copied_artifacts_are_the_ones_the_selfcheck_copies():
    script = load_script(SKILL, "case_artifacts")
    tuples = [
        {elt.value for elt in node.elts}
        for node in ast.walk(ast.parse(source(SELFCHECK)))
        if isinstance(node, ast.Tuple) and node.elts and all(isinstance(e, ast.Constant) for e in node.elts)
    ]
    assert set(script.ARTIFACTS) in tuples
    assert set(script.ARTIFACTS) <= set(script.KINDS)


def test_documented_paths_come_from_the_broker_and_runner():
    broker, selfcheck, runner = source(BROKER), source(SELFCHECK), source(RUNNER)
    assert 'ws / "selfcheck_out" / "requests" / rid' in broker
    assert 'f"resp_{rid}.json"' in broker and 'f"done_{rid}"' in broker and "detached_<id>" in broker
    assert 'workspace / "selfcheck_out" / "requests" / rid' in selfcheck
    assert '"your_artifacts": own' in selfcheck and '/ "model_layers"' in selfcheck
    for key in ("barrier_tier", "barrier_status", "tiers", "numeric", "trace_check", "failure", "declined",
                "sim_console_tail", "first_mismatch", "your_value"):
        assert f'"{key}"' in selfcheck, key
    assert '"runs" / (suite or suite_for(target)) / cap["name"] / "capsule_result.json"' in runner
    for name in ("numeric_report.yaml", "capsule_result.json", "run_manifest.yaml", "instruction_trace.json"):
        assert f'"{name}"' in runner, name
    from aet.core.run_paths import RunPaths

    paths = RunPaths.from_run_dir(skill_dir(SKILL), skill_dir(SKILL))
    assert (paths.generated.name, paths.logs.name, paths.artifacts_dir.name) == ("generated", "logs", "artifacts")
    text = (skill_dir(SKILL) / "SKILL.md").read_text() + (skill_dir(SKILL) / "reference" / "layout.md").read_text()
    for documented in ("selfcheck_out/requests/RID/CASE/", ".qa_channel/resp_RID.json", "model_layers/runs/SUITE/CASE/",
                       "your_artifacts", "--attach RID"):
        assert documented in text, documented


def _case(root, name, *, opcodes, classes, loops, ops):
    root.mkdir(parents=True)
    (root / "command_buffer.json").write_text(json.dumps({
        "tensors": {"A": {"shape": [4, 4], "dtype": "i8", "role": "input"}},
        "commands": [{"opcode": op, "operands": {}, "attributes": {}} for op in opcodes]}))
    (root / "instruction_trace.json").write_text(json.dumps({
        "instructions": [{"index": i, "class": c} for i, c in enumerate(classes)],
        "summary": {"class_histogram": {c: classes.count(c) for c in set(classes)},
                    "counted_loops": {"functions": [{"function": "entry", "loops": [
                        {"header_block": h, "trip_count": t} for h, t in loops]}]}}}))
    lines = ["builtin.module {", "  llvm.func @entry() {"]
    lines += [f'    "{op}"() : () -> ()' for op in ops] + ["    llvm.return", "  }", "}"]
    (root / "lowered.llvm.mlir").write_text("\n".join(lines) + "\n")


def test_list_joins_each_request_copy_to_its_redacted_row(tmp_path):
    script = load_script(SKILL, "case_artifacts")
    ws = tmp_path / "ws"
    old = ws / "selfcheck_out" / "requests" / "1_a" / "C"
    new = ws / "selfcheck_out" / "requests" / "2_b" / "C"
    _case(old, "C", opcodes=["X"], classes=["K"], loops=[(1, 4)], ops=["llvm.add"])
    _case(new, "C", opcodes=["X", "Y"], classes=["K", "K"], loops=[(1, 8)], ops=["llvm.add", "llvm.mul"])
    for path in old.iterdir():
        os.utime(path, (time.time() - 100, time.time() - 100))
    (ws / ".qa_channel").mkdir()
    (ws / ".qa_channel" / "resp_2_b.json").write_text(json.dumps({
        "selfcheck_submission": "variants/v1",
        "per_capsule": [{"capsule": "C", "pass": False, "tiers": {"L1": "fail"}, "unlisted": "dropped",
                         "numeric": {"status": "fail", "mismatch_count": 3, "per_output": {}},
                         "failure": {"category": "functional_mismatch", "detail": "x" * 900}}]}))
    rows = script.copies(ws, "C")
    assert [row["request_id"] for row in rows] == ["2_b", "1_a"]
    verdict = rows[0]["verdict"]
    assert verdict["submission"] == "variants/v1" and verdict["numeric"] == {"status": "fail", "mismatch_count": 3}
    assert "unlisted" not in verdict and len(verdict["failure"]["detail"]) == 300
    assert rows[1]["verdict"] is None
    assert {f["name"]: f["kind"] for f in rows[0]["files"]} == {
        "command_buffer.json": "compiler output", "instruction_trace.json": "harness decode",
        "lowered.llvm.mlir": "compiler output"}
    assert script.main(["list", "C", "--ws", str(ws)]) == 0
    assert script.main(["list", "missing", "--ws", str(ws)]) == 1

    report = script.diff(old, new)
    assert report["files"] == {"command_buffer.json": "differs", "instruction_trace.json": "differs",
                               "lowered.llvm.mlir": "differs"}
    assert report["commands"]["opcode_delta"] == {"Y": [0, 1]} and report["commands"]["first_difference"] == 1
    assert report["trace"]["class_delta"] == {"K": [1, 2]}
    assert report["trace"]["loops"] == [["entry:bb1x4"], ["entry:bb1x8"]]
    assert report["lowered"]["op_delta"] == {"llvm.mul": [0, 1]} and report["lowered"]["functions"] == [["entry"]] * 2
    assert script.diff(old, old)["files"]["lowered.llvm.mlir"] == "identical"


def test_full_run_directories_are_read_through_generated(tmp_path):
    script = load_script(SKILL, "case_artifacts")
    run = tmp_path / "ws" / "selfcheck_out" / "model_layers" / "runs" / "suite" / "C"
    _case(run / "generated", "C", opcodes=["X"], classes=["K"], loops=[], ops=["llvm.add"])
    (run / "capsule_result.json").write_text("{}")
    rows = script.copies(tmp_path / "ws", "C")
    assert len(rows) == 1 and rows[0]["scope"].startswith("model-layer")
    assert {"capsule_result.json", "lowered.llvm.mlir"} <= {f["name"] for f in rows[0]["files"]}
    assert script.diff(run, run / "generated")["files"]["command_buffer.json"] == "identical"
