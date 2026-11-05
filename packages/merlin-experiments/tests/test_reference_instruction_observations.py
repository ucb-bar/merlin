"""Actual RV64/selected-ISA and emitted-IR contrasts survive the bounded broker reply."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from merlin_experiments.phase1.feedback.result_details import load_row
from merlin_experiments.phase1.tools import selfcheck as client
from test_reference_analysis_tool import _request, _world
from test_reference_machine_artifacts import observer

from merlin.verify.lean_contract import document_digest


def selected_world(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    fixture["facts"].write_text(
        json.dumps(
            dict(
                interfaces=[dict(name="funct_decode_table", custom_opcode=43, legal_funct=[4], names={"4": "RAW_NAME"})]
            )
        )
    )
    decoder = dict(
        semantics=observer(),
        target="array_fixture",
        structure=dict(instruction_roles={"FETCH_PANEL": ["operand_load"]}),
        decoder_inputs=[dict(path="selected_facts", sha256=hashlib.sha256(fixture["facts"].read_bytes()).hexdigest())],
    )
    monkeypatch.setattr(analysis, "_operand_decoder", lambda *args, **kwargs: decoder)
    return fixture


def listing(height):
    custom = 4 << 25 | 6 << 20 | 3 << 12 | 43
    return (
        "00000000 <kernel>:\n"
        f" 0: {height << 20 | 6 << 7 | 55:08x} lui t1, {height << 8}\n"
        f" 4: {custom:08x} .insn r 0x2b,3,4,x0,x0,t1\n"
        " 8: 00008067 ret\n"
    )


def test_fresh_machine_values_reach_real_analysis_and_detail_summary(tmp_path, monkeypatch):
    fixture = selected_world(tmp_path, monkeypatch)
    (tmp_path / "reference-kernels/kernel.dis").write_text(listing(4))
    candidate = tmp_path / 'artifacts' / 'kernel.dis'
    candidate.parent.mkdir()
    candidate.write_text(listing(16))
    request = _request(
        "compare",
        "reference-kernels/kernel.dis",
        symbol="kernel",
        limit=1,
        candidate=dict(path=(Path('artifacts') / 'kernel.dis').as_posix(), symbol="kernel"),
    )
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        response = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
    assert response["status"] == "analyzed", response
    report = response["report"]
    row = report["observation_comparison"]["field_differences"][0]
    assert (row["instruction"], row["field"]) == ("FETCH_PANEL", "height")
    assert row["reference"]["samples"] == [dict(value=4, sites=1)]
    assert row["candidate"]["samples"] == [dict(value=16, sites=1)]
    assert report["reference"]["regions"][0]["cca"]["instructions"]["fields"]
    assert report["semantic_compatibility"] == report["schedule_consumption"] == "unknown"
    # Force the same immutable detail publication and bounded summary used by
    # native transport, without compiling or grading a workload.
    large = copy.deepcopy(response)
    large["report"]["reference"]["transport_fixture"] = "x" * (80 << 10)
    large["report_identity"] = document_digest(large["report"])
    compact = analysis._bounded_response(large, workspace=tmp_path, pretty=True)
    assert len(json.dumps(compact, indent=2).encode()) < 64 << 10
    assert compact["report"]["observation_comparison"]["field_differences"][0] == row
    full = load_row(compact, workspace=tmp_path)
    assert full["report"]["observation_comparison"] == report["observation_comparison"]
    assert compact["semantic_equivalence"] == compact["performance"] == "unknown"
    assert compact["qualification_issued"] is False


def test_emitted_compiler_fields_use_the_same_observation_comparison(tmp_path, monkeypatch):
    fixture = selected_world(tmp_path, monkeypatch)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    for name, height in (("reference", 4), ("candidate", 16)):
        (directory / (name + ".mlir")).write_text(
            "module { llvm.func @kernel_entry() { "
            "%slot = llvm.mlir.constant(0 : i64) : i64\n"
            f"%height = llvm.mlir.constant({height << 20} : i64) : i64\n"
            '"llvm.inline_asm"(%slot, %height) <{asm_string = '
            '".insn r 0x2b, 3, 4, x0, $0, $1", constraints = "r,r", has_side_effects}> '
            ": (i64, i64) -> ()\nllvm.return } }"
        )
    request = _request(
        "compare",
        (Path('artifacts') / 'reference.mlir').as_posix(),
        format="emitted_mlir",
        observation_fields=["FETCH_PANEL.height"],
        candidate=dict(path=(Path('artifacts') / 'candidate.mlir').as_posix(), format="emitted_mlir"),
    )
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        result = analysis.handle(request, workspace=tmp_path)
    assert result["status"] == "analyzed", result
    comparison = result["report"]["observation_comparison"]
    assert comparison["compared_fields"] == 1
    assert comparison["field_differences"][0]["field"] == "height"
    assert comparison["count_basis"] == "static_ir_sites"
    assert comparison["semantic_alignment"] == comparison["performance"] == "unknown"


@pytest.mark.parametrize("fields", ["ISA.field", [""], ["x" * 513], ["ISA.field"] * 9])
def test_broker_refuses_invalid_field_selectors(tmp_path, fields):
    fixture = _world(tmp_path)
    request = _request("compare", "reference-kernels/trace.txt", observation_fields=fields,
                       candidate=dict(path=(Path('artifacts') / 'other.dis').as_posix()))
    with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
        result = analysis.handle(request, workspace=tmp_path)
    assert result["status"] == "unknown" and result["reason"] == "invalid_analysis_request"


def test_copied_stdlib_client_field_request_reaches_real_broker_and_returns_values(tmp_path, monkeypatch):
    fixture = selected_world(tmp_path, monkeypatch)
    (tmp_path / "reference-kernels/kernel.dis").write_text(listing(4))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "kernel.dis").write_text(listing(16))
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    assert shim.read_bytes() == Path(client.__file__).read_bytes()
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    argv = [
        sys.executable,
        str(shim),
        "--reference-analysis",
        "compare",
        "--reference-path",
        "reference-kernels/kernel.dis",
        "--reference-symbol",
        "kernel",
        "--candidate-path",
        (Path('artifacts') / 'kernel.dis').as_posix(),
        "--candidate-symbol",
        "kernel",
        "--reference-observation-field",
        "FETCH_PANEL.height",
        "--wait-budget",
        "5",
    ]
    process = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        requests = []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests
        request = json.loads(requests[0].read_bytes())
        assert request["reference_analysis"]["observation_fields"] == ["FETCH_PANEL.height"]
        rid = request["request_id"]
        with reference.invocation(fixture["path"], fixture["sha"], tasks=False):
            response = broker._reference_analysis_response(
                request,
                workspace=tmp_path,
                facts=fixture["facts"],
                compiler_sha256=fixture["compiler"],
                rid=rid,
                started_ns=time.time_ns(),
            )
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        result = json.loads(stdout)
        row = result["report"]["observation_comparison"]["field_differences"][0]
        assert row["reference"]["samples"][0]["value"] == 4
        assert row["candidate"]["samples"][0]["value"] == 16
        assert result["qualification_issued"] is False and result["semantic_equivalence"] == "unknown"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
