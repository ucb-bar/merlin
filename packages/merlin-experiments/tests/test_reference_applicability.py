"""Bank-arm applicability over a real published CCA bank, the compare refusals and the client path."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_bank as bank_mode
from merlin_experiments.phase1.feedback import reference_experiments
from merlin_experiments.phase1.tools import selfcheck as client
from test_cca_bank_admission import RAW, _conditions, _occurrence, _report, _request, world  # noqa: F401


def _candidate(world):  # noqa: F811
    unit, region, folder = _occurrence(world["banks"]["cca-4"])
    workspace = world["run"]["workspace"]
    (workspace / "artifacts").mkdir(exist_ok=True)
    shutil.copyfile(world["repo"] / "banks/cca-4" / folder / "instructions.dis", workspace / 'artifacts' / 'kernel.dis')
    return unit, region, dict(path=(Path('artifacts') / 'kernel.dis').as_posix(), format="disassembly")


def test_applicability_reports_differences_and_routes(world):  # noqa: F811
    unit, region, candidate = _candidate(world)
    response = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate,
                                        shape=dict(m=32, k=32, n=32)), **world["run"])
    assert response["status"] == "analyzed", response
    assert response["report_in_detail"] and len(json.dumps(response)) < 64 << 10
    report = _report(response, world["run"]["workspace"])
    assert report["kind"] == "reference_applicability"
    assert report["reference"]["unit_id"] == unit and report["reference"]["region_id"] == region
    result = report["applicability"]
    assert result["candidate_shape"] == dict(value=dict(m=32, k=32, n=32), family="dense", source="author_declared")
    assert result["selected_geometry"]["status"] == "unknown"  # no facts selected in this fixture
    for row in result["differences"]:
        assert row["status"] in ("observed", "observed_partial", "inferred", "unknown") and row["evidence"]
        for route in row.get("routes", []):
            assert route["admission"]["status"] in ("unknown", "not_declared", "not_applicable", "satisfied",
                                                    "violated")
    assert report["semantic_equivalence"] == report["performance"] == "unknown"
    assert response["report"]["differences"] and response["report"]["choice"]


def _refusals(world):  # noqa: F811
    unit, region, candidate = _candidate(world)
    other = world["banks"]["cca-8"]["units"]
    excluded = next(u["kernel_id"] for u in other
                    if u["kernel_id"] not in {v["kernel_id"] for v in world["banks"]["cca-4"]["units"]})
    return [dict(entry=excluded, region=region, candidate=candidate),
            dict(entry="machine-" + "0" * 64, region=region, candidate=candidate),
            dict(entry=unit, region="region-unknown", candidate=candidate),
            dict(path="reference-kernels/models/raw.elf", candidate=candidate),
            dict(entry=unit, region=region, candidate=dict(path="../outside.dis")),
            dict(entry=unit, region=region),
            dict(entry=unit, candidate=candidate, region=region, symbol="kernel_entry")]


def test_refusals_are_identical_to_compare(world):  # noqa: F811
    for fields in _refusals(world):
        compare = analysis.handle(_request("compare", **fields), **world["run"])
        applicable = analysis.handle(_request("applicability", **fields), **world["run"])
        assert compare["status"] != "analyzed", fields
        assert (applicable["status"], applicable["reason"]) == (compare["status"], compare["reason"]), fields


def test_owned_left_side_shape_and_raw_arm_are_refused(world, monkeypatch):  # noqa: F811
    unit, region, candidate = _candidate(world)
    owned = analysis.handle(_request("applicability", (Path('artifacts') / 'kernel.dis').as_posix(), format="disassembly",
                                     candidate=candidate), **world["run"])
    assert owned["reason"] == bank_mode.OUTSIDE
    for shape in (dict(m=0, k=1, n=1), dict(rows=4), [4, 4, 4]):
        bad = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate,
                                       shape=shape), **world["run"])
        assert bad["reason"] == "invalid_analysis_request", shape
    stray = analysis.handle(_request("lift", entry=unit, shape=dict(m=1, k=1, n=1)), **world["run"])
    assert stray["reason"] == "invalid_analysis_request"
    monkeypatch.setenv(A.ENV, RAW)
    monkeypatch.setattr(bank_mode, "open_bank", lambda *a, **k: pytest.fail("raw arm read a bank"))
    denied = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate),
                             **world["run"])
    assert denied["status"] == "denied"
    # A non-bank reference arm refuses both bank-only operations before reading anything.
    monkeypatch.setattr(reference_experiments, "_current", lambda: "selected")
    monkeypatch.setattr(reference_experiments, "_load", lambda selected: (dict(identity="p"), None))
    for operation in bank_mode.BANK_OPERATIONS:
        refused = analysis.handle(_request(operation, entry=unit, region=region, candidate=candidate),
                                  **world["run"])
        assert refused["reason"] == "bank_operation_requires_cca_bank_arm", refused


def test_copied_client_reaches_applicability_over_the_qa_channel(world):  # noqa: F811
    unit, region, _ = _candidate(world)
    workspace = world["run"]["workspace"]
    shim = workspace / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    channel = workspace / ".qa_channel"
    channel.mkdir()
    argv = [sys.executable, str(shim), "--reference-analysis", "applicability", "--reference-entry", unit,
            "--reference-region", region, "--candidate-path", (Path('artifacts') / 'kernel.dis').as_posix(), "--candidate-format",
            "disassembly", "--candidate-shape", "m=16,k=16,n=16", "--wait-budget", "10"]
    process = subprocess.Popen(argv, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline, requests = time.monotonic() + 10, []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests
        request = json.loads(requests[0].read_bytes())
        assert request["reference_analysis"]["shape"] == dict(m=16, k=16, n=16)
        A.check_broker_action(request["action"])
        rid = request["request_id"]
        response = broker._reference_analysis_response(
            request, workspace=workspace, facts=None, compiler_sha256="0" * 64, rid=rid,
            started_ns=time.time_ns(), bundle_manifest=world["run"]["bundle_manifest"], repo=world["repo"])
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        result, _ = json.JSONDecoder().raw_decode(stdout)
        assert result["status"] == "analyzed" and result["report"]["kind"] == "reference_applicability"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _target():
    """The bank's target is the product folder it was published under (conditions/<label> three below it)."""
    return _conditions()["cca-4"].parents[2].name


def _run(world):  # noqa: F811
    return dict(world["run"], target=_target())


def _schedule_document(world, name=(Path('artifacts') / 'schedule.json').as_posix()):  # noqa: F811
    """A stream document in the author tool's format, holding the candidate's own recorded command stream."""
    from merlin.compile.analysis.disassembly import tokenize_text
    from merlin.compile.analysis.riscv_machine import finite_control_counts
    from merlin.kernels.schedule_effects import SCHEMA, _digest

    workspace = world["run"]["workspace"]
    rows = tokenize_text((workspace / 'artifacts' / 'kernel.dis').read_text())
    root, bank = bank_mode.open_bank(workspace, world["run"]["bundle_manifest"], world["repo"],
                                     bank_mode.selected_pin())
    selected = bank_mode._bank_decoder(root, bank)
    decoder = analysis._operand_decoder(selected["contract"], selected["facts"], _target())
    opcode = decoder["semantics"].isa_constants(decoder["target"])["CUSTOM_OPCODE"]
    stream = finite_control_counts(rows, custom_opcode=opcode, record_commands=True).get("command_stream") or []
    document = dict(schema=SCHEMA, status="derived", basis="test", arguments=[], stream=stream,
                    entries=len(stream), truncated=False, stream_sha256=_digest(stream), annotations=None)
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return name, document


def test_schedule_and_binary_give_a_summary_first_three_way_reply(world):  # noqa: F811
    from merlin.kernels.schedule_effects import ORDER_FACTS

    unit, region, candidate = _candidate(world)
    schedule, _ = _schedule_document(world)
    response = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate,
                                        candidate_schedule=schedule), **_run(world))
    assert response["status"] == "analyzed", response
    brief = response["report"]["schedule_attribution"]
    assert next(iter(response["report"])) == "schedule_attribution"  # summary first
    assert set(brief["order_facts"]) <= set(ORDER_FACTS) and brief["timing"] == "not_claimed"
    assert brief["emission"]["realized"] is True and brief["emission"]["compared"] > 0, brief["emission"]
    assert brief["order_facts"]["issue_overlap"]["status"] == "observed", brief["order_facts"]
    kinds = [row["kind"] for row in brief["differences"]]
    assert kinds == sorted(kinds, key=lambda kind: kind != "order")
    assert {row["attribution"] for row in brief["differences"]} <= {
        "lowering_design_choice", "introduced_by_emission", "unknown"}
    assert len(json.dumps(response)) < 64 << 10
    full = _report(response, world["run"]["workspace"])["schedule_attribution"]
    assert full["input"]["path"] == schedule and "comparisons" in full["attribution"]
    alone = analysis.handle(_request("compare", entry=unit, region=region, candidate_schedule=schedule),
                            **_run(world))
    assert alone["status"] == "analyzed", alone
    assert alone["report"]["kind"] == "reference_schedule_comparison"
    assert alone["report"]["schedule_attribution"]["emission"]["realized"] is None


def test_schedule_documents_keep_the_candidate_refusals(world):  # noqa: F811
    unit, region, candidate = _candidate(world)
    name, document = _schedule_document(world)
    for path in ("submission/schedule.json", "variants/schedule.json", "../schedule.json"):
        refused = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate,
                                           candidate_schedule=path), **_run(world))
        assert refused["status"] != "analyzed" and refused["reason"] in (
            "candidate_schedule_outside_owned_artifacts", "analysis_input_unavailable_invalid_or_unsupported"), path
    document["stream"] = document["stream"][1:]
    (world["run"]["workspace"] / name).write_text(json.dumps(document))
    edited = analysis.handle(_request("applicability", entry=unit, region=region, candidate=candidate,
                                      candidate_schedule=name), **_run(world))
    assert edited["reason"] == "schedule_stream_digest_mismatch" and edited["next_action"]
    lifted = analysis.handle(_request("lift", entry=unit, candidate_schedule=name), **_run(world))
    assert lifted["reason"] == "invalid_analysis_request"


def test_copied_client_sends_schedule_and_binary_through_the_broker(world):  # noqa: F811
    unit, region, _ = _candidate(world)
    schedule, _ = _schedule_document(world, "variants/v1/schedule.json")
    workspace = world["run"]["workspace"]
    shim = workspace / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    channel = workspace / ".qa_channel"
    channel.mkdir()
    argv = [sys.executable, str(shim), "--reference-analysis", "applicability", "--reference-entry", unit,
            "--reference-region", region, "--candidate-path", (Path('artifacts') / 'kernel.dis').as_posix(), "--candidate-format",
            "disassembly", "--candidate-schedule", schedule, "--wait-budget", "10"]
    process = subprocess.Popen(argv, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline, requests = time.monotonic() + 10, []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests
        request = json.loads(requests[0].read_bytes())
        assert request["reference_analysis"]["candidate_schedule"] == schedule
        A.check_broker_action(request["action"])
        rid = request["request_id"]
        response = broker._reference_analysis_response(
            request, workspace=workspace, facts=None, compiler_sha256="0" * 64, rid=rid,
            started_ns=time.time_ns(), bundle_manifest=world["run"]["bundle_manifest"], repo=world["repo"],
            target=_target())
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        result, _ = json.JSONDecoder().raw_decode(stdout)
        assert result["status"] == "analyzed" and "schedule_attribution" in result["report"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    bad = subprocess.run([sys.executable, str(shim), "--reference-analysis", "applicability", "--reference-entry",
                          unit, "--reference-region", region, "--candidate-schedule", "submission/s.json"],
                         cwd=workspace, capture_output=True, text=True, timeout=20)
    assert bad.returncode != 0 and "--candidate-schedule must be" in bad.stderr
