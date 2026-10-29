"""Paired analysis selects actual regions and retains semantic/consumer evidence gaps."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from merlin_experiments.phase1.feedback import reference_map
from merlin_experiments.phase1.tools import selfcheck as client
from test_reference_analysis_tool import _request, _world

DISASSEMBLY = '''00000100 <kernel>:
 100: 0200307b .insn r 0x7b,3,1,x0,x0,x0
 104: 00008082 ret
'''


def _candidate(tmp_path, text=DISASSEMBLY):
    path = tmp_path / "variants/trial/kernel.dis"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


def _compare(path="reference-kernels/trace.txt", **kw):
    return _request("compare", path, offset=1,
                    candidate=dict(path="variants/trial/kernel.dis", symbol="kernel"), **kw)


def test_real_paired_static_analysis_does_not_call_empty_cca_agreement(tmp_path):
    fixture = _world(tmp_path)
    candidate = _candidate(tmp_path)
    with reference.invocation(fixture["path"], fixture["sha"]):
        first = analysis.handle(_compare(), workspace=tmp_path, facts=fixture["facts"],
                                compiler_sha256=fixture["compiler"])
        candidate.write_text(DISASSEMBLY.replace(" 104: 00008082 ret", " 104: 0001 nop\n 106: 00008082 ret"))
        changed = analysis.handle(_compare(), workspace=tmp_path, facts=fixture["facts"])
    assert first["status"] == changed["status"] == "analyzed"
    report = first["report"]
    assert report["reference"]["regions"][0]["selected_decode"]["counts"]["load"] == 1
    assert report["candidate"]["regions"][0]["schedule_model"]["stage_witnesses"]
    assert report["command_site_comparison"]["load"] == dict(reference=1, candidate=1)
    assert report["cca"]["status"] == "unknown" and report["cca"]["denominator"] == 0
    assert not report["cca"]["divergences"]
    assert report["schedule_consumption"] == report["semantic_compatibility"] == "unknown"
    assert changed["report"]["candidate"]["input"]["sha256"] != report["candidate"]["input"]["sha256"]
    assert changed["report_identity"] != first["report_identity"]
    assert changed["report"]["instruction_site_delta"] == 1
    assert first["all_pass"] is None and first["qualification_issued"] is False


def test_pair_requires_a_selected_region_and_rejects_unknown_symbols(tmp_path):
    fixture = _world(tmp_path)
    _candidate(tmp_path)
    request = _compare()
    request["reference_analysis"]["offset"] = 0
    with reference.invocation(fixture["path"], fixture["sha"]):
        ambiguous = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
        request["reference_analysis"]["offset"] = 1
        request["reference_analysis"]["candidate"]["symbol"] = "absent"
        missing = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
    assert ambiguous["reason"] == "analysis_compare_requires_one_region"
    assert missing["reason"] == "analysis_symbol_not_unique"
    assert missing["gap"]["kind"] == "evidence_or_adapter_gap" and missing["next_action"]


@pytest.mark.parametrize("name,text,locator", [
    ("trace.txt", None, dict(lines=[2, 3])),
    ("trace.dis", DISASSEMBLY + DISASSEMBLY.replace("kernel", "second"), dict(lines=[4, 6])),
])
def test_pinned_map_locator_selects_leaf_without_human_extraction(tmp_path, monkeypatch, name, text, locator):
    fixture = _world(tmp_path)
    _candidate(tmp_path)
    if text:
        (tmp_path / "reference-kernels" / name).write_text(text)
    entry = dict(id="leaf", file=name, locator=locator,
                 operation=dict(kind="matmul", dims=dict(M=16, K=16, N=16)), techniques=[],
                 uses_hardware_loops=False, confidence="inferred", evidence="fixture listing rows")
    document = dict(schema=reference_map.SCHEMA, reference_manifest_sha256="f" * 64, entries=[entry])
    map_path = tmp_path / "selected-map.json"
    map_path.write_text(json.dumps(document))
    monkeypatch.setenv(reference_map.ENV, json.dumps(dict(path=str(map_path),
        sha256=hashlib.sha256(map_path.read_bytes()).hexdigest())))
    request = _compare(path="", entry="leaf")
    request["reference_analysis"]["offset"] = 0
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
        request["reference_analysis"]["path"] = "reference-kernels/schedule.py"
        wrong = analysis.handle(request, workspace=tmp_path, facts=fixture["facts"])
    assert result["status"] == "analyzed"
    assert result["report"]["reference"]["listed_instructions"] == 2
    if name.endswith(".dis"):
        assert result["report"]["reference"]["regions"][0]["symbol"] == "second"
    assert result["report"]["reference"]["reference_entry"]["semantic_compatibility"] == "unknown"
    assert wrong["reason"] == "analysis_entry_path_mismatch"


def test_source_lines_select_second_function_and_listing_function_requires_a_real_locator(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    source = tmp_path / "reference-kernels/schedule.py"
    source.write_text("def first():\n    return 1\n\ndef second():\n    return 2\n")
    entry = dict(id="source", file="schedule.py", locator=dict(lines=[4, 5]),
        operation=dict(kind="matmul"), techniques=[], uses_hardware_loops=False,
        confidence="inferred", evidence="fixture")
    document = dict(schema=reference_map.SCHEMA, reference_manifest_sha256="f" * 64, entries=[entry])
    path = tmp_path / "map.json"
    def select():
        path.write_text(json.dumps(document))
        monkeypatch.setenv(reference_map.ENV, json.dumps(dict(path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    select()
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("lift", "", entry="source"), workspace=tmp_path)
        assert result["report"]["functions"][0]["symbol"] == "second"
        entry.update(file="trace.txt", locator=dict(function="invisible_symbol"))
        select()
        unknown = analysis.handle(_request("lift", "", entry="source"), workspace=tmp_path)
    assert unknown["reason"] == "analysis_locator_unsupported"


def test_schedule_dialect_decisions_are_not_realization(tmp_path):
    fixture = _world(tmp_path)
    path = tmp_path / "submission/plan.mlir"
    path.write_text('builtin.module { "schedule.plan"() <{sym_name = "candidate"}> ({}) : () -> () }')
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("candidate", "submission/plan.mlir", format="schedule_mlir"),
                                 workspace=tmp_path)
    assert result["status"] == "analyzed"
    assert result["report"]["kind"] == "schedule_decisions"
    assert result["report"]["decisions"][0]["operation"] == "schedule.plan"
    assert result["report"]["realized"] is False
    assert "consumer" in result["report"]["unknowns"][0]
    assert result["artifact_to_current_compiler_binding"] == "unknown"


def test_elf_function_extent_includes_local_labels_and_excludes_equal_pc_sections(tmp_path, monkeypatch):
    """Real objdump output includes relocation labels inside the selected function."""
    from merlin.compile.analysis.disassembly import objdump_bin

    clang = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not clang or "objdump" not in Path(objdump_bin()).name or Path(objdump_bin()).name == "objdump":
        pytest.skip("a RISC-V compiler and disassembler are required")
    fixture = _world(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    source = tmp_path / "function.S"
    source.write_text('''
.section .text.selected,"ax",@progbits
.globl kernel
.type kernel,@function
kernel:
  .insn 4, 0x0200307b
.Lpcrel_hi0:
  auipc a0, %pcrel_hi(object)
  addi a0, a0, %pcrel_lo(.Lpcrel_hi0)
  .insn 4, 0x0200307b
  ret
.size kernel, .-kernel
.section .text.other,"ax",@progbits
.globl decoy
.type decoy,@function
decoy:
  .insn 4, 0x0200307b
  .insn 4, 0x0200307b
  .insn 4, 0x0200307b
  ret
.size decoy, .-decoy
.section .data,"aw",@progbits
object: .word 0
''')
    output = tmp_path / "submission/kernel.o"
    subprocess.run([clang, "--target=riscv64-unknown-elf", "-march=rv64gc", "-c", str(source),
                    "-o", str(output)], check=True, capture_output=True, timeout=30)
    text = subprocess.check_output([objdump_bin(), "-d", "-M", "no-aliases", str(output)], text=True)
    assert "<.Lpcrel_hi0>:" in text and "<decoy>:" in text
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("candidate", "submission/kernel.o", format="elf", symbol="kernel"),
                                 workspace=tmp_path, facts=fixture["facts"])
    assert result["status"] == "analyzed"
    region, = result["report"]["regions"]
    assert region["selected_decode"]["counts"]["load"] == 2
    assert region["listed_instructions"] == 5
    assert region["static_function_extent"]["section"] == ".text.selected"
    assert region["cfg"]["trip_counts"] == region["schedule_model"]["realization"] == "unknown"
    assert region["cca"]["dispatch"] is None

    # The same complete textual extraction must not fall back to the setup
    # prefix when the caller names the symbol and local relocation labels exist.
    scope = subprocess.check_output([objdump_bin(), '-d', '-M', 'no-aliases',
        '--section=.text.selected', str(output)], text=True)
    lifted = analysis._lift(scope.encode(), 'disassembly', dict(path='selected.dis', sha256='a'*64,
        size_bytes=len(scope.encode())),
        facts=fixture['facts'], offset=0, limit=1, candidate=False, symbol='kernel')
    text_region, = lifted['regions']
    assert text_region['listed_instructions'] == 5
    assert text_region['selected_decode']['counts']['load'] == 2
    assert text_region['symbol_status'] == 'listing_label_inferred_static_span'


def test_elf_without_retained_function_boundary_is_an_actionable_gap(tmp_path, monkeypatch):
    fixture = _world(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    (tmp_path / "submission/kernel.o").write_bytes(b"not an ELF")
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("candidate", "submission/kernel.o", format="elf", symbol="kernel"),
                                 workspace=tmp_path, facts=fixture["facts"])
    assert result["status"] == "unknown"
    assert result["reason"] == "analysis_elf_symbol_boundary_unavailable"
    assert result["next_action"] and result["qualification_issued"] is False


def test_wordless_mapped_span_uses_annotation_without_inventing_symbols(tmp_path):
    fixture = _world(tmp_path)
    raw = b"100: add a0,a1,a2\n104: .insn 4,0x0200307b\n108: ret\n"
    pin = dict(path="reference-kernels/expert.dis", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    report = analysis._lift(raw, "disassembly", pin, facts=fixture["facts"], offset=0, limit=1,
        candidate=False, symbol="mapped_function", locator=dict(lines=[1, 3]), mapped_symbol=True)
    region = report["regions"][0]
    assert region["symbol"] == "unlabelled" and region["mapped_function"] == "mapped_function"
    assert region["symbol_status"] == "not_in_listing"
    assert region["schedule_model"]["command_sites"] == 1
    assert region["cfg"]["status"] == "unknown" and region["cfg"]["missing_encodings"] == 2
    with pytest.raises(ValueError, match="analysis_symbol_not_unique"):
        analysis._lift(raw, "disassembly", pin, facts=fixture["facts"], offset=0, limit=1,
                       candidate=False, symbol="invented", locator=dict(lines=[1, 3]))


@pytest.mark.skipif(not os.environ.get("MERLIN_TEST_JACK_DISASSEMBLY"), reason="operator-owned raw reference")
def test_real_jack_fx_conv_27_selected_span():
    """Opt-in regression against retained raw bytes, without publishing the private fixture."""
    raw = Path(os.environ["MERLIN_TEST_JACK_DISASSEMBLY"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "4e953c8a5c58aefeb6bdb11a92a0fc06a471d2da636d0a7354ad92c596fd5e55"
    pin = dict(path="reference-kernels/jack.dis", sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    report = analysis._lift(raw, "disassembly", pin, facts=Path(os.environ["MERLIN_TEST_JACK_FACTS"]),
        offset=0, limit=1, candidate=False, symbol="fx_conv_27", mapped_symbol=True,
        locator=dict(lines=[96260, 101034]))
    region = report["regions"][0]
    counts = region["selected_decode"]["counts"]
    assert region["listed_instructions"] == 4775
    assert region["selected_decode"]["endpoint_insns"] == 710
    assert counts["COMPUTE_AND_STAY_CMD"] == 219
    assert counts["COMPUTE_AND_FLIP_CMD"] + counts["COMPUTE_AND_STAY_CMD"] == 309
    assert region["cfg"]["status"] == "unknown"
    assert region["cca"]["dispatch"] is None


@pytest.mark.parametrize("raw,format,reason", [
    ("[]", "semantic_region", "analysis_semantic_contract_invalid"),
    ("null", "semantic_region", "analysis_semantic_contract_invalid"),
    ('{"version":1,"operations":[null]}', "semantic_region", "analysis_semantic_contract_invalid"),
    ("not MLIR", "schedule_mlir", "analysis_schedule_invalid"),
    ('builtin.module { "schedule.plan"() ({}) : () -> () }', "schedule_mlir", "analysis_schedule_invalid"),
])
def test_malformed_records_are_actionable_gaps_not_broker_crashes(tmp_path, raw, format, reason):
    fixture = _world(tmp_path)
    (tmp_path / "submission/input.txt").write_text(raw)
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(_request("candidate", "submission/input.txt", format=format), workspace=tmp_path)
    assert result["status"] == "unknown" and result["reason"] == reason
    assert result["next_action"] and result["gap"]["retryable"] is False


def _region():
    numeric = dict(accumulator_dtype="i32", bias="none", scale=0.25, activation="none",
                   rounding="nearest_even", saturation="signed_i8")
    return dict(region_id="qkv", complete=True,
        tensors=[dict(key="x", shape=[2, 4], dtype="i8", layout="row")]
            + [dict(key="w" + key, shape=[4, 3], dtype="i8", layout="row") for key in ("q", "k", "v")]
            + [dict(key=key, shape=[2, 3], dtype="i8", layout="row") for key in ("q", "k", "v")],
        operations=[dict(key=key, op="matmul", inputs=["x", "w" + key], outputs=[key], numeric=numeric)
                    for key in ("q", "k", "v")],
        inputs=["x", "wq", "wk", "wv"], outputs=["q", "k", "v"], external_uses=dict(q=1, k=1, v=1),
        boundary=dict(scope="complete_region", packing="included", preparation="included",
                      conversions="included", synchronization="included", readout="included",
                      input_identity="same-input", runtime_identity="same-runtime"))


@pytest.mark.parametrize("change, semantic, measurement", [
    ("same", "equivalent", "equivalent"), ("scale", "mismatch", "equivalent"),
    ("layout", "mismatch", "equivalent"), ("liveouts", "mismatch", "equivalent"),
    ("packing", "equivalent", "mismatch"), ("incomplete", "unknown", "equivalent")])
def test_declared_contract_comparison_preserves_fusion_numeric_and_timing_boundaries(
        tmp_path, change, semantic, measurement):
    fixture = _world(tmp_path)
    expert = _region()
    ours = copy.deepcopy(expert)
    if change == "scale":
        ours["operations"][0]["numeric"]["scale"] = 0.5
    elif change == "layout":
        ours["tensors"][0]["layout"] = "packed"
    elif change == "liveouts":
        ours["outputs"] = ["q"]
        ours["external_uses"] = dict(q=1)
    elif change == "packing":
        ours["boundary"]["packing"] = "excluded"
    elif change == "incomplete":
        ours["complete"] = False
    (tmp_path / "reference-kernels/region.json").write_text(json.dumps(expert))
    (tmp_path / "submission/region.json").write_text(json.dumps(ours))
    request = _request("compare", "reference-kernels/region.json", format="semantic_region",
                       candidate=dict(path="submission/region.json", format="semantic_region"))
    with reference.invocation(fixture["path"], fixture["sha"]):
        result = analysis.handle(request, workspace=tmp_path)
    assert result["status"] == "analyzed"
    declared = result["report"]["declared_contracts"]
    assert declared["status"] == semantic and declared["measurement_status"] == measurement
    assert result["semantic_equivalence"] == result["performance"] == "unknown"
    assert result["report"]["artifact_origin"] == "unverified"


def test_standalone_compare_client_roundtrip(tmp_path):
    fixture = _world(tmp_path)
    _candidate(tmp_path)
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    argv = [sys.executable, str(shim), "--reference-analysis", "compare", "--reference-path",
            "reference-kernels/trace.txt", "--reference-offset", "1", "--candidate-path",
            "variants/trial/kernel.dis", "--candidate-symbol", "kernel", "--wait-budget", "5"]
    process = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        requests = []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(.01)
        assert requests
        request = json.loads(requests[0].read_text())
        assert request["reference_analysis"]["candidate"]["symbol"] == "kernel"
        rid = request["request_id"]
        with reference.invocation(fixture["path"], fixture["sha"]):
            response = broker._reference_analysis_response(request, workspace=tmp_path, facts=fixture["facts"],
                compiler_sha256=fixture["compiler"], rid=rid, started_ns=time.time_ns())
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stdout + stderr
        assert json.loads(stdout)["report"]["kind"] == "reference_candidate_comparison"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
