from __future__ import annotations

from verification_witnesses.functional_gate import (
    FG,
    GATE,
    PASS_TRANSCRIPT,
    Path,
    _fake_bundle,
    _fake_toolchain,
    _script,
    json,
    pytest,
    stat,
)


def test_recipe_builds_stages_and_passes_on_the_real_transcript(tmp_path):
    toolchain = _fake_toolchain(tmp_path, transcript=PASS_TRANSCRIPT)
    bundle = _fake_bundle(tmp_path)
    result = FG.run_functional_gate(
        "llvm.func @main() {}\n",
        {"commands": [], "params": {}},
        model_payload_dir=bundle / "payload",
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "work",
        timeout=30,
    )
    assert result.status == FG.STATUS_PASSED, result.reason
    assert result.stage == "evaluate" and result.simulation_executed is True
    assert result.simulator_returncode == 0
    assert result.fields["top1"] == "258"
    assert result.elf_sha256 and result.transcript_sha256 and result.artifact_sha256
    work = tmp_path / "work"
    assert (work / "compiler" / "model.llvm.mlir").read_text() == "llvm.func @main() {}\n"
    assert (work / "compiler" / "kernel.o").is_file()
    assert (work / "payload" / "single_run_harness.c").is_file()
    assert (work / "link_args.txt").read_text().strip() == "link args: /opt/sdk"
    assert (work / "simulate.log").read_text() == PASS_TRANSCRIPT
    spike_args = (tmp_path / "tools" / "spike_args.txt").read_text()
    assert spike_args.startswith("--isa=rv64gc_zicntr --extension=accel ")
    assert spike_args.endswith("model_1.elf")
    assert not (work / "model_1.elf").exists()
    record = result.to_dict()
    assert record["schema"] == FG.RESULT_SCHEMA
    assert record["excludes_candidate"] is False
    assert record["expected"] == GATE.to_dict()
    assert record["toolchain"]["simulator_isa"] == "rv64gc_zicntr"
    assert json.dumps(record)


def test_absent_toolchain_or_payload_is_not_run_before_anything_is_spent(tmp_path):
    toolchain = _fake_toolchain(tmp_path, transcript=PASS_TRANSCRIPT)
    missing = FG.FunctionalGateToolchain(**{**toolchain.__dict__, "simulator": tmp_path / "nope"})
    result = FG.run_functional_gate(
        "x",
        None,
        model_payload_dir=_fake_bundle(tmp_path),
        toolchain=missing,
        gate_spec=GATE,
        workdir=tmp_path / "w1",
        timeout=5,
    )
    assert (result.status, result.stage) == (FG.STATUS_NOT_RUN, "preflight")
    assert "simulator=" in result.reason and (not (tmp_path / "w1").exists())
    result = FG.run_functional_gate(
        "x",
        None,
        model_payload_dir=tmp_path / "absent",
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "w2",
        timeout=5,
    )
    assert (result.status, result.stage) == (FG.STATUS_NOT_RUN, "preflight")
    assert "payload absent" in result.reason
    declined = {"declined": {"reason": "unsupported rank"}}
    result = FG.run_functional_gate(
        "x",
        declined,
        model_payload_dir=_fake_bundle(tmp_path / "b2"),
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "w3",
        timeout=5,
    )
    assert result.status == FG.STATUS_NOT_RUN and "declined" in result.reason
    assert result.simulation_executed is False and result.to_dict()["excludes_candidate"] is False


def test_simulator_that_exits_nonzero_without_reporting_is_failed(tmp_path):
    toolchain = _fake_toolchain(
        tmp_path,
        transcript=PASS_TRANSCRIPT,
        simulator_body='echo "MERLIN_PROFILE warmup begin"\necho "*** FAILED *** (tohost = 1)"\nexit 1\n',
    )
    result = FG.run_functional_gate(
        "x",
        None,
        model_payload_dir=_fake_bundle(tmp_path),
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "work",
        timeout=30,
    )
    assert result.status == FG.STATUS_FAILED
    assert result.simulation_executed is True and result.simulator_returncode == 1
    assert result.to_dict()["excludes_candidate"] is True


def test_simulator_timeout_is_not_run_with_partial_fields(tmp_path):
    toolchain = _fake_toolchain(
        tmp_path,
        transcript=PASS_TRANSCRIPT,
        simulator_body='echo "MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90"\nsleep 20\n',
    )
    result = FG.run_functional_gate(
        "x",
        None,
        model_payload_dir=_fake_bundle(tmp_path),
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "work",
        timeout=2,
    )
    assert (result.status, result.stage) == (FG.STATUS_NOT_RUN, "simulate")
    assert "budget" in result.reason
    assert result.simulation_executed is False
    assert result.fields.get("checksum_fnv1a64") == "c6e777c3fe0aae90"
    assert result.to_dict()["excludes_candidate"] is False
