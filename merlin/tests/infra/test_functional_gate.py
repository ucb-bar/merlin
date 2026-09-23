"""The functional gate reads a simulator transcript structurally and never reports success by default.

Every transcript below is a REAL Spike run of the ResNet-50 W8A8 whole-model ELF, copied verbatim
except for two tokens that name the target (the ``MERLIN_COMPILER`` tag and the simulator's
trailing extension banner); neither is a result or verdict line. Four real shapes, because each
one classifies differently and the differences are the whole point:

* ``PASS_TRANSCRIPT``  -- the accepted revision (2026-09-09 hand validation);
* ``FAIL_TRANSCRIPT``  -- a miscompiled revision: every logit wrong, ``bad=1000 top1=556``, and
  eight ``MERLIN_MISMATCH`` lines, a line type the parser must not confuse for a result line;
* ``TRAP_TRANSCRIPT``  -- a revision that TRAPPED: the runner's ``*** FAILED ***`` banner and no
  result line at all. This one found a real defect -- it was classified ``not_run``, which left a
  candidate that never even ran eligible to be selected as best;
* ``ABORT_TRANSCRIPT`` -- a revision whose emitted loop bounds the hardware refused, aborting
  during warmup with neither a result line nor a verdict.

The remaining cases are one-token mutations of the PASS text, which is the point: a gate that
could not fail on ``bad=1`` is not a gate.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from merlin.perf import functional_gate as FG

PASS_TRANSCRIPT = """\
MERLIN_MODEL resnet50_w8a8_torchao_pt2e
MERLIN_COMPILER phase2_reduction_resident_kernel
MERLIN_INPUT universal_resnet50_dog_sha256_92994d4d2219bc51
MERLIN_INVOCATIONS warmup=1 measured=1 batch=1
MERLIN_INPUT_PROLOGUE timed=1 owner=phase2_reduction_resident_kernel kind=static_pt2e_w8a8
MERLIN_PROFILE warmup begin
MERLIN_PROFILE warmup end rc=0
MERLIN_WARM_RESET timed=0 kind=zero_mutable_arena bytes=90083520
MERLIN_PROFILE measured begin
MERLIN_METRIC cycles=691647813
MERLIN_METRIC instret=691647817
MERLIN_METRIC main_ld_cycles=5841166
MERLIN_METRIC main_st_cycles=5862797
MERLIN_METRIC main_ex_cycles=5813796
MERLIN_METRIC exe_active_cycle=5830844
MERLIN_METRIC loop_matmul_active_cycles=5836613
MERLIN_METRIC rdma_bytes_rec=13
MERLIN_METRIC wdma_bytes_sent=11
MERLIN_METRIC reservation_station_active_cycles=5860767
MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90
MERLIN_RESULT max_abs_bits=00000000 max_rel_bits=00000000 qdq_max_abs_bits=3e31e2aa
MERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258
MERLIN_PROFILE measured end rc=0
PASS: warm-then-measured Merlin W8A8 ResNet-50 and all-logit check
Accelerator extension configured with:
    dim = 16
"""

FAIL_TRANSCRIPT = """\
MERLIN_MODEL resnet50_w8a8_torchao_pt2e
MERLIN_COMPILER phase2_bf62c927_accel
MERLIN_INPUT universal_resnet50_dog_sha256_92994d4d2219bc51
MERLIN_INVOCATIONS warmup=1 measured=1 batch=1
MERLIN_INPUT_PROLOGUE timed=1 owner=phase2_bf62_kernel kind=static_pt2e_w8a8
MERLIN_PROFILE warmup begin
MERLIN_PROFILE warmup end rc=0
MERLIN_WARM_RESET timed=0 kind=zero_mutable_arena bytes=90083520
MERLIN_PROFILE measured begin
MERLIN_MISMATCH index=0 got_bits=bf48e933 expected_bits=3e2478bd
MERLIN_MISMATCH index=1 got_bits=bdbe3e2c expected_bits=be33a940
MERLIN_MISMATCH index=2 got_bits=bf617a5c expected_bits=bd9936dc
MERLIN_MISMATCH index=3 got_bits=bf483ba8 expected_bits=be533fac
MERLIN_MISMATCH index=4 got_bits=bf0c6490 expected_bits=bf128de6
MERLIN_MISMATCH index=5 got_bits=bf0aa2fb expected_bits=be5893b5
MERLIN_MISMATCH index=6 got_bits=bf872922 expected_bits=be8aa5ee
MERLIN_MISMATCH index=7 got_bits=bf434b30 expected_bits=3da49436
MERLIN_METRIC cycles=1438624346
MERLIN_METRIC instret=1438624350
MERLIN_METRIC main_ld_cycles=3801783
MERLIN_METRIC main_st_cycles=3793931
MERLIN_METRIC main_ex_cycles=3824021
MERLIN_METRIC exe_active_cycle=3810502
MERLIN_METRIC loop_matmul_active_cycles=3797513
MERLIN_METRIC rdma_bytes_rec=7
MERLIN_METRIC wdma_bytes_sent=10
MERLIN_METRIC reservation_station_active_cycles=3800398
MERLIN_RESULT checksum_fnv1a64=694ab3a308942b47
MERLIN_RESULT max_abs_bits=40e97424 max_rel_bits=468534a7 qdq_max_abs_bits=40e96315
MERLIN_RESULT logits_checked=1000 bad=1000 nonfinite=0 top1=556 expected_top1=258
MERLIN_PROFILE measured end rc=1
FAIL: output differs from independent PT2E integer golden
Accelerator extension configured with:
    dim = 16
*** FAILED *** (tohost = 1)
"""

TRAP_TRANSCRIPT = """\
*** FAILED *** (tohost = 1)
MERLIN_PROFILE warmup begin
MERLIN_PROFILE warmup end rc=0
MERLIN_PROFILE measured begin
MERLIN_PROFILE measured end rc=1
"""

ABORT_TRANSCRIPT = """\
MERLIN_MODEL resnet50_w8a8_torchao_pt2e
MERLIN_COMPILER phase2_bf62c927_accel
MERLIN_INPUT universal_resnet50_dog_sha256_92994d4d2219bc51
MERLIN_INVOCATIONS warmup=1 measured=1 batch=1
MERLIN_INPUT_PROLOGUE timed=1 owner=phase2_bf62_kernel kind=static_pt2e_w8a8
MERLIN_PROFILE warmup begin
Accelerator extension configured with:
    dim = 16
LOOP_WS bounds were too large for double-buffering
"""

# The model's exact gate, as the hand validation applied it.
GATE = FG.FunctionalGateSpec(
    {"checksum_fnv1a64": "c6e777c3fe0aae90", "max_abs_bits": "00000000", "bad": 0, "top1": 258}
)


def test_parses_the_real_pass_transcript_structurally():
    parsed = FG.parse_simulator_transcript(PASS_TRANSCRIPT)
    assert len(parsed.result_lines) == 3
    assert parsed.fields == {
        "checksum_fnv1a64": "c6e777c3fe0aae90",
        "max_abs_bits": "00000000",
        "max_rel_bits": "00000000",
        "qdq_max_abs_bits": "3e31e2aa",
        "logits_checked": "1000",
        "bad": "0",
        "nonfinite": "0",
        "top1": "258",
        "expected_top1": "258",
    }
    assert parsed.verdict == "PASS"
    assert parsed.verdict_line.startswith("PASS: warm-then-measured")
    assert parsed.duplicate_fields == {}
    # METRIC/PROFILE lines are not results and must not leak into the fields
    assert "cycles" not in parsed.fields


def test_real_pass_transcript_passes_the_exact_gate():
    status, reason, comparisons = FG.evaluate_transcript(FG.parse_simulator_transcript(PASS_TRANSCRIPT), GATE)
    assert status == FG.STATUS_PASSED
    assert all(row["ok"] for row in comparisons)
    assert [row["field"] for row in comparisons] == ["checksum_fnv1a64", "max_abs_bits", "bad", "top1"]
    assert "PASS" in reason


def test_fail_transcript_fails_and_names_every_mismatched_field():
    parsed = FG.parse_simulator_transcript(FAIL_TRANSCRIPT)
    assert parsed.verdict == "FAIL"
    assert parsed.verdict_line == "FAIL: output differs from independent PT2E integer golden"
    assert parsed.fields["bad"] == "1000" and parsed.fields["top1"] == "556"
    # The eight MERLIN_MISMATCH lines carry key=value tokens of their own (index, got_bits,
    # expected_bits, the last repeated eight times). Only MERLIN_RESULT lines are results; a
    # looser prefix match would silently pollute the gated fields with per-logit debug output.
    assert len(parsed.result_lines) == 3
    assert not {"index", "got_bits", "expected_bits"} & set(parsed.fields)
    status, reason, comparisons = FG.evaluate_transcript(parsed, GATE)
    assert status == FG.STATUS_FAILED
    bad = {row["field"]: row for row in comparisons if not row["ok"]}
    assert set(bad) == {"checksum_fnv1a64", "max_abs_bits", "bad", "top1"}
    assert bad["bad"]["reason"] == "1000 != 0"
    assert bad["top1"]["reason"] == "556 != 258"
    assert "FAIL" in reason


@pytest.mark.parametrize(
    "mutation,expected_status,expect_in_reason",
    [
        # ONE bad logit while the harness still prints PASS: the field decides, not the banner.
        (("bad=0 nonfinite", "bad=1 nonfinite"), FG.STATUS_FAILED, "bad=1"),
        # Right numerics, wrong class: still a failure.
        (("top1=258 expected", "top1=257 expected"), FG.STATUS_FAILED, "top1=257"),
        # A checksum that differs in one hex digit.
        (("c6e777c3fe0aae90", "c6e777c3fe0aae91"), FG.STATUS_FAILED, "checksum_fnv1a64"),
        # max_abs_bits nonzero while bad=0 (a harness tolerance would mask this; the gate does not).
        (("max_abs_bits=00000000", "max_abs_bits=00000001"), FG.STATUS_FAILED, "max_abs_bits"),
        # The harness printed FAIL although every field matched: the harness verdict dominates.
        (("PASS: warm", "FAIL: warm"), FG.STATUS_FAILED, "harness verdict FAIL"),
        # The riscv-tests trap banner after a PASS line: FAIL anywhere dominates PASS.
        (("    dim = 16\n", "    dim = 16\n*** FAILED *** (tohost = 1)\n"), FG.STATUS_FAILED, "FAILED"),
        # No verdict at all (the program stopped after reporting): not a pass.
        (
            ("PASS: warm-then-measured Merlin W8A8 ResNet-50 and all-logit check\n", ""),
            FG.STATUS_FAILED,
            "no PASS verdict",
        ),
    ],
)
def test_single_token_mutations_of_the_pass_transcript_fail(mutation, expected_status, expect_in_reason):
    old, new = mutation
    mutated = PASS_TRANSCRIPT.replace(old, new)
    assert mutated != PASS_TRANSCRIPT, "the mutation must change the transcript"
    status, reason, _ = FG.evaluate_transcript(FG.parse_simulator_transcript(mutated), GATE)
    assert status == expected_status, reason
    assert expect_in_reason in reason


def test_trapped_program_with_no_result_line_is_failed_not_not_run():
    """A candidate that TRAPS must be excluded, not filed as "no evidence".

    Regression for a real defect: the no-result-line check ran first, so a transcript carrying
    the runner's ``*** FAILED ***`` banner and nothing else returned ``not_run`` -- and
    ``not_run`` does not exclude a candidate from selection. A program that never reached its own
    reporting is the clearest possible failure.
    """
    parsed = FG.parse_simulator_transcript(TRAP_TRANSCRIPT)
    assert parsed.result_lines == () and parsed.verdict == "FAIL"
    status, reason, _ = FG.evaluate_transcript(parsed, GATE)
    assert status == FG.STATUS_FAILED
    assert "did not reach its own reporting" in reason
    assert FG.gate_result_for_selection({"status": status})[0] is True


def test_program_that_aborted_without_reporting_is_decided_by_the_exit_status():
    """No result line and no verdict: the simulator's exit status is the only evidence there is."""
    parsed = FG.parse_simulator_transcript(ABORT_TRANSCRIPT)
    assert parsed.result_lines == () and parsed.verdict is None
    # it ran and died -> the candidate's defect
    status, reason, _ = FG.evaluate_transcript(parsed, GATE, simulator_returncode=1)
    assert status == FG.STATUS_FAILED and "exited 1 without printing" in reason
    # it exited cleanly, or the exit status is unknown -> ambiguous, and never success
    for returncode in (0, None):
        status, reason, _ = FG.evaluate_transcript(parsed, GATE, simulator_returncode=returncode)
        assert status == FG.STATUS_NOT_RUN and "no MERLIN_RESULT line" in reason


def test_transcript_without_result_lines_is_not_run_never_passed():
    stripped = "\n".join(line for line in PASS_TRANSCRIPT.splitlines() if not line.startswith("MERLIN_RESULT")) + "\n"
    parsed = FG.parse_simulator_transcript(stripped)
    assert parsed.result_lines == () and parsed.verdict == "PASS"
    status, reason, comparisons = FG.evaluate_transcript(parsed, GATE, simulator_returncode=0)
    assert status == FG.STATUS_NOT_RUN
    assert comparisons == []
    assert "MERLIN_RESULT" in reason
    # and an empty transcript (simulator died before printing) is the same non-answer
    assert FG.evaluate_transcript(FG.parse_simulator_transcript(""), GATE)[0] == FG.STATUS_NOT_RUN


def test_gate_naming_a_field_the_harness_never_prints_is_not_run():
    spec = FG.FunctionalGateSpec({"bad": 0, "no_such_field": 1})
    status, reason, comparisons = FG.evaluate_transcript(FG.parse_simulator_transcript(PASS_TRANSCRIPT), spec)
    assert status == FG.STATUS_NOT_RUN
    assert "no_such_field" in reason
    absent = [row for row in comparisons if not row["present"]]
    assert [row["field"] for row in absent] == ["no_such_field"]
    # ... unless the program also never reached a verdict, which is the program stopping short
    truncated = PASS_TRANSCRIPT.split("MERLIN_RESULT logits_checked")[0]
    status, reason, _ = FG.evaluate_transcript(FG.parse_simulator_transcript(truncated), GATE)
    assert status == FG.STATUS_FAILED
    assert "stopped before reporting" in reason and "bad" in reason


def test_bounded_gate_is_declared_in_the_spec_not_the_module():
    bounded = FG.FunctionalGateSpec(
        {
            "max_abs_bits": {"max": 0x100, "radix": 16},
            "bad": {"max": 5},
            "top1": {"equals": 258},
            "checksum_fnv1a64": {"equals": "C6E777C3FE0AAE90"},  # text equality is case-insensitive
        }
    )
    assert FG.evaluate_transcript(FG.parse_simulator_transcript(PASS_TRANSCRIPT), bounded)[0] == FG.STATUS_PASSED
    within = PASS_TRANSCRIPT.replace("max_abs_bits=00000000", "max_abs_bits=000000ff").replace(
        "bad=0 nonfinite", "bad=5 nonfinite"
    )
    assert FG.evaluate_transcript(FG.parse_simulator_transcript(within), bounded)[0] == FG.STATUS_PASSED
    over = within.replace("bad=5 nonfinite", "bad=6 nonfinite")
    status, reason, _ = FG.evaluate_transcript(FG.parse_simulator_transcript(over), bounded)
    assert status == FG.STATUS_FAILED and "6 > max 5" in reason
    with pytest.raises(ValueError):
        FG.FunctionalGateSpec({})
    with pytest.raises(ValueError):
        FG.FunctionalGateSpec({"bad": {"radix": 16}})  # a bound with no bound
    with pytest.raises(ValueError):
        FG.FunctionalGateSpec({"bad": True})


def test_duplicate_fields_keep_first_value_and_are_recorded():
    doubled = PASS_TRANSCRIPT + "MERLIN_RESULT bad=7\n"
    parsed = FG.parse_simulator_transcript(doubled)
    assert parsed.fields["bad"] == "0"
    assert parsed.duplicate_fields == {"bad": ["0", "7"]}


def test_selection_only_excludes_a_failed_gate():
    assert FG.gate_result_for_selection({"status": "failed", "reason": "bad=1000"}) == (True, "bad=1000")
    assert FG.gate_result_for_selection({"status": "passed"}) == (False, None)
    assert FG.gate_result_for_selection({"status": "not_run"}) == (False, None)
    assert FG.gate_result_for_selection(None) == (False, None)
    with pytest.raises(ValueError):
        FG.FunctionalGateResult(status="ok", reason="", stage="x")


# --------------------------------------------------------------------------------------------
# run_functional_gate: the recipe, with a fake toolchain standing in for translate/clang/link/sim
# --------------------------------------------------------------------------------------------


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\nset -u\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _fake_bundle(tmp_path: Path, *, link_exit: int = 0, elf_count: int = 1) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "payload").mkdir(parents=True)
    (bundle / "payload" / "single_run_harness.c").write_text("int main(void){return 0;}\n")
    (bundle / "runtime" / "include").mkdir(parents=True)
    (bundle / "runtime" / "crt.S").write_text("\n")
    _script(
        bundle / "build_elf.sh",
        f"""\
here=$(cd "$(dirname "$0")" && pwd -P)
[[ -f "$here/compiler/kernel.o" ]] || {{ echo "missing required file: kernel.o" >&2; exit 2; }}
[[ -f "$here/payload/single_run_harness.c" ]] || exit 2
echo "link args: $*" > "$here/link_args.txt"
for i in $(seq 1 {elf_count}); do printf 'ELF%s' "$i" > "$here/model_$i.elf"; done
exit {link_exit}
""",
    )
    return bundle


def _fake_toolchain(
    tmp_path: Path, *, transcript: str, translate_exit: int = 0, clang_exit: int = 0, simulator_body: str | None = None
) -> FG.FunctionalGateToolchain:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "transcript.txt").write_text(transcript, encoding="utf-8")
    # mlir-translate --mlir-to-llvmir <in> -o <out>
    _script(
        tools / "mlir-translate",
        f"""\
[[ "$1" == "--mlir-to-llvmir" ]] || exit 9
cp "$2" "$4"
exit {translate_exit}
""",
    )
    # clang ... -c <in> -o <out>
    _script(
        tools / "clang",
        f"""\
out=""; while [[ $# -gt 0 ]]; do if [[ "$1" == "-o" ]]; then out="$2"; shift; fi; shift; done
[[ -n "$out" ]] && echo "obj" > "$out"
exit {clang_exit}
""",
    )
    _script(
        tools / "spike",
        simulator_body
        or f"""\
printf '%s' "$*" > "$(dirname "$0")/spike_args.txt"
cat "$(dirname "$0")/transcript.txt"
exit 0
""",
    )
    return FG.FunctionalGateToolchain(
        mlir_translate=tools / "mlir-translate",
        clang=tools / "clang",
        simulator=tools / "spike",
        clang_target="riscv64-unknown-elf",
        march="rv64gc",
        mabi="lp64d",
        simulator_isa="rv64gc_zicntr",
        simulator_extension="accel",
        link_script_args=("/opt/sdk",),
        env={"MERLIN_FAKE": "1"},
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
    # the payload copy, the artifact and the object are staged the way the hand recipe did it
    work = tmp_path / "work"
    assert (work / "compiler" / "model.llvm.mlir").read_text() == "llvm.func @main() {}\n"
    assert (work / "compiler" / "kernel.o").is_file()
    assert (work / "payload" / "single_run_harness.c").is_file()
    assert (work / "link_args.txt").read_text().strip() == "link args: /opt/sdk"
    assert (work / "simulate.log").read_text() == PASS_TRANSCRIPT
    # simulator flags come from the caller's toolchain, never from this module
    spike_args = (tmp_path / "tools" / "spike_args.txt").read_text()
    assert spike_args.startswith("--isa=rv64gc_zicntr --extension=accel ")
    assert spike_args.endswith("model_1.elf")
    assert not (work / "model_1.elf").exists()  # keep_elf defaults to False
    record = result.to_dict()
    assert record["schema"] == FG.RESULT_SCHEMA
    assert record["excludes_candidate"] is False
    assert record["expected"] == GATE.to_dict()
    assert record["toolchain"]["simulator_isa"] == "rv64gc_zicntr"
    assert json.dumps(record)  # serializable as an iteration receipt


def test_recipe_fails_on_the_fail_transcript(tmp_path):
    toolchain = _fake_toolchain(tmp_path, transcript=FAIL_TRANSCRIPT)
    result = FG.run_functional_gate(
        "llvm.func @main() {}\n",
        None,
        model_payload_dir=_fake_bundle(tmp_path),
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=tmp_path / "work",
        timeout=30,
    )
    assert result.status == FG.STATUS_FAILED
    assert result.simulation_executed is True
    assert result.to_dict()["excludes_candidate"] is True
    assert result.fields["bad"] == "1000" and result.fields["top1"] == "556"


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
    assert "simulator=" in result.reason and not (tmp_path / "w1").exists()
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


def test_candidate_artifact_that_does_not_translate_compile_or_link_is_failed(tmp_path):
    for index, (stage, kwargs, bundle_kwargs) in enumerate(
        (
            ("translate", {"translate_exit": 1}, {}),
            ("compile", {"clang_exit": 1}, {}),
            ("link", {}, {"link_exit": 1}),
            ("link", {}, {"elf_count": 2}),
        )
    ):
        root = tmp_path / f"case{index}_{stage}"
        root.mkdir()
        toolchain = _fake_toolchain(root, transcript=PASS_TRANSCRIPT, **kwargs)
        result = FG.run_functional_gate(
            "x",
            None,
            model_payload_dir=_fake_bundle(root, **bundle_kwargs),
            toolchain=toolchain,
            gate_spec=GATE,
            workdir=root / "work",
            timeout=30,
        )
        assert (result.status, result.stage) == (FG.STATUS_FAILED, stage), result.reason
        assert result.simulation_executed is False
    # the link script's OWN preflight refusal (exit 2) is the environment, not the candidate
    root = tmp_path / "env"
    root.mkdir()
    toolchain = _fake_toolchain(root, transcript=PASS_TRANSCRIPT)
    result = FG.run_functional_gate(
        "x",
        None,
        model_payload_dir=_fake_bundle(root, link_exit=2),
        toolchain=toolchain,
        gate_spec=GATE,
        workdir=root / "work",
        timeout=30,
    )
    assert (result.status, result.stage) == (FG.STATUS_NOT_RUN, "link")


def test_simulator_that_exits_nonzero_without_reporting_is_failed(tmp_path):
    toolchain = _fake_toolchain(
        tmp_path,
        transcript=PASS_TRANSCRIPT,
        simulator_body="""\
echo "MERLIN_PROFILE warmup begin"
echo "*** FAILED *** (tohost = 1)"
exit 1
""",
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
        simulator_body="""\
echo "MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90"
sleep 20
""",
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


def test_config_file_resolves_relative_paths_and_carries_its_digest(tmp_path):
    (tmp_path / "bin").mkdir()
    document = {
        "schema": FG.CONFIG_SCHEMA,
        "model_payload_dir": "bundle",
        "toolchain": {
            "mlir_translate": "bin/mlir-translate",
            "clang": "/abs/clang",
            "simulator": "bin/sim",
            "clang_target": "riscv64-unknown-elf",
            "march": "rv64gc",
            "mabi": "lp64d",
            "simulator_isa": "rv64gc_zicntr",
            "simulator_extension": "accel",
            "link_script_args": ["/opt/sdk"],
        },
        "gate": {"checksum_fnv1a64": "c6e777c3fe0aae90", "bad": 0},
        "timeout_seconds": 90,
    }
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(document))
    config = FG.load_functional_gate_config(path)
    assert config.model_payload_dir == tmp_path / "bundle"
    assert config.toolchain.mlir_translate == tmp_path / "bin" / "mlir-translate"
    assert config.toolchain.clang == Path("/abs/clang")
    assert config.toolchain.simulator_extension == "accel"
    assert config.gate_spec.expectations == {"checksum_fnv1a64": "c6e777c3fe0aae90", "bad": 0}
    assert config.timeout_seconds == 90.0 and config.keep_elf is False
    assert config.source_path == path.resolve() and len(config.source_sha256) == 64
    assert config.to_dict()["gate"]["expectations"]["bad"] == 0
    with pytest.raises(ValueError, match="timeout_seconds"):
        FG.FunctionalGateConfig.from_mapping({**document, "timeout_seconds": 0})
    with pytest.raises(ValueError, match="toolchain is missing"):
        FG.FunctionalGateConfig.from_mapping({**document, "toolchain": {"clang": "c"}})
