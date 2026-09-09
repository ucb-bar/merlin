"""A promoted FireSim result exists only when all queue and UART proofs agree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf.firesim_receipt import (
    FireSimReceiptError,
    RECEIPT_SCHEMA,
    VALIDATION_POLICY_SCHEMA,
    main,
    parse_queued_firesim_receipt,
    write_queued_firesim_receipt,
)


_JOB_ID = 535
_WORKLOAD = "synthetic-whole-model"
_SUCCESS_MARKERS = (
    "VALIDATION output_digest=0123456789abcdef",
    "VALIDATION status=PASS",
)


def _write(path: Path, text: str, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def _client(*, state: str = "DONE", exit_code: int = 0,
            workload: str = _WORKLOAD, job_id: int = _JOB_ID) -> str:
    return "\n".join((
        (f"[firesim-queue] job_id={job_id} kind=runworkload-full "
         "user=test priority=5 project=test state=QUEUED"),
        f"[firesim-queue]   workload={workload} bootbinary=model.elf",
        f"[firesim-queue] job_id={job_id} state=RUNNING phase=RUNNING",
        (f"[firesim-queue] job_id={job_id} terminal state={state} "
         f"exit_code={exit_code} wall=1.0s"),
    )) + "\n"


# The daemon log is the one the daemon actually wrote (job 535, phase and command lines vendored
# verbatim), not a hand-written approximation.  The invented fixture this replaces omitted the
# LEADING_KILL banner entirely and scoped the leading kill to INFRASETUP, so the receipt required
# a lifecycle no real job has ever emitted and had never sealed a run.
_OBSERVED_DAEMON_LOG = (
    merlin_dir() / "tests/data/firesim_queue/job535_daemon_phase_skeleton.log")


def _daemon() -> str:
    text = _OBSERVED_DAEMON_LOG.read_text(encoding="utf-8")
    assert f"job_id={_JOB_ID} " in text, "the vendored daemon log must belong to _JOB_ID"
    return text


def _uart(*, cycles: int = 987654) -> str:
    return "\n".join((
        "boot noise",
        "MERLIN_INVOCATIONS warmup=1 measured=1",
        "MERLIN_PROFILE warmup begin",
        "MERLIN_PROFILE warmup end rc=0",
        "MERLIN_PROFILE measured begin",
        *_SUCCESS_MARKERS,
        f"METRIC cycles {cycles}",
        "MERLIN_PROFILE measured end rc=0",
        "DONE",
    )) + "\n"


def _inputs(tmp_path: Path) -> dict[str, object]:
    queue = _write(tmp_path / "queue" / "bin" / "firesim-queue", "#!/bin/sh\n", executable=True)
    submission = [
        str(queue), "runworkload-full", "--chipyard", "/targets/chipyard",
        "--workload", _WORKLOAD, "--stage-from", "/artifacts/model.elf",
    ]
    submission_path = _write(
        tmp_path / "evidence" / "submission.json",
        json.dumps(submission, indent=2) + "\n",
    )
    policy = {
        "schema": VALIDATION_POLICY_SCHEMA,
        "policy_id": "synthetic-target/exact-output-digest-v1",
        "workload": _WORKLOAD,
        "success_markers": list(_SUCCESS_MARKERS),
    }
    policy_path = _write(
        tmp_path / "evidence" / "validation-policy.json",
        json.dumps(policy, indent=2) + "\n",
    )
    return {
        "queue_client_log": _write(
            tmp_path / "evidence" / "queue-client.log", _client()),
        "queue_daemon_log": _write(
            tmp_path / "queue" / "jobs" / str(_JOB_ID) / "stdout.log", _daemon()),
        "uart_log": _write(tmp_path / "evidence" / "uartlog", _uart()),
        "expected_queue_executable": queue,
        "expected_submission_json": submission_path,
        "expected_job_id": _JOB_ID,
        "expected_workload": _WORKLOAD,
        "validation_policy_json": policy_path,
    }


def _parse(tmp_path: Path, **updates):
    inputs = _inputs(tmp_path)
    inputs.update(updates)
    return parse_queued_firesim_receipt(**inputs)


def test_parses_and_atomically_writes_a_content_bound_receipt(tmp_path: Path) -> None:
    receipt = _parse(tmp_path)
    destination = tmp_path / "receipt.json"

    write_queued_firesim_receipt(receipt, destination)
    first = destination.read_bytes()
    write_queued_firesim_receipt(receipt, destination)

    assert destination.read_bytes() == first
    assert list(tmp_path.glob(".receipt.json.*.tmp")) == []
    document = json.loads(first)
    assert document["schema"] == RECEIPT_SCHEMA
    assert document["status"] == "passed"
    assert document["queue_receipt"]["queue_job_id"] == _JOB_ID
    assert document["queue_receipt"]["warm_profile"]["total_compute_cycles"] == 987654
    assert [row["command"][1] for row in document["verification"]["daemon_lifecycle"]] == [
        "kill", "infrasetup", "runworkload", "kill"]
    assert [row["marker"] for row in document["verification"]["uart"]["correctness_markers"]] \
        == list(_SUCCESS_MARKERS)

    evidence = document["evidence"]
    assert evidence["queue_executable"]["sha256"] == hashlib.sha256(
        Path(evidence["queue_executable"]["path"]).read_bytes()).hexdigest()
    assert evidence["expected_submission"]["sha256"] == hashlib.sha256(
        Path(evidence["expected_submission"]["path"]).read_bytes()).hexdigest()
    assert evidence["validation_policy"]["sha256"] == hashlib.sha256(
        Path(evidence["validation_policy"]["path"]).read_bytes()).hexdigest()
    for log in document["queue_receipt"]["logs"].values():
        assert log["sha256"] == hashlib.sha256(Path(log["path"]).read_bytes()).hexdigest()


def test_writer_rechecks_evidence_and_refuses_post_parse_drift(tmp_path: Path) -> None:
    receipt = _parse(tmp_path)
    Path(receipt.queue_receipt.logs[2].path).write_text(_uart(cycles=1), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        write_queued_firesim_receipt(receipt, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize(("state", "exit_code"), [
    ("FAILED", 1), ("CANCELLED", 1), ("DONE", 1),
])
def test_client_must_prove_done_with_zero_exit(
        tmp_path: Path, state: str, exit_code: int) -> None:
    inputs = _inputs(tmp_path)
    _write(Path(inputs["queue_client_log"]), _client(state=state, exit_code=exit_code))

    with pytest.raises(FireSimReceiptError, match="state=DONE and exit_code=0"):
        parse_queued_firesim_receipt(**inputs)


def test_client_job_and_workload_must_match_exactly(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write(Path(inputs["queue_client_log"]), _client(job_id=410))
    with pytest.raises(FireSimReceiptError, match="job id 410"):
        parse_queued_firesim_receipt(**inputs)

    _write(Path(inputs["queue_client_log"]), _client(workload="another-model"))
    with pytest.raises(FireSimReceiptError, match="matching workload"):
        parse_queued_firesim_receipt(**inputs)


@pytest.mark.parametrize("daemon_edit", [
    lambda text: text.replace("Running: kill", "Running: placeholder", 1).replace(
        "Running: infrasetup", "Running: kill", 1).replace(
            "Running: placeholder", "Running: infrasetup", 1),
    lambda text: text.replace("Running: runworkload", "Running: status\nRunning: runworkload"),
    lambda text: text.replace("phase=RUNNING", "phase=TEARDOWN"),
    lambda text: text.replace(f"job_id={_JOB_ID}", f"job_id={_JOB_ID + 1}", 1),
    # the leading kill must stay scoped to its own LEADING_KILL banner
    lambda text: text.replace("phase=LEADING_KILL", "phase=INFRASETUP", 1),
])
def test_daemon_requires_exact_phase_scoped_lifecycle(tmp_path: Path, daemon_edit) -> None:
    inputs = _inputs(tmp_path)
    daemon = Path(inputs["queue_daemon_log"])
    _write(daemon, daemon_edit(daemon.read_text(encoding="utf-8")))

    with pytest.raises(FireSimReceiptError, match="queue daemon|kill -> infrasetup"):
        parse_queued_firesim_receipt(**inputs)


@pytest.mark.parametrize("uart", [
    _uart().replace("MERLIN_PROFILE warmup end rc=0", "MERLIN_PROFILE warmup end rc=3"),
    _uart().replace("MERLIN_PROFILE measured end rc=0", "MERLIN_PROFILE measured end rc=4"),
    _uart().replace("METRIC cycles 987654", "METRIC cycles 0"),
    _uart().replace("METRIC cycles 987654", "METRIC cycles 1\nMETRIC cycles 2"),
    _uart().replace("METRIC cycles 987654\n", "").replace(
        "MERLIN_PROFILE warmup begin", "METRIC cycles 987654\nMERLIN_PROFILE warmup begin"),
    _uart().replace("MERLIN_INVOCATIONS warmup=1 measured=1", "MERLIN_INVOCATIONS warmup=0 measured=1"),
])
def test_uart_requires_one_successful_warm_then_measured_cycle_window(
        tmp_path: Path, uart: str) -> None:
    inputs = _inputs(tmp_path)
    _write(Path(inputs["uart_log"]), uart)

    with pytest.raises(FireSimReceiptError, match="UART|cycle metric|warm|order"):
        parse_queued_firesim_receipt(**inputs)


@pytest.mark.parametrize("uart", [
    _uart().replace(_SUCCESS_MARKERS[0] + "\n", ""),
    _uart().replace(_SUCCESS_MARKERS[0], _SUCCESS_MARKERS[0] + "\n" + _SUCCESS_MARKERS[0]),
    _uart().replace(
        _SUCCESS_MARKERS[0] + "\n", "", 1).replace(
            "MERLIN_PROFILE measured begin\n",
            _SUCCESS_MARKERS[0] + "\nMERLIN_PROFILE measured begin\n"),
])
def test_correctness_is_owned_by_the_workload_policy_and_must_precede_metric(
        tmp_path: Path, uart: str) -> None:
    inputs = _inputs(tmp_path)
    _write(Path(inputs["uart_log"]), uart)

    with pytest.raises(FireSimReceiptError, match="correctness marker"):
        parse_queued_firesim_receipt(**inputs)


def test_submission_and_policy_are_exact_content_bound_inputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    submission_path = Path(inputs["expected_submission_json"])
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    submission[submission.index(_WORKLOAD)] = "another-model"
    _write(submission_path, json.dumps(submission))
    with pytest.raises(FireSimReceiptError, match="exactly one --workload"):
        parse_queued_firesim_receipt(**inputs)

    inputs = _inputs(tmp_path / "policy-mismatch")
    policy_path = Path(inputs["validation_policy_json"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["workload"] = "another-model"
    _write(policy_path, json.dumps(policy))
    with pytest.raises(FireSimReceiptError, match="policy workload"):
        parse_queued_firesim_receipt(**inputs)


def test_all_inputs_and_output_must_be_absolute_plain_files(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    relative = Path("relative-client.log")
    inputs["queue_client_log"] = relative
    with pytest.raises(FireSimReceiptError, match="absolute plain file"):
        parse_queued_firesim_receipt(**inputs)

    receipt = _parse(tmp_path / "output")
    with pytest.raises(FireSimReceiptError, match="absolute path"):
        write_queued_firesim_receipt(receipt, Path("receipt.json"))


def test_cli_writes_the_same_verified_document(tmp_path: Path, capsys) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "cli-receipt.json"
    arguments = [
        "--queue-client-log", str(inputs["queue_client_log"]),
        "--queue-daemon-log", str(inputs["queue_daemon_log"]),
        "--uart-log", str(inputs["uart_log"]),
        "--queue-executable", str(inputs["expected_queue_executable"]),
        "--submission-json", str(inputs["expected_submission_json"]),
        "--job-id", str(inputs["expected_job_id"]),
        "--workload", str(inputs["expected_workload"]),
        "--validation-policy", str(inputs["validation_policy_json"]),
        "--output", str(output),
    ]

    assert main(arguments) == 0
    assert capsys.readouterr().out.strip() == str(output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


# --- the four protocol constants, pinned to logs the queue and the harness actually wrote ---
#
# Every one of these constants had been written from a guess, and each guess made the receipt
# unsatisfiable in a way no test could see, because the tests graded against the same guess.
# There is no sealed receipt anywhere under out/artifacts/, which is what that costs.  These
# tests parse the vendored real logs with the production parsers, so a constant can only be
# wrong if the vendored bytes are wrong.

_OBSERVED_UART_LOG = (
    merlin_dir() / "tests/data/firesim_queue/job610_uart_marker_skeleton.log")


def test_queue_phase_sequence_is_the_sequence_the_daemon_prints() -> None:
    from merlin.perf.execution_policy import FIRESIM_QUEUE_PHASES
    from merlin.perf.firesim_receipt import _daemon_phase

    text = _OBSERVED_DAEMON_LOG.read_text(encoding="utf-8")
    observed: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        phase = _daemon_phase(line, number, _JOB_ID)
        if phase is not None and (not observed or observed[-1] != phase):
            observed.append(phase)

    assert tuple(observed) == FIRESIM_QUEUE_PHASES
    # LEADING_KILL is the banner the constant used to omit, which alone made every run unsealable.
    assert "LEADING_KILL" in FIRESIM_QUEUE_PHASES


def test_lifecycle_commands_are_scoped_to_the_phases_the_daemon_scoped_them_to() -> None:
    from merlin.perf.firesim_receipt import _verify_daemon

    commands = _verify_daemon(_OBSERVED_DAEMON_LOG.read_text(encoding="utf-8"), _JOB_ID)

    assert tuple((phase, command) for phase, command, _line in commands) == (
        ("LEADING_KILL", "kill"),
        ("INFRASETUP", "infrasetup"),
        ("RUNNING", "runworkload"),
        ("TEARDOWN", "kill"),
    )


def test_uart_cycle_metric_is_the_bare_form_the_harness_prints() -> None:
    text = _OBSERVED_UART_LOG.read_text(encoding="utf-8")
    metrics = [line for line in text.splitlines() if line.startswith("METRIC")]

    assert metrics == ["METRIC cycles 33085199302"]
    assert "MERLIN_METRIC" not in text, "the harness prints a bare METRIC, not MERLIN_METRIC"


def test_invocation_line_carries_no_trailing_batch_field() -> None:
    from merlin.perf.firesim_receipt import _INVOCATION_LINE

    lines = [
        line for line in _OBSERVED_UART_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("MERLIN_INVOCATIONS")
    ]

    assert lines == [_INVOCATION_LINE]


def test_observed_profile_window_is_the_constant_the_receipt_requires() -> None:
    from merlin.perf.firesim_receipt import _PROFILE_LINES

    lines = tuple(
        line for line in _OBSERVED_UART_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("MERLIN_PROFILE")
    )

    assert lines == _PROFILE_LINES


def test_whole_model_harness_still_prints_no_line_a_policy_could_name() -> None:
    """The remaining reason this harness's runs cannot be sealed, asserted so it fails loudly.

    `UartValidationPolicy` requires at least one exact success marker, and refuses any marker
    starting with MERLIN_PROFILE / MERLIN_INVOCATIONS / METRIC.  The whole-model harness that
    produced job 610 publishes its verdict only implicitly, by withholding the METRIC when
    validation fails, and otherwise prints raw `OUT <tensor> ...` value dumps.  So no policy can
    be written for it and `parse_queued_firesim_receipt` cannot seal the run even now that the
    phase constants match the daemon.

    This is a harness gap, not a receipt gap: the fix is for the gate renderer to emit an explicit
    validation line.  When it does, this test fails and should be replaced by one that seals a real
    run end to end.
    """
    reserved = ("MERLIN_PROFILE", "MERLIN_INVOCATIONS", "METRIC")
    candidates = [
        line for line in _OBSERVED_UART_LOG.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and not line.startswith(reserved)
    ]

    assert candidates == [], (
        "the harness now prints a line a validation policy could name; seal a real run instead")
