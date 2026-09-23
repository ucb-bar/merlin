"""A whole-model CYCLE CLAIM seals only when the run proved it computed the model.

Three things had to be true at once before a cycle number on this route could be sealed, and none
of them was:

1. the harness printed ``GROUP_MODEL_TOTAL cycles: N`` and no ``MERLIN_INVOCATIONS`` line, so
   ``firesim_receipt._verify_uart`` rejected its UART outright -- **no run of this program could
   ever have been sealed**, whatever the queue did and whatever it computed. The two FPGA logs
   vendored beside these tests are real runs and neither carries a line the parser would accept;
2. the only validation policy for this model was v1, whose entire content is a list of exact marker
   lines -- an argmax over a thousand classes and one cosine;
3. one oracle group checksum was on record, so a v2 policy could not be authored without guessing
   the other seventy.

The evidence here is measured, not synthesised. ``merlin/tests/data/firesim_queue/`` vendors the
protocol lines of two real runs on `alveo_u250_firesim_gemmini_rocket_30mhz`: job **731**, correct,
and job **730**, the run that was wrong. The tests re-frame each run's own device numbers in the
protocol the fixed harness prints -- via the harness's own renderer -- and feed them to the
production parser. So the numbers under test were produced by the device, and only the wrapper is
new code. A third log is the console of a real BUILD of the fixed harness, run on the Spike
functional model -- the first UART anything has emitted in the new protocol, and the one place the
wrapper itself is observed rather than rendered.

WHAT JOB 730 ACTUALLY DID, since it is easy to overstate. It published 23,787,829 cycles, under
both declared cycle thresholds, with an argmax marker **byte-identical** to a correct run's, and
every one of its 71 group checksums wrong, starting at group 1. Its *cosine* marker did move
(997,981 -> 988,504 ppm), so the v1 policy's second marker would have caught it had anything run
that policy -- nothing did, because blocker 1 made the run unsealable in the first place. The case
for v2 is therefore not "v1 would have sealed job 730". It is that an argmax is a one-bit check and
a cosine is a coarse aggregate: an error that corrupted **all 71** group outputs moved the cosine
by under 1%, and a smaller one would leave both markers intact.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf.firesim_receipt import (
    VALIDATION_POLICY_SCHEMA,
    FireSimReceiptError,
    parse_queued_firesim_receipt,
    write_queued_firesim_receipt,
)

_JOB_ID = 535
#: The workload every one of these runs was really submitted under (its queue record's
#: ``"workload"``), so the policy's own ``workload`` field is checked against an observed value.
_WORKLOAD = "merlin-checkpoint"
_POLICY = merlin_dir() / "contract" / "firesim_uart_policies" / "resnet50_group_model_cycle_claim_v2.json"
_DATA = merlin_dir() / "tests" / "data" / "firesim_queue"
#: The one oracle checksum published independently of this policy, in the job-730 post-mortem
#: (see the phase-2 measurement contract and hardware provenance records).
_ORACLE_GROUP_1 = 5_663_048
#: What job 730 printed for group 1, beside a byte-identical argmax marker.
_JOB_730_GROUP_1 = 5_652_929
_JOB_731_CYCLES = 24_319_110


def _harness():
    """The harness script, loaded from its path: it is a script, not an importable package module."""
    path = merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts" / "group_model_program.py"
    spec = importlib.util.spec_from_file_location("group_model_program", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _observed(job: int) -> dict:
    """One real run's own published numbers, parsed structurally from its vendored UART."""
    text = (_DATA / f"job{job}_group_model_uart_skeleton.log").read_text(encoding="utf-8")
    steps: list[tuple[str, str]] = []
    checksums: dict[str, int] = {}
    cycles: dict[str, int] = {}
    total = argmax = want = cosine = None
    for raw in text.splitlines():
        if raw.startswith("#"):
            continue
        tokens = raw.split()
        fields = {key: value for key, _, value in (token.partition("=") for token in tokens) if value}
        if tokens[:1] == ["GM_GROUP"]:
            group, kind = tokens[1], tokens[2]
            steps.append((group, kind))
            checksums[group] = int(fields["sum"])
            cycles[group] = int(tokens[3])
        elif tokens[:1] == ["GM_ARGMAX"]:
            argmax, want = int(fields["got"]), int(fields["want"])
        elif tokens[:1] == ["GM_COSINE_PPM"]:
            cosine = int(tokens[1])
        elif tokens[:1] == ["GROUP_MODEL_TOTAL"]:
            total = int(tokens[-1])
    return {
        "steps": steps,
        "checksums": checksums,
        "group_cycles": cycles,
        "cycles": total,
        "argmax": argmax,
        "want": want,
        "cosine_ppm": cosine,
        "text": text,
    }


def _render(run: dict, *, checksums: dict[str, int] | None = None, label: str | None = None) -> str:
    """That run's own device numbers, in the protocol the fixed harness prints.

    Rendered by the harness, off the same format strings its C ``printf`` calls are built from, so
    a change to what the program publishes changes what these tests feed the parser instead of
    leaving them agreeing with a stale transcript.
    """
    return "\n".join(
        _harness().uart_lines(
            run["steps"],
            cycles=run["cycles"],
            checksums=run["checksums"] if checksums is None else checksums,
            group_cycles=run["group_cycles"],
            argmax=run["argmax"],
            want=run["want"],
            cosine_ppm=run["cosine_ppm"],
            window_label=label or _window()["label"],
        )
    )


def _reframed(job: int, **updates) -> str:
    return "boot noise\n" + _render(_observed(job), **updates) + "\nDONE\n"


def _policy_document() -> dict:
    return json.loads(_POLICY.read_text(encoding="utf-8"))


def _window() -> dict:
    return _policy_document()["per_window"][0]


def _write(path: Path, text: str, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


def _client() -> str:
    return (
        "\n".join(
            (
                f"[firesim-queue] job_id={_JOB_ID} kind=runworkload-full "
                "user=test priority=5 project=test state=QUEUED",
                f"[firesim-queue]   workload={_WORKLOAD} bootbinary=group_model_program.elf",
                f"[firesim-queue] job_id={_JOB_ID} state=RUNNING phase=RUNNING",
                f"[firesim-queue] job_id={_JOB_ID} terminal state=DONE exit_code=0 wall=1.0s",
            )
        )
        + "\n"
    )


def _daemon() -> str:
    """The daemon log job 535 actually wrote, vendored verbatim; not a hand-written approximation."""
    return (_DATA / "job535_daemon_phase_skeleton.log").read_text(encoding="utf-8")


def _inputs(tmp_path: Path, *, uart: str | None = None, policy: dict | None = None) -> dict[str, object]:
    queue = _write(tmp_path / "queue" / "bin" / "firesim-queue", "#!/bin/sh\n", executable=True)
    submission = [
        str(queue), "runworkload-full", "--chipyard", "/targets/chipyard",
        "--workload", _WORKLOAD, "--stage-from", "/artifacts/group_model_program.elf",
    ]  # fmt: skip
    return {
        "queue_client_log": _write(tmp_path / "evidence" / "queue-client.log", _client()),
        "queue_daemon_log": _write(tmp_path / "queue" / "jobs" / str(_JOB_ID) / "stdout.log", _daemon()),
        "uart_log": _write(tmp_path / "evidence" / "uartlog", uart if uart is not None else _reframed(731)),
        "expected_queue_executable": queue,
        "expected_submission_json": _write(
            tmp_path / "evidence" / "submission.json", json.dumps(submission, indent=2) + "\n"
        ),
        "expected_job_id": _JOB_ID,
        "expected_workload": _WORKLOAD,
        "validation_policy_json": _write(
            tmp_path / "evidence" / "validation-policy.json",
            json.dumps(policy if policy is not None else _policy_document(), indent=2) + "\n",
        ),
    }


# --------------------------------------------------------------- the oracle is a measured fact
def test_the_generated_oracle_is_what_the_device_actually_computed() -> None:
    """THE claim the policy file rests on, checked against hardware rather than asserted.

    The checksums were produced by a numpy emulation on a host.  This says they are also, group for
    group, what a real FPGA run of the compiled program published -- so the file is a measured fact
    about this device, not a statement about a numpy program that happens to be checked in.
    """
    declared = _window()["checksums"]
    correct = _observed(731)["checksums"]

    assert len(declared) == 71, "one checksum per device group of the capture, not a sample"
    assert sorted(int(group) for group in declared) == list(range(1, 72)), "no gaps: a missing group is undeclared"
    assert correct == declared, "every group the device computed matches the generated oracle"


def test_the_policy_declares_everything_its_v1_sibling_did_and_the_group_sums_besides() -> None:
    document, window = _policy_document(), _window()

    assert document["schema"] == "merlin_firesim_uart_validation_policy_v2"
    assert document["workload"] == _WORKLOAD, "the workload every one of these jobs was submitted under"
    assert window["checksums"]["1"] == _ORACLE_GROUP_1
    assert window["markers"] == ["GM_ARGMAX got=21 want=21 agrees=1", "GM_COSINE_PPM 997981"]
    assert document["checksum_line"] == {"prefix": "GM_GROUP", "value_key": "sum", "group_token_offset": 1}


def test_the_policy_records_what_produced_it() -> None:
    """A checksum set whose provenance is not recorded cannot be regenerated or refuted."""
    sidecar = json.loads(_POLICY.with_suffix(".provenance.json").read_text(encoding="utf-8"))

    assert sidecar["capture"]["linalg.mlir_sha256"]
    # WITHOUT THE FACTS ARTIFACT the same capture forms 70 host regions and no device group at all,
    # so the checksum set is a property of these facts as much as of the capture.
    assert sidecar["rtl_facts"]["sha256"] and not sidecar["rtl_facts"]["sha256"].startswith("UNKNOWN")
    assert sidecar["compiler_provenance"]["source_digest"]
    assert sidecar["emulation"] == {"argmax": 21, "want": 21, "cosine_ppm": 997981}


# --------------------------------------------------------------- blocker 1, pinned to real bytes
@pytest.mark.parametrize("job", [730, 731])
def test_no_run_of_the_old_harness_could_be_sealed_however_it_computed(job: int) -> None:
    """The reason this route had no sealed receipt, asserted against two real FPGA logs.

    One of these runs was right and one was wrong, and it made no difference: neither publishes a
    line the receipt parser will read as a measurement.
    """
    published = [line for line in _observed(job)["text"].splitlines() if not line.startswith("#")]

    assert any(line.startswith("GROUP_MODEL_TOTAL cycles:") for line in published)
    assert not any(line.startswith("METRIC") for line in published), "no metric the parser would read"
    assert not any(line.startswith("MERLIN_INVOCATIONS") for line in published)


def test_the_two_real_runs_printed_the_same_argmax_marker() -> None:
    """What a marker-only policy can see, and what it cannot.

    The correct run and the wrong one published byte-identical argmax lines.  The wrong run's
    cosine DID move -- so v1's second marker was not blind here -- but it moved under 1% for an
    error that changed every one of the 71 group outputs, which is the measure of how coarse that
    check is.
    """
    correct, wrong = _observed(731), _observed(730)

    assert (correct["argmax"], correct["want"]) == (wrong["argmax"], wrong["want"]) == (21, 21)
    assert (correct["cosine_ppm"], wrong["cosine_ppm"]) == (997981, 988504)
    assert sum(1 for group, value in wrong["checksums"].items() if value != correct["checksums"][group]) == 71


# --------------------------------------------------------------- the protocol, as emitted
def test_a_real_build_of_the_fixed_harness_emits_a_uart_that_seals(tmp_path: Path) -> None:
    """The one end nothing else here covers: bytes a BUILD of this program actually printed.

    Every other UART in this file is the harness's renderer applied to a run's numbers.  This one
    is the console of a real build -- the 71-group capture, the vendor library, the riscv64 toolchain
    -- executed on the Spike functional model, filtered to its protocol lines and vendored verbatim.
    Until this run the ``METRIC cycles N`` line and the ``MERLIN_WINDOW`` frame were things the
    harness had been CHANGED to print and nothing had printed; ``firesim_batch`` still describes the
    frame as "PROPOSED, NOT OBSERVED".

    IT IS NOT A HARDWARE MEASUREMENT.  Spike is a functional model and its cycle figure is an
    instruction count; no performance claim may cite it, and this test asserts nothing about the
    value.  What it establishes is that the emission and the parser agree on real bytes.
    """
    text = (_DATA / "spike_group_model_uart_skeleton.log").read_text(encoding="utf-8")
    published = [line for line in text.splitlines() if not line.startswith("#")]

    assert published[0] == "MERLIN_INVOCATIONS warmup=1 measured=1", "byte-exact, no trailing batch= field"
    assert published[1] == "MERLIN_WINDOW begin label=group_model" and published[-1].endswith("label=group_model")
    assert sum(1 for line in published if line.startswith("METRIC")) == 1
    assert "GROUP_MODEL_TOTAL" not in text, "the spelling nothing could parse is gone from the built program"

    receipt = parse_queued_firesim_receipt(**_inputs(tmp_path, uart=text), cycle_claim_window="group_model")

    assert receipt.cycle_claim is not None
    assert len(receipt.cycle_claim.group_checksums) == 71
    # The emission change did not disturb what the program computes: this build agrees with the
    # oracle on every group, as seven FPGA runs of earlier builds do.
    assert dict(receipt.cycle_claim.group_checksums) == {
        group: int(value) for group, value in _window()["checksums"].items()
    }


# --------------------------------------------------------------- the job-730 run, on the seal path
def test_the_real_job_730_run_is_refused_and_the_real_correct_run_is_not(tmp_path: Path) -> None:
    """THE reason this path exists, run against the actual UARTs rather than a mutation."""
    sealed = parse_queued_firesim_receipt(
        **_inputs(tmp_path / "correct", uart=_reframed(731)), cycle_claim_window="group_model"
    )
    assert sealed.cycle_claim is not None
    assert sealed.queue_receipt.warm_profile.total_compute_cycles == _JOB_731_CYCLES

    with pytest.raises(FireSimReceiptError) as refusal:
        parse_queued_firesim_receipt(
            **_inputs(tmp_path / "wrong", uart=_reframed(730)), cycle_claim_window="group_model"
        )
    message = str(refusal.value)
    assert "RAN and was wrong" in message, "a run that ran and was wrong is not an absence"
    assert f"group 1 summed {_JOB_730_GROUP_1} against an oracle {_ORACLE_GROUP_1}" in message
    assert "declared marker(s) not present exactly once" in message, "its cosine marker moved too"


def test_one_wrong_group_checksum_is_enough_to_refuse_a_claim(tmp_path: Path) -> None:
    """The mutation in isolation: the correct run with group 1 replaced by what job 730 printed.

    Job 730 got every group wrong, because group 1's error propagates.  This narrows it to the one
    difference, so the test cannot pass for the wrong reason -- the argmax and the cosine markers
    are the correct run's here, and the claim is still refused.
    """
    run = _observed(731)
    correct = _render(run)
    mutated = _render(run, checksums=dict(run["checksums"]) | {"1": _JOB_730_GROUP_1})
    differing = [(a, b) for a, b in zip(correct.splitlines(), mutated.splitlines(), strict=True) if a != b]

    assert len(differing) == 1 and differing[0][0].startswith("GM_GROUP 1 "), "one integer, nothing else"

    with pytest.raises(FireSimReceiptError, match="RAN and was wrong") as refusal:
        parse_queued_firesim_receipt(**_inputs(tmp_path, uart=mutated + "\n"), cycle_claim_window="group_model")
    assert "declared marker(s) not present exactly once" not in str(refusal.value), "only the checksum is wrong"


def test_a_sealed_cycle_claim_records_what_it_proved(tmp_path: Path) -> None:
    receipt = parse_queued_firesim_receipt(**_inputs(tmp_path), cycle_claim_window="group_model")
    destination = write_queued_firesim_receipt(receipt, tmp_path / "receipt.json")
    document = json.loads(destination.read_text(encoding="utf-8"))

    assert document["status"] == "passed"
    assert document["queue_receipt"]["warm_profile"]["total_compute_cycles"] == _JOB_731_CYCLES
    claim = document["verification"]["cycle_claim"]
    assert claim["window"] == "group_model"
    assert claim["group_checksums_verified"] == 71
    assert claim["group_checksums"]["1"] == _ORACLE_GROUP_1
    assert claim["checksum_line"]["prefix"] == "GM_GROUP"


def test_the_claim_and_its_correctness_evidence_must_be_the_same_window(tmp_path: Path) -> None:
    """A metric published outside the claimed window's frame is a number about something else."""
    uart = _reframed(731).replace("MERLIN_WINDOW end label=group_model", "MERLIN_WINDOW end label=other_window")

    with pytest.raises(FireSimReceiptError, match="opens and never closes"):
        parse_queued_firesim_receipt(**_inputs(tmp_path, uart=uart), cycle_claim_window="group_model")


def test_a_cycle_claim_naming_a_window_the_policy_does_not_declare_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FireSimReceiptError, match="declares no window"):
        parse_queued_firesim_receipt(**_inputs(tmp_path), cycle_claim_window="some_other_window")


# --------------------------------------------------------------- v1: refused for the claim, not globally
def _v1_policy() -> dict:
    """The markers-only shape, spelled exactly as the tracked v1 sibling spells it."""
    return {
        "schema": VALIDATION_POLICY_SCHEMA,
        "policy_id": "resnet50_group_model_correctness_v2",
        "workload": _WORKLOAD,
        "success_markers": ["GM_ARGMAX got=21 want=21 agrees=1", "GM_COSINE_PPM 997981"],
    }


def test_a_v1_policy_is_refused_for_a_cycle_claim(tmp_path: Path) -> None:
    with pytest.raises(FireSimReceiptError, match="per-window group checksums"):
        parse_queued_firesim_receipt(**_inputs(tmp_path, policy=_v1_policy()), cycle_claim_window="group_model")


def test_the_same_v1_policy_still_seals_on_the_legacy_path(tmp_path: Path) -> None:
    """v1 is refused for the CLAIM, not for the schema.

    Receipts sealed under it remain valid; refusing it outright would retroactively invalidate them
    to fix a claim they never made.  The same policy and the same UART that cannot support a cycle
    number still seal the thing v1 actually proves -- that the run printed the lines it named.
    """
    v1 = _v1_policy()

    receipt = parse_queued_firesim_receipt(**_inputs(tmp_path, policy=v1))

    assert receipt.cycle_claim is None
    document = receipt.to_dict()
    assert "cycle_claim" not in document["verification"], "a legacy receipt carries no claim it did not make"
    assert [row["marker"] for row in document["verification"]["uart"]["correctness_markers"]] == v1["success_markers"]


def test_a_v2_policy_offered_without_a_claim_is_refused_rather_than_read_as_v1(tmp_path: Path) -> None:
    """The mirror of the rule above: the legacy path reads v1 documents and nothing else.

    A v2 document quietly read as v1 would seal a receipt whose ``success_markers`` were absent and
    whose checksums were never compared -- the strongest policy in the repository, doing nothing.
    """
    with pytest.raises(FireSimReceiptError, match="validation policy keys must be exactly"):
        parse_queued_firesim_receipt(**_inputs(tmp_path))
