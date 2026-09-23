"""A checkpoint is a table of SEALED queue-owned runs; anything less is reported as not sealed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf import firesim_checkpoint as FC
from merlin.perf.firesim_receipt import VALIDATION_POLICY_SCHEMA

_WORKLOAD = "synthetic-workload"
_BOOT = "synthetic.elf"
_MARKERS = ("VALIDATION output_digest=0123456789abcdef", "VALIDATION status=PASS")
_OBSERVED_DAEMON_LOG = merlin_dir() / "tests/data/firesim_queue/job535_daemon_phase_skeleton.log"
_HELP = (
    "runworkload-full atomic FireSim lifecycle: daemon owns the full kill ->\n"
    "  infrasetup -> runworkload -> kill sequence\n"
    "usage: [--chipyard C] [--workload W] [--bootbinary B] [--stage-from S] [--priority P]\n"
    "  [--project P] [--timeout T] [--hw-config H] [--hwdb-config-artifact A]\n"
)


def _uart(cycles: int, markers=_MARKERS) -> str:
    return (
        "\n".join(
            (
                "MERLIN_INVOCATIONS warmup=1 measured=1",
                "MERLIN_PROFILE warmup begin",
                "MERLIN_PROFILE warmup end rc=0",
                "MERLIN_PROFILE measured begin",
                *markers,
                f"METRIC cycles {cycles}",
                "MERLIN_PROFILE measured end rc=0",
                "DONE",
            )
        )
        + "\n"
    )


class FakeQueue:
    """Stands in for the queue client: same argv, same log shapes, no FPGA."""

    def __init__(
        self,
        host: FC.QueueHost,
        *,
        help_text: str = _HELP,
        busy: tuple[int, ...] = (),
        cycles: dict[str, int] | None = None,
        broken: tuple[str, ...] = (),
    ):
        self.host, self.help_text, self.busy = host, help_text, busy
        self.cycles, self.broken = cycles or {}, broken
        self.next_job, self.submitted = 535, []

    def __call__(self, argv, **_kwargs):
        if argv[1:] in (["--help"], ["runworkload-full", "--help"]):
            return subprocess.CompletedProcess(argv, 0, self.help_text, "")
        if argv[1:] == ["status"]:
            lines = [f"[firesim-queue] job_id={job} state=RUNNING" for job in self.busy]
            return subprocess.CompletedProcess(argv, 0, "\n".join(lines) or "[firesim-queue] no jobs match", "")
        assert argv[1] == "runworkload-full"
        elf = Path(argv[argv.index("--stage-from") + 1])
        job, self.next_job = self.next_job, self.next_job + 1
        self.submitted.append(elf.name)
        daemon = self.host.queue_state_root / "jobs" / str(job) / "stdout.log"
        daemon.parent.mkdir(parents=True)
        daemon.write_text(
            _OBSERVED_DAEMON_LOG.read_text(encoding="utf-8").replace("job_id=535 ", f"job_id={job} "), encoding="utf-8"
        )
        uart = (
            self.host.chipyard
            / "sims/firesim/deploy/results-workload"
            / f"2026-{_WORKLOAD}-q{job}"
            / f"{_WORKLOAD}0"
            / "uartlog"
        )
        uart.parent.mkdir(parents=True)
        markers = _MARKERS[:1] if elf.name in self.broken else _MARKERS
        uart.write_text(_uart(self.cycles.get(elf.name, 1000), markers), encoding="utf-8")
        client = (
            "\n".join(
                (
                    f"[firesim-queue] job_id={job} kind=runworkload-full user=t priority=5 state=QUEUED",
                    f"[firesim-queue]   workload={_WORKLOAD} bootbinary={_BOOT}",
                    f"[firesim-queue] job_id={job} terminal state=DONE exit_code=0 wall=1.0s",
                )
            )
            + "\n"
        )
        return subprocess.CompletedProcess(argv, 0, client, None)


def _plan(tmp_path: Path, elves: dict[str, bytes], **overrides) -> FC.CheckpointPlan:
    queue = tmp_path / "install" / "bin" / "firesim-queue"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/bin/sh\n", encoding="utf-8")
    queue.chmod(0o755)
    for directory in ("state", "chipyard"):
        (tmp_path / directory).mkdir()
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema": VALIDATION_POLICY_SCHEMA,
                "policy_id": "synthetic/exact-output-digest-v1",
                "workload": _WORKLOAD,
                "success_markers": list(_MARKERS),
            }
        ),
        encoding="utf-8",
    )
    entries = []
    for name, payload in elves.items():
        (tmp_path / name).write_bytes(payload)
        entries.append(
            {
                "label": name.removesuffix(".elf"),
                "model": "synthetic_model",
                "experiment": "checkpoint",
                "elf": str(tmp_path / name),
                "validation_policy": str(policy),
            }
        )
    document = {
        "schema": FC.MANIFEST_SCHEMA,
        "checkpoint": "synthetic",
        "host": {
            "queue_executable": str(queue),
            "queue_state_root": str(tmp_path / "state"),
            "chipyard": str(tmp_path / "chipyard"),
        },
        "design": {"hw_config": "design_a", "substrate": "firesim_design_a"},
        "workload": {"name": _WORKLOAD, "bootbinary": _BOOT},
        "entries": entries,
    }
    document.update(overrides)
    return FC.load_plan(document)


def _run(plan: FC.CheckpointPlan, tmp_path: Path, queue: FakeQueue) -> dict:
    return FC.run_checkpoint(plan, evidence_root=lambda entry: tmp_path / "evidence" / entry.label, runner=queue)


def test_a_checkpoint_row_is_a_sealed_receipt(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"a.elf": b"a"})
    table = _run(plan, tmp_path, FakeQueue(plan.host, cycles={"a.elf": 4242}))

    (row,) = table["entries"]
    assert (table["sealed"], table["total"], row["status"], row["cycles"]) == (1, 1, "sealed", 4242)
    receipt = json.loads(Path(row["receipt"]).read_text(encoding="utf-8"))
    assert receipt["queue_receipt"]["queue_job_id"] == row["job_id"]
    assert "--hw-config" in receipt["queue_receipt"]["queue_submission"]
    # The evidence outlives the queue's own job directory.
    assert str(tmp_path / "evidence") in receipt["queue_receipt"]["logs"]["queue_daemon"]["path"]


def test_a_queue_that_would_ignore_the_bitstream_flag_is_never_submitted_to(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"a.elf": b"a"})
    queue = FakeQueue(plan.host, help_text=_HELP.replace("[--hw-config H]", ""))
    with pytest.raises(FC.CheckpointError, match="--hw-config"):
        _run(plan, tmp_path, queue)
    assert queue.submitted == []


def test_a_busy_shared_queue_is_left_alone(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"a.elf": b"a"})
    queue = FakeQueue(plan.host, busy=(77,))
    with pytest.raises(FC.CheckpointError, match="busy"):
        _run(plan, tmp_path, queue)
    assert queue.submitted == []


def test_identical_bytes_are_measured_once(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"a.elf": b"same", "b.elf": b"same"})
    queue = FakeQueue(plan.host)
    table = _run(plan, tmp_path, queue)
    assert queue.submitted == ["a.elf"]
    assert [row["status"] for row in table["entries"]] == ["sealed", "duplicate_elf"]


def test_a_run_that_does_not_prove_its_output_is_not_a_number(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"bad.elf": b"bad", "good.elf": b"good"})
    table = _run(plan, tmp_path, FakeQueue(plan.host, broken=("bad.elf",)))
    bad, good = table["entries"]
    assert bad["status"] == "not_sealed" and "cycles" not in bad
    assert good["status"] == "sealed"  # one broken program does not cost the others their slot
    assert (table["sealed"], table["total"]) == (1, 2)


def test_checkpoints_on_different_designs_are_not_compared(tmp_path: Path) -> None:
    plan = _plan(tmp_path, {"a.elf": b"a"})
    before = _run(plan, tmp_path, FakeQueue(plan.host, cycles={"a.elf": 2000}))
    after = dict(before, entries=[dict(before["entries"][0], cycles=1000)])
    assert FC.compare(after, before) == [
        {"label": "a", "cycles": 1000, "previous_cycles": 2000, "ratio": 0.5, "same_elf": True}
    ]
    with pytest.raises(FC.CheckpointError, match="not comparable"):
        FC.compare(dict(after, hw_config="design_b"), before)


def test_the_bitstream_is_named_and_the_client_is_not_a_symlink(tmp_path: Path) -> None:
    with pytest.raises(FC.CheckpointError, match="hw_config"):
        _plan(tmp_path, {"a.elf": b"a"}, design={"hw_config": "", "substrate": "firesim_design_a"})
    link = tmp_path / "firesim-queue"
    link.symlink_to(tmp_path / "install" / "bin" / "firesim-queue")
    with pytest.raises(FC.CheckpointError, match="plain file"):
        FC.QueueHost(queue_executable=link, queue_state_root=tmp_path / "state", chipyard=tmp_path / "chipyard")


def test_host_policy_shapes_the_client_environment_and_nothing_else(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher"
    launcher.mkdir()
    env = FC.client_environment(
        path_prefix=[launcher], drop=["HOME"], base={"HOME": "/home/x", "PATH": "/usr/bin", "KEEP": "1"}
    )
    assert env == {"PATH": f"{launcher}:/usr/bin", "KEEP": "1"}
    with pytest.raises(FC.CheckpointError, match="absolute directory"):
        FC.client_environment(path_prefix=[tmp_path / "missing"])
    with pytest.raises(FC.CheckpointError, match="unknown client_env"):
        _plan(tmp_path, {"a.elf": b"a"}, client_env={"shell": "yes"})


def test_a_program_in_an_older_uart_dialect_is_observed_and_never_called_sealed(tmp_path: Path) -> None:
    old_dialect = (
        "\n".join(
            (
                "MERLIN_PROFILE warmup begin",
                "MERLIN_PROFILE warmup end rc=0",
                "MERLIN_PROFILE measured begin",
                "conv 1 cycles: 11",
                "Total cycles: 4242 (100%)",
                "PASS",
                "MERLIN_PROFILE measured end rc=0",
            )
        )
        + "\n"
    )
    how = {
        "cycles_prefix": "Total cycles:",
        "after": "MERLIN_PROFILE measured begin",
        "before": "MERLIN_PROFILE measured end rc=0",
        "markers": ["PASS"],
    }
    assert FC.observe_cycles(old_dialect, how) == 4242
    with pytest.raises(FC.CheckpointError, match="exactly one 'PASS'"):  # noqa: PT012
        FC.observe_cycles(old_dialect.replace("PASS\n", ""), how)
    with pytest.raises(FC.CheckpointError, match="expected one"):
        FC.observe_cycles(old_dialect.replace("Total cycles: 4242 (100%)", "Total cycles: 1\nTotal cycles: 2"), how)

    plan = _plan(tmp_path, {"old.elf": b"old"})
    entry = plan.entries[0]
    plan = FC.CheckpointPlan(
        **{
            **plan.__dict__,
            "entries": (
                FC.CheckpointEntry(
                    label=entry.label,
                    model=entry.model,
                    experiment=entry.experiment,
                    elf=entry.elf,
                    validation_policy=entry.validation_policy,
                    observe=how,
                ),
            ),
        }
    )

    class OldDialectQueue(FakeQueue):
        def __call__(self, argv, **kwargs):
            result = super().__call__(argv, **kwargs)
            if "--stage-from" in argv:  # a submission, not the contract's --help
                uart = next((self.host.chipyard / "sims/firesim/deploy/results-workload").rglob("uartlog"))
                uart.write_text(old_dialect, encoding="utf-8")
            return result

    table = _run(plan, tmp_path, OldDialectQueue(plan.host))
    (row,) = table["entries"]
    assert (row["status"], row["observed_cycles"], table["sealed"]) == ("observed", 4242, 0)
    assert "cycles" not in row and "UART" in row["reason"]


_WINDOW = {
    "cycles_prefix": "METRIC cycles",
    "after": "MERLIN_PROFILE measured begin",
    "before": "MERLIN_PROFILE measured end rc=0",
    "markers": list(_MARKERS),
}


def _with_observe(plan: FC.CheckpointPlan, how: dict) -> FC.CheckpointPlan:
    entry = plan.entries[0]
    observed = FC.CheckpointEntry(
        label=entry.label,
        model=entry.model,
        experiment=entry.experiment,
        elf=entry.elf,
        validation_policy=entry.validation_policy,
        observe=how,
    )
    return FC.CheckpointPlan(**{**plan.__dict__, "entries": (observed,)})


class CutOffQueue(FakeQueue):
    """A job that closes its measured window and is then killed at its time limit while it drains."""

    def __call__(self, argv, **kwargs):
        result = super().__call__(argv, **kwargs)
        if "--stage-from" not in argv:
            return result
        job = self.next_job - 1
        finished = next((self.host.chipyard / "sims/firesim/deploy/results-workload").rglob("uartlog"))
        live = self.host.queue_state_root / "jobs" / str(job) / "simulation" / "sim_slot_0" / "uartlog"
        live.parent.mkdir(parents=True)
        live.write_text(finished.read_text(encoding="utf-8") + "OUT Y1 18080 64 1 2 3", encoding="utf-8")
        finished.unlink()  # a killed job never publishes its results directory
        client = result.stdout.replace("terminal state=DONE exit_code=0", "terminal state=TIMEOUT exit_code=-15")
        return subprocess.CompletedProcess(argv, 1, client, None)


def test_a_window_closed_before_the_time_limit_is_read_only_where_the_entry_said_so(tmp_path: Path) -> None:
    plan = _with_observe(_plan(tmp_path, {"drains.elf": b"drains"}), {**_WINDOW, FC.JOB_MAY_NOT_FINISH: True})
    (row,) = _run(plan, tmp_path, CutOffQueue(plan.host, cycles={"drains.elf": 33085}))["entries"]
    # Its own status: weaker than observed, which is weaker than sealed, and it names the job's fate.
    assert (row["status"], row["observed_cycles"], row["job_state"]) == (FC.OBSERVED_INCOMPLETE, 33085, "TIMEOUT")
    assert "cycles" not in row and Path(row["evidence_dir"], "uartlog").is_file()


def test_without_the_declaration_a_cut_off_job_is_not_a_number(tmp_path: Path) -> None:
    # The mutation: the same job, the same complete window, no declaration made before the run.
    plan = _with_observe(_plan(tmp_path, {"drains.elf": b"drains"}), dict(_WINDOW))
    (row,) = _run(plan, tmp_path, CutOffQueue(plan.host, cycles={"drains.elf": 33085}))["entries"]
    assert row["status"] == "not_sealed" and "observed_cycles" not in row
    # The evidence is kept all the same, so the refusal can be audited.
    assert Path(row["evidence_dir"], "uartlog").is_file() and row["job_state"] == "TIMEOUT"


def test_a_job_that_already_ran_is_read_only_if_it_staged_these_bytes_on_this_design(tmp_path: Path) -> None:
    plan = _with_observe(_plan(tmp_path, {"drains.elf": b"drains"}), {**_WINDOW, FC.JOB_MAY_NOT_FINISH: True})
    entry = plan.entries[0]
    job = plan.host.queue_state_root / "jobs" / "701"
    staged = job / "deploy_overlay" / "workloads" / _WORKLOAD / _BOOT
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"drains")
    (job / "simulation" / "sim_slot_0").mkdir(parents=True)
    (job / "simulation" / "sim_slot_0" / "uartlog").write_text(_uart(33085) + "OUT Y1 18080 64 1 2", encoding="utf-8")
    record = {
        "workload": _WORKLOAD,
        "bootbinary": _BOOT,
        "hw_config": plan.hw_config,
        "hwdb_config_artifact_sha256": None,
    }
    (job / "runworkload-full.json").write_text(json.dumps(record), encoding="utf-8")

    row = FC.observe_finished_job(plan, entry, 701, tmp_path / "evidence" / "existing")
    assert (row["status"], row["observed_cycles"], row["read_from_existing_job"]) == (
        FC.OBSERVED_INCOMPLETE,
        33085,
        True,
    )

    staged.write_bytes(b"other bytes")
    with pytest.raises(FC.CheckpointError, match="did not stage this entry's executable"):
        FC.observe_finished_job(plan, entry, 701, tmp_path / "evidence" / "other")
    staged.write_bytes(b"drains")
    (job / "runworkload-full.json").write_text(json.dumps({**record, "hw_config": "design_b"}), encoding="utf-8")
    with pytest.raises(FC.CheckpointError, match="not this plan's"):
        FC.observe_finished_job(plan, entry, 701, tmp_path / "evidence" / "design")
