"""Board batches: several candidates and the reference CONTROL in one board job, the control-drift rule,
and board outages that defer rather than refuse."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import batch as B
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured.identity import normalize_build_record, read_json, write_json_atomic

from merlin.perf import whole_model_verdict as V

SEPARATOR = "=== MERLIN_VARIANT"

BATCH_SIM = f'''
import json, sys
for index, program in enumerate(json.loads(open(sys.argv[-1]).read()), 1):
    print("{SEPARATOR}", index)
    print("MERLIN_INVOCATIONS warmup=1 measured=1")
    total = 0
    for g, v in sorted(program["groups"].items(), key=lambda kv: int(kv[0])):
        cycles = int(v["cycles"] * program.get("scale", 1.0))
        print(f"GM_GROUP {{g}} matmul {{cycles}} sum={{v['sum']}} fnv1a={{v['fnv']}}")
        print(f"GM_WORDS {{g}} bytes=16 digest={{v['sum']}}")
        total += cycles
    print(f"FM full model cycles: {{total}}")
    print(f"GM_ARGMAX got={{program['argmax']}} want=3 agrees=1")
    print("MERLIN_WINDOW end label=model")
'''


class FakeDriver:
    """The target driver's batch linker and console splitter, over JSON "ELFs"."""

    def __init__(self, scales=None):
        self.scales = dict(scales or {})
        self.linked = []

    def link_batch(self, variants, out, **kw):
        out.mkdir(parents=True, exist_ok=True)
        programs = []
        for variant in variants:
            program = json.loads(Path(variant["elf"]).read_text())
            program["scale"] = self.scales.get(variant["label"], 1.0)
            programs.append(program)
        self.linked.append([v["label"] for v in variants])
        elf = out / "batch.elf"
        elf.write_text(json.dumps(programs))
        return {"elf": str(elf)}

    def split_batch(self, text, count):
        blocks, current, index = {}, [], None
        for line in text.splitlines():
            if line.startswith(SEPARATOR):
                if index is not None:
                    blocks[index] = "\n".join(current)
                index, current = int(line.split()[-1]), []
            else:
                current.append(line)
        if index is not None:
            blocks[index] = "\n".join(current)
        return blocks


def _board_job(tmp_path: Path, store: Path, name: str, *, machine: dict, argmax: int = 3, **fields) -> Path:
    """A job built and graded locally, waiting for the board, exactly as the worker leaves one.  Each
    ``name`` is its own package bytes (a note differs), the same program."""
    builder = FX.write_builder(tmp_path)
    pkg = FX.package(tmp_path, name, argmax=argmax, doc=name)
    job_dir = FX.job_dir_for(store, pkg, machine=machine, builder=builder, **fields)
    spec = {}
    exec(FX.BUILDER_SOURCE, spec)
    raw = spec["build"](job_dir / "package", target="toy", out_dir=job_dir / "build_timing")
    build = normalize_build_record(raw, package_sha256=json.loads((job_dir / "job.json").read_text())["package_sha256"])
    local_log = job_dir / "run_local" / "uart.log"
    local_log.parent.mkdir(parents=True)
    local_log.write_text(
        subprocess.run([*FX.write_sim(tmp_path), build["elf"]], capture_output=True, text=True, check=True).stdout
    )
    for stub in ("program.o", "syscalls.o"):
        (job_dir / stub).write_text("o")
    job = json.loads((job_dir / "job.json").read_text())
    write_json_atomic(
        job_dir / J.BOARD_REQUEST,
        {
            "builds": {"timing": build, "local": build},
            "identities": {"timing": {}, "local": {}},
            "local_run": {"completed": True, "uart_log": str(local_log)},
            "local_device": {"machine": "spike"},
            "check": None,
            "package_sha256": job["package_sha256"],
            "variant": {
                "label": job["package_sha256"],
                "elf": build["elf"],
                "program_object": str(job_dir / "program.o"),
                "objects": [],
                "supports": [str(job_dir / "syscalls.o")],
                "program": {"compiler": "/bin/true", "flags": [], "link_flags": [], "link_script": "x.ld"},
            },
        },
    )
    job.update(state=J.BOARD, board_ready_epoch=1.0)
    write_json_atomic(job_dir / "job.json", job)
    return job_dir


def _batched_machine(tmp_path: Path, *, control: dict | None = None) -> dict:
    sim = tmp_path / "batch_sim.py"
    sim.write_text(BATCH_SIM)
    timing = {
        "kind": "spike",
        "target": "toy",
        "command": [sys.executable, str(sim)],
        "environment": {},
        "identity_files": [],
    }
    return {
        "kind": "batched",
        "target": "toy",
        "timing": timing,
        "local": FX.spike_machine(tmp_path),
        "control": control or {},
    }


def _device(spec: dict) -> str:
    from merlin_experiments.phase2.whole_model_measured.machines import machine_from_spec

    return machine_from_spec(spec["timing"]).identity().binary_sha256


def _solo(control_dir: Path, *, device_sha256: str, cycles: int = 300, **fields) -> dict:
    """The control's SOLO board reading, as the board's own result records it."""
    request = read_json(control_dir / J.BOARD_REQUEST)
    return {
        "timing_status": V.TIMING_MEASURED,
        "verdict": {"whole_window_cycles": cycles},
        "run": {"uart_log": str(control_dir / "run_local" / "uart.log")},
        "build": {"elf_sha256": request["builds"]["timing"]["elf_sha256"]},
        "device": {"binary_sha256": device_sha256},
        "finished_at": "20261005T120000Z",
        **fields,
    }


def _control(tmp_path: Path, store: Path, **solo_fields) -> dict:
    """The reference arm's own board request and its SOLO result, as a launch declares the control."""
    machine = _batched_machine(tmp_path)
    control_dir = _board_job(tmp_path, tmp_path / "control_store", "vendor", machine=machine)
    solo = _solo(control_dir, device_sha256=_device(machine), **solo_fields)
    write_json_atomic(control_dir / "solo_result.json", solo)
    return {
        "board_request": str(control_dir / J.BOARD_REQUEST),
        "solo_result": str(control_dir / "solo_result.json"),
        "cycles_tolerance": 0.02,
    }


def test_the_control_check_needs_both_the_cycles_and_every_groups_bytes():
    block = "\n".join(["GM_GROUP 1 k 100 sum=1 fnv1a=1", "GM_WORDS 1 bytes=4 digest=5", "FM full model cycles: 101"])
    assert B.control_check({}, block, 100, {"1": (4, 5)})["ok"]
    assert not B.control_check({}, block, 90, {"1": (4, 5)})["ok"]
    moved = B.control_check({}, block, 100, {"1": (4, 6)})
    assert not moved["ok"] and moved["groups_whose_bytes_moved"] == ["1"]
    assert B.control_check({"unstable_groups": ["1"]}, block, 100, {"1": (4, 6)})["ok"]
    assert not B.control_check({}, None, 100, {})["ok"]


def test_a_batch_waits_for_its_controls_solo_measurement(tmp_path):
    store = tmp_path / "store"
    control = _control(tmp_path, store)
    solo = Path(control["solo_result"])
    kept = solo.read_text()
    solo.unlink()
    spec = _batched_machine(tmp_path, control=control)
    _board_job(tmp_path, store, "a", machine=spec)
    assert not B.worth_starting(store, {**spec, "batch_wait_seconds": 0}, clock=1e12)
    assert "no readable solo result" in read_json(store / B.CONTROL_PREFLIGHT)["reason"]
    solo.write_text("{}")  # a file is not a reading
    assert not B.worth_starting(store, {**spec, "batch_wait_seconds": 0}, clock=1e12)
    solo.write_text(kept)
    assert B.worth_starting(store, {**spec, "batch_wait_seconds": 0}, clock=1e12)
    write_json_atomic(store / B.BOARD_OUTAGE, {"retry_after_epoch": 2e12})
    assert not B.worth_starting(store, {**spec, "batch_wait_seconds": 0}, clock=1e12)


def test_the_control_preflight_refuses_every_reading_that_cannot_judge_this_batch(tmp_path):
    """Each way a 'solo result' can fail to be a solo board reading of this program on this device is
    refused by name; only the real one passes."""
    control = _control(tmp_path, tmp_path / "store")
    solo_path = Path(control["solo_result"])
    good = read_json(solo_path)
    device = good["device"]["binary_sha256"]
    assert B.control_preflight(control, device_sha256=device)["ok"]
    for mutation, why in (
        ({"timing_status": V.TIMING_MEASURED_INVALID}, "not a MEASURED reading"),
        ({"verdict": {}}, "no whole-window cycle count"),
        ({"batch": {"size": 3}}, "inside a batch of 3"),
        ({"build": {"elf_sha256": "f" * 64}}, "the batch links"),
        ({"device": {"binary_sha256": "e" * 64}}, "another machine"),
    ):
        write_json_atomic(solo_path, {**good, **mutation})
        verdict = B.control_preflight(control, device_sha256=device)
        assert not verdict["ok"] and why in verdict["reason"], (mutation, verdict)
        assert verdict["reason"].startswith(B.INFRA_CONTROL_UNMEASURED)


def test_a_control_measured_on_another_machine_holds_the_batch_before_any_board_job(tmp_path):
    store = tmp_path / "store"
    control = _control(tmp_path, store, device={"binary_sha256": "d" * 64})
    spec = _batched_machine(tmp_path, control=control)
    jobs = [_board_job(tmp_path, store, name, machine=spec) for name in ("a", "b")]
    driver = FakeDriver()
    B.batch_main(store, driver=driver)
    assert driver.linked == []  # nothing linked, nothing run
    for job_dir in jobs:
        job = read_json(job_dir / "job.json")
        assert job["state"] == J.BOARD and "no board job was spent" in job["notice"]
        assert job["control_preflight_holds"] and not (job_dir / "result.json").exists()
    assert "another machine" in read_json(store / B.CONTROL_PREFLIGHT)["reason"]


def test_the_drift_tolerance_is_the_machines_own_and_is_recorded(tmp_path):
    """A control 3% off its solo reading drifts under the declared 2% -- unless this device's own solo
    repeats of one program already spread 4% across days, which the batch then records as its basis."""
    store = tmp_path / "store"
    control = _control(tmp_path, store, finished_at="20260930T120000Z")
    spec = _batched_machine(tmp_path, control=control)
    jobs = [_board_job(tmp_path, store, name, machine=spec) for name in ("a", "b")]
    B.batch_main(store, driver=FakeDriver(scales={"control": 1.03}))
    assert all(read_json(j / "job.json").get("control_drifts") for j in jobs)
    # Solo repeats of the control's program on this device: two the same day, one two days later, 4% off.
    solo = read_json(Path(control["solo_result"]))
    for name, day, cycles in (("r1", "20261001", 300), ("r2", "20261001", 301), ("r3", "20261003", 312)):
        repeat = store / f"repeat_{name}"
        repeat.mkdir(parents=True)
        document = {**solo, "verdict": {"whole_window_cycles": cycles}, "finished_at": f"{day}T120000Z"}
        write_json_atomic(repeat / "result.json", document)
    for job_dir in jobs:
        job = read_json(job_dir / "job.json")
        job.update(solo=False, control_drifts=[])
        write_json_atomic(job_dir / "job.json", job)
    B.batch_main(store, driver=FakeDriver(scales={"control": 1.03}))
    result = read_json(jobs[0] / "result.json")
    rule = result["batch"]["control"]["tolerance_rule"]
    assert result["batch"]["control"]["ok"] and rule["basis"] == "cross_day_solo_spread" and rule["declared"] == 0.02


def test_a_batch_with_its_control_in_tolerance_finishes_every_candidate_from_its_own_block(tmp_path):
    store = tmp_path / "store"
    spec = _batched_machine(tmp_path, control=_control(tmp_path, store))
    jobs = [_board_job(tmp_path, store, name, machine=spec, argmax=3) for name in ("a", "b")]
    (store / "batches").mkdir()
    driver = FakeDriver()
    B.batch_main(store, driver=driver)
    assert driver.linked[0][0] == "control"  # the first batch puts it first; the next, last
    for job_dir in jobs:
        result = read_json(job_dir / "result.json")
        assert result["timing_status"] == V.TIMING_MEASURED and result["objective_cycles"] == 300
        assert result["batch"]["control"]["ok"] and result["batch"]["size"] == 3
        assert read_json(job_dir / "job.json")["state"] == J.DONE


def test_a_drifted_control_sends_each_candidate_back_alone_once_then_refuses(tmp_path):
    store = tmp_path / "store"
    spec = _batched_machine(tmp_path, control=_control(tmp_path, store))
    jobs = [_board_job(tmp_path, store, name, machine=spec) for name in ("a", "b")]
    (store / "batches").mkdir()
    B.batch_main(store, driver=FakeDriver(scales={"control": 1.05}))
    for job_dir in jobs:
        job = read_json(job_dir / "job.json")
        assert job["state"] == J.BOARD and job["solo"] is True and len(job["control_drifts"]) == 1
        assert job["notice"] == J.CONTROL_DRIFT_NOTICE and not (job_dir / "result.json").exists()
    # A job that already used its re-measurement is refused by the drift, with the reason.
    for job_dir in jobs:
        job = read_json(job_dir / "job.json")
        job["solo"] = False
        write_json_atomic(job_dir / "job.json", job)
    B.batch_main(store, driver=FakeDriver(scales={"control": 1.05}))
    refused = [read_json(job_dir / "result.json") for job_dir in jobs]
    assert all(
        r["timing_status"] == V.TIMING_REFUSED and "control" in str(r["verdict"].get("refusal")) for r in refused
    )


def test_a_board_that_never_ran_the_batch_defers_every_job(tmp_path, monkeypatch):
    from merlin_experiments.phase2.whole_model_measured import machines as M

    store = tmp_path / "store"
    spec = _batched_machine(tmp_path)
    job_dir = _board_job(tmp_path, store, "a", machine=spec, solo=True)
    monkeypatch.setattr(
        M.SpikeMachine,
        "run",
        lambda self, elf, workdir, *, timeout_s: {
            "completed": False,
            M.INFRA_BOARD_UNAVAILABLE: True,
            "incomplete_reason": "no FPGA",
        },
    )
    B.batch_main(store, driver=FakeDriver())
    job = read_json(job_dir / "job.json")
    assert job["state"] == J.BOARD and job["notice"] == J.BOARD_UNAVAILABLE_NOTICE and job["board_losses"]
    assert B.board_outage(store)["failures"] and not (job_dir / "result.json").exists()


def test_a_job_whose_board_objects_are_gone_is_rebuilt_not_linked(tmp_path):
    store = tmp_path / "store"
    job_dir = _board_job(tmp_path, store, "a", machine=_batched_machine(tmp_path), solo=True)
    (job_dir / "program.o").unlink()
    driver = FakeDriver()
    B.batch_main(store, driver=driver)
    job = read_json(job_dir / "job.json")
    assert job["state"] == J.PENDING and job["board_rebuilds"] and not driver.linked
    assert read_json(job_dir / J.ATTEMPTS_DIR / "0" / J.ATTEMPT_RECORD)["kind"] == "unlinkable_attempt"


# --------------------------------------------------------------- solo jobs never starve a new group


def test_a_capped_run_of_solo_repeats_does_not_starve_a_new_candidate_group(tmp_path):
    """Any waiting solo job used to preempt a pending multi-variant group unconditionally, so a long
    run of solo repeats (a control-drift requeue, a chain of confirmations) could starve a freshly
    queued group for as long as solo work kept arriving. The cap forces the group through instead,
    and a forced (or ordinary) multi-variant pick resets the streak it is counted against."""
    machine = _batched_machine(tmp_path)
    store = tmp_path / "store"
    for i in range(2):
        _board_job(tmp_path, store, f"m{i}", machine=machine)
    for i in range(B.MAX_CONSECUTIVE_SOLO_BATCHES + 5):
        _board_job(tmp_path, store, f"solo{i}", machine=machine, solo=True)
    picks = []
    for _ in range(B.MAX_CONSECUTIVE_SOLO_BATCHES + 1):
        chosen = B.batch_candidates(store, batch_size=8)
        picks.append(len(chosen))
        B.record_batch_kind(store, solo=len(chosen) == 1 and bool(chosen[0].get("solo")))
    assert picks[: B.MAX_CONSECUTIVE_SOLO_BATCHES] == [1] * B.MAX_CONSECUTIVE_SOLO_BATCHES
    assert picks[B.MAX_CONSECUTIVE_SOLO_BATCHES] == 2, "the pending group is forced through at the cap"
    assert B._solo_streak(store) == 0, "a forced multi-variant pick resets the streak"
    # The next round sees solo work again, unconstrained -- the cap protected one group, not the board.
    assert len(B.batch_candidates(store, batch_size=8)) == 1


def test_a_fresh_promotions_repeat_outranks_an_older_one_waiting_in_the_same_queue(tmp_path):
    """Two solo jobs, the same age: the one requested at the objective's elevated
    PROMOTED_REPEAT_PRIORITY (a freshly-promoted best's confirmation) is picked first, never by
    arrival order alone."""
    machine = _batched_machine(tmp_path)
    store = tmp_path / "store"
    _board_job(tmp_path, store, "old_repeat", machine=machine, solo=True, priority=0)
    _board_job(tmp_path, store, "fresh_repeat", machine=machine, solo=True, priority=B.PROMOTED_REPEAT_PRIORITY)
    (chosen,) = B.batch_candidates(store, batch_size=8)
    assert chosen.get("priority") == B.PROMOTED_REPEAT_PRIORITY
