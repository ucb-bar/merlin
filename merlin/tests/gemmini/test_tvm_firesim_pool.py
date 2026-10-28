"""The manager contract must fail closed and serialize physical slots."""

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common.paths import repo_root

ADAPTER = """
import hashlib,json,sys,time,signal
from pathlib import Path
request,result,mode=map(str,sys.argv[1:])
request=json.loads(Path(request).read_text());folder=Path(result).parent
def ident(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def cancel(*_):
    Path(result).write_text(json.dumps(dict(status="cancelled",remote_stopped=True,elf_sha256=request["elf"]["sha256"])))
    sys.exit(0)
if mode=="cancel":signal.signal(signal.SIGTERM,cancel)
start=time.monotonic();time.sleep(.15)
(folder/"interval.json").write_text(json.dumps([start,time.monotonic()]))
boot=folder/"boot.json"
boot.write_text(json.dumps(dict(elf_sha256=request["elf"]["sha256"] if mode!="boot" else "wrong")))
first=999999999 if mode=="startup" else 123
log=folder/"guest.log"
log.write_text(f"BOOT_LOADING_CYCLES=999999999\\nDSE_CYCLES first={first} second=100\\n"
               +("DSE_PASS\\n" if mode!="marker" else ""))
value=dict(status="passed",guest_exit=0,elf_sha256=request["elf"]["sha256"],hardware_sha256=request["hardware"].get("sha256"),transcript=ident(log),boot_evidence=ident(boot),wall_limit_enforced=True,remote_stopped=mode!="running")
if mode=="sha":value["transcript"]["sha256"]="wrong"
if mode=="deadline":value["wall_limit_enforced"]=False
Path(result).write_text(json.dumps(value))
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/comparisons/tvm"))
    spec = importlib.util.spec_from_file_location(
        "firesim_pool_test", repo_root() / "examples/gemmini/comparisons/tvm/firesim_job_pool.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(ADAPTER)
    hardware = tmp_path / "hardware.json"
    hardware.write_text('{"scope":"unit test only; no hardware execution"}')
    elf = tmp_path / "tiny.elf"
    elf.write_bytes(b"unit test ELF")

    def make(mode="ok", slots=1, with_receipt=True, **kwargs):
        config = tmp_path / (mode + str(slots) + ".json")
        value = dict(
            persistent_manager=True,
            adapter_sources=[module.identity(adapter)],
            adapter_cancels_remote_on_timeout=True,
            slots=[
                dict(
                    id=str(i),
                    lock=str(tmp_path / f"slot-{i}.lock"),
                    command=[sys.executable, str(adapter), "{request}", "{result}", mode],
                )
                for i in range(slots)
            ],
        )
        if with_receipt:
            value["hardware_receipt"] = str(hardware)
        config.write_text(json.dumps(value))
        return module.FireSimPool(config, deadline_epoch=time.time() + 20, **kwargs)

    return module, make, elf


def test_valid_warm_interval(setup, tmp_path):
    _, make, elf = setup
    assert make().run(elf, tmp_path / "job") == 100


@pytest.mark.parametrize("mode", ["boot", "marker", "sha", "deadline", "running"])
def test_reject_unbound_or_unbounded_result(setup, tmp_path, mode):
    _, make, elf = setup
    with pytest.raises(ValueError):
        make(mode).run(elf, tmp_path / "job")


def test_request_uses_wall_watchdog_without_incumbent_filter(setup, tmp_path):
    _, make, elf = setup
    pool = make(finish_active_runs=True, active_run_timeout_seconds=10)
    assert pool.run(elf, tmp_path / "job", wall_timeout=lambda: 3) == 100
    request = json.loads((tmp_path / "job/request.json").read_text())
    assert request["wall_seconds"] == 3
    assert request["schema"] == "gemmini_firesim_probe_request_v4"
    assert request["incumbent_cycle_pruning"] is False
    assert "measured_cycle_limit" not in request and "live_limit" not in request


def test_same_slot_serializes_across_pool_instances(setup, tmp_path):
    _, make, elf = setup
    pools = [make(), make()]
    with ThreadPoolExecutor(2) as executor:
        assert list(executor.map(lambda i: pools[i].run(elf, tmp_path / f"job-{i}"), [0, 1])) == [100, 100]
    intervals = sorted(json.loads((tmp_path / f"job-{i}/interval.json").read_text()) for i in range(2))
    assert intervals[0][1] <= intervals[1][0]


def test_separate_slots_run_concurrently(setup, tmp_path):
    _, make, elf = setup
    pool = make(slots=2)
    with ThreadPoolExecutor(2) as executor:
        assert list(executor.map(lambda i: pool.run(elf, tmp_path / f"job-{i}"), [0, 1])) == [100, 100]
    intervals = sorted(json.loads((tmp_path / f"job-{i}/interval.json").read_text()) for i in range(2))
    assert intervals[1][0] < intervals[0][1]


def test_deadline_prevents_launch(setup, tmp_path):
    _, make, elf = setup
    pool = make()
    pool.deadline = time.time() - 1
    with pytest.raises(TimeoutError):
        pool.run(elf, tmp_path / "job")
    assert not (tmp_path / "job/request.json").exists()


def test_unconfirmed_timeout_quarantines_slot(setup, tmp_path):
    _, make, elf = setup
    pool = make()
    with pytest.raises(TimeoutError):
        pool.run(elf, tmp_path / "timed-out", wall_timeout=.05)
    assert (tmp_path / "slot-0.lock.quarantine").exists()
    with pytest.raises(RuntimeError, match="quarantined"):
        make().run(elf, tmp_path / "next-job")


def test_confirmed_timeout_cancellation_allows_slot_reuse(setup, tmp_path):
    _, make, elf = setup
    with pytest.raises(TimeoutError):
        make("cancel").run(elf, tmp_path / "timed-out", wall_timeout=.1)
    assert not (tmp_path / "slot-0.lock.quarantine").exists()
    assert make().run(elf, tmp_path / "next-job") == 100


def test_startup_and_warmup_are_not_the_cost_label(setup, tmp_path):
    _, make, elf = setup
    assert make("startup").run(elf, tmp_path / "job") == 100
    request = json.loads((tmp_path / "job/request.json").read_text())
    assert request["startup_included"] is False
    assert "max_total_cycles" not in request


def test_provided_ddr3_bitstream_needs_no_memory_setup_or_receipt(setup, tmp_path):
    _, make, elf = setup
    assert make(with_receipt=False).run(elf, tmp_path / "job") == 100
    request = json.loads((tmp_path / "job/request.json").read_text())
    assert request["expected_memory_model"] == "DDR3"
    assert request["configure_memory"] is False


def test_finish_active_uses_separate_watchdog_and_records_drain(setup, tmp_path, monkeypatch):
    from types import SimpleNamespace

    module, make, elf = setup
    pool = make(finish_active_runs=True, active_run_timeout_seconds=10)
    clock = {"now": 100.0}
    pool.deadline = 101.0
    original_popen = module.subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        wait = process.wait
        def completed(*args, **kwargs):
            result = wait(*args, **kwargs)
            clock["now"] = 103.0
            return result
        process.wait = completed
        return process

    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock["now"], sleep=time.sleep, monotonic=time.monotonic))
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    folder = tmp_path / "draining"
    assert pool.run(elf, folder) == 100
    request = json.loads((folder / "request.json").read_text())
    assert request["admission_deadline_epoch"] == 101
    assert request["deadline_epoch"] == 110
    assert request["finish_active_runs"] is True
    assert json.loads((folder / "occupancy.json").read_text())["post_budget_seconds"] == 2
    with pytest.raises(TimeoutError, match="deadline"):
        pool.run(elf, tmp_path / "late")
    assert not (tmp_path / "late/request.json").exists()


@pytest.mark.parametrize("timeout", [None, 0, -1, float("nan"), float("inf")])
def test_drain_requires_bounded_job_watchdog(setup, timeout):
    _, make, _ = setup
    with pytest.raises(ValueError, match="watchdog"):
        make(finish_active_runs=True, active_run_timeout_seconds=timeout)


def test_quarantine_created_during_lease_acquisition_prevents_launch(setup, tmp_path, monkeypatch):
    module, make, elf = setup
    pool = make()
    original = module.fcntl.flock

    def quarantine_on_acquire(handle, operation):
        original(handle, operation)
        if operation == module.fcntl.LOCK_EX | module.fcntl.LOCK_NB:
            (tmp_path / "slot-0.lock.quarantine").write_text("unconfirmed previous job")

    monkeypatch.setattr(module.fcntl, "flock", quarantine_on_acquire)
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.run(elf, tmp_path / "race")
    assert not (tmp_path / "race/request.json").exists()
