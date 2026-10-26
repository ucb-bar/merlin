"""The manager contract must fail closed and serialize physical slots."""

import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

ADAPTER = """
import hashlib,json,sys,time
from pathlib import Path
request,result,mode=map(str,sys.argv[1:])
request=json.loads(Path(request).read_text());folder=Path(result).parent
def ident(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
start=time.monotonic();time.sleep(.15)
(folder/"interval.json").write_text(json.dumps([start,time.monotonic()]))
boot=folder/"boot.json";boot.write_text(json.dumps(dict(elf_sha256=request["elf"]["sha256"] if mode!="boot" else "wrong")))
first=999999999 if mode=="startup" else 123
log=folder/"guest.log";log.write_text(f"BOOT_LOADING_CYCLES=999999999\\nDSE_CYCLES first={first} second=100\\n"+("DSE_PASS\\n" if mode!="marker" else ""))
value=dict(status="passed",guest_exit=0,elf_sha256=request["elf"]["sha256"],hardware_sha256=request["hardware"].get("sha256"),transcript=ident(log),boot_evidence=ident(boot),wall_limit_enforced=True,measured_limit_enforced=mode!="limit")
if mode=="sha":value["transcript"]["sha256"]="wrong"
if mode=="deadline":value["wall_limit_enforced"]=False
Path(result).write_text(json.dumps(value))
"""


@pytest.fixture
def setup(tmp_path):
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

    def make(mode="ok", slots=1, with_receipt=True):
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
        return module.FireSimPool(config, deadline_epoch=time.time() + 20)

    return module, make, elf


def test_valid_warm_interval(setup, tmp_path):
    _, make, elf = setup
    assert make().run(elf, tmp_path / "job", measured_limit=101) == 100


@pytest.mark.parametrize("mode", ["boot", "marker", "sha", "limit", "deadline"])
def test_reject_unbound_or_unbounded_result(setup, tmp_path, mode):
    _, make, elf = setup
    with pytest.raises(ValueError):
        make(mode).run(elf, tmp_path / "job", measured_limit=101)


@pytest.mark.parametrize("limit", [99, 100])
def test_reject_completed_slower_or_equal_candidate(setup, tmp_path, limit):
    _, make, elf = setup
    with pytest.raises(TimeoutError):
        make().run(elf, tmp_path / "job", measured_limit=limit)


def test_live_refresh_requires_backend_confirmation(setup, tmp_path):
    _, make, elf = setup
    with pytest.raises(ValueError, match="live incumbent"):
        make().run(
            elf,
            tmp_path / "job",
            measured_limit=lambda: 101,
            live_limit=dict(path=str(tmp_path / "limits.json"), key="test"),
        )


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


def test_wall_timeout_releases_slot(setup, tmp_path):
    _, make, elf = setup
    pool = make()
    pool.deadline = time.time() + 0.03
    with pytest.raises(TimeoutError):
        pool.run(elf, tmp_path / "timed-out")
    assert make().run(elf, tmp_path / "next-job") == 100


def test_startup_and_warmup_do_not_consume_active_cycle_limit(setup, tmp_path):
    _, make, elf = setup
    assert make("startup").run(elf, tmp_path / "job", measured_limit=101) == 100
    request = json.loads((tmp_path / "job/request.json").read_text())
    assert request["startup_included"] is False
    assert "max_total_cycles" not in request


def test_provided_ddr3_bitstream_needs_no_memory_setup_or_receipt(setup, tmp_path):
    _, make, elf = setup
    assert make(with_receipt=False).run(elf, tmp_path / "job") == 100
    request = json.loads((tmp_path / "job/request.json").read_text())
    assert request["expected_memory_model"] == "DDR3"
    assert request["configure_memory"] is False


def test_live_incumbent_can_tighten_after_submission(setup, tmp_path):
    _, make, elf = setup
    live = tmp_path / "limits.json"
    live.write_text('{"test":99}')
    pool = make()
    # The test wrapper does not implement live control, so supply that explicit
    # acknowledgement here and still require the client to reject a stale cost.
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        ADAPTER.replace("wall_limit_enforced=True,", "wall_limit_enforced=True,moving_limit_enforced=True,")
    )
    module = setup[0]
    pool.config["adapter_sources"] = [module.identity(adapter)]
    with pytest.raises(TimeoutError):
        pool.run(elf, tmp_path / "job", measured_limit=101, live_limit=dict(path=str(live), key="test"))
