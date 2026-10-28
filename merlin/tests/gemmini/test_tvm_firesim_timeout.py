"""Median watchdog calibration, context isolation and stable per-job limits."""

import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def policy(tmp_path):
    path = repo_root() / "examples/gemmini/comparisons/tvm/firesim_timeout.py"
    spec = importlib.util.spec_from_file_location("median_watchdog_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def make(**overrides):
        options = dict(bootstrap_seconds=20, ceiling_seconds=60, binding={"program": "a", "hardware": "b"})
        options.update(overrides)
        return module.MedianWatchdog(tmp_path / "history.json", **options)
    return make


def calibrate(watchdog, key="a", samples=(4, 5, 6)):
    for seconds in samples:
        assert watchdog.timeout(key, baseline=True) == 20
        watchdog.observe(key, seconds, baseline=True)


def test_three_baselines_are_required_and_workload_histories_are_separate(policy):
    watchdog = policy()
    with pytest.raises(ValueError, match="three"):
        watchdog.timeout("a")
    calibrate(watchdog)
    assert watchdog.timeout("a") == 15
    with pytest.raises(ValueError, match="three"):
        watchdog.timeout("b")
    with pytest.raises(ValueError, match="already calibrated"):
        watchdog.observe("a", 2, baseline=True)


def test_recent_window_floor_ceiling_and_existing_job_limit(policy):
    watchdog = policy(ceiling_seconds=25)
    calibrate(watchdog)
    admitted_limit = watchdog.timeout("a")
    for _ in range(8):
        watchdog.observe("a", 8)
    assert watchdog.timeout("a") == 15
    watchdog.observe("a", 8)
    assert watchdog.timeout("a") == 24
    assert admitted_limit == 15  # A previously admitted job does not tighten.
    for _ in range(9):
        watchdog.observe("a", 1)
    assert watchdog.timeout("a") == 15
    for _ in range(9):
        watchdog.observe("a", 20)
    assert watchdog.timeout("a") == 25


def test_inconsistent_pilot_ceiling_refuses_instead_of_clipping_baseline(policy):
    watchdog = policy(ceiling_seconds=25)
    calibrate(watchdog, samples=(9, 10, 11))
    with pytest.raises(ValueError, match="below 3x"):
        watchdog.timeout("a")


def test_resume_preserves_history_and_refuses_changed_context(policy):
    watchdog = policy()
    calibrate(watchdog)
    for _ in range(9):
        watchdog.observe("a", 8)
    assert policy().timeout("a") == 24
    with pytest.raises(ValueError, match="another policy"):
        policy(ceiling_seconds=120)
    with pytest.raises(ValueError, match="another policy"):
        policy(binding={"program": "changed", "hardware": "b"})


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_durations_never_enter_history(policy, value):
    watchdog = policy()
    with pytest.raises(ValueError, match="positive finite"):
        watchdog.observe("a", value, baseline=True)
    assert not watchdog.path.exists()


def test_threaded_successes_keep_one_atomic_bounded_history(policy):
    watchdog = policy()
    calibrate(watchdog)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: watchdog.observe("a", 8), range(12)))
    state = json.loads(watchdog.path.read_text())["workloads"]["a"]
    assert state["successes"] == 15
    assert state["recent"] == [8] * 9
    assert policy().timeout("a") == 24


@pytest.fixture
def measurements(tmp_path, monkeypatch, policy):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("candidate_measurements_test", folder / "tune_firesim_chunks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Pool:
        hardware = {"scope": "unit test"}
        calls = []
        fail = False

        def run(self, elf, output, *, wall_timeout):
            self.calls.append(wall_timeout())
            if self.fail:
                raise TimeoutError("unit-test hang")
            output.mkdir(parents=True)
            index = len(self.calls) - 1
            cycles = [100, 120, 110, 150][min(index, 3)]
            (output / "result.json").write_text(json.dumps({"cycles": cycles}))
            (output / "occupancy.json").write_text(json.dumps({"occupied_seconds": [4, 5, 6, 8][min(index, 3)]}))
            return cycles

    def artifact(name):
        path = tmp_path / name
        path.mkdir()
        elfs = {}
        for label in ("valid", "bad-oracle"):
            elf = path / (label + ".elf")
            elf.write_bytes((name + label).encode())
            elfs[label] = module.identity(elf)
        metadata = path / "artifact.json"
        metadata.write_text(json.dumps(dict(native=True, folder=str(path), workload="a", elfs=elfs)))
        return metadata

    pool, watchdog = Pool(), policy()
    return module.CandidateMeasurements(pool, watchdog), pool, watchdog, artifact


def test_baselines_are_real_runs_and_completed_slower_candidate_is_retained(measurements):
    run, pool, watchdog, artifact = measurements
    baseline = artifact("baseline")
    assert run.calibrate(baseline) == 110
    assert pool.calls == [20, 20, 20]
    assert run(baseline) == 110  # Exact cache reuse adds no timing observation.
    assert len(pool.calls) == 3
    slow = artifact("slow")
    assert run(slow) == 150
    assert pool.calls == [20, 20, 20, 15]
    assert json.loads((slow.parent / "measurement.json").read_text())["cycles"] == 150
    assert watchdog.state["workloads"]["a"]["recent"] == [8]
    assert run(slow) == 150 and len(pool.calls) == 4


def test_timeout_is_not_a_successful_duration_or_cycle_label(measurements):
    run, pool, watchdog, artifact = measurements
    run.calibrate(artifact("baseline"))
    pool.fail = True
    failed = artifact("timeout")
    with pytest.raises(TimeoutError):
        run(failed)
    assert watchdog.state["workloads"]["a"]["recent"] == []
    assert not (failed.parent / "measurement.json").exists()


def test_cached_control_elf_identity_is_still_checked(measurements):
    run, pool, _, artifact = measurements
    baseline = artifact("baseline")
    run.calibrate(baseline)
    (baseline.parent / "bad-oracle.elf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed candidate ELF"):
        run(baseline)
    assert len(pool.calls) == 3


def test_python_lowering_change_invalidates_duration_history(measurements, tmp_path):
    from types import SimpleNamespace
    _, _, watchdog, _ = measurements
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    spec = importlib.util.spec_from_file_location("source_binding_test", folder / "tune_firesim_chunks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = tmp_path / "python/tvm/relax/backend/contrib"
    backend.mkdir(parents=True)
    lowering = backend / "gemmini_search.py"
    lowering.write_text("original")
    apps = tmp_path / "apps"
    apps.mkdir()
    (apps / "measure.py").write_text("builder")
    tvm = SimpleNamespace(__file__=str(tmp_path / "python/tvm/__init__.py"))
    measure = SimpleNamespace(__file__=str(apps / "measure.py"))
    before = module.measurement_sources(tvm, measure)
    bound = dict(python_sources=before)
    first = module.MedianWatchdog(watchdog.path, bootstrap_seconds=20, ceiling_seconds=60, binding=bound)
    calibrate(first)
    lowering.write_text("changed")
    with pytest.raises(ValueError, match="another policy"):
        module.MedianWatchdog(watchdog.path, bootstrap_seconds=20, ceiling_seconds=60,
                              binding=dict(python_sources=module.measurement_sources(tvm, measure)))
