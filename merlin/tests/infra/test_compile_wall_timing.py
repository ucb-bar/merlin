"""Observed host intervals separate real lease waits from process wall time."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common import proc
from merlin.common.cpu_pool import shared_compute_pool
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe


def _tool(tmp_path):
    script = tmp_path / "tool.py"
    script.write_text("import sys,time; from pathlib import Path\n"
                      "time.sleep(.04)\n"
                      "Path(sys.argv[3]).write_bytes(Path(sys.argv[1]).read_bytes())\n")
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"unchanged actual bytes")
    argv = [sys.executable, "-I", "-S", "-B", str(script), str(source), "-o", str(output)]
    recipe = CompilationRecipe(tmp_path, producer=Path(proc.__file__).resolve())
    return recipe, argv, [script, source], output


def _duration(event):
    return (event["finished_ns"] - event["started_ns"]) / 1e9


def test_real_queued_process_separates_acquisition_from_process_interval(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(min(os.sched_getaffinity(0))))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    recipe, argv, inputs, output = _tool(tmp_path)
    attempted = threading.Event()

    def runner(argv):
        attempted.set()
        with shared_compute_pool(timeout_s=5):
            return proc.run_checked(argv)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with shared_compute_pool():
            future = pool.submit(recipe.run, argv, runner=runner, inputs=inputs, output=output)
            assert attempted.wait(5)
            assert not future.done()
            threading.Event().wait(.08)
        assert future.result(timeout=5).returncode == 0
    recipe.completed(output)
    document = json.loads(recipe.path.read_text())
    events = document["commands"][0]["timing"]["events"]
    acquire = [x for x in events if x["kind"] == "compute_acquire"]
    processes = [x for x in events if x["kind"] == "process"]
    runner_event = next(x for x in events if x["label"] == "command runner")
    total = next(x for x in events if x["label"] == "compilation command")
    assert len(acquire) == len(processes) == 1
    assert _duration(acquire[0]) >= .06 and _duration(processes[0]) >= .03
    assert _duration(total) >= _duration(runner_event) >= _duration(acquire[0]) + _duration(processes[0])
    assert output.read_bytes() == inputs[-1].read_bytes()
    verify_completed_recipe(recipe.path, executable=output,
                            expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest())


def test_unobserved_runner_has_no_invented_process_or_compute_measurement(tmp_path):
    recipe, argv, inputs, output = _tool(tmp_path)
    recipe.run(argv, runner=lambda argv: subprocess.run(argv, check=True), inputs=inputs, output=output)
    events = json.loads(recipe.path.read_text())["commands"][0]["timing"]["events"]
    assert {x["kind"] for x in events} == {"phase"}
    assert {x["label"] for x in events} == {"command runner", "compilation command"}


def test_slot_timeout_records_wait_without_inventing_process_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(min(os.sched_getaffinity(0))))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    recipe, argv, inputs, output = _tool(tmp_path)

    def runner(argv):
        with shared_compute_pool(timeout_s=.04):
            return proc.run_checked(argv)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with shared_compute_pool():
            future = pool.submit(recipe.run, argv, runner=runner, inputs=inputs, output=output)
            with pytest.raises(TimeoutError):
                future.result(timeout=5)
    command = json.loads(recipe.path.read_text())["commands"][0]
    events = command["timing"]["events"]
    acquire = next(x for x in events if x["kind"] == "compute_acquire")
    assert command["status"] == "failed" and acquire["status"] == "failed"
    assert _duration(acquire) >= .03 and not any(x["kind"] == "process" for x in events)
    assert not output.exists()


def test_legacy_recipe_without_optional_timings_remains_verifiable(tmp_path):
    recipe, argv, inputs, output = _tool(tmp_path)
    recipe.run(argv, runner=proc.run_checked, inputs=inputs, output=output)
    recipe.completed(output)
    document = json.loads(recipe.path.read_text())
    for command in document["commands"]:
        command.pop("timing")
    document["schema"] = "merlin.compilation_recipe.v1"
    recipe.path.write_text(json.dumps(document))
    verify_completed_recipe(recipe.path, executable=output,
                            expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest())


def test_failed_and_nested_phases_retain_actual_outcomes(tmp_path):
    recipe, _, _, _ = _tool(tmp_path)
    with I.observe_timings() as outer:
        with pytest.raises(RuntimeError, match="failure"):
            with recipe.phase("preparation"):
                with I.timed_event("phase", "nested"):
                    raise RuntimeError("failure")
    phases = json.loads(recipe.path.read_text())["phases"]
    events = phases[0]["timing"]["events"]
    assert events == outer and [x["label"] for x in events] == ["nested", "preparation"]
    assert all(x["status"] == "failed" for x in events)
    I.verify_timing_document(phases[0]["timing"])
    assert recipe.record["status"] == "prepared"


@pytest.mark.parametrize("mutation", [
    "negative", "bool", "missing", "extra", "unknown", "schema", "outside", "pid", "root", "status",
])
def test_mutated_timing_does_not_pass_recipe_verification(tmp_path, mutation):
    recipe, argv, inputs, output = _tool(tmp_path)
    recipe.run(argv, runner=proc.run_checked, inputs=inputs, output=output)
    recipe.completed(output)
    document = json.loads(recipe.path.read_text())
    timing = document["commands"][0]["timing"]
    event = timing["events"][0]
    if mutation == "negative":
        event["finished_ns"] = event["started_ns"] - 1
    elif mutation == "bool":
        event["started_ns"] = True
    elif mutation == "missing":
        event.pop("finished_ns")
    elif mutation == "extra":
        event["hardware_cycles"] = 0
    elif mutation == "unknown":
        event["kind"] = "hardware_cycles"
    elif mutation == "schema":
        timing["schema"] = "unknown"
    elif mutation == "outside":
        event["finished_ns"] = timing["events"][-1]["finished_ns"] + 1
    elif mutation == "pid":
        event["pid"] += 1
    elif mutation == "root":
        timing["events"][-1]["label"] = "not the command"
    else:
        timing["events"][-1]["status"] = "failed"
    recipe.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="timing"):
        verify_completed_recipe(recipe.path, executable=output,
                                expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest())
