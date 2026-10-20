"""Ordinary launcher selection custody; compiler and native work is intercepted."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_measurement_execution as M

from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.perf.component_measurement_stream import RawMeasurementPlan
from merlin.targetgen.contract import process_execution as process_owner
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.contract.process_execution import RecordedProcessExecution
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy


def renderer(cb, *, inputs, readback_policy):
    raise AssertionError("source selection controls must not render or compile")


def parser(console):
    raise AssertionError("source selection controls must not execute or parse native output")


@pytest.fixture
def selected(tmp_path, monkeypatch):
    # These individually owned files select no runtime. The fixed process's
    # verification can run, but compiler, runner and real accounting cannot.
    tool = tmp_path / "unused-tool"
    tool.write_text("#!/bin/sh\nexit 9\n")
    tool.chmod(0o700)
    script = tmp_path / "unused-link.ld"
    script.write_text("/* no link is launched */\n")
    own_source = Path(__file__).resolve()
    process_source = Path(process_owner.__file__).resolve()
    process = RecordedProcessExecution(
        tool,
        ("{elf}",),
        tmp_path,
        (("PATH", "/usr/bin:/bin"),),
        tmp_path / "process",
        "stdout",
        tuple((str(path), file_digest(path)) for path in (tool, process_source)),
    )
    pins = (*process.source_pins, (str(own_source), file_digest(own_source)))
    execution = FunctionalExecutionService(
        "fixture", "unexecuted", process.run_elf, parser, pins, '{"scope":"source control only"}', process
    )
    build = BuildOnlyService("fixture", HarnessBuildRecipe(tool, (), (), script, 0), renderer, pins)
    original = {}
    for name in ("package", "capsule", "contract"):
        root = tmp_path / name
        root.mkdir()
        (root / "original.txt").write_text(name + "\n")
        original[name] = root
    arguments = dict(
        plan=RawMeasurementPlan(COMPLETE_STAGES, 8192, 4096),
        out_dir=tmp_path / "measurement",
        native_environments=(("unexecuted", (("PATH", "/usr/bin:/bin"),)),),
        package_dir=original["package"],
        capsule_dir=original["capsule"],
        contract_root=original["contract"],
        target="fixture",
        build_service=build,
        execution_service=execution,
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        source_verifier=renderer,
        timeout_s=10,
    )
    calls, accounting = [], []

    def ordinary(**kwargs):
        calls.append(kwargs)
        output = kwargs["out_dir"]
        output.mkdir()
        for name in ("fixture.elf", "console.txt", "result.json"):
            (output / name).write_text("unexecuted source-control fixture\n")
        return {
            "target": "fixture",
            "elf": {"path": str(output / "fixture.elf")},
            "console": {"path": str(output / "console.txt")},
            "emission": {},
        }

    @contextmanager
    def observed(*args, **kwargs):
        accounting.append(kwargs)
        yield SimpleNamespace(returned=lambda: None)

    def collect(**kwargs):
        return {"scope": "intercepted source control; no execution or qualification", "plan": kwargs["plan"].record()}

    monkeypatch.setattr(M, "execute_component", ordinary)
    monkeypatch.setattr(M.I, "observe_call", observed)
    monkeypatch.setattr(M.I, "run", lambda *a, **k: pytest.fail("source control launched a native process"))
    monkeypatch.setattr(M, "collect_component_measurement", collect)
    return arguments, calls, accounting, ordinary, collect


@pytest.mark.parametrize("change", ["order", "console", "frame", "policy"])
def test_changed_selection_during_ordinary_work_refuses_before_accounting(selected, monkeypatch, change):
    arguments, calls, accounting, ordinary, _ = selected

    def changed(**kwargs):
        result = ordinary(**kwargs)
        if change == "policy":
            object.__setattr__(arguments["readback_policy"], "transport", COHERENT_DUMP_V1)
        else:
            field, value = {
                "order": ("stage_order", tuple(reversed(COMPLETE_STAGES))),
                "console": ("max_console_bytes", 16384),
                "frame": ("max_frame_bytes", 8192),
            }[change]
            object.__setattr__(arguments["plan"], field, value)
        return result

    monkeypatch.setattr(M, "execute_component", changed)
    with pytest.raises(ValueError, match="observation selection changed"):
        M.execute_component_measurement(**arguments)
    assert len(calls) == 1 and accounting == []
    assert not (arguments["out_dir"] / "raw_measurement.json").exists()


@pytest.mark.parametrize("change", ["plan", "policy"])
def test_changed_selection_during_accounting_refuses_before_publication(selected, monkeypatch, change):
    arguments, calls, accounting, _, collect = selected

    def changed(**kwargs):
        result = collect(**kwargs)
        if change == "plan":
            object.__setattr__(arguments["plan"], "max_console_bytes", 16384)
        else:
            object.__setattr__(arguments["readback_policy"], "transport", COHERENT_DUMP_V1)
        return result

    monkeypatch.setattr(M, "collect_component_measurement", changed)
    with pytest.raises(ValueError, match="observation selection changed"):
        M.execute_component_measurement(**arguments)
    assert len(calls) == len(accounting) == 1
    assert not (arguments["out_dir"] / "raw_measurement.json").exists()


@pytest.mark.parametrize("order", [COMPLETE_STAGES, tuple(reversed(COMPLETE_STAGES))])
def test_unchanged_explicit_order_reaches_existing_accounting(selected, order):
    arguments, calls, accounting, _, _ = selected
    arguments["plan"] = replace(arguments["plan"], stage_order=order)
    original = arguments["plan"].record()
    result = M.execute_component_measurement(**arguments)
    assert len(calls) == len(accounting) == 1
    assert result["plan"] == accounting[0]["arguments"]["plan"] == original
    assert json.loads((arguments["out_dir"] / "raw_measurement.json").read_text()) == result


def test_missing_recorded_selection_refuses_before_ordinary_work(selected):
    arguments, calls, accounting, _, _ = selected
    arguments["execution_service"] = replace(arguments["execution_service"], process_transport=None)
    with pytest.raises(ValueError, match="actual selected recorded process"):
        M.execute_component_measurement(**arguments)
    assert calls == accounting == [] and not arguments["out_dir"].exists()


def test_mismatched_transport_refuses_before_ordinary_work(selected):
    arguments, calls, accounting, _, _ = selected
    arguments["readback_policy"] = ReadbackPolicy(COHERENT_DUMP_V1)
    with pytest.raises(ValueError, match="complete text full-value"):
        M.execute_component_measurement(**arguments)
    assert calls == accounting == [] and not arguments["out_dir"].exists()
