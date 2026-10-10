"""Actual ordinary CPU, original histories and raw object custody only."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from merlin_experiments.phase2.component_measurement_execution import (
    collect_component_measurement,
    execute_component_measurement,
)
from merlin_experiments.phase2.component_runtime_fixture import prepare_source_control

from merlin.common import invocation_record as I
from merlin.common.paths import data_path, module_source_path, runtime_dir
from merlin.perf.component_coherent_measurement import CoherentMeasurementPlan
from merlin.runtime.backends.base import parse_console
from merlin.runtime.direct_kernel_counter import DirectKernelCounterPlan
from merlin.runtime.direct_kernel_harness import DirectKernelAbi
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.runtime.direct_kernel_phases import DirectKernelPhasePlan
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.contract.prepared_process_readback import PreparedProcessReadbackPlan
from merlin.targetgen.contract.process_execution import RecordedProcessExecution
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, ReadbackPolicy


def load_fixture(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load_fixture("coherent_measurement_control")
original_control = load_fixture("measurement_execution_control")
ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def selection():
    original = CompileOnlySourceAbi(
        tuple(CompileOnlyTensor(name, (1, 3), "f32") for name in ("a", "b")),
        tuple(CompileOnlyTensor(name, (1, 3), "f32") for name in ("sum", "copy", "identity")),
    )
    invocation = DirectKernelInvocationPlan(original, 2, "history", "whole_count")
    counter = DirectKernelCounterPlan("raw_counter", "raw_control", "samples", 2, 1024)
    phase = DirectKernelPhasePlan(counter, "phases", 8192)
    abi = DirectKernelAbi("control_entry", "finish_copy", 8, "little", "primary_context_id")
    return PreparedProcessReadbackPlan(abi, invocation, counter, phase, 0, 8192, 65536)


@pytest.fixture
def ordinary(tmp_path, request, monkeypatch):
    from merlin_experiments.phase2 import component_runtime_controls as controls

    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    selected = os.environ.get("MERLIN_TEST_STOCK_CPU_SIMULATOR")
    cc = os.environ.get("MERLIN_TEST_RISCV_GCC")
    environment_path = os.environ.get("MERLIN_TEST_MEASUREMENT_ENV_SELECTION")
    if not selected or not cc or not environment_path or not toolchain.mlir_translate().is_file():
        pytest.skip("requires explicit stock CPU, translation and object/link selections")
    monkeypatch.setattr(base, "get_backend", lambda *args: pytest.fail("coherent observation discovered a backend"))
    cc, simulator = Path(cc).resolve(strict=True), Path(selected).resolve(strict=True)
    compilation = json.loads(Path(environment_path).read_text())
    # The package's selected conftest declares this empty default before build.
    compilation.setdefault("MERLIN_TARGET_PATH", "")
    compilation.update(PYTEST_VERSION=pytest.__version__, PYTEST_CURRENT_TEST=request.node.nodeid + " (call)")
    environments = tuple(
        (stage, tuple(sorted(values.items())))
        for stage, values in {
            **{
                stage: ENV
                for stage in (
                    "parse",
                    "lower_interface_to_target",
                    "emit_command_buffer",
                    "emit_target_artifact",
                    "coherent_stock_cpu",
                    "coherent_elf_symbols",
                )
            },
            **{stage: compilation for stage in ("llvm_translation", "object", "harness_object", "elf")},
        }.items()
    )
    baremetal = runtime_dir() / "baremetal/spike"
    stock = tuple(baremetal / name for name in ("crt.S", "htif.c", "libc_min.c"))
    support = tmp_path / "support.c"
    config = tmp_path / "runner-selection.json"
    config.write_text(
        json.dumps(
            {
                "tool": str(simulator),
                "tool_sha256": file_digest(simulator),
                "recorder": str(Path(I.__file__).resolve()),
                "recorder_sha256": file_digest(Path(I.__file__)),
            },
            sort_keys=True,
        )
    )
    runner = Path(__file__).with_name("coherent_measurement_runner.py").resolve()
    script = baremetal / "link.ld"
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json", "capsule.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    target = tmp_path / "target.yaml"
    target.write_text("target: cpu_diagnostic\n")

    def assemble(mode="complete"):
        plan = selection()
        # Derive the concrete output roster from the original parsed source,
        # not the candidate's returned packet or predicted output values.
        source = (
            "module { func.func @main(%a: tensor<1x3xf32>, %b: tensor<1x3xf32>) "
            "-> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) { "
            "%sum = arith.addf %a, %b : tensor<1x3xf32> "
            "func.return %sum, %b, %a : tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32> } }"
        )
        original_cb = controls.primitive_buffer(controls.parse_primitive(source), "cpu_diagnostic")
        support.write_text(control.support_source(plan, original_cb, mode=mode))
        recipe = HarnessBuildRecipe(
            cc,
            (baremetal,),
            (*stock, support),
            script,
            0x80000000,
            ("-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib"),
            ldflags=("-Wl,--wrap=control_entry", "-Wl,--wrap=htif_exit"),
            kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
            header_dependencies=(baremetal / "htif.h",),
        )
        build = BuildOnlyService(
            "cpu_diagnostic",
            recipe,
            partial(control.render, plan=plan),
            tuple(
                (str(path), file_digest(path))
                for path in (
                    Path(control.__file__),
                    cc,
                    script,
                    *stock,
                    support,
                    baremetal / "htif.h",
                    *plan.source_paths(),
                )
            ),
        )
        original = prepare_source_control(
            name="source_correspondence.positive",
            root=tmp_path / "original",
            build_service=build,
            contract_root=contract,
            target_descriptor=target,
        )
        symbols = Path(os.environ.get("MERLIN_TEST_READELF", "/usr/bin/readelf")).resolve(strict=True)
        reader = control.Reader(plan, symbols)
        process_pins = tuple(
            (str(path), file_digest(path))
            for path in (
                Path("/usr/bin/env"),
                Path(sys.executable).resolve(),
                runner,
                config,
                simulator,
                symbols,
                Path(I.__file__).resolve(),
                module_source_path("merlin.targetgen.contract.process_execution"),
                *plan.source_paths(),
            )
        )
        process = RecordedProcessExecution(
            Path("/usr/bin/env"),
            (sys.executable, "-I", str(runner), str(config), "{elf}", "{request}", "{output}"),
            tmp_path,
            tuple(ENV.items()),
            tmp_path / "process",
            "combined",
            process_pins,
            prepared_readback=plan,
        )
        service_pins = (
            *process_pins,
            *(
                (str(path), file_digest(path))
                for path in (
                    Path(control.__file__),
                    module_source_path("merlin.runtime.backends.base"),
                    module_source_path("merlin.perf.component_coherent_measurement"),
                )
            ),
        )
        execution = FunctionalExecutionService(
            "cpu_diagnostic",
            "stock_cpu_diagnostic",
            process.run_elf,
            parse_console,
            service_pins,
            json.dumps({"scope": "stock CPU raw diagnostic only"}),
            process,
        )
        return dict(
            plan=CoherentMeasurementPlan(plan, 65536),
            out_dir=tmp_path / "measurement",
            package_dir=original.grade_arguments["package_dir"],
            capsule_dir=original.capsule_root,
            contract_root=contract,
            target="cpu_diagnostic",
            build_service=build,
            execution_service=execution,
            source_verifier=original_control.verify_original,
            readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
            memory_readback=reader,
            timeout_s=60,
            native_environments=environments,
        )

    return assemble


def run(arguments):
    with P.scoped_package_executor(original_control.ActualCommands()):
        return execute_component_measurement(**arguments)


def recollect(arguments):
    result = arguments["out_dir"] / "ordinary/result.json"
    return collect_component_measurement(
        result_path=result,
        plan=arguments["plan"],
        build_service=arguments["build_service"],
        execution_service=arguments["execution_service"],
        native_environments=arguments["native_environments"],
        original_inputs=json.loads(result.read_text())["inputs"],
    )


def test_ordinary_coherent_full_roster_and_histories(ordinary):
    arguments = ordinary()
    observed = run(arguments)
    raw = observed["raw_events"]
    assert observed["schema"] == "merlin.component_measurement_execution.v2"
    assert raw["completed_count"] == 2 and len(raw["objects"]) == 39 and raw["object_bytes"] == 476
    assert all(row["status"] == "pass" for row in raw["per_call_numeric_reports"])
    assert raw["call_outputs"][0] == raw["call_outputs"][1] == observed["replayed_outputs"]
    assert raw["cold"] is raw["warm"] is None and len(raw["unknown"]) == 8
    assert raw["decoder_products"]["payload"] == observed["actual_consumption"]["prepared_readback"]["output"]
    assert raw["counter_samples"]["call_samples"][0][2:] == (0, 1)
    assert recollect(arguments) == observed


def test_real_empty_completion_cannot_hide_bad_histories(ordinary):
    arguments = ordinary("completion")
    with pytest.raises(ValueError, match="per-call numerical"):
        run(arguments)
    result = json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())
    assert result["numeric_report"]["status"] == "pass"
    native = I.verify(
        Path(
            arguments["execution_service"].process_transport.consumption(
                elf=Path(result["elf"]["path"]), console="DONE\n"
            )["record"]["path"]
        )
    )
    assert native["returncode"] == 0


@pytest.mark.parametrize("mode", ["missing", "partial"])
def test_actual_incomplete_native_output_refuses(ordinary, mode):
    arguments = ordinary(mode)
    with pytest.raises(subprocess.CalledProcessError, match="non-zero exit status"):
        run(arguments)
    records = list((arguments["out_dir"] / "ordinary").rglob("invocation.json"))
    native = [json.loads(path.read_text()) for path in records]
    assert any(row["stage"] == "coherent_stock_cpu" and row["returncode"] == 0 for row in native)


@pytest.mark.parametrize("member", ["outputs", "numeric_report", "readback_memory"])
def test_resigned_saved_data_cannot_replace_actual_decode(ordinary, member):
    arguments = ordinary()
    run(arguments)
    path = arguments["out_dir"] / "ordinary/result.json"
    result = json.loads(path.read_text())
    if member == "outputs":
        result["native"]["outputs"]["out0"][0][0] = 100.0
    elif member == "numeric_report":
        result["numeric_report"]["mismatch_count"] = 1
    else:
        result["native"]["readback_memory"]["status"] = "incomplete"
    path.write_text(json.dumps(result))
    with pytest.raises(Exception):
        recollect(arguments)


def test_changed_plan_refuses_before_execution(ordinary):
    arguments = ordinary()
    arguments["plan"] = replace(
        arguments["plan"],
        readback=replace(
            arguments["plan"].readback, invocation_plan=replace(arguments["plan"].readback.invocation_plan, count=3)
        ),
    )
    with pytest.raises(ValueError, match="same explicitly selected"):
        run(arguments)
    assert not arguments["out_dir"].exists()


@pytest.mark.parametrize("mode", ["input", "counter_count"])
def test_real_native_input_or_count_defect_refuses(ordinary, mode):
    arguments = ordinary(mode)
    with pytest.raises(ValueError):
        run(arguments)
    result = json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())
    assert result["numeric_report"]["status"] == "pass"
    assert (
        arguments["execution_service"].consumption(elf=Path(result["elf"]["path"]), console="DONE\n")[
            "prepared_readback"
        ]["payload_bytes"]
        == 476
    )


def test_counter_wrap_is_retained_without_elapsed_cost(ordinary):
    observed = run(ordinary("wrapping"))
    counters = observed["raw_events"]["counter_samples"]
    assert set(counters["call_samples"][0][:2]) == {2**64 - 1, 0}
    samples = [sample for phase in observed["raw_events"]["harness_phases"].values() for sample in phase]
    assert any(end < start for start, end, before, after in samples)
    assert observed["raw_events"]["cold"] is observed["raw_events"]["warm"] is None


@pytest.mark.parametrize("member", ["elf", "payload", "request", "decoder"])
def test_changed_consumed_product_refuses_replay(ordinary, member):
    arguments = ordinary()
    observed = run(arguments)
    prepared = observed["actual_consumption"]["prepared_readback"]
    pins = {
        "elf": observed["actual_consumption"]["elf"],
        "payload": prepared["output"],
        "request": prepared["request"],
        "decoder": observed["raw_events"]["decoder_products"]["decoder_product"],
    }
    path = Path(pins[member]["path"])
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    try:
        with pytest.raises((ValueError, RuntimeError)):
            recollect(arguments)
    finally:
        # Preserve every original native invocation product for independent reopening.
        path.write_bytes(original)
    assert recollect(arguments) == observed


def test_changed_explicit_environment_refuses_replay(ordinary):
    arguments = ordinary()
    run(arguments)
    arguments["native_environments"] = tuple(
        (stage, tuple(sorted({**dict(values), "LC_ALL": "changed"}.items())))
        for stage, values in arguments["native_environments"]
    )
    with pytest.raises(ValueError, match="environment"):
        recollect(arguments)


def test_complete_per_call_metadata_budget_precedes_expansion():
    selected = selection()
    plan = CoherentMeasurementPlan(
        replace(selected, invocation_plan=replace(selected.invocation_plan, count=400)), 8192
    )
    with pytest.raises(ValueError, match="per-call metadata"):
        plan.record()


@pytest.mark.parametrize("change", ["frame", "counter", "phase", "console"])
def test_unsupported_coherent_plan_refuses(change):
    plan = CoherentMeasurementPlan(selection(), 8192)
    if change == "console":
        plan = replace(plan, max_console_bytes=True)
    else:
        plan = replace(
            plan,
            readback=replace(
                plan.readback,
                **{
                    "frame": {"frame_bytes": 32},
                    "counter": {"counter_plan": None},
                    "phase": {"phase_plan": None},
                }[change],
            ),
        )
    with pytest.raises(ValueError, match="complete typed"):
        plan.record()
