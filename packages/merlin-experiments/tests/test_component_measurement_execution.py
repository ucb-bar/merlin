"""Actual ordinary CPU output/event controls, with unqualified stage semantics."""

import importlib.util
import json
import os
import shutil
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
from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.perf.component_measurement_stream import RawMeasurementPlan
from merlin.runtime.backends.base import parse_console
from merlin.runtime.direct_kernel_harness import DirectKernelAbi
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.contract.process_execution import RecordedProcessExecution
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, FULL_VALUES_B64, ReadbackPolicy

spec = importlib.util.spec_from_file_location(
    "measurement_execution_control", Path(__file__).with_name("measurement_execution_control.py")
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


@pytest.fixture
def ordinary(tmp_path, monkeypatch, request):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    selected = os.environ.get("MERLIN_TEST_STOCK_CPU_SIMULATOR")
    cc = os.environ.get("MERLIN_TEST_RISCV_GCC")
    environment_file = os.environ.get("MERLIN_TEST_MEASUREMENT_ENV_SELECTION")
    if (
        not selected
        or not cc
        or not environment_file
        or not toolchain.mlir_translate().is_file()
        or not Path(toolchain.clang()).is_file()
    ):
        pytest.skip("requires explicit stock CPU, translation and object/link tools")
    monkeypatch.setattr(base, "get_backend", lambda *args: pytest.fail("measurement discovered a backend"))
    simulator, cc = Path(selected).resolve(strict=True), Path(cc).resolve(strict=True)
    compilation_environment = json.loads(Path(environment_file).read_text())
    # Exact pytest-owned additions are derived from this selected test/tool,
    # never recovered from a native process environment or digest alone.
    compilation_environment.update(
        PYTEST_VERSION=pytest.__version__, PYTEST_CURRENT_TEST=request.node.nodeid + " (call)"
    )
    native_environments = tuple(
        (stage, tuple(sorted(environment.items())))
        for stage, environment in {
            **{
                stage: {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
                for stage in ("parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact")
            },
            **{stage: compilation_environment for stage in ("llvm_translation", "object", "harness_object", "elf")},
        }.items()
    )
    baremetal = runtime_dir() / "baremetal/spike"
    stock = tuple(baremetal / name for name in ("crt.S", "htif.c", "libc_min.c"))
    support = tmp_path / "raw_events.c"
    script = baremetal / "link.ld"
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json", "capsule.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    target = tmp_path / "target.yaml"
    target.write_text("target: cpu_diagnostic\n")

    def assemble(mode):
        support.write_text(control.support_source(mode))
        recipe = HarnessBuildRecipe(
            cc,
            (baremetal,),
            (*stock, support),
            script,
            0x80000000,
            ("-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib"),
            ldflags=("-Wl,--wrap=htif_exit",),
            kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
            header_dependencies=(baremetal / "htif.h",),
        )
        pins = tuple(
            (str(path), file_digest(path))
            for path in (
                Path(control.__file__),
                cc,
                script,
                *stock,
                support,
                baremetal / "htif.h",
                module_source_path("merlin.runtime.direct_kernel_harness"),
            )
        )
        build = BuildOnlyService(
            "cpu_diagnostic",
            recipe,
            partial(control.render, abi=DirectKernelAbi("control_entry", None, 8, "little", "primary_context_id")),
            pins,
        )
        original = prepare_source_control(
            name="source_correspondence.positive",
            root=tmp_path / "original",
            build_service=build,
            contract_root=contract,
            target_descriptor=target,
        )
        process = RecordedProcessExecution(
            simulator,
            ("--isa=rv64gc", "{elf}"),
            tmp_path,
            (("PATH", "/usr/bin:/bin"), ("LC_ALL", "C")),
            tmp_path / "process",
            "combined",
            tuple(
                (str(path), file_digest(path))
                for path in (simulator, module_source_path("merlin.targetgen.contract.process_execution"))
            ),
        )
        execution = FunctionalExecutionService(
            "cpu_diagnostic",
            "stock_cpu_diagnostic",
            process.run_elf,
            parse_console,
            (
                *process.source_pins,
                (
                    str(module_source_path("merlin.runtime.backends.base")),
                    file_digest(module_source_path("merlin.runtime.backends.base")),
                ),
            ),
            json.dumps({"scope": "stock CPU diagnostic; no stage or timer qualification"}),
            process,
        )
        return dict(
            plan=RawMeasurementPlan(COMPLETE_STAGES, 8192, 4096),
            out_dir=tmp_path / "measurement",
            package_dir=original.grade_arguments["package_dir"],
            capsule_dir=original.capsule_root,
            contract_root=contract,
            target="cpu_diagnostic",
            build_service=build,
            execution_service=execution,
            source_verifier=control.verify_original,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
            timeout_s=60,
            native_environments=native_environments,
        )

    return assemble


def run(arguments):
    with P.scoped_package_executor(control.ActualCommands()):
        return execute_component_measurement(**arguments)


def test_fresh_cpu_consumes_complete_original_outputs_and_raw_events(ordinary):
    arguments = ordinary("complete")
    observed = run(arguments)
    assert observed["numeric_report"]["mismatch_count"] == 0
    assert set(observed["numeric_report"]["per_output"]) == {"copy", "identity", "sum"}
    assert sum(row["n_elements"] for row in observed["numeric_report"]["per_output"].values()) == 9
    assert len(observed["raw_events"]["events"]) == 22
    assert observed["raw_events"]["cold"] is observed["raw_events"]["warm"] is None
    native = I.verify(Path(observed["actual_consumption"]["record"]["path"]))
    assert native["inputs"] == [observed["elf"]]
    assert native["argv"][-1] == observed["elf"]["path"]
    product = arguments["out_dir"] / "raw_measurement.json"
    for stage, values in arguments["native_environments"]:
        assert observed["native_environments"][stage] == I.environment_identity(dict(values))
    assert os.environ["MERLIN_TEST_MEASUREMENT_ENV_SELECTION"] not in product.read_text()
    assert all("values" not in identity for identity in observed["native_environments"].values())
    accounting = next((arguments["out_dir"] / "accounting").rglob("invocation.json"))
    actual_accounting = I.verify(accounting)
    assert {"path": str(product), "sha256": file_digest(product)} in actual_accounting["outputs"]
    assert actual_accounting["arguments"]["native_environments"] == observed["native_environments"]
    assert actual_accounting["arguments"]["consumption_environment"] == observed["consumption_environment"]
    assert actual_accounting["arguments"]["original_input_trees_sha256"]
    for module in (
        "merlin.targetgen.capsule_golden",
        "merlin.targetgen.native_component_execution",
        "merlin.targetgen.contract.readback_policy",
        "merlin.runtime.backends.base",
    ):
        path = module_source_path(module)
        assert {"path": str(path), "sha256": file_digest(path)} in actual_accounting["dependencies"]


@pytest.mark.parametrize("mode", ["missing", "moved"])
def test_actual_missing_or_moved_events_refuse_with_original_numeric_pass(ordinary, mode):
    arguments = ordinary(mode)
    with pytest.raises(ValueError, match="roster|order"):
        run(arguments)
    result = json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())
    assert result["numeric_report"]["mismatch_count"] == 0
    assert not (arguments["out_dir"] / "raw_measurement.json").exists()
    assert (
        json.loads(next((arguments["out_dir"] / "accounting").rglob("invocation.json")).read_text())["status"]
        == "interrupted"
    )


@pytest.mark.parametrize("changed", ["elf", "console", "compiler", "record_alias"])
def test_changed_consumed_products_cannot_be_recollected(ordinary, changed):
    arguments = ordinary("complete")
    observed = run(arguments)
    if changed in ("elf", "console"):
        Path(observed[changed]["path"]).write_bytes(b"changed actual bytes")
    elif changed == "compiler":
        (arguments["package_dir"] / "driver.py").write_text("raise SystemExit(0)\n")
    else:
        external = arguments["out_dir"].parent / "excluded.json"
        external.write_text("not read as an invocation")
        (arguments["out_dir"] / "ordinary/added").mkdir()
        (arguments["out_dir"] / "ordinary/added/invocation.json").symlink_to(external)
    with pytest.raises(ValueError, match="changed|membership"):
        collect_component_measurement(
            result_path=arguments["out_dir"] / "ordinary/result.json",
            plan=arguments["plan"],
            build_service=arguments["build_service"],
            execution_service=arguments["execution_service"],
            native_environments=arguments["native_environments"],
            original_inputs=json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())["inputs"],
        )


@pytest.mark.parametrize("product", ["result", "console"])
def test_actual_file_growth_at_open_refuses_before_decode(ordinary, monkeypatch, product):
    arguments = ordinary("complete")
    observed = run(arguments)
    selected = (
        arguments["out_dir"] / "ordinary/result.json" if product == "result" else Path(observed["console"]["path"])
    )
    limit = 8 * 1024 * 1024 if product == "result" else arguments["plan"].max_console_bytes
    original_open = Path.open
    selected_opens = 0

    def grow_at_open(path, mode="r", *args, **kwargs):
        nonlocal selected_opens
        if path == selected and mode == "rb":
            selected_opens += 1
            if selected_opens == 2:
                with original_open(path, "ab") as stream:
                    stream.write(b"x" * (limit + 1))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_at_open)
    with pytest.raises(ValueError, match="byte budget"):
        collect_component_measurement(
            result_path=arguments["out_dir"] / "ordinary/result.json",
            plan=arguments["plan"],
            build_service=arguments["build_service"],
            execution_service=arguments["execution_service"],
            native_environments=arguments["native_environments"],
            original_inputs=json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())["inputs"],
        )
    assert selected_opens == 2


def test_unrecorded_callback_refuses_before_compiler(ordinary):
    arguments = ordinary("complete")
    arguments["execution_service"] = replace(arguments["execution_service"], process_transport=None)
    with pytest.raises(ValueError, match="recorded process"):
        run(arguments)
    assert not arguments["out_dir"].exists()


def test_unsupported_coherent_replay_refuses_before_compiler(ordinary):
    arguments = ordinary("complete")
    arguments["readback_policy"] = ReadbackPolicy(COHERENT_DUMP_V1)
    with pytest.raises(ValueError, match="complete text"):
        run(arguments)
    assert not arguments["out_dir"].exists()


@pytest.mark.parametrize("defect", ["missing", "changed"])
def test_actual_environment_selection_must_reopen_in_full(ordinary, defect):
    arguments = ordinary("complete")
    run(arguments)
    selection = arguments["native_environments"]
    if defect == "missing":
        selection = tuple(row for row in selection if row[0] != "object")
    else:
        selection = tuple(
            (stage, (*values, ("UNSELECTED_VALUE", "not actually consumed"))) if stage == "object" else (stage, values)
            for stage, values in selection
        )

    with pytest.raises(ValueError, match="environment"):
        collect_component_measurement(
            result_path=arguments["out_dir"] / "ordinary/result.json",
            plan=arguments["plan"],
            build_service=arguments["build_service"],
            execution_service=arguments["execution_service"],
            native_environments=selection,
            original_inputs=json.loads((arguments["out_dir"] / "ordinary/result.json").read_text())["inputs"],
        )


@pytest.mark.parametrize("defect", ["report", "report_roster", "output", "output_roster", "metrics", "scalar_type"])
def test_forged_saved_numerical_results_refuse_with_unchanged_native_stream(ordinary, defect):
    arguments = ordinary("complete")
    observed = run(arguments)
    result_path = arguments["out_dir"] / "ordinary/result.json"
    result = json.loads(result_path.read_text())
    original_inputs = result["inputs"]
    console_pin = dict(observed["console"])
    process_pin = dict(observed["actual_consumption"]["record"])
    if defect == "report":
        result["numeric_report"]["max_abs_error"] = 17
    elif defect == "report_roster":
        result["numeric_report"]["per_output"].pop("copy")
    elif defect == "output":
        first = next(iter(result["native"]["outputs"]))
        result["native"]["outputs"][first][0][0] += 1
    elif defect == "output_roster":
        result["native"]["outputs"].pop(next(iter(result["native"]["outputs"])))
    elif defect == "metrics":
        result["native"]["raw_metrics"]["invented"] = 17
    else:
        result["numeric_report"]["mismatch_count"] = False
    result_path.write_text(json.dumps(result, sort_keys=True))
    with pytest.raises(ValueError, match="outputs|numerical report"):
        collect_component_measurement(
            result_path=result_path,
            plan=arguments["plan"],
            build_service=arguments["build_service"],
            execution_service=arguments["execution_service"],
            native_environments=arguments["native_environments"],
            original_inputs=original_inputs,
        )
    assert file_digest(Path(console_pin["path"])) == console_pin["sha256"]
    assert file_digest(Path(process_pin["path"])) == process_pin["sha256"]
    I.verify(Path(process_pin["path"]))


def test_resigned_original_selection_cannot_replace_original_reference(ordinary):
    arguments = ordinary("complete")
    run(arguments)
    result_path = arguments["out_dir"] / "ordinary/result.json"
    result = json.loads(result_path.read_text())
    original_inputs = json.loads(json.dumps(result["inputs"]))
    result["inputs"]["capsule"].pop(next(iter(result["inputs"]["capsule"])))
    result_path.write_text(json.dumps(result, sort_keys=True))
    with pytest.raises(ValueError, match="original input selection"):
        collect_component_measurement(
            result_path=result_path,
            plan=arguments["plan"],
            build_service=arguments["build_service"],
            execution_service=arguments["execution_service"],
            native_environments=arguments["native_environments"],
            original_inputs=original_inputs,
        )
