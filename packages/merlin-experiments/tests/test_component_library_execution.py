"""Explicit public library transport through a real CPU-only numerical run.

The private primitive compiler and source-built stock simulator are diagnostic
selections. No device extension, target backend or runtime qualification exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
from merlin_experiments.phase2 import component_runtime_controls as controls
from merlin_experiments.phase2.component_runtime_fixture import prepare_source_control

from merlin.common import invocation_record as I
from merlin.common.paths import data_path, module_source_path, runtime_dir
from merlin.runtime.backends.base import parse_console
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.execution_service import FunctionalExecutionService
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy
from merlin.targetgen.native_component_execution import execute_component


class ActualCommands:
    def __init__(self):
        self.calls = []

    def build_package(self, package, **kwargs):
        assert not package.manifest.get("build")

    def run_entrypoint(self, package, name, source, output=None, *, timeout, invocation_directory, **kwargs):
        self.calls.append(name)
        argv = P._resolve_argv(package, name, source, output)
        return I.run(
            [sys.executable, "-I", "-B", *argv[1:]],
            directory=invocation_directory,
            stage=name,
            inputs=(Path(source),),
            outputs=(Path(output),) if output is not None else (),
            dependencies=tuple(package.directory.rglob("*.py")),
            cwd=package.directory,
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def verify_original(*, source, lowered_mlir, **kwargs):
    return controls.verify_primitive_llvm(source.read_text(), lowered_mlir.read_text(), entry_symbol="control_entry")


class StockCpuRunner:
    def __init__(self, simulator, root):
        self.simulator, self.root = simulator, root
        self.calls = 0

    def run(self, elf, *, timeout, **kwargs):
        self.calls += 1
        result = I.run(
            [str(self.simulator), "--isa=rv64gc", str(elf)],
            directory=self.root,
            stage="explicit_stock_cpu",
            inputs=(Path(elf),),
            dependencies=(self.simulator, Path(__file__).resolve()),
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        result.check_returncode()
        return result.stdout


@pytest.fixture
def numerical_transport(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import base

    selected = os.environ.get("MERLIN_TEST_STOCK_CPU_SIMULATOR")
    compiler = os.environ.get("MERLIN_TEST_RISCV_GCC")
    if not selected or not compiler or not toolchain.mlir_translate().is_file():
        pytest.skip("requires explicitly selected stock CPU simulator and ordinary build tools")
    if not Path(toolchain.clang()).is_file():
        pytest.skip("requires explicitly selected LLVM object compiler")
    monkeypatch.setattr(base, "get_backend", lambda *args: pytest.fail("independent CPU run discovered a backend"))
    simulator, cc = Path(selected).resolve(strict=True), Path(compiler).resolve(strict=True)
    baremetal = runtime_dir() / "baremetal" / "spike"
    sources = tuple(baremetal / name for name in ("crt.S", "htif.c", "libc_min.c"))
    script = baremetal / "link.ld"
    owner = Path(__file__).resolve()
    build_pins = tuple(
        (str(path), file_digest(path))
        for path in (
            owner,
            cc,
            script,
            *sources,
            baremetal / "htif.h",
            module_source_path("merlin.runtime.direct_kernel_harness"),
        )
    )
    recipe = HarnessBuildRecipe(
        cc,
        (baremetal,),
        sources,
        script,
        0x80000000,
        ("-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin", "-nostdlib"),
        kernel_stack_frame=KernelStackFramePolicy("control_entry", 1024),
        header_dependencies=(baremetal / "htif.h",),
    )
    build = BuildOnlyService(
        "cpu_diagnostic",
        recipe,
        partial(render_direct_kernel, abi=DirectKernelAbi("control_entry", None, 8, "little", "primary_context_id")),
        build_pins,
    )
    contract = tmp_path / "contract"
    (contract / "schemas").mkdir(parents=True)
    for name in ("manifest.schema.json", "command_buffer.schema.json", "capsule.schema.json"):
        shutil.copyfile(data_path("contract") / "schemas" / name, contract / "schemas" / name)
    target = tmp_path / "target.yaml"
    target.write_text("target: cpu_diagnostic\n")
    fixture = prepare_source_control(
        name="source_correspondence.positive",
        root=tmp_path / "original",
        build_service=build,
        contract_root=contract,
        target_descriptor=target,
    )
    library_root = tmp_path / "selected-library"
    (library_root / "merlin").mkdir(parents=True)
    (library_root / "merlin/__init__.py").write_text("")
    (library_root / "merlin/portable.py").write_text("def identity(value):\n    return value\n")
    library = freeze_compiler_library(
        library_root,
        review_id="explicit diagnostic API review; no generality or runtime authority",
        public_modules=("merlin.portable",),
        sources=(("merlin/__init__.py", "merlin"), ("merlin/portable.py", "merlin.portable")),
    )
    candidate = fixture.grade_arguments["package_dir"]
    driver = candidate / "driver.py"
    text = driver.read_text().replace(
        "source = Path(source_path).read_text()", "source = identity(Path(source_path).read_text())"
    )
    driver.write_text(
        text.replace(
            "from __future__ import annotations\n",
            "from __future__ import annotations\nimport sys\n"
            + f"sys.path.insert(0, {str(library_root)!r})\nfrom merlin.portable import identity\n",
            1,
        )
    )
    runner = StockCpuRunner(simulator, tmp_path / "native")
    execution = FunctionalExecutionService(
        "cpu_diagnostic",
        "stock_cpu_diagnostic",
        runner.run,
        parse_console,
        tuple(
            (str(path), file_digest(path))
            for path in (owner, simulator, module_source_path("merlin.runtime.backends.base"))
        ),
        json.dumps(
            {
                "tool_sha256": file_digest(simulator),
                "scope": "CPU-only functional diagnostic, no device or timing authority",
            }
        ),
    )
    return (
        dict(
            package_dir=candidate,
            capsule_dir=fixture.capsule_root,
            contract_root=contract,
            target="cpu_diagnostic",
            out_dir=tmp_path / "execution",
            build_service=build,
            execution_service=execution,
            source_verifier=verify_original,
            readback_policy=ReadbackPolicy(FULL_VALUES_B64),
            timeout_s=60,
        ),
        library,
        library_root,
        runner,
    )


def test_selected_library_runs_original_complete_cpu_outputs(numerical_transport):
    arguments, library, root, runner = numerical_transport
    commands = ActualCommands()
    with P.scoped_package_executor(commands):
        result = execute_component(**arguments, compiler_library=library, compiler_library_root=root)
    assert result["status"] == "numeric_match_diagnostic"
    assert result["numeric_report"]["status"] == "pass" and runner.calls == 1
    assert len(commands.calls) == 4
    assert result["compiler_library"]["contract_sha256"] == library.sha256
    assert set(result["numeric_report"]["per_output"]) == {"sum", "copy", "identity"}
    assert sum(row["n_elements"] for row in result["numeric_report"]["per_output"].values()) == 9
    assert result["numeric_report"]["mismatch_count"] == 0
    assert Path(result["elf"]["path"]).is_file()
    member = root / "merlin/portable.py"
    lowering = next(row for row in result["invocations"] if row["stage"] == "component_source_lowering")
    record = I.verify(Path(lowering["record"]["path"]))
    assert any(row["path"] == str(member) and row["sha256"] == file_digest(member) for row in record["dependencies"])


@pytest.mark.parametrize("selection", ["absent", "changed"])
def test_unselected_or_changed_library_refuses_before_commands(numerical_transport, selection):
    arguments, library, root, runner = numerical_transport
    keywords = {}
    if selection == "changed":
        (root / "merlin/portable.py").write_text("def identity(value):\n    return None\n")
        keywords = {"compiler_library": library, "compiler_library_root": root}
    commands = ActualCommands()
    with P.scoped_package_executor(commands), pytest.raises((ValueError, RuntimeError, P.CertFailure)):
        execute_component(**arguments, **keywords)
    assert commands.calls == [] and runner.calls == 0
