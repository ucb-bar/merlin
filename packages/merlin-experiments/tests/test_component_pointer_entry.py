"""Actual ordinary source dispatch keeps the shared pointer calling contract.

The host ELF control executes the emitted primitive via stock translation and
the shared harness. It grants no target runtime, accelerator, or stage authority.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from merlin_experiments.phase2.component_runtime_control_execution import PrivateRuntimeControlExecutor
from merlin_experiments.phase2.contracts import exact_tree_record

from merlin.common import invocation_record
from merlin.common.paths import runtime_dir
from merlin.runtime.direct_kernel_harness import DirectKernelAbi, render_direct_kernel
from merlin.runtime.out_b64 import OutB64Decoder
from merlin.targetgen import native_component_execution as native
from merlin.targetgen import package_runtime as P
from merlin.targetgen.bundle_harness import emitted_entry_arity
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy

_spec = importlib.util.spec_from_file_location(
    "private_pointer_entry_source_fixture", Path(__file__).with_name("test_component_runtime_support.py")
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
prepared = _fixture.prepared


def _arguments(context, fixture, output):
    return dict(
        package_dir=fixture.grade_arguments["package_dir"],
        capsule_dir=fixture.capsule_root,
        contract_root=context.contract_root,
        target=context.build_service.target,
        out_dir=output,
        build_service=context.build_service,
        execution_service=context.execution_service,
        source_verifier=context._source_verifier,
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        timeout_s=30,
    )


def _mutated(value, defect):
    if defect == "integer":
        value = value.replace("%ptr0: !llvm.ptr", "%carrier: i64", 1)
        return value.replace(") {\n", ") {\n    %ptr0 = llvm.inttoptr %carrier : i64 to !llvm.ptr\n", 1)
    if defect == "variadic":
        return value.replace("%ptr4: !llvm.ptr)", "%ptr4: !llvm.ptr, ...)", 1)
    if defect == "return":
        value = value.replace("%ptr4: !llvm.ptr) {", "%ptr4: !llvm.ptr) -> i32 {", 1)
        return value.replace("llvm.return", "%zero = llvm.mlir.constant(0 : i32) : i32\n llvm.return %zero : i32", 1)
    return value.replace("llvm.func @", "llvm.func " + defect + " @", 1)


@pytest.mark.parametrize("defect", ["integer", "variadic", "return", "fastcc", "internal"])
def test_same_arity_entry_changes_refuse_before_source_proof_or_link(prepared, tmp_path, monkeypatch, defect):
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "ordinary"
    args = _arguments(prepared, fixture, output)
    driver = args["package_dir"] / "driver.py"
    # Evaluator-only counterfactual, independently frozen before ordinary entry
    # dispatch. It is not a candidate compiler seed or an admitted experiment.
    driver.write_text(
        driver.read_text().replace(
            'if __name__ == "__main__":',
            "_selected_emit = emit_primitive_llvm\n"
            "def emit_primitive_llvm(program, entry_symbol):\n"
            "    value = _selected_emit(program, entry_symbol)\n"
            + "\n".join("    " + line for line in _mutation_lines(defect))
            + '\nif __name__ == "__main__":',
        )
    )

    def unselected(*_args, **_kwargs):
        pytest.fail("changed entry reached source proof or native link")

    args["source_verifier"] = unselected
    monkeypatch.setattr(compiler, "run_on_oracle", unselected)
    candidate = args["package_dir"]
    with P.scoped_package_executor(
        PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    ):
        with pytest.raises(native.NativeComponentExecutionError, match="C pointer entry ABI"):
            native.execute_component(**args)
    artifact = (output / "generated/lowered.llvm.mlir").read_text()
    assert emitted_entry_arity(artifact, entry_symbol="control_entry") == 5
    result = json.loads((output / "result.json").read_bytes())
    assert result["status"] == "unavailable" and "numeric_report" not in result
    assert not (output / "build").exists()
    records = [invocation_record.verify(path) for path in output.rglob("invocation.json")]
    assert {"parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact"} <= {
        row["stage"] for row in records
    }


def _mutation_lines(defect):
    # Serialize only literal source transformations, never an import or tool
    # callback into the private compiler's process.
    from merlin_experiments.phase2.component_runtime_controls import emit_primitive_llvm, parse_primitive

    source = "module { func.func @f(%a: tensor<1xf32>, %b: tensor<1xf32>) -> "
    source += "(tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) { func.return %a, %b, %a : "
    source += "tensor<1xf32>, tensor<1xf32>, tensor<1xf32> } }"
    before = emit_primitive_llvm(parse_primitive(source), "control_entry")
    after = _mutated(before, defect)
    # The original body stays untouched; derive the explicit header/return
    # replacements from the parsed source-produced function.
    old_lines, new_lines = before.splitlines(), after.splitlines()
    header = next(line for line in old_lines if "llvm.func" in line)
    changed_header = next(line for line in new_lines if "llvm.func" in line)
    changed = [f"value = value.replace({header!r}, {changed_header!r}, 1)"]
    if defect == "integer":
        changed.append(
            "value = value.replace(') {\\n', ') {\\n    %ptr0 = llvm.inttoptr %carrier : i64 to !llvm.ptr\\n', 1)"
        )
    elif defect == "return":
        changed.append(
            "value = value.replace('llvm.return', '%zero = llvm.mlir.constant(0 : i32) : i32\\n "
            "llvm.return %zero : i32', 1)"
        )
    return [*changed, "return value"]


def test_ordinary_source_artifact_executes_all_original_outputs_on_native_host(prepared, tmp_path, monkeypatch):
    clang, translate = os.environ.get("MERLIN_CLANG"), os.environ.get("MERLIN_MLIR_TRANSLATE")
    if not clang or not translate:
        pytest.skip("requires explicitly selected stock LLVM translator and native clang")
    clang, translate = Path(clang).resolve(strict=True), Path(translate).resolve(strict=True)
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "original")
    output = tmp_path / "ordinary"
    args = _arguments(prepared, fixture, output)

    def host_diagnostic(bound, artifact, **kwargs):
        work = kwargs["workdir"]
        work.mkdir()
        mlir = output / "generated/lowered.llvm.mlir"
        assert artifact == mlir.read_text()
        llvm, obj, harness, elf = (work / name for name in ("kernel.ll", "kernel.o", "harness.c", "control.elf"))
        header, codec = work / "htif.h", work / "out_b64.h"
        header.write_text(
            "void console_init(void);void htif_puts(const char*);void htif_exit(int) __attribute__((noreturn));\n"
        )
        codec.write_bytes((runtime_dir() / "baremetal/out_b64.h").read_bytes())
        harness.write_text(
            render_direct_kernel(
                bound,
                inputs=kwargs["inputs"],
                readback_policy=kwargs["readback_policy"],
                abi=DirectKernelAbi("control_entry", None, 8, "little", "void"),
            )
        )
        support = work / "support.c"
        support.write_text(
            "#include <stdio.h>\n#include <stdlib.h>\nvoid console_init(void){}\n"
            "void htif_puts(const char*s){fputs(s,stdout);}\nvoid htif_exit(int code){exit(code);}\n"
        )
        environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        records = []
        for argv, inputs, products, stage in (
            (
                [str(translate), "--mlir-to-llvmir", str(mlir), "-o", str(llvm)],
                (mlir,),
                (llvm,),
                "native_host_translation",
            ),
            (
                [str(clang), "-Wno-override-module", "-c", str(llvm), "-o", str(obj)],
                (llvm,),
                (obj,),
                "native_host_object",
            ),
            (
                [str(clang), str(harness), str(support), str(obj), "-o", str(elf)],
                (harness, support, obj, header, codec),
                (elf,),
                "native_host_link",
            ),
            ([str(elf)], (elf,), (), "native_host_execution"),
        ):
            process = invocation_record.run(
                argv,
                directory=work,
                stage=stage,
                inputs=inputs,
                outputs=products,
                dependencies=(clang, translate),
                env=environment,
                timeout=kwargs["execution_deadline"].remaining(),
                check=True,
                capture_output=True,
                text=True,
            )
            records.append(process)
        console = records[-1].stdout
        (work / "oracle_console.log").write_text(console)
        values, decoder = {}, OutB64Decoder()
        for line in console.splitlines():
            decoder.consume(line.split(), values)
        decoder.require_closed()
        assert console.splitlines().count("DONE") == 1
        assert tuple(values) == tuple(bound["kernel_abi"]["outputs"])
        for path in work.rglob("invocation.json"):
            invocation_record.require_environment(path, environment=environment)
        from merlin.runtime.backends.base import decode_float_readback
        from merlin.runtime.commandbuffer import declared_output_dtypes

        values = decode_float_readback(values, declared_output_dtypes(bound))
        return {"elf": str(elf), "outputs": values, "console": console}

    monkeypatch.setattr(compiler, "run_on_oracle", host_diagnostic)
    candidate = args["package_dir"]
    with P.scoped_package_executor(
        PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    ):
        result = native.execute_component(**args)
    assert result["numeric_report"]["status"] == "pass"
    assert "unqualified" in result["scope"]
    assert len(result["invocations"]) >= 9
