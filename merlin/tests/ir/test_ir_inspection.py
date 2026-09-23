"""Native printer views are inspection-only and never change compiler input."""

import hashlib
import json
import subprocess
import warnings

import pytest

from merlin.common.ir_audit import IrAudit, audit_mode
from merlin.llvmlower import pipeline
from merlin.llvmlower.ir_inspection import bind_inspection
from merlin.xdsl_dialects.lowering.input_workload import build_input_module


@pytest.mark.parametrize("value", [None, 0, 1, "", "off", "invalid", [], {}])
def test_invalid_mode_is_not_a_truthy_fallback(value, tmp_path):
    from merlin.compile_core import compile_core_mlir

    with pytest.raises(ValueError, match="ir_audit must"):
        audit_mode(value)
    with pytest.raises(ValueError, match="ir_audit must"):
        compile_core_mlir(None, ir_audit=value, workdir=tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_compact_staged_route_retains_views_without_changing_output(tmp_path):
    from merlin.xdsl_dialects.lowering.pipeline import lower_module

    plain = lower_module(build_input_module())
    compact = lower_module(build_input_module(), workdir=tmp_path, ir_audit="compact")
    assert compact.command_buffer == plain.command_buffer
    index_path = next(tmp_path.glob("ir-audit-*/index.json"))
    index = json.loads(index_path.read_text())
    assert len(index["stages"]) == 6
    assert all(stage["file"] is None and stage["compact_status"] == "recorded" for stage in index["stages"])
    for stage in index["stages"]:
        view = stage["inspection"]
        assert not view["executable"] and view["parent_sha256"] == stage["sha256"]
        content = (index_path.parent / view["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == view["sha256"]
        assert b"NOT EXECUTABLE" in content
    assert not list(index_path.parent.glob("*.mlir"))


def test_warning_as_error_keeps_successful_lowering_terminal(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="hash-only"):
            with IrAudit(tmp_path, enabled="compact", producer="unsupported", source=__file__) as audit:
                audit.stage("input", "unprinted IR")
    index = json.loads(next(tmp_path.glob("ir-audit-*/index.json")).read_text())
    assert index["outcome"] == "completed"
    assert len(index["stages"]) == 1


def test_warning_as_error_preserves_primary_lowering_failure(monkeypatch, tmp_path):
    from merlin.xdsl_dialects.lowering import pipeline as staged

    original = RuntimeError("original lowering failure")

    def refuse(module):
        raise original

    monkeypatch.setattr(staged, "lower_to_schedule", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(RuntimeError) as raised:
            staged.lower_module(build_input_module(), workdir=tmp_path, ir_audit="compact")
    assert raised.value is original
    assert not getattr(original, "__notes__", [])  # Supported prefix views need no hash-only warning.
    index = json.loads(next(tmp_path.glob("ir-audit-*/index.json")).read_text())
    assert index["outcome"] == "failed"
    assert index["failure_type"] == "RuntimeError"
    assert [stage["name"] for stage in index["stages"]] == ["input", "contract"]
    assert all(stage["inspection"]["file"] for stage in index["stages"])


@pytest.mark.parametrize("mode", ["compact", "both"])
def test_xdsl_dense_inspection_elides_without_mutating_module(tmp_path, mode):
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, ModuleOp, TensorType, i32

    from merlin.xdsl_dialects._common import text
    from merlin.xdsl_dialects.ir_inspection import record_stage

    dense = DenseIntOrFPElementsAttr.from_list(TensorType(i32, [128]), list(range(128)))
    module = ModuleOp([ConstantOp(dense)])
    exact = text(module)
    with IrAudit(tmp_path, enabled=mode, producer="xdsl-test", source=__file__) as audit:
        record_stage(audit, "dense", module)
        # Collecting native views cannot erase these stage-local views.
        audit.collect_views()
    assert text(module) == exact
    record = audit.record["stages"][0]
    assert record["sha256"] == hashlib.sha256(exact.encode()).hexdigest()
    view = (audit.directory / record["inspection"]["file"]).read_text()
    (tensor,) = record["inspection"]["tensors"]
    assert tensor["file"] in view and tensor["sha256"] in view
    assert "tensor<128xi32>" in view
    assert (audit.directory / tensor["file"]).read_bytes() == dense.data.data
    assert "NOT EXECUTABLE" in view
    if mode == "both":
        assert (audit.directory / record["file"]).read_text() == exact
    else:
        assert record["file"] is None


@pytest.mark.parametrize("mode", ["exact", "compact", "both"])
def test_whole_model_xdsl_preprocessing_records_each_rewrite_without_changing_output(tmp_path, mode):
    from merlin.llvmlower.passes_xdsl import preprocess_text

    module = "module { func.func @forward(%x: f32) -> f32 { return %x : f32 } }"
    plain = preprocess_text(module)
    with IrAudit(tmp_path, enabled=mode, producer="preprocessing", source=__file__) as audit:
        observed = preprocess_text(module, audit=audit)
    assert observed == plain
    assert [stage["name"] for stage in audit.record["stages"]] == [
        "xdsl-parsed",
        "xdsl-pruned",
        "xdsl-quant-lowered",
        "xdsl-c-interface",
    ]
    final = audit.record["stages"][-1]
    assert final["sha256"] == hashlib.sha256(observed[0].encode()).hexdigest()
    for stage in audit.record["stages"]:
        if mode != "compact":
            assert (audit.directory / stage["file"]).is_file()
        if mode != "exact":
            assert (audit.directory / stage["inspection"]["file"]).is_file()


def test_whole_model_xdsl_preprocessing_failure_keeps_completed_prefix(tmp_path, monkeypatch):
    from merlin.llvmlower import passes_xdsl

    def refuse(module):
        raise RuntimeError("rewrite refused")

    monkeypatch.setattr(passes_xdsl, "lower_quant_ext", refuse)
    with pytest.raises(RuntimeError, match="rewrite refused"):
        with IrAudit(tmp_path, enabled="compact", producer="preprocessing", source=__file__) as audit:
            passes_xdsl.preprocess_text("module {}", audit=audit)
    assert audit.record["outcome"] == "failed"
    assert [stage["name"] for stage in audit.record["stages"]] == ["xdsl-parsed", "xdsl-pruned"]


@pytest.mark.parametrize("emit", [pipeline.EMIT_TRANSLATE, pipeline.EMIT_DUMP])
@pytest.mark.parametrize("variant", ["plain", "activation", "scalarize"])
def test_all_runner_variants_bind_live_operation_hooks(emit, variant, tmp_path):
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER

    features = frozenset({"vectorized_transcendental_activation"}) if variant == "activation" else frozenset()
    passes = SCALARIZE_MARKER if variant == "scalarize" else "canonicalize"
    code = pipeline._select_runner(passes, features, emit=emit, inspection_dir=str(tmp_path), keep_exact=True)
    assert code.count("_write_view(module.operation,") == 2
    # Only the instrumentation helper itself calls the native parser directly.
    assert code.count("PassManager.parse(") == 1
    assert "tree_printing_dir_path=str(segment)" in code
    assert "__MERLIN_INSPECT_" not in code
    compile(code, "<inspection-runner>", "exec")
    plain = pipeline._select_runner(passes, features, emit=emit)
    assert "_write_view" not in plain
    compile(plain, "<plain-runner>", "exec")


def test_missing_template_hook_refuses():
    with pytest.raises(ValueError, match="must declare one"):
        bind_inspection("print('not a supported runner')", "/unused")


@pytest.mark.parametrize("textual", [False, True])
def test_real_dense_native_views_do_not_change_compiler_result(tmp_path, textual):
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import m2m_python

    python = m2m_python()
    if not python.is_file():
        pytest.skip("compiler Python unavailable")
    if subprocess.run([str(python), "-c", "import torch_mlir"], capture_output=True).returncode:
        pytest.skip("torch-mlir unavailable")
    values = ", ".join(str(n) for n in range(4096))
    module = (
        "module { func.func @forward() -> tensor<4096xi32> { "
        "%c = arith.constant dense<[" + values + "]> : tensor<4096xi32> "
        "return %c : tensor<4096xi32> } }"
    )
    plain = lower_model(module, tmp_path / "off", targets=(), textual=textual)
    for mode in ("exact", "compact", "both"):
        lowered = lower_model(module, tmp_path / mode, targets=(), textual=textual, ir_audit=mode)
        assert lowered.ll_path.read_bytes() == plain.ll_path.read_bytes()
        index_path = next((tmp_path / mode).glob("ir-audit-*/index.json"))
        index = json.loads(index_path.read_text())
        xdsl_stages = [stage for stage in index["stages"] if stage["name"].startswith("xdsl-")]
        assert len(xdsl_stages) == (0 if textual else 4)
        if mode == "exact":
            assert not index.get("inspection_views")
            continue
        views = index["inspection_views"]
        passes = index["pass_inspection"]
        assert passes["files"], "native passes must produce actual inspection files"
        assert passes["segments"][0]["directory"] == "passes/segment-0000"
        assert passes["segments"][0]["pipeline"].startswith("builtin.module(")
        assert passes["limits"]["large_elements_limit"] == 64
        for record in passes["files"]:
            assert record["executable"] is False
            payload = (index_path.parent / record["file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert [view["name"] for view in views] == ["00-parsed", "01-lowered"]
        parsed = views[0]
        content = (index_path.parent / parsed["file"]).read_bytes()
        assert b"INSPECTION ONLY" in content and b"NOT EXECUTABLE IR" in content
        assert parsed["executable"] is False
        assert len(content) < parsed["parent"]["bytes"] // 4
        assert parsed["sha256"] == hashlib.sha256(content).hexdigest()
        assert "large_elements_limit" in parsed["limits"]
        if mode == "both":
            exact = (index_path.parent / parsed["parent"]["file"]).read_bytes()
            assert hashlib.sha256(exact).hexdigest() == parsed["parent"]["sha256"]
        else:
            assert parsed["parent"]["file"] is None
            assert not list((index_path.parent / "native").glob("*.mlir"))
            assert all(stage["file"] is None for stage in index["stages"])


def test_compact_prefix_is_collected_after_real_pass_failure(tmp_path):
    from merlin.llvmlower.toolchain import m2m_python

    if not m2m_python().is_file():
        pytest.skip("compiler Python unavailable")
    with pytest.raises(pipeline.PipelineError):
        with IrAudit(tmp_path, enabled="compact", producer="fixture", source=__file__) as audit:
            pipeline.lower_to_llvm_ir("module {}", workdir=tmp_path / "compile", pipeline="no-such-pass", audit=audit)
    index = json.loads(next(tmp_path.glob("ir-audit-*/index.json")).read_text())
    assert index["outcome"] == "failed"
    assert [view["name"] for view in index["inspection_views"]] == ["00-parsed"]


def test_real_pass_failure_preserves_native_inspection_prefix(tmp_path):
    from merlin.llvmlower.toolchain import m2m_python

    if not m2m_python().is_file():
        pytest.skip("compiler Python unavailable")
    module = "module { func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> { return %x : tensor<4xf32> } }"
    with pytest.raises(pipeline.PipelineError):
        with IrAudit(tmp_path, enabled="compact", producer="fixture", source=__file__) as audit:
            pipeline.lower_to_llvm_ir(
                module, workdir=tmp_path / "compile", pipeline="canonicalize,convert-func-to-llvm", audit=audit
            )
    index_path = next(tmp_path.glob("ir-audit-*/index.json"))
    index = json.loads(index_path.read_text())
    assert index["outcome"] == "failed"
    assert index["pass_inspection"]["files"]
    assert any("canonicalize" in entry["file"] for entry in index["pass_inspection"]["files"]), index["pass_inspection"]
    assert all((index_path.parent / entry["file"]).is_file() for entry in index["pass_inspection"]["files"])
