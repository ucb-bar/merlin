"""Named-stage evidence preserves real lowering bytes and execution."""

import hashlib
import json
import subprocess

import pytest

from merlin.common.ir_audit import IrAudit
from merlin.compile_core import compile_core_mlir
from merlin.xdsl_dialects._common import text
from merlin.xdsl_dialects.lowering import pipeline
from merlin.xdsl_dialects.lowering.input_workload import build_input_module


def _index(work):
    indexes = list(work.glob("ir-audit-*/index.json"))
    assert len(indexes) == 1
    return indexes[0].parent, json.loads(indexes[0].read_text())


def test_real_staged_lowering_audit_parity_and_execution(tmp_path):
    module = build_input_module(reuse=2, m=2, k=3, n=2)
    original = text(module)
    plain = compile_core_mlir(module, target="toy_npu", workdir=tmp_path / "plain")
    audited = compile_core_mlir(module, target="toy_npu", workdir=tmp_path / "audit", ir_audit=True)
    assert not (tmp_path / "plain").exists()
    assert text(module) == original
    assert audited.staged.command_buffer == plain.staged.command_buffer
    assert pipeline.execute(audited.staged)["correct"]
    directory, index = _index(tmp_path / "audit")
    assert index["outcome"] == "completed"
    assert [s["name"] for s in index["stages"]] == ["input", "contract", "schedule", "interface", "target", "runtime"]
    for stage, a, b in zip(index["stages"], audited.staged.modules(), plain.staged.modules(), strict=True):
        payload = (directory / stage["file"]).read_bytes()
        assert payload == text(a).encode() == text(b).encode()
        assert stage["sha256"] == hashlib.sha256(payload).hexdigest()
        assert stage["representation"] == "exact-ir"


def test_staged_failure_retains_completed_prefix(monkeypatch, tmp_path):
    def refuse(module):
        raise RuntimeError("synthetic refusal")

    monkeypatch.setattr(pipeline, "lower_to_schedule", refuse)
    with pytest.raises(RuntimeError, match="synthetic refusal"):
        pipeline.lower_module(build_input_module(), workdir=tmp_path, ir_audit=True)
    _, index = _index(tmp_path)
    assert index["outcome"] == "failed"
    assert index["failure_type"] == "RuntimeError"
    assert [s["name"] for s in index["stages"]] == ["input", "contract"]


def test_in_memory_lowering_never_creates_audit(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    pipeline.lower_module(build_input_module(), ir_audit=True)
    assert list(tmp_path.iterdir()) == []


def test_declared_sidecars_preserve_exact_bits_and_require_presence(tmp_path):
    weights = tmp_path / "weights.safetensors"
    payload = bytes(range(256))
    weights.write_bytes(payload)
    manifest = tmp_path / "weights.safetensors.manifest.json"
    manifest.write_bytes(b'{"storage": "opaque test bytes; never converted"}')
    with IrAudit(
        tmp_path / "audit", enabled=True, producer="fixture", source=__file__, sidecars=(weights, manifest)
    ) as audit:
        audit.stage("input", "builtin.module {}")
    assert weights.read_bytes() == payload
    _, index = _index(tmp_path / "audit")
    assert index["sidecars"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert index["sidecars"][0]["path"] == str(weights.resolve())
    with pytest.raises(FileNotFoundError):
        pipeline.lower_module(
            build_input_module(), workdir=tmp_path / "missing", ir_audit=True, audit_sidecars=(tmp_path / "absent",)
        )
    assert not (tmp_path / "missing").exists()


def test_sidecar_mutation_refuses_and_keeps_failure_evidence(tmp_path):
    weights = tmp_path / "weights"
    weights.write_bytes(b"original")
    with pytest.raises(ValueError, match="sidecar changed"):
        with IrAudit(
            tmp_path / "audit", enabled=True, producer="fixture", source=__file__, sidecars=(weights,)
        ) as audit:
            audit.stage("input", "builtin.module {}")
            weights.write_bytes(b"changed")
    _, index = _index(tmp_path / "audit")
    assert index["outcome"] == "failed"
    assert index["failure_type"] == "SidecarChanged"


def test_repeated_invocations_preserve_prior_audit(tmp_path):
    for _ in range(2):
        with IrAudit(tmp_path, enabled=True, producer="fixture", source=__file__) as audit:
            audit.stage("input", "builtin.module {}")
    assert len(list(tmp_path.glob("ir-audit-*/index.json"))) == 2


def test_failure_to_write_outcome_preserves_original_exception(monkeypatch, tmp_path):
    original = RuntimeError("primary lowering diagnostic")
    with pytest.raises(RuntimeError) as raised:
        with IrAudit(tmp_path, enabled=True, producer="fixture", source=__file__) as audit:

            def unwritable():
                raise OSError("fixture disk full")

            monkeypatch.setattr(audit, "_flush", unwritable)
            raise original
    assert raised.value is original
    assert "IR audit outcome could not be written: OSError" in original.__notes__


@pytest.fixture
def mocked_model_lowering(monkeypatch):
    from merlin.llvmlower import lower

    def preprocess(source, **kwargs):
        return source + "\n// preprocessed", {"synthetic": True}

    def pipeline(source, *, audit, **kwargs):
        audit.stage("mock-scheduled", source)
        return "define void @forward() { ret void }\n"

    def codegen(source, output):
        output.write_bytes(b"synthetic artifact; not an executable")
        return output

    monkeypatch.setattr(lower, "preprocess_text", preprocess)
    monkeypatch.setattr(lower, "lower_to_llvm_ir", pipeline)
    monkeypatch.setattr(lower, "build_host_shared", codegen)
    monkeypatch.setattr(lower, "compile_ll", lambda *a, **k: pytest.fail("unexpected target compiler"))
    return lower


def test_lower_result_returns_completed_current_audit_index(mocked_model_lowering, tmp_path):
    result = mocked_model_lowering.lower_model("module {}", tmp_path, ir_audit=True, static_arena=False)
    directory, index = _index(tmp_path)
    assert result.audit_index == directory / "index.json"
    assert index["outcome"] == "completed"
    assert index["producer"] == "merlin.llvmlower.lower_model"
    assert [stage["name"] for stage in index["stages"]] == ["input", "upstream", "mock-scheduled", "llvm-final"]
    assert result.host_so.read_bytes() == b"synthetic artifact; not an executable"
    assert (directory / index["stages"][-1]["file"]).read_bytes() == result.ll_path.read_bytes()


def test_lower_result_disabled_audit_is_none(mocked_model_lowering, tmp_path):
    result = mocked_model_lowering.lower_model("module {}", tmp_path, ir_audit=False, static_arena=False)
    assert result.audit_index is None
    assert not list(tmp_path.glob("ir-audit-*"))


def test_lower_result_repeated_workdir_returns_distinct_immutable_indices(mocked_model_lowering, tmp_path):
    first = mocked_model_lowering.lower_model("module {}", tmp_path, ir_audit=True, static_arena=False)
    original = first.audit_index.read_bytes()
    second = mocked_model_lowering.lower_model("module {}\n// next", tmp_path, ir_audit=True, static_arena=False)
    assert first.audit_index != second.audit_index
    assert first.audit_index.read_bytes() == original
    for result in (first, second):
        assert json.loads(result.audit_index.read_text())["outcome"] == "completed"
    assert set(tmp_path.glob("ir-audit-*/index.json")) == {first.audit_index, second.audit_index}


@pytest.mark.parametrize("failure", ["preprocess", "pipeline", "codegen"])
def test_lower_model_failure_retains_prefix_without_returning_result(
    mocked_model_lowering, tmp_path, monkeypatch, failure
):
    lower = mocked_model_lowering

    def refuse(*args, **kwargs):
        if failure == "pipeline":
            kwargs["audit"].stage("failed-pipeline-prefix", "module {}")
        raise RuntimeError("synthetic " + failure + " refusal")

    selected = {"preprocess": "preprocess_text", "pipeline": "lower_to_llvm_ir", "codegen": "build_host_shared"}
    monkeypatch.setattr(lower, selected[failure], refuse)
    results = []
    with pytest.raises(RuntimeError, match="synthetic " + failure + " refusal"):
        results.append(lower.lower_model("module {}", tmp_path, ir_audit=True, static_arena=False))
    assert results == []
    _, index = _index(tmp_path)
    assert index["outcome"] == "failed"
    assert index["failure_type"] == "RuntimeError"
    expected = {
        "preprocess": ["input"],
        "pipeline": ["input", "upstream", "failed-pipeline-prefix"],
        "codegen": ["input", "upstream", "mock-scheduled", "llvm-final"],
    }
    assert [stage["name"] for stage in index["stages"]] == expected[failure]


def test_real_llvm_lowering_parity(tmp_path):
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import m2m_python

    python = m2m_python()
    if not python.is_file():
        pytest.skip("compiler Python unavailable")
    probe = subprocess.run([str(python), "-c", "import torch_mlir"], capture_output=True)
    if probe.returncode:
        pytest.skip("torch-mlir unavailable")
    module = "module { func.func @forward(%x: f32) -> f32 { return %x : f32 } }"
    plain = lower_model(module, tmp_path / "plain", targets=(), textual=True)
    audited = lower_model(module, tmp_path / "audit", targets=(), textual=True, ir_audit=True)
    assert plain.ll_path.read_bytes() == audited.ll_path.read_bytes()
    directory, index = _index(tmp_path / "audit")
    assert index["outcome"] == "completed"
    assert [s["name"] for s in index["stages"]] == [
        "input",
        "upstream",
        "upstream-scheduled",
        "llvm-translated",
        "llvm-normalized",
        "llvm-final",
    ]
    assert (directory / index["stages"][-1]["file"]).read_bytes() == audited.ll_path.read_bytes()
    assert not list((tmp_path / "plain").glob("ir-audit-*"))
    command = index["commands"][0]
    assert command["argv"][0] == str(python)
    assert command["argv"][4]
    assert command["executable"]["sha256"]
    assert "clang" in command["toolchain_observation"]
