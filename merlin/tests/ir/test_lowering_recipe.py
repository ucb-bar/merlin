"""Execute lowering with its effective settings recorded and failures unqualified."""

from __future__ import annotations

import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower import pipeline, toolchain

SOURCE = """module {
  func.func @add(%a: f32, %b: f32) -> f32 {
    %c = arith.addf %a, %b : f32
    return %c : f32
  }
}
"""


@pytest.fixture
def lowering_tools():
    if not toolchain.m2m_python().is_file() or not shutil.which(str(toolchain.clang())):
        pytest.skip("upstream lowering/native compiler unavailable")


def _read(work):
    return json.loads((work / "lowering_recipe.json").read_text())


def test_effective_fusion_settings_and_actual_returned_code(tmp_path, monkeypatch, lowering_tools):
    monkeypatch.setenv("MERLIN_TEST_API_TOKEN", "private-canary-material")
    monkeypatch.setenv("UNRELATED_SECRET", "unrelated-canary-material")
    receipts = []
    for enabled in (False, True):
        for name in ("MERLIN_FUSE_POST", "MERLIN_GENERALIZE_BEFORE_FUSE"):
            if enabled:
                monkeypatch.setenv(name, "1")
            else:
                monkeypatch.delenv(name, raising=False)
        work = tmp_path / str(enabled)
        llvm = pipeline.lower_to_llvm_ir(SOURCE, workdir=work)
        record = _read(work)
        receipts.append(record)
        assert record["status"] == "returned"
        assert record["returned_llvm_ir"]["sha256"] == hashlib.sha256(llvm.encode()).hexdigest()
        assert record["sources"]["prepared_mlir"]["sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest()
        command = record["commands"][0]
        assert command["environment"]["MERLIN_TEST_API_TOKEN"] == "<redacted>"
        serialized = json.dumps(record)
        assert "private-canary-material" not in serialized
        assert "unrelated-canary-material" not in serialized
        for identity in (command["executable"], *record["sources"].values()):
            assert identity["sha256"] == hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()

        ll, shared = work / "returned.ll", work / "returned.so"
        ll.write_text(llvm)
        subprocess.run(
            [str(toolchain.clang()), "-O2", "-shared", "-fPIC", str(ll), "-o", str(shared)],
            check=True,
            capture_output=True,
        )
        add = ctypes.CDLL(str(shared)).add
        add.argtypes, add.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_float
        assert add(2.5, -0.75) == 1.75

    first, second = [record["commands"][0] for record in receipts]
    assert first["argv"][4] != second["argv"][4]
    assert "MERLIN_FUSE_POST" not in first["environment"]
    assert second["environment"]["MERLIN_FUSE_POST"] == "1"
    assert first["environment_digest"] != second["environment_digest"]


def test_failed_reused_workdir_does_not_retain_successful_return(tmp_path, lowering_tools):
    pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path)
    assert _read(tmp_path)["status"] == "returned"
    with pytest.raises(pipeline.PipelineError, match="upstream lowering failed"):
        pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path, pipeline="builtin.module(no-such-lowering-pass)")
    record = _read(tmp_path)
    assert record["status"] == "invoked"
    assert "returned_llvm_ir" not in record
    assert record["commands"][0]["argv"][4] == "builtin.module(no-such-lowering-pass)"


def test_missing_compiler_reused_workdir_does_not_retain_success(tmp_path, monkeypatch, lowering_tools):
    pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path)
    monkeypatch.setattr(pipeline, "m2m_python", lambda: tmp_path / "missing-python")
    with pytest.raises(FileNotFoundError):
        pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path)
    record = _read(tmp_path)
    assert record["status"] == "prepared"
    assert "returned_llvm_ir" not in record


def test_refused_feature_drops_previous_success_receipt(tmp_path, lowering_tools):
    pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path)
    with pytest.raises(KeyError, match="unknown"):
        pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path, features={"not-a-registered-compiler-feature"})
    assert not (tmp_path / "lowering_recipe.json").exists()


@pytest.mark.parametrize("mutation", [None, "before", "after", "report"])
def test_ordinary_ciface_annotation_retains_upstream_and_checks_exact_change(tmp_path, mutation):
    from merlin.llvmlower.ciface_inline import inline_wide_ciface_implementations
    from merlin.llvmlower.lowering_recipe import LoweringRecipe, bind_ciface_annotation

    arguments = ", ".join(f"i64 %a{i}" for i in range(257))
    before = f"define void @wide({arguments}) {{\n ret void\n}}\n"
    before += "define void @_mlir_ciface_wide(ptr %p) {\n ret void\n}\n"
    recipe = LoweringRecipe(tmp_path, features=(), sources={})
    recipe.returned(before)
    after, report = inline_wide_ciface_implementations(before)
    assert after != before
    actual_before = before + "\n" if mutation == "before" else before
    actual_after = after + "\n" if mutation == "after" else after
    if mutation == "report":
        report["count"] = 0
    if mutation is not None:
        with pytest.raises(ValueError, match="ABI annotation differs"):
            bind_ciface_annotation(recipe.path, actual_before, actual_after, report)
        assert "ciface_boundary_inline" not in _read(tmp_path)
        return
    bind_ciface_annotation(recipe.path, before, after, report)
    result = _read(tmp_path)
    assert result["upstream_returned_llvm_ir"]["sha256"] == hashlib.sha256(before.encode()).hexdigest()
    assert result["returned_llvm_ir"]["sha256"] == hashlib.sha256(after.encode()).hexdigest()
    assert result["ciface_boundary_inline"]["report"]["count"] == 1
