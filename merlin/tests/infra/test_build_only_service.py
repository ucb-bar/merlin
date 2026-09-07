"""The optional compile service has no runtime registry/cache dependency."""
import builtins
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, load_build_package
from merlin.targetgen.contract import compile as compiler


def service(tmp_path):
    source = tmp_path / "renderer.py"
    source.write_text("trusted build fixture")
    recipe = HarnessBuildRecipe(Path("/usr/bin/cc"), (), (), Path("/fixture/link.ld"), 0,
                                ("-march=fixture",))
    return BuildOnlyService("fixture", recipe, lambda cb, **kwargs: "C text",
        ((str(source), hashlib.sha256(source.read_bytes()).hexdigest()),))


def test_pure_branch_never_imports_registry_or_cache(tmp_path, monkeypatch):
    cap = service(tmp_path)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("merlin.runtime.backends") or "build_cache" in name:
            raise AssertionError("pure build attempted runtime registry/cache import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    calls = []
    def obj(text, work, **kwargs):
        calls.append(("object", kwargs))
        return work / "kernel.o"
    def link(cb, obj, work, **kwargs):
        calls.append(("link", kwargs))
        return work / "kernel.elf"
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", obj)
    monkeypatch.setattr(compiler, "link_elf", link)
    assert compiler.compile_lowered_to_elf({}, "llvm", tmp_path, target="fixture",
        inputs={"a": [1]}, _build_service=cap) == tmp_path / "kernel.elf"
    assert [name for name, _ in calls] == ["object", "link"]
    assert all(kwargs["_build_service"] is cap for _, kwargs in calls)


@pytest.mark.parametrize("defect", ["stale", "target", "inputs", "authority"])
def test_invalid_capability_before_compile(tmp_path, monkeypatch, defect):
    cap = service(tmp_path)
    kwargs = {"target": "fixture", "inputs": {"a": [1]}, "_build_service": cap}
    if defect == "stale":
        Path(cap.source_pins[0][0]).write_text("changed")
    elif defect == "target":
        kwargs["target"] = "other"
    elif defect == "inputs":
        kwargs["inputs"] = None
    else:
        kwargs["prepack_authorizations"] = {}
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **k: pytest.fail("compiled"))
    with pytest.raises(ValueError):
        compiler.compile_lowered_to_elf({}, "llvm", tmp_path, **kwargs)


def test_package_reuse_refuses_changed_source(tmp_path):
    initializer = tmp_path / "__init__.py"
    initializer.write_text("VALUE = 1")
    assert load_build_package(initializer).VALUE == 1
    initializer.write_text("VALUE = 2")
    with pytest.raises(ValueError, match="changed"):
        load_build_package(initializer)


def test_build_translation_refuses_non_llvm_before_tool(tmp_path, monkeypatch):
    cap = service(tmp_path)
    monkeypatch.setattr(compiler.subprocess, "run", lambda *a, **k: pytest.fail("translation ran"))
    text = 'builtin.module { %x = "builtin.unrealized_conversion_cast"() : () -> i32 }'
    with pytest.raises(ValueError, match="LLVM/Builtin"):
        compiler.llvm_mlir_to_object(text, tmp_path, target="fixture", _build_service=cap)


def test_legacy_object_path_retains_original_lowering(tmp_path, monkeypatch):
    from merlin.llvmlower import pipeline, codegen
    calls = []
    def lower(text, *, workdir):
        calls.append((text, workdir))
        return "; existing lowering"
    def compile_ll(source, output, target, *, extra_flags):
        assert source.read_text() == "; existing lowering"
        assert target == "riscv" and extra_flags == ()
        return output
    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lower)
    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    assert compiler.llvm_mlir_to_object("legacy source", tmp_path) == tmp_path / "kernel.o"
    assert calls == [("legacy source", tmp_path)]
