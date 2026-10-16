"""A slow device kernel must not erase the rest of the build's failure roster."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower import device_build as DB


@pytest.fixture
def isolated_build(tmp_path, monkeypatch):
    from merlin.common.digest import sha256_text
    from merlin.common.tree_hash import hash_tree
    from merlin.llvmlower import device_shim, exact_offload, toolchain
    from merlin.targetgen import oot_runner
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
    from merlin.targetgen.lowering_coverage import contraction_interface

    target = "timeout_test_device"
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.yaml").write_text("test fixture\n")
    digest = hash_tree(package)["sha256"]
    interface = contraction_interface(2, 5, 4, target=target, operand_mlir="i8", accum_mlir="i32")
    commands = parse_interface_mlir(interface)
    commands["commands"] = commands["commands"][:-1]  # Exact resident kernel ABI has no EVICT command.
    interface = emit_interface_mlir(commands)
    calls = []
    state = {"timeout_stage": None}
    monkeypatch.setattr(DB, "objects_buildable", lambda _target: None)
    monkeypatch.setattr(DB, "_objcopy", lambda: "test-objcopy")
    monkeypatch.setattr(exact_offload, "_package_sha256", lambda _path: digest)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _target: SimpleNamespace(symbol="kernel"))
    monkeypatch.setattr(oot_runner, "load_package", lambda _path: SimpleNamespace(target=target))
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: "test-translate")
    monkeypatch.setattr(toolchain, "clang", lambda: "test-clang")

    def emit(_package, _entrypoint, path, *, timeout):
        calls.append(("emit", path.stem))
        if path.name.startswith("a_slow.") and state["timeout_stage"] == "emit":
            raise subprocess.TimeoutExpired(["test-emitter"], timeout)
        return subprocess.CompletedProcess([], 0, stdout="module { llvm.func @kernel() { llvm.return } }", stderr="")

    def run(argv, *, timeout):
        output = Path(argv[-1])
        stage = {"test-translate": "translate", "test-objcopy": "rename"}.get(argv[0], "clang")
        if output.name == "device_shim.o":
            stage = "shim"
        calls.append((stage, output.name))
        if state["timeout_stage"] == stage and (output.name.startswith("a_slow.") or stage == "shim"):
            # A timed-out tool may already have left an incomplete file. It is never a built object.
            output.write_bytes(b"incomplete")
            raise subprocess.TimeoutExpired(argv, timeout, stderr=b"interrupted tool")
        output.write_bytes(b"successful synthetic output")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def shim(_target, signatures, _dtypes, **_kwargs):
        return SimpleNamespace(text="/* synthetic shim */", symbols=tuple(signatures), skipped=())

    monkeypatch.setattr(oot_runner, "run_entrypoint", emit)
    from merlin.targetgen import capsule_common

    def lower(pkg, path, generated, *, timeout, **kwargs):
        artifact = emit(pkg, "emit_target_artifact", path, timeout=timeout).stdout
        generated.mkdir(parents=True, exist_ok=True)
        (generated / "lowered.llvm.mlir").write_text(artifact)
        return {}, artifact

    monkeypatch.setattr(capsule_common, "lower_interface", lower)
    monkeypatch.setattr(DB, "_run", run)
    monkeypatch.setattr(device_shim, "emit_translation_unit", shim)

    def build():
        return DB.build_device_objects(
            target,
            {symbol: (2, 4, 5) for symbol in ("a_slow", "b_fast")},
            {symbol: ("i8", "i8", "i32") for symbol in ("a_slow", "b_fast")},
            package_dir=package,
            workdir=tmp_path / "build",
            operand_dtype="i8",
            accum_dtype="i32",
            cflags=["--synthetic-test-flags"],
            expected_interfaces={
                symbol: {"mlir": interface, "sha256": sha256_text(interface)} for symbol in ("a_slow", "b_fast")
            },
            package_sha256=digest,
            timeout=3,
        )

    return build, state, calls


@pytest.mark.parametrize("stage", ["emit", "translate", "clang", "rename"])
def test_kernel_timeout_preserves_other_kernel_and_names_stage(isolated_build, stage):
    build, state, calls = isolated_build
    state["timeout_stage"] = stage
    result = build()
    assert result.ok
    assert set(result.kernels) == {"b_fast"}
    assert set(result.built_from) == {"b_fast"}
    assert all(not path.name.startswith("a_slow.") for path in result.objects)
    assert result.skipped[0][0] == "a_slow"
    assert "timed out" in result.skipped[0][1] and "3" in result.skipped[0][1]
    expected_stage = {
        "emit": "package lowering",
        "translate": "mlir-translate",
        "clang": "clang",
        "rename": "symbol rename",
    }
    assert expected_stage[stage] in result.skipped[0][1]
    assert ("emit", "b_fast.iface") in calls


def test_shim_timeout_cannot_be_a_successful_device_build(isolated_build):
    build, state, _calls = isolated_build
    state["timeout_stage"] = "shim"
    result = build()
    assert not result.ok
    assert result.shim_object is None
    assert set(result.kernels) == {"a_slow", "b_fast"}
    assert all(path.name != "device_shim.o" for path in result.objects)
    assert result.skipped[-1][0] == "shim"
    assert "timed out" in result.skipped[-1][1]


def test_real_tool_timeout_is_nonzero_and_never_returns_partial_success():
    result = DB._run_build_tool([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05)
    assert result.returncode == 124
    assert result.stdout == ""
    assert "timed out" in result.stderr and "incomplete outputs are not admitted" in result.stderr


def test_non_timeout_infrastructure_errors_are_not_hidden(monkeypatch):
    def fail(_argv, *, timeout):
        raise PermissionError("tool is not executable")

    monkeypatch.setattr(DB, "_run", fail)
    with pytest.raises(PermissionError, match="not executable"):
        DB._run_build_tool(["not-a-tool"], timeout=3)
