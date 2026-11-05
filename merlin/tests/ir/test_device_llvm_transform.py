"""Opted-in original LLVM stages affect actual device object bytes only.

Package/interface setup is synthetic. Translation, optimization, import, object
compilation and native execution are real; no target qualification is issued.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from device_llvm_configuration_fixture import CLANG, SOURCE, TRANSLATOR, blocks, chain, imported, run, verify

from merlin.common import invocation_record as I
from merlin.llvmlower import device_build as DB
from merlin.llvmlower import device_object_optimization as T
from merlin.llvmlower import toolchain

pytest_plugins = ("test_device_build_parallel",)


def _public_artifact(directory):
    source, artifact = directory / "original.ll", directory / "original.mlir"
    source.write_text(SOURCE)
    run([TRANSLATOR(), "--import-llvm", "--mlir-print-op-generic", source, "-o", artifact])
    return artifact.read_text()


def _original_tools(monkeypatch):
    monkeypatch.setattr(toolchain, "clang", CLANG)
    monkeypatch.setattr(toolchain, "mlir_translate", TRANSLATOR)

    def execute(argv, *, timeout):
        return subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=timeout)

    monkeypatch.setattr(DB, "_run_build_tool", execute)


def _setter_relocations(path):
    """Actual executable-section ELF64 LE symbol relocations, no ISA guesses."""
    import struct

    raw = path.read_bytes()
    assert raw[:6] == b"\x7fELF\x02\x01"
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", raw)
    sections = [struct.unpack_from("<IIQQQQIIQQ", raw, header[6] + i * header[11]) for i in range(header[12])]
    total = 0
    for section in sections:
        if section[1] not in (4, 9) or not sections[section[7]][2] & 4:
            continue
        symbols = sections[section[6]]
        strings = sections[symbols[6]]
        for offset in range(section[4], section[4] + section[5], section[9]):
            info = struct.unpack_from("<QQ", raw, offset)[1]
            name_offset = struct.unpack_from("<I", raw, symbols[4] + (info >> 32) * symbols[9])[0]
            start = strings[4] + name_offset
            total += raw[start : raw.index(b"\0", start)] == b"set_register"
    return total


@pytest.mark.parametrize("length", [7, 19])
def test_selected_llvm_is_consumed_by_actual_object_and_preserves_complete_state(
    synthetic_build, tmp_path, monkeypatch, length
):
    build, _, _, _, artifacts, owner = synthetic_build
    _original_tools(monkeypatch)
    artifacts["a"] = _public_artifact(tmp_path)
    results = {}
    for arm, enabled in (("baseline", False), ("candidate", True)):
        selected, runtime = chain(tmp_path / arm, enabled=enabled)
        visited = []
        original_transform = selected.stages[0].transform

        def transform(source, work):
            visited.append(threading.get_ident())
            return original_transform(source, work)

        stage = replace(selected.stages[0], transform=transform)
        selected = replace(selected, stages=(stage,))
        # The replacement has the same original test source closure explicitly.
        from merlin.llvmlower.host_transform_chain import HostLLVMArtifact

        contract = replace(stage.contract, sources=(*stage.contract.sources, HostLLVMArtifact.capture(Path(__file__))))
        selected = replace(selected, promised_stages=(contract,), stages=(replace(stage, contract=contract),))
        original_selection = T.freeze_transform_chain(selected)
        output = tmp_path / arm / "built"
        # Existing fixture selects one workdir; isolate actual products per arm.
        original_builder = DB.build_device_objects

        def redirected(*args, **kwargs):
            kwargs["workdir"] = output
            return original_builder(*args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(DB, "build_device_objects", redirected)
            result = build(names=("a",), codegen_target="host", cflags=["-O0", "-fPIC"], llvm_transform_chain=selected)
        assert result.ok, result.skipped
        assert visited == [owner]
        custody = result.custody["a"]
        emission = custody["llvm_transform"]
        record = json.loads(Path(emission["chain"]["path"]).read_text())
        evidence = json.loads(Path(record["stages"][0]["evidence"][0]["path"]).read_text())
        assert evidence["actual_removed_setters"] == (2 if enabled else 0)
        optimized = json.loads(Path(custody["optimization"]["path"]).read_text())
        assert optimized["inputs"][0]["path"] == emission["selected_path"]
        assert optimized["inputs"][0]["sha256"] == emission["selected_sha256"]
        for role in ("translation", "optimization", "optimized_import", "compilation", "rename"):
            assert hashlib.sha256(Path(custody[role]["path"]).read_bytes()).hexdigest() == custody[role]["sha256"]
            I.verify(Path(custody[role]["path"]))
        library = output / "native.so"
        run([CLANG(), "-shared", "-fPIC", result.objects[0], runtime, "-o", library])
        function = getattr(ctypes.CDLL(str(library)), result.kernels["a"])
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool]
        state = np.arange(length, dtype=np.uint32)
        calls = []
        for epoch in range(3):
            x = np.asarray([((i * 17 + epoch) % 257) - 128 for i in range(length)], dtype=np.int32)
            original = x.copy(), state.copy()
            guards = np.full(3 * length + 32, 0x12345678, dtype=np.uint32)
            actual = guards[16:-16].reshape(3, length)
            expected = np.empty_like(actual)
            expected_state = state.copy()
            for row, scale in enumerate((3, 3, 5)):
                for i in range(length):
                    expected[row, i] = ((int(x[i]) + int(expected_state[i])) * scale) % (1 << 32)
                    expected_state[i] = (int(expected_state[i]) + 1) % (1 << 32)
            function(x.ctypes.data, actual.ctypes.data, state.ctypes.data, length, bool(epoch % 2))
            np.savez(
                output / f"actual-{epoch}.npz",
                before_x=original[0],
                before_state=original[1],
                x=x,
                state=state,
                history=actual,
                guards=guards,
            )
            assert actual.tobytes() == expected.tobytes() and state.tobytes() == expected_state.tobytes()
            assert x.tobytes() == original[0].tobytes()
            assert np.all(guards[:16] == 0x12345678) and np.all(guards[-16:] == 0x12345678)
            calls.append([hashlib.sha256(a.tobytes()).hexdigest() for a in (actual, state)])
        T.recheck_transform_chain(selected, original_selection, emission)
        results[arm] = {
            "custody": custody,
            "object_setter_relocations": _setter_relocations(result.objects[0]),
            "calls": calls,
            "native": str(library),
            "native_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    assert results["baseline"]["calls"] == results["candidate"]["calls"]
    assert results["baseline"]["object_setter_relocations"] == 7
    assert results["candidate"]["object_setter_relocations"] == 5
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "arms": results,
                "scope": "actual neutral CPU objects; provider-owned synthetic effects only",
                "timing": None,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("mutation", ["attach", "replace", "proof", "callback", "owner"])
def test_original_transform_selection_cannot_change_after_freeze(tmp_path, mutation):
    original, runtime = chain(tmp_path / "owner")
    frozen = T.freeze_transform_chain(None if mutation == "attach" else original)
    selected = original
    if mutation == "replace":
        selected = replace(original)
    elif mutation == "proof":
        runtime.write_text(runtime.read_text() + "\n/* changed proof */\n")
    elif mutation == "callback":
        object.__setattr__(original.stages[0], "verify_emission", original.stages[0].transform)
    elif mutation == "owner":
        # An in-memory implementation change has unchanged on-disk source.
        original.stages[0].transform.__code__ = original.stages[0].transform.__code__.replace(co_name="changed")
    with pytest.raises(ValueError):
        T.recheck_transform_chain(selected, frozen)


@pytest.mark.parametrize("where", ["unknown", "cfg_join"])
def test_actual_emission_cannot_transfer_state_across_unknown_or_cfg_boundary(tmp_path, where):
    from xdsl.dialects import llvm

    original = tmp_path / "source.ll"
    original.write_text(SOURCE)
    module = imported(original, tmp_path)
    actual = module.clone()
    body = blocks(actual)
    if where == "cfg_join":
        removed = [next(op for op in body[-1].ops if type(op) is llvm.CallOp)]
    else:
        seen_unknown, removed = False, []
        for op in body[0].ops:
            if type(op) is llvm.CallOp and op.callee is not None:
                name = op.callee.root_reference.data
                if name == "unknown":
                    seen_unknown = True
                elif seen_unknown and name == "set_register":
                    removed.append(op)
    for op in removed:
        op.parent.erase_op(op)
    with pytest.raises(ValueError, match="omitted write changes live state"):
        verify(module, actual)


@pytest.mark.parametrize("kind", ["source", "selected", "evidence", "proof"])
def test_mutation_after_emission_check_never_reaches_an_accepted_object(synthetic_build, tmp_path, monkeypatch, kind):
    import runpy

    from merlin.common.paths import merlin_dir

    helpers = runpy.run_path(str(merlin_dir() / "tests/runtime/test_host_transform_chain.py"))
    build, tool, _, calls, _, _ = synthetic_build
    selected = helpers["_chain"](helpers["_stage"](tmp_path, "copy"))
    changed = []

    def mutate(argv, *, timeout):
        result = tool(argv, timeout=timeout)
        if "--import-llvm" in argv:
            stage = tmp_path / "build/a.transform/stage_0_copy"
            target = {
                "source": tmp_path / "build/a.ll",
                "selected": stage / "selected.ll",
                "evidence": stage / "emission.json",
                "proof": tmp_path / "copy_permission.json",
            }[kind]
            target.write_bytes(target.read_bytes() + b"changed after checker")
            changed.append(target)
        return result

    monkeypatch.setattr(DB, "_run_build_tool", mutate)
    if kind == "proof":
        with pytest.raises(ValueError, match="identity changed"):
            build(names=("a",), llvm_transform_chain=selected)
    else:
        result = build(names=("a",), llvm_transform_chain=selected)
        assert not result.ok and result.shim_object is None
    assert changed
    assert not any(name.endswith(".raw.o") for _, name, _ in calls)


def test_original_declared_source_and_bound_checker_owner_are_retained(tmp_path):
    from merlin.llvmlower.host_transform_chain import HostLLVMArtifact

    original, _ = chain(tmp_path / "original")
    source = tmp_path / "selected_checker_dependency.py"
    source.write_text("# original independently selected checker dependency\n")
    stage = original.stages[0]
    contract = replace(stage.contract, sources=(*stage.contract.sources, HostLLVMArtifact.capture(source)))
    original = replace(original, promised_stages=(contract,), stages=(replace(stage, contract=contract),))
    frozen = T.freeze_transform_chain(original)
    source.write_text("# changed selected checker dependency\n")
    with pytest.raises(ValueError, match="identity changed"):
        T.recheck_transform_chain(original, frozen)

    class Checker:
        def verify(self, *args):
            return stage.verify_emission(*args)

    source.write_text("# original independently selected checker dependency\n")
    owner = Checker()
    contract = replace(contract, sources=(*contract.sources, HostLLVMArtifact.capture(Path(__file__))))
    stage = replace(stage, contract=contract, verify_emission=owner.verify)
    original = replace(original, promised_stages=(contract,), stages=(stage,))
    frozen = T.freeze_transform_chain(original)
    object.__setattr__(stage, "verify_emission", Checker().verify)
    with pytest.raises(ValueError, match="callback or contract changed"):
        T.recheck_transform_chain(original, frozen)


@pytest.mark.parametrize("role", ["device_build.py", "device_build_jobs.py", "device_object_optimization.py"])
def test_original_transport_sources_are_pinned_before_package_callbacks(synthetic_build, tmp_path, monkeypatch, role):
    from merlin.targetgen import capsule_common

    build, _, _, calls, _, _ = synthetic_build
    selected, _ = chain(tmp_path / "owner")
    original_directory = Path(T.__file__).parent
    copies = tmp_path / "original_core"
    copies.mkdir()
    for name in ("device_build.py", "device_build_jobs.py", "device_object_optimization.py"):
        (copies / name).write_bytes((original_directory / name).read_bytes())
    # Isolated byte-identical snapshots avoid editing live implementation files.
    monkeypatch.setattr(T, "__file__", str(copies / "device_object_optimization.py"))
    original = capsule_common.lower_interface

    def change(*args, **kwargs):
        value = original(*args, **kwargs)
        (copies / role).write_bytes((copies / role).read_bytes() + b"changed after original freeze")
        return value

    monkeypatch.setattr(capsule_common, "lower_interface", change)
    result = build(names=("a",), llvm_transform_chain=selected)
    assert not result.ok and not calls
    assert "identity changed" in result.skipped[0][1]


@pytest.mark.parametrize("reason", ["empty", "nonllvm", "invalid_workers"])
def test_selected_chain_cannot_be_ignored_or_skip_original_worker_validation(
    synthetic_build, tmp_path, monkeypatch, reason
):
    build, _, emitted, calls, _, _ = synthetic_build
    selected, _ = chain(tmp_path / "owner")
    names = () if reason == "empty" else ("a",)
    if reason == "nonllvm":
        monkeypatch.setattr(DB, "objects_buildable", lambda _: "selected non-LLVM artifact transport")
    if reason == "invalid_workers":
        from merlin.llvmlower import device_build_jobs

        monkeypatch.setattr(
            device_build_jobs, "build_workers", lambda: (_ for _ in ()).throw(ValueError("original worker validation"))
        )
    with pytest.raises(ValueError, match="consumer|original worker validation"):
        build(names=names, llvm_transform_chain=selected)
    assert not emitted and not calls
