"""Original selected stages through the ordinary prepared-device builder.

Neutral fixture compilation is real CPU execution. It supplies no device
instruction effects, target numerical qualification, timing or author grant.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from device_llvm_configuration_fixture import CLANG, chain

from merlin.common import invocation_record as I
from merlin.llvmlower import device_build as DB
from merlin.llvmlower import device_preparation as DP
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.llvmlower.device_offload import DeviceRewrite, Routed

pytest_plugins = ("test_device_build_parallel",)


def source(work):
    work.mkdir(parents=True, exist_ok=True)
    DeviceRewrite(
        "synthetic", signatures={"a": (2, 2, 2)}, routed=(Routed("a", (2, 2), (2,), ("i8", "i8", "i32")),)
    ).write_sidecar(work)


def prepared(device, frozen, work, flags=lambda: ["-O0", "-fPIC"]):
    return DP.build_prepared_device_objects(
        device,
        work,
        offload_selection=frozen,
        device_admission_selection=None,
        target_configuration_selection=None,
        device_ir_budget=None,
        shim_cflags=flags,
    )


@pytest.mark.parametrize("value", [{}, "selected", object()])
def test_routing_rejects_untyped_transform_chain(tmp_path, value):
    with pytest.raises(TypeError, match="original typed"):
        DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=value)


@pytest.mark.parametrize("damage", ["drop", "replace", "proof", "attach"])
def test_preparation_reopens_original_chain_before_package_callbacks(tmp_path, monkeypatch, damage):
    selected, proof = chain(tmp_path / "original")
    device = DB.DeviceRouting(
        "synthetic", tmp_path, "i8", "i32", llvm_transform_chain=None if damage == "attach" else selected
    )
    frozen = DP.freeze_device_offload_selection(device)
    if damage == "proof":
        proof.write_text(proof.read_text() + "\n/* changed original proof */\n")
    else:
        object.__setattr__(device, "llvm_transform_chain", None if damage == "drop" else replace(selected))
    monkeypatch.setattr(DP, "_artifact_payload", lambda *_: pytest.fail("changed chain reached source preparation"))
    with pytest.raises(ValueError):
        DP.require_device_offload_selection(device, frozen)


def test_selected_catalog_or_empty_route_cannot_silently_ignore_chain(tmp_path, monkeypatch):
    selected, _ = chain(tmp_path / "original")
    device = DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=selected)
    frozen = DP.freeze_device_offload_selection(device)
    monkeypatch.setattr(DB, "require_device_elf_admission", lambda *_: None)
    with pytest.raises(ValueError, match="routed package-emitted LLVM"):
        prepared(device, frozen, tmp_path)
    source(tmp_path)
    side = tmp_path / "device_signatures.json"
    data = json.loads(side.read_text())
    data["catalog_manifest"] = "original-catalog"
    side.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="routed package-emitted LLVM"):
        prepared(device, frozen, tmp_path)


def test_lazy_flag_callback_cannot_drop_chain_before_object_builder(tmp_path, monkeypatch):
    selected, _ = chain(tmp_path / "original")
    device = DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=selected)
    frozen = DP.freeze_device_offload_selection(device)
    source(tmp_path)
    monkeypatch.setattr(DB, "require_device_elf_admission", lambda *_: None)
    monkeypatch.setattr(
        DB, "build_device_objects", lambda *_args, **_kwargs: pytest.fail("changed chain reached builder")
    )

    def flags():
        object.__setattr__(device, "llvm_transform_chain", None)
        return []

    with pytest.raises(ValueError):
        prepared(device, frozen, tmp_path, flags)


def test_builder_cannot_drop_a_selected_emission(synthetic_build, tmp_path, monkeypatch):
    selected, _ = chain(tmp_path / "original")
    device = DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=selected)
    frozen = DP.freeze_device_offload_selection(device)
    source(tmp_path)
    monkeypatch.setattr(DB, "require_device_elf_admission", lambda *_: None)
    monkeypatch.setattr(
        DB,
        "build_device_objects",
        lambda *_args, **_kwargs: DB.DeviceBuild(
            "synthetic", objects=(tmp_path / "unused.o",), shim_object=tmp_path / "unused.o", kernels={"a": "kernel"}
        ),
    )
    with pytest.raises(ValueError, match="not consumed"):
        prepared(device, frozen, tmp_path)


def test_prepared_builder_consumes_real_selected_llvm_object_and_link_recipe(synthetic_build, tmp_path, monkeypatch):
    from test_device_llvm_transform import _original_tools, _public_artifact, _setter_relocations

    _, _, _, _, artifacts, _ = synthetic_build
    _original_tools(monkeypatch)
    artifacts["a"] = _public_artifact(tmp_path)
    original = DB.build_device_objects
    forwarded = []

    def cpu_fixture(*args, **kwargs):
        # Only this neutral test substitutes the platform. Production forwarding
        # retains the ordinary RISC-V call and its original cflags unchanged.
        assert kwargs["codegen_target"] == "riscv"
        forwarded.append(kwargs["llvm_transform_chain"])
        kwargs["codegen_target"] = "host"
        return original(*args, **kwargs)

    monkeypatch.setattr(DB, "build_device_objects", cpu_fixture)
    monkeypatch.setattr(DB, "require_device_elf_admission", lambda *_: None)
    rows = {}
    for arm, enabled in (("baseline", False), ("candidate", True)):
        work = tmp_path / arm
        selected, runtime = chain(work / "owner", enabled=enabled)
        device = DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=selected)
        frozen = DP.freeze_device_offload_selection(device)
        source(work)
        _, _, built, products = prepared(device, frozen, work)
        assert built.ok, built.skipped
        emission = built.custody["a"]["llvm_transform"]
        optimized = I.verify(Path(built.custody["a"]["optimization"]["path"]))
        assert optimized["inputs"][0]["path"] == emission["selected_path"]
        assert optimized["inputs"][0]["sha256"] == emission["selected_sha256"]
        for role in ("translation", "optimization", "optimized_import", "compilation", "rename"):
            I.verify(Path(built.custody["a"][role]["path"]))
        recipe = CompilationRecipe(work, producer=Path(DP.__file__))
        recipe.bind_preparation("device_llvm_chain", Path(emission["chain"]["path"]))
        library = work / "native.so"
        recipe.run(
            [CLANG(), "-shared", "-fPIC", str(built.objects[0]), str(runtime), "-o", str(library)],
            runner=lambda argv: subprocess.run(argv, capture_output=True),
            inputs=(built.objects[0], runtime),
            output=library,
        )
        recipe.completed(library)
        verify_completed_recipe(
            recipe.path, executable=library, expected_recipe_sha256=hashlib.sha256(recipe.path.read_bytes()).hexdigest()
        )
        function = getattr(ctypes.CDLL(str(library)), built.kernels["a"])
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool]
        state = np.arange(7, dtype=np.uint32)
        calls = []
        for epoch in range(3):
            x = np.arange(7, dtype=np.int32) - epoch
            before_x, before_state = x.copy(), state.copy()
            guards = np.full(21 + 32, 0x12345678, dtype=np.uint32)
            history = guards[16:-16].reshape(3, 7)
            expected, expected_state = np.empty_like(history), state.copy()
            for row, scale in enumerate((3, 3, 5)):
                for i in range(7):
                    expected[row, i] = ((int(x[i]) + int(expected_state[i])) * scale) % (1 << 32)
                    expected_state[i] = (int(expected_state[i]) + 1) % (1 << 32)
            function(x.ctypes.data, history.ctypes.data, state.ctypes.data, 7, bool(epoch % 2))
            assert history.tobytes() == expected.tobytes() and state.tobytes() == expected_state.tobytes()
            assert x.tobytes() == before_x.tobytes()
            assert np.all(guards[:16] == 0x12345678) and np.all(guards[-16:] == 0x12345678)
            np.savez(
                work / f"actual-{epoch}.npz",
                x=x,
                before_state=before_state,
                state=state,
                history=history,
                guards=guards,
            )
            calls.append([hashlib.sha256(a.tobytes()).hexdigest() for a in (history, state)])
        DP.require_device_offload_selection(device, frozen)
        rows[arm] = {
            "products": products,
            "recipe": str(recipe.path),
            "calls": calls,
            "setter_relocations": _setter_relocations(built.objects[0]),
        }
    assert forwarded[0] is not forwarded[1]
    assert rows["baseline"]["calls"] == rows["candidate"]["calls"]
    assert [rows[a]["setter_relocations"] for a in ("baseline", "candidate")] == [7, 5]
    (tmp_path / "pair.json").write_text(json.dumps(rows, indent=2) + "\n")


@pytest.mark.parametrize("changed", [False, True])
def test_chain_only_final_boundary_reopens_after_original_admission(tmp_path, monkeypatch, changed):
    selected, _ = chain(tmp_path / "original")
    device = DB.DeviceRouting("synthetic", tmp_path, "i8", "i32", llvm_transform_chain=selected)
    frozen = DP.freeze_device_offload_selection(device)
    monkeypatch.setattr(DB, "freeze_device_elf_custody", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(DB, "revalidate_device_elf_custody", lambda *_: None)

    def original_admission(*args):
        if changed:
            object.__setattr__(device, "llvm_transform_chain", None)

    monkeypatch.setattr(DB, "revalidate_device_target_artifacts", original_admission)

    def finish():
        DP.finish_device_offload_selection(
            device,
            frozen,
            admission=None,
            result=None,
            elf=None,
            linked_sha256=None,
            target_configuration=None,
            artifacts=None,
        )

    if changed:
        with pytest.raises(ValueError):
            finish()
    else:
        finish()
