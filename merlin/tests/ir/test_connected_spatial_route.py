"""Ordinary source/host/object transport; no target numerical execution."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_connected_spatial_host import fixture, helper
from xdsl.dialects import func, llvm, memref
from xdsl.parser import Parser

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.llvmlower import device_build as B
from merlin.llvmlower import device_preparation as P
from merlin.llvmlower import device_shim as S
from merlin.llvmlower.connected_spatial_route import (
    ConnectedSpatialRoute,
    prepare_host,
    select_emission,
    with_private_bridge,
)
from merlin.llvmlower.device_offload import load_sidecar
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.xdsl_dialects._common import text


def route(tmp_path):
    module, rewrite, plan, writer, workspace, _ = fixture()
    selected = ConnectedSpatialRoute(("producer", "pool"), plan, writer, workspace, (1, 1, 4, 1), "selected_raw_helper")
    rewrite = replace(rewrite, granularity="group", signatures={"producer": (50, 17, 27), "pool": (50, 17, 1)})
    device = B.DeviceRouting(
        "ordinary_selected_target", tmp_path, "i8", "i32", granularity="group", connected_spatial=selected
    )
    return module, rewrite, device


@pytest.mark.parametrize("damage", ["drop", "replace", "attach", "plan", "effects"])
def test_original_source_and_ownership_selection_is_frozen_before_callbacks(tmp_path, damage):
    _, _, device = route(tmp_path)
    owner = device.connected_spatial
    if damage == "attach":
        device = replace(device, connected_spatial=None)
    frozen = P.freeze_device_offload_selection(device)
    if damage in ("drop", "replace", "attach"):
        object.__setattr__(
            device, "connected_spatial", None if damage == "drop" else owner if damage == "attach" else replace(owner)
        )
    elif damage == "plan":
        object.__setattr__(owner.plan, "spatial", replace(owner.plan.spatial, padding=(0, 0, 0, 0)))
    else:
        object.__setattr__(owner.workspace, "synchronous_before_return", False)
    with pytest.raises(ValueError):
        P.require_device_offload_selection(device, frozen)


@pytest.mark.parametrize("field", ["post_offload_transform", "offload_selection", "catalog_manifest"])
def test_spatial_selection_cannot_be_attached_to_an_opaque_or_distinct_source_route(tmp_path, field):
    _, _, device = route(tmp_path)
    with pytest.raises(ValueError, match="group route"):
        replace(device, **{field: object()})


@pytest.mark.parametrize("early", [False, True])
def test_ordinary_preparation_groups_current_ssa_before_freezing_sidecar(tmp_path, monkeypatch, early):
    from merlin.llvmlower import device_offload as O

    module, rewrite, device = route(tmp_path)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(text(module, generic=True))
    original = prepared.read_bytes()
    rewrite.write_sidecar(tmp_path)
    calls = []

    def ordinary_rewrite(*args, **kwargs):
        calls.append((args, kwargs))
        rewrite.write_sidecar(tmp_path)
        return rewrite

    monkeypatch.setattr(O, "rewrite_prepared_file", ordinary_rewrite)
    monkeypatch.setattr(S, "kernel_abi_for", lambda _: S.KernelAbi("logical_kernel", version=2))
    frozen = P.freeze_device_offload_selection(device)
    selected_path = P.prepare_device_route(device, prepared, tmp_path, early_complete_groups=early, selection=frozen)
    assert len(calls) == int(not early)
    assert (tmp_path / "connected_spatial/original-input.mlir").read_bytes() == original
    selected = parse_mlir_file(selected_path)
    forward = selected.body.block.first_op
    assert forward.function_type == module.body.block.first_op.function_type
    assert len([op for op in forward.body.block.ops if isinstance(op, memref.AllocOp)]) == 1
    source = load_sidecar(tmp_path)
    assert list(source["signatures"]) == [device.connected_spatial.writer.symbol]
    assert len(source["entries"][device.connected_spatial.writer.symbol]["components"]) == 2
    assert len([op for op in selected.walk() if isinstance(op, func.CallOp)]) == 2
    assert "logical_kernel_0" in frozen.spatial_host.scratch.bridge_source
    P.require_device_offload_selection(device, frozen)
    selected_path.write_text(selected_path.read_text() + "\n")
    with pytest.raises(ValueError, match="realized source or sidecar changed"):
        P.require_device_offload_selection(device, frozen)


def test_checked_final_artifact_is_kept_and_only_closed_helper_selected(tmp_path, monkeypatch):
    module, rewrite, device = route(tmp_path)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(text(module, generic=True))
    monkeypatch.setattr(S, "kernel_abi_for", lambda _: S.KernelAbi("logical_kernel", version=2))
    _, host = prepare_host(device, device.connected_spatial, prepared, rewrite, tmp_path / "host")
    emitted = helper()
    checked = text(emitted, generic=True)
    derived, receipt = select_emission(checked, device.connected_spatial, host, entry_symbol="kernel_entry")
    assert text(emitted, generic=True) == checked
    assert receipt["checked_producer_sha256"] == hashlib.sha256(checked.encode()).hexdigest()
    assert "selected_raw_helper" not in derived and "kernel_entry" in derived
    assert receipt["region"]["operations"] == 2  # original memory fence + return
    unit = with_private_bridge(
        S.ShimUnit("/* other ordinary shims */"),
        device.connected_spatial,
        host,
        symbol=device.connected_spatial.writer.symbol,
        kernel="logical_kernel_0",
    )
    assert unit.symbols == (device.connected_spatial.writer.symbol,)
    assert unit.kernels == ("logical_kernel_0",)
    assert "malloc" not in unit.text and "free(" not in unit.text
    with pytest.raises(ValueError, match="original host-bound kernel"):
        with_private_bridge(
            S.ShimUnit(""),
            device.connected_spatial,
            host,
            symbol=device.connected_spatial.writer.symbol,
            kernel="wrong_kernel",
        )
    with pytest.raises(ValueError, match="calls"):
        select_emission(
            text(helper(imports=True), generic=True), device.connected_spatial, host, entry_symbol="kernel_entry"
        )


def test_original_prepared_driver_delivers_the_actual_host_to_object_build(tmp_path, monkeypatch):
    from merlin.llvmlower import device_offload as O

    module, rewrite, device = route(tmp_path)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(text(module, generic=True))
    rewrite.write_sidecar(tmp_path)
    monkeypatch.setattr(O, "rewrite_prepared_file", lambda *a, **kw: rewrite)
    monkeypatch.setattr(S, "kernel_abi_for", lambda _: S.KernelAbi("logical_kernel", version=2))
    frozen = P.freeze_device_offload_selection(device)
    P.prepare_device_route(device, prepared, tmp_path, selection=frozen)
    monkeypatch.setattr(B, "require_device_elf_admission", lambda *a: None)

    class ReachedOrdinaryObjectBuilder(Exception):
        pass

    def receive(*args, **kwargs):
        assert kwargs["connected_spatial"] == (device.connected_spatial, frozen.spatial_host)
        assert kwargs["entries"] == frozen.spatial_host.rewrite.entries
        assert kwargs["call_buffers"] == frozen.spatial_host.rewrite.call_buffers
        raise ReachedOrdinaryObjectBuilder

    monkeypatch.setattr(B, "build_device_objects", receive)
    with pytest.raises(ReachedOrdinaryObjectBuilder):
        P.build_prepared_device_objects(
            device,
            tmp_path,
            offload_selection=frozen,
            device_admission_selection=None,
            target_configuration_selection=None,
            device_ir_budget=None,
            shim_cflags=lambda: [],
        )


def test_normal_object_builder_compiles_only_private_helper_and_keeps_strict_guard(
    tmp_path, monkeypatch, upstream_host_tools
):
    """Actual stock object transport using a fence stub, never a numerical witness."""
    from merlin.compile import mesh
    from merlin.llvmlower import device_build_jobs as J
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen import package_runtime as PR

    module, rewrite, device = route(tmp_path)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text(text(module, generic=True))
    monkeypatch.setattr(S, "kernel_abi_for", lambda _: S.KernelAbi("logical_kernel", version=2))
    _, host = prepare_host(device, device.connected_spatial, prepared, rewrite, tmp_path / "host")
    symbol = device.connected_spatial.writer.symbol
    checked = text(helper(), generic=True)
    monkeypatch.setattr(B, "objects_buildable", lambda _: None)
    monkeypatch.setattr(B, "executable_kernel_entry", lambda _: "kernel_entry")
    monkeypatch.setattr(mesh, "_mesh_tile_binding", lambda *a, **kw: None)
    monkeypatch.setattr(PR, "load_package", lambda _: SimpleNamespace(target=device.device))
    monkeypatch.setattr(
        CS,
        "build",
        lambda *args: (
            {"operation": {"op": "spatial_producer_pool", "attributes": {"out": "Y"}}},
            "source-interface-stub",
        ),
    )
    monkeypatch.setattr(S, "logical_buffers", lambda *a, **kw: ((), ""))
    monkeypatch.setattr(J, "build_workers", lambda: 1)

    def checked_lower(_pkg, _interface, output, **kwargs):
        output.mkdir(parents=True)
        (output / "lowered.llvm.mlir").write_text(checked)
        return {}, checked

    monkeypatch.setattr(CC, "lower_interface", checked_lower)
    result = B.build_device_objects(
        device.device,
        host.rewrite.signatures,
        {symbol: ("i8", "i8", "i8")},
        package_dir=tmp_path,
        workdir=tmp_path / "objects",
        operand_dtype="i8",
        accum_dtype="i32",
        entries=host.rewrite.entries,
        call_buffers=host.rewrite.call_buffers,
        cflags=["-O2"],
        codegen_target="x86",
        connected_spatial=(device.connected_spatial, host),
    )
    assert result.ok and not result.skipped
    receipt = result.custody[symbol]["connected_spatial"]
    assert receipt["original_object_binding"]["status"] == "object_symbol_binding_verified"
    work = tmp_path / "objects"
    assert (work / f"{symbol}.lowering/lowered.llvm.mlir").read_text() == checked
    selected = (work / f"{symbol}.device.mlir").read_text()
    assert "selected_raw_helper" not in selected
    assert [
        f.sym_name.data
        for f in Parser(make_llvm_context(), selected).parse_module().body.block.ops
        if isinstance(f, llvm.FuncOp)
    ] == ["kernel_entry"]
    assert json.loads((work / f"{symbol}.private-helper.json").read_text())["checked_producer_sha256"] == (
        hashlib.sha256(checked.encode()).hexdigest()
    )
