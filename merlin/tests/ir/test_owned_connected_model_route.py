"""Owned model entry reaches existing original-SSA spatial preparation."""

from dataclasses import replace

import pytest
from test_connected_spatial_route import route
from xdsl.dialects import memref

from merlin.compile.owned_host_model import OwnedHostModelCompiler
from merlin.llvmlower import device_preparation, device_shim
from merlin.llvmlower.device_offload import load_sidecar
from merlin.xdsl_dialects._common import text


def test_owned_model_selected_route_reaches_actual_original_host_preparation(tmp_path, monkeypatch):
    from merlin.compile import baremetal_model, route_before_build

    module, rewrite, selected_device = route(tmp_path)
    original_device = replace(selected_device, connected_spatial=None)
    selected = selected_device.connected_spatial
    host = tmp_path / "fixed_host"
    candidate = tmp_path / "candidate"
    capture = tmp_path / "capture"
    for path in (host, candidate, capture):
        path.mkdir()
    (host / "fixed.txt").write_text("original selected host")
    original_text = text(module, generic=True)
    (capture / "model.mlir").write_text(original_text)
    service = OwnedHostModelCompiler.select(
        host_package=host, target=selected_device.device,
        board_catalog=tmp_path / "boards.yaml", board="board", dts=tmp_path / "board.dts",
    )
    admission = object()

    def plan(target, source, **kwargs):
        assert target == selected_device.device and source == original_text
        assert kwargs["linked_elf_admission"] is admission
        assert kwargs["require_all_accelerator_groups"] is True
        return {"device_routing": original_device, "offload": {"groups": 2}}

    monkeypatch.setattr(route_before_build, "plan_before_build", plan)
    monkeypatch.setattr(device_shim, "kernel_abi_for", lambda _: device_shim.KernelAbi("logical_kernel", version=2))

    def build(**kwargs):
        assert kwargs["capture"] == capture and kwargs["package"] == host
        assert kwargs["respect_captured_quantization_scope"] is True
        assert kwargs["readback"] == "full"
        actual = kwargs["device"]
        assert actual == replace(original_device, connected_spatial=selected)
        assert original_device.connected_spatial is None
        # Continue through real current-source grouping, ownership and sidecar
        # preparation. Only the outer expensive board/object driver is replaced.
        work = kwargs["output"]
        work.mkdir()
        prepared = work / "prepared.mlir"
        prepared.write_text(original_text)
        rewrite.write_sidecar(work)
        frozen = device_preparation.freeze_device_offload_selection(actual)
        path = device_preparation.prepare_device_route(
            actual, prepared, work, early_complete_groups=True, selection=frozen,
        )
        device_preparation.require_device_offload_selection(actual, frozen)
        owner = frozen.spatial_host
        assert owner.scratch.module.body.block.first_op.function_type == module.body.block.first_op.function_type
        assert owner.scratch.plan_sha256 == selected.plan.identity
        assert owner.source_entry["components"][0]["acc_scale"] == rewrite.entries["producer"]["acc_scale"]
        assert owner.source_entry["components"][1]["acc_scale"] == rewrite.entries["pool"]["acc_scale"]
        forward = owner.scratch.module.body.block.first_op
        assert len([op for op in forward.body.block.ops if isinstance(op, memref.AllocOp)]) == 1
        sidecar = load_sidecar(work)
        assert set(sidecar["entries"]) == {selected.writer.symbol}
        assert path.is_file() and (work / "connected_spatial/original-input.mlir").read_text() == original_text
        assert (capture / "model.mlir").read_text() == original_text
        return {"status": "original_host_prepared", "source_components": 2}

    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    assert service.compile(
        capture=capture, candidate_package=candidate, datapath="int8",
        linked_elf_admission=admission, output=tmp_path / "build", arena_mb=1,
        require_all_accelerator_groups=True, connected_spatial=selected,
    ) == {"status": "original_host_prepared", "source_components": 2}


@pytest.mark.parametrize("damage", ["opaque_route", "different_selection"])
def test_owned_model_does_not_accept_opaque_or_replace_distinct_selection(tmp_path, monkeypatch, damage):
    from merlin.compile import baremetal_model, route_before_build

    module, _, device = route(tmp_path)
    host, candidate, capture = (tmp_path / name for name in ("host", "candidate", "capture"))
    for path in (host, candidate, capture):
        path.mkdir()
    (host / "fixed.txt").write_text("original selected host")
    (capture / "model.mlir").write_text(text(module, generic=True))
    service = OwnedHostModelCompiler.select(
        host_package=host, target=device.device,
        board_catalog=tmp_path / "boards", board="board", dts=tmp_path / "dts",
    )
    selected = object() if damage == "opaque_route" else replace(device.connected_spatial)
    monkeypatch.setattr(
        route_before_build, "plan_before_build", lambda *a, **kw: {"device_routing": device, "offload": {}}
    )
    monkeypatch.setattr(baremetal_model, "compile_saved_model", lambda **kw: pytest.fail("invalid route reached build"))
    with pytest.raises((TypeError, ValueError), match="typed spatial route|distinct connected"):
        service.compile(
            capture=capture, candidate_package=candidate, datapath="int8", linked_elf_admission=object(),
            output=tmp_path / "out", arena_mb=1, connected_spatial=selected,
        )
