"""The fixed host provider cannot be replaced through candidate build inputs."""

from pathlib import Path

import pytest

from merlin.compile.owned_host_model import OwnedHostModelCompiler


def fixture_service(tmp_path):
    host = tmp_path / "host"
    candidate = tmp_path / "device"
    capture = tmp_path / "capture"
    for path in (host, candidate, capture):
        path.mkdir()
    (host / "host.mlir").write_text("fixed host provider")
    (candidate / "device.mlir").write_text("candidate device provider")
    (capture / "model.mlir").write_text("original model source")
    service = OwnedHostModelCompiler.select(
        host_package=host, target="device", board_catalog=tmp_path / "boards.yaml",
        board="board", dts=tmp_path / "board.dts",
    )
    return service, candidate, capture


def test_original_source_and_fixed_host_reach_ordinary_build(tmp_path, monkeypatch):
    from merlin.compile import baremetal_model, route_before_build
    from merlin.llvmlower.device_offload import BY_GROUP

    service, candidate, capture = fixture_service(tmp_path)
    route = object()
    admission = object()
    def plan(target, source, **kwargs):
        assert source == "original model source"
        assert kwargs["device_package"] == str(candidate)
        assert kwargs["linked_elf_admission"] is admission
        assert kwargs["granularity"] == BY_GROUP
        return {"device_routing": route, "offload": {"groups": 1}}
    monkeypatch.setattr(route_before_build, "plan_before_build", plan)
    def build(**kwargs):
        assert kwargs["package"] == service.host_package
        assert kwargs["capture"] == capture
        assert kwargs["device"] is route
        assert kwargs["readback"] == "full"
        assert kwargs["respect_captured_quantization_scope"] is True
        return {"status": "compiled"}
    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    assert service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                           linked_elf_admission=admission, output=tmp_path / "out", arena_mb=1) == {"status": "compiled"}


@pytest.mark.parametrize("case", ["same_provider", "changed_provider", "missing_admission"])
def test_untrusted_provider_substitution_refuses_before_build(tmp_path, case):
    service, candidate, capture = fixture_service(tmp_path)
    admission = object()
    if case == "same_provider":
        candidate = service.host_package
    elif case == "changed_provider":
        (service.host_package / "host.mlir").write_text("changed host provider")
    else:
        admission = None
    with pytest.raises(ValueError):
        service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                        linked_elf_admission=admission, output=tmp_path / "out", arena_mb=1)
