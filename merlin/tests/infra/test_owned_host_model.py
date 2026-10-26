"""The fixed host provider cannot be replaced through candidate build inputs."""

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


@pytest.mark.parametrize("require_all_groups", [None, True])
@pytest.mark.parametrize("host_math", ["native", "portable_exp_trig"])
@pytest.mark.parametrize("share_preparations", [False, True])
@pytest.mark.parametrize("dense_views", [False, True])
def test_original_source_and_fixed_host_reach_ordinary_build(
    tmp_path, monkeypatch, require_all_groups, host_math, share_preparations, dense_views
):
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
        assert kwargs["require_all_accelerator_groups"] is (require_all_groups is True)
        return {"device_routing": route, "offload": {"groups": 1}}
    monkeypatch.setattr(route_before_build, "plan_before_build", plan)
    def build(**kwargs):
        assert kwargs["package"] == service.host_package
        assert kwargs["capture"] == capture
        assert kwargs["device"] is route
        assert kwargs["readback"] == "full"
        assert kwargs["respect_captured_quantization_scope"] is True
        assert kwargs["host_math_policy"] == host_math
        assert kwargs["share_exact_tensor_preparations"] is share_preparations
        assert kwargs["dense_input_views"] is dense_views
        return {"status": "compiled"}
    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    assert service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                           linked_elf_admission=admission, output=tmp_path / "out", arena_mb=1,
                           host_math_policy=host_math,
                           share_exact_tensor_preparations=share_preparations,
                           dense_input_views=dense_views,
                           **({} if require_all_groups is None else
                              {"require_all_accelerator_groups": require_all_groups})) == {"status": "compiled"}


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


def test_unknown_math_policy_refuses_before_source_routing(tmp_path):
    service, candidate, capture = fixture_service(tmp_path)
    with pytest.raises(ValueError, match="host_math_policy"):
        service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                        linked_elf_admission=object(), output=tmp_path / "out", arena_mb=1,
                        host_math_policy="invented")


@pytest.mark.parametrize("choice", [0, 1, None, "true"])
def test_tensor_preparation_choice_refuses_before_source_routing(tmp_path, choice):
    service, candidate, capture = fixture_service(tmp_path)
    with pytest.raises(ValueError, match="tensor preparation sharing"):
        service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                        linked_elf_admission=object(), output=tmp_path / "out", arena_mb=1,
                        share_exact_tensor_preparations=choice)


@pytest.mark.parametrize("choice", [0, 1, None, "true"])
def test_dense_input_view_choice_refuses_before_source_routing(tmp_path, choice):
    service, candidate, capture = fixture_service(tmp_path)
    with pytest.raises(ValueError, match="dense input views"):
        service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                        linked_elf_admission=object(), output=tmp_path / "out", arena_mb=1,
                        dense_input_views=choice)
