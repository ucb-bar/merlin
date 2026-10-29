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
@pytest.mark.parametrize("op_profile", [False, True])
def test_original_source_and_fixed_host_reach_ordinary_build(
    tmp_path, monkeypatch, require_all_groups, host_math, share_preparations, dense_views, op_profile
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
        assert kwargs["op_profile"] is op_profile
        return {"status": "compiled"}
    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    assert service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                           linked_elf_admission=admission, output=tmp_path / "out", arena_mb=1,
                           host_math_policy=host_math,
                           share_exact_tensor_preparations=share_preparations,
                           dense_input_views=dense_views,
                           op_profile=op_profile,
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


def test_original_pre_group_layout_policy_reaches_actual_device_route(tmp_path, monkeypatch):
    import yaml

    from merlin.compile import baremetal_model, route_before_build
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.llvmlower.host_graph_policy import read_host_graph_policy

    service, candidate, capture = fixture_service(tmp_path)
    software = tmp_path/'software.yaml'
    software.write_text(yaml.safe_dump({'schema': 'merlin.software_spec.v1', 'status': 'reviewed',
                                       'target': service.target}))
    policy = read_host_graph_policy(software, target=service.target, entry='forward')
    route = DeviceRouting(device=service.target, package_dir=candidate, operand_dtype='i8', accum_dtype='i32',
                          require_complete_groups=True, granularity="group")
    monkeypatch.setattr(route_before_build, 'plan_before_build', lambda *a, **kw: {
        'device_routing': route, 'offload': {'groups': 1}})
    observed = []
    monkeypatch.setattr(baremetal_model, 'compile_saved_model', lambda **kw: observed.append(kw) or {})
    service.compile(capture=capture, candidate_package=candidate, datapath='int8',
                    linked_elf_admission=object(), output=tmp_path/'out', arena_mb=1,
                    require_all_accelerator_groups=True, pre_group_layout=policy, intermediate_storage=policy)
    assert route.pre_group_layout is None
    assert observed[0]['device'].pre_group_layout is policy
    assert observed[0]['device'].intermediate_storage is policy
    # Partial/source-projected routes need an independent group correspondence;
    # they cannot silently move this policy after their earlier binding.
    with pytest.raises(ValueError, match='deferred'):
        service.compile(capture=capture, candidate_package=candidate, datapath='int8',
                        linked_elf_admission=object(), output=tmp_path/'partial', arena_mb=1,
                        pre_group_layout=policy)


def test_existing_bounded_rne_request_reaches_original_host_builder(tmp_path, monkeypatch):
    from merlin.compile import baremetal_model, route_before_build

    service, candidate, capture = fixture_service(tmp_path)
    monkeypatch.setattr(route_before_build, "plan_before_build", lambda *a, **kw: {
        "device_routing": object(), "offload": {"groups": 1},
    })
    observed = []
    monkeypatch.setattr(baremetal_model, "compile_saved_model", lambda **kw: observed.append(kw) or {})
    service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                    linked_elf_admission=object(), output=tmp_path / "out", arena_mb=1,
                    host_bounded_rne=True)
    assert observed[0]["host_bounded_rne"] is True
    assert observed[0]["host_math_policy"] == "native"
    assert observed[0]["respect_captured_quantization_scope"] is True
    with pytest.raises(ValueError, match="host bounded RNE"):
        service.compile(capture=capture, candidate_package=candidate, datapath="int8",
                        linked_elf_admission=object(), output=tmp_path / "bad", arena_mb=1,
                        host_bounded_rne="true")


@pytest.mark.parametrize('kind', ['sibling_source', 'independent_heads'])
def test_composed_source_keeps_original_layout_policy_and_ordinary_route(tmp_path, monkeypatch, kind):
    import yaml
    from merlin.compile import baremetal_model, route_before_build
    from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
    from merlin.llvmlower.host_graph_policy import read_host_graph_policy
    from merlin.llvmlower.sibling_source_route import SiblingSourceRoute

    service, candidate, capture = fixture_service(tmp_path)
    software = tmp_path/'software.yaml'
    software.write_text(yaml.safe_dump({'schema':'merlin.software_spec.v1','status':'reviewed',
                                       'target':service.target}))
    policy = read_host_graph_policy(software,target=service.target,entry='forward',connected_layout=True)
    effects = BorrowedBufferEffects('original synchronous source owner',True,True,True)
    selected = (SiblingSourceRoute('0'*64,effects) if kind == 'sibling_source'
                else HeadBatchSourceRoute('0'*64,effects))
    route = DeviceRouting(device=service.target,package_dir=candidate,operand_dtype='i8',accum_dtype='i32',
                          require_complete_groups=True,granularity='group')
    monkeypatch.setattr(route_before_build,'plan_before_build',lambda *a,**kw:{
        'device_routing':route,'offload':{'groups':1}})
    observed = []
    monkeypatch.setattr(baremetal_model,'compile_saved_model',lambda **kw:observed.append(kw) or {})
    service.compile(capture=capture,candidate_package=candidate,datapath='int8',
                    linked_elf_admission=object(),output=tmp_path/'out',arena_mb=1,
                    require_all_accelerator_groups=True,pre_group_layout=policy,**{kind:selected})
    actual = observed[0]['device']
    assert actual.pre_group_layout is policy and actual.sibling_source is selected
    assert actual.require_complete_groups and actual.granularity == 'group'


@pytest.mark.parametrize("name", ["host_roundeven", "host_elementwise_fusion"])
def test_existing_host_codegen_choices_forward_only_true_and_refuse_nonbool(tmp_path, monkeypatch, name):
    from merlin.compile import baremetal_model, route_before_build

    service, candidate, capture = fixture_service(tmp_path)
    route = object()
    monkeypatch.setattr(route_before_build, "plan_before_build", lambda *a, **kw: {
        "device_routing": route, "offload": {"groups": 1}})
    calls = []
    monkeypatch.setattr(baremetal_model, "compile_saved_model", lambda **kw: calls.append(kw) or {})
    args = dict(capture=capture, candidate_package=candidate, datapath="int8",
                linked_elf_admission=object(), output=tmp_path / "out", arena_mb=1)
    service.compile(**args)
    service.compile(**args, **{name: False})
    assert name not in calls[0] and calls[0] == calls[1]
    service.compile(**args, **{name: True})
    assert calls[2] == {**calls[0], name: True}
    for bad in (1, "true", None):
        with pytest.raises(ValueError, match="explicit Boolean"):
            service.compile(**args, **{name: bad})
    assert len(calls) == 3
