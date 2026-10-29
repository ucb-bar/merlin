"""Optimization requests cannot replace original source eligibility or ABI checks."""

import json
from types import SimpleNamespace

import pytest
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.common import mlir_query as mq
from merlin.targetgen.owned_host_model_execution import (
    _composition_preferences,
    _rederive_composition_selection,
    _resolve_source_composition,
    _resolve_weight_preparation,
)
from merlin.xdsl_dialects._common import text


def source():
    module = mq.parse('''module {
      func.func @forward(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>, %w: tensor<2x2xi8>)
          -> (tensor<2x2xi32>, tensor<2x2xi32>) {
        %zero = arith.constant 0 : i32
        %i = tensor.splat %zero : tensor<2x2xi32>
        %q = linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%a, %b : tensor<2x2xi8>, tensor<2x2xi8>) outs(%i : tensor<2x2xi32>) {
          ^bb0(%x:i8, %y:i8, %acc:i32):
            %xx = arith.extsi %x : i8 to i32
            %yy = arith.extsi %y : i8 to i32
            %p = arith.muli %xx, %yy : i32
            %sum = arith.addi %acc, %p : i32
            linalg.yield %sum : i32
          } -> tensor<2x2xi32>
        %z = arith.constant 0 : i32
        %j = tensor.splat %z : tensor<2x2xi32>
        %r = linalg.generic {indexing_maps = [affine_map<(m,n,k)->(m,k)>,
          affine_map<(m,n,k)->(k,n)>, affine_map<(m,n,k)->(m,n)>],
          iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%a, %w : tensor<2x2xi8>, tensor<2x2xi8>) outs(%j : tensor<2x2xi32>) {
          ^bb0(%x:i8, %y:i8, %acc:i32):
            %xx = arith.extsi %x : i8 to i32
            %yy = arith.extsi %y : i8 to i32
            %p = arith.muli %xx, %yy : i32
            %sum = arith.addi %acc, %p : i32
            linalg.yield %sum : i32
          } -> tensor<2x2xi32>
        return %q, %r : tensor<2x2xi32>, tensor<2x2xi32>
      }
    }''')
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    for index, root in enumerate(roots):
        root.attributes.update({"prov.region_id": StringAttr(f"original-region-{index}"),
                                "prov.source_node_ids": ArrayAttr([StringAttr(f"original-node-{index}")])})
    groups = [SimpleNamespace(index=index, root=root, placement="test-endpoint")
              for index, root in zip((3, 7), roots, strict=True)]
    return module, groups


def test_source_owner_resolves_complete_roster_without_granting_execution():
    module, groups = source()
    before = text(module, generic=True)
    route, record = _resolve_source_composition(
        module, groups, {3, 7}, preferences={"shared_input_siblings": True},
    )
    assert route is not None and record["original_source_groups"] == [3, 7]
    assert record["status"] == "source_selected_not_qualified"
    assert "independently" in route.effects.qualification_identity
    assert text(module, generic=True) == before
    # Another independently eligible group cannot disappear behind the shared call.
    route, record = _resolve_source_composition(
        module, groups, {3, 7, 11}, preferences={"shared_input_siblings": True},
    )
    assert route is None and record["baseline_retained"] is True


def test_missing_original_identity_retains_baseline_instead_of_inventing_a_tag():
    module, groups = source()
    del groups[1].root.attributes["prov.source_node_ids"]
    route, record = _resolve_source_composition(
        module, groups, {3, 7}, preferences={"shared_input_siblings": True},
    )
    assert route is None and record["status"] == "not_applicable"
    assert "provenance" in record["reason"]


def test_preference_file_is_closed_and_carries_no_source_or_effect_authority(tmp_path):
    defaults = {"shared_input_siblings": False, "prepack_weights": False}
    assert _composition_preferences(tmp_path) == (defaults, None)
    path = tmp_path / "source_composition.json"
    request = {"schema": "merlin.source-composition-preferences.v1",
               "shared_input_siblings": True, "prepack_weights": True}
    path.write_text(json.dumps(request))
    selected, pin = _composition_preferences(tmp_path)
    assert selected == {key: True for key in defaults}
    assert pin["path"] == str(path) and len(pin["sha256"]) == 64
    for extra in ({"source_groups": [3, 7]}, {"complete_before_return": True}, {"prepack_weights": 1}):
        path.write_text(json.dumps({**request, **extra}))
        with pytest.raises(ValueError, match="unsupported"):
            _composition_preferences(tmp_path)
    path.write_text('{"schema": "first", "schema": "second"}')
    with pytest.raises(ValueError, match="repeat"):
        _composition_preferences(tmp_path)


def test_indirect_preferences_refuse_and_disabled_request_does_not_inspect_source(tmp_path):
    destination = tmp_path / "other.json"
    destination.write_text("{}")
    (tmp_path / "source_composition.json").symlink_to(destination)
    with pytest.raises(ValueError, match="direct"):
        _composition_preferences(tmp_path)
    assert _resolve_source_composition(None, None, None, preferences={"shared_input_siblings": False}) == (
        None, {"status": "not_requested"},
    )


def test_required_supporting_operation_in_selected_group_retains_baseline():
    module, groups = source()
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    indices = {id(op): index for index, op in enumerate(function.body.block.ops)}
    obligations = [{"source_op_index": indices[id(group.root)], "source_group": group.index,
                    "region_id": group.root.attributes["prov.region_id"].data,
                    "semantic_family": "contraction"} for group in groups]

    def resolve(rows):
        return _resolve_source_composition(module, groups, {3, 7},
                                           preferences={"shared_input_siblings": True}, eligible_source_regions=rows)

    assert resolve(obligations)[0] is not None
    for extra in ({**obligations[0], "semantic_family": "elementwise_map"},
                  {**obligations[0], "source_op_index": 0}):
        choice, record = resolve([*obligations, extra])
        assert choice is None and record["baseline_retained"] is True
        assert "supporting" in record["reason"]


def test_optional_prepack_without_hoistable_source_keeps_valid_baseline(tmp_path):
    module, _ = source()
    (tmp_path / "model.mlir").write_text(text(module, generic=True))
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "shape": [2, 2], "dtype": "int8"},
        "1": {"kind": "param", "weight": "w0", "shape": [2, 2], "dtype": "int8"},
        "2": {"kind": "param", "weight": "w1", "shape": [2, 2], "dtype": "int8"},
    }))
    selected, record = _resolve_weight_preparation(tmp_path, requested=True)
    assert selected is False and record["baseline_retained"] is True
    assert "no hoistable" in record["reason"]
    assert _resolve_weight_preparation(None, requested=False) == (False, {"status": "not_requested"})


@pytest.mark.parametrize("share_preparations", [False, True])
def test_saved_composition_reopens_actual_candidate_and_original_owner_instead_of_flat_join(
    tmp_path, share_preparations
):
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    module, groups = source()
    package = tmp_path / "candidate"
    package.mkdir()
    preference = package / "source_composition.json"
    preference.write_text(json.dumps({"schema": "merlin.source-composition-preferences.v1",
                                      "shared_input_siblings": True, "prepack_weights": False}))
    if share_preparations:
        request = json.loads(preference.read_text())
        request["share_exact_tensor_preparations"] = True
        preference.write_text(json.dumps(request))
    function = next(op for op in module.body.block.ops if op.name == "func.func")
    indices = {id(op): i for i, op in enumerate(function.body.block.ops)}
    placement = {"eligible_source_regions": [
        {"source_op_index": indices[id(group.root)], "source_group": group.index,
         "region_id": group.root.attributes["prov.region_id"].data, "semantic_family": "contraction"}
        for group in groups]}
    preferences, pin = _composition_preferences(package)
    original, resolved = _resolve_source_composition(
        module, groups, {3, 7}, preferences=preferences, eligible_source_regions=placement["eligible_source_regions"],
    )
    selection = {**resolved, "preferences": preferences, "preference_file": pin,
                 "weight_preparation": {"status": "not_requested"}}
    receipt = {"inputs": {"device": {"name": "test-endpoint", "package": str(package),
                                     "package_tree": strict_tree_sha256(package)}, "prepack_weights": False}}

    if share_preparations:
        receipt["inputs"]["share_exact_tensor_preparations"] = True

    def audit(record=selection):
        return _rederive_composition_selection(tmp_path, receipt, record, module=module, groups=groups,
                                               placement=placement, target="test-endpoint")

    assert audit().record() == original.record()
    with pytest.raises(ValueError, match="independently resolved"):
        audit({**selection, "original_source_groups": [3]})
    receipt["inputs"]["prepack_weights"] = True
    with pytest.raises(ValueError, match="independently resolved"):
        audit()
    receipt["inputs"]["prepack_weights"] = False
    receipt["inputs"]["share_exact_tensor_preparations"] = not share_preparations
    with pytest.raises(ValueError, match="independently resolved"):
        audit()
    if share_preparations:
        receipt["inputs"]["share_exact_tensor_preparations"] = True
    else:
        receipt["inputs"].pop("share_exact_tensor_preparations")
    receipt["inputs"]["dense_input_views"] = True
    with pytest.raises(ValueError, match="independently resolved"):
        audit()
    receipt["inputs"].pop("dense_input_views")
    preference.write_text(preference.read_text() + " ")
    with pytest.raises(ValueError, match="compiled snapshot"):
        audit()


@pytest.mark.parametrize("choice", [False, True])
@pytest.mark.parametrize("option", [
    "share_exact_tensor_preparations", "dense_input_views", "op_profile", "group_profile",
    "propagate_layouts", "reuse_intermediate_storage", "whole_head_batches", "shared_rhs_row_views", "host_bounded_rne",
    "host_roundeven", "host_elementwise_fusion"])
def test_exact_preparation_sharing_is_a_pinned_optional_request(tmp_path, choice, option):
    request = {"schema": "merlin.source-composition-preferences.v1",
               "shared_input_siblings": False, "prepack_weights": False,
               option: choice}
    path = tmp_path / "source_composition.json"
    path.write_text(json.dumps(request))
    selected, pin = _composition_preferences(tmp_path)
    assert selected == {key: value for key, value in request.items() if key != "schema"}
    from hashlib import sha256

    assert pin["sha256"] == sha256(path.read_bytes()).hexdigest()
    request[option] = 1
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="unsupported"):
        _composition_preferences(tmp_path)


def test_coarse_profile_reports_gaps_without_ranking_functional_ticks():
    from merlin.targetgen.owned_host_model_execution import _group_profile

    record = {"build": {"inputs": {"group_profile": True}}}
    console = (b"GROUP_ID 0 first\nGROUP 0 12\nGAP 0 3\nGROUP_ID 1 second\nGROUP 1 8\n"
               b"GAP 1 4\nGAP 2 5\nMETRIC group_calls 2\nMETRIC group_calls_dropped 0\nDONE\n")
    result = _group_profile(record, console)
    assert result["status"] == "observed"
    assert result["group_counter_ticks"] == 20
    assert result["between_group_counter_ticks"] == 12
    assert result["total_counter_ticks"] == 32
    assert result["unit"] == "counter_ticks"
    assert not result["timing_qualified"] and not result["performance_ranking_eligible"]
    assert _group_profile({"build": {"inputs": {}}}, console) is None
    for bad in (b"", console.replace(b"GAP 1 4\n", b""),
                console.replace(b"group_calls_dropped 0", b"group_calls_dropped 1")):
        assert _group_profile(record, bad)["status"] == "unavailable"


def test_bounded_rne_request_uses_existing_callback_proof_not_old_observer_permissions(tmp_path, monkeypatch):
    from pathlib import Path

    from merlin.common.digest import sha256_file
    from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform
    from merlin.llvmlower.toolchain import clang
    from merlin.targetgen.source_composition_preferences import host_preparation, require_host_preparation

    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    monkeypatch.delenv("MERLIN_HOST_BOUNDED_RNE", raising=False)
    request = {"host_bounded_rne": True}
    options, selected = host_preparation(request, target="device")
    assert options == request
    assert selected["host_bounded_rne"]["effect_grant"] is False
    original = tmp_path / "source.ll"
    original.write_text("""define i8 @quant(float %x) {
  %lo = call float @llvm.maximum.f32(float %x, float -128.0)
  %hi = call float @llvm.minimum.f32(float %lo, float 127.0)
  %rounded = call float @llvm.roundeven.f32(float %hi)
  %result = fptosi float %rounded to i8
  ret i8 %result
}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
declare float @llvm.roundeven.f32(float)
""")
    merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True)(
        original, tmp_path / "host_llvm")
    report = tmp_path / "host_llvm/receipt.json"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"preparation": {"host_bounded_rne": {
        "path": str(report), "sha256": sha256_file(report),
    }}}))
    built = {"inputs": {"host_bounded_rne": {"host_isa": "rv64gc", "combine_clamp": True}},
             "output": {"compilation_recipe": {"path": str(recipe), "sha256": sha256_file(recipe)}}}
    require_host_preparation(built, options, selected)
    require_host_preparation({"inputs": {}}, {"host_bounded_rne": False})
    with pytest.raises(ValueError, match="selected request"):
        require_host_preparation({"inputs": {}}, options)
    report.write_text("{}")
    with pytest.raises(ValueError, match="source proof changed"):
        require_host_preparation(built, options)


@pytest.mark.parametrize("composed", [False, True])
def test_layout_request_selects_connected_planning_from_original_software(tmp_path, monkeypatch, composed):
    import yaml
    from merlin_experiments.corpus import numeric_policy
    from merlin.targetgen import capsule_runner, software_spec
    from merlin.targetgen.source_composition_preferences import host_preparation
    from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA

    recipe, software = tmp_path / "recipe.yaml", tmp_path / "software.yaml"
    recipe.write_text("schema: fixture\n")
    software.write_text(yaml.safe_dump({
        "schema": "merlin.software_spec.v1", "status": "reviewed", "target": "neutral",
        "host_optimization": {"schema": POLICY_SCHEMA, "permitted_stages": ["logical_host"],
                              "permitted_optimizations": ["propagate_layout"],
                              "independent_element_order_unobserved": True},
    }))
    frozen = []
    def original_input(path, snapshot):
        assert snapshot == "original-snapshot"
        frozen.append(path)
        return path
    monkeypatch.setattr(numeric_policy, "numeric_profile_path", lambda *a, **kw: recipe)
    monkeypatch.setattr(software_spec, "software_spec_path_for_recipe", lambda *a, **kw: software)
    monkeypatch.setattr(capsule_runner, "_model_frozen_input", original_input)
    experiment = SimpleNamespace(numeric_profile="fixture", _source_root=lambda: tmp_path)
    options, selected = host_preparation(
        {"propagate_layouts": True}, target="neutral", composed=composed,
        lane=(experiment, None, {"run_snapshot": "original-snapshot"}),
    )
    assert frozen == [recipe, software]
    policy = options["pre_group_layout"]
    policy.revalidate()
    assert policy.connected_layout is True
    assert policy.software_bytes == software.read_bytes()
    assert selected["propagate_layouts"]["permissions_admitted"] is True
    assert selected["propagate_layouts"]["status"] == "policy_selected_not_applied"
    assert selected["propagate_layouts"]["source_stage"] == (
        "after_source_projection" if composed else "before_group_binding")


def original_descriptor_effects_factory(source, built):
    raise AssertionError("selection must not qualify or invoke the original owner")


@pytest.mark.parametrize("composed,permitted", [(False, True), (True, True), (False, False)])
def test_storage_request_joins_only_explicit_original_factory(tmp_path, monkeypatch, composed, permitted):
    import yaml
    from merlin_experiments.corpus import numeric_policy
    from merlin.targetgen import capsule_runner, software_spec
    from merlin.targetgen.source_composition_preferences import host_preparation
    from merlin.llvmlower.host_optimization_request import POLICY_SCHEMA

    recipe, software = tmp_path/'recipe.yaml',tmp_path/'software.yaml'
    recipe.write_text('schema: original\n')
    software.write_text(yaml.safe_dump({
        'schema':'merlin.software_spec.v1','status':'reviewed','target':'neutral',
        'host_optimization':{'schema':POLICY_SCHEMA,'permitted_stages':['llvm'],
            'permitted_optimizations':['static_arena'],'sequential_execution':permitted,
            'successful_allocations':True,'non_reentrant_entry':True}}))
    monkeypatch.setattr(numeric_policy,'numeric_profile_path',lambda *a,**kw:recipe)
    monkeypatch.setattr(software_spec,'software_spec_path_for_recipe',lambda *a,**kw:software)
    monkeypatch.setattr(capsule_runner,'_model_frozen_input',lambda path,snapshot:path)
    lane=(SimpleNamespace(numeric_profile='original',_source_root=lambda:tmp_path),None,{})
    kwargs=dict(target='neutral',composed=composed,lane=lane)
    ordinary, original = host_preparation({'reuse_intermediate_storage':True},**kwargs)
    assert 'descriptor_results' not in ordinary and 'descriptor_effects' not in ordinary
    assert 'descriptor_results' not in original['reuse_intermediate_storage']
    options, chosen = host_preparation({'reuse_intermediate_storage':True},
        descriptor_effects=original_descriptor_effects_factory,**kwargs)
    row=chosen['reuse_intermediate_storage']['descriptor_results']
    assert row['effect_grant'] is False
    if permitted and not composed:
        assert options['descriptor_results'] is True
        assert options['descriptor_effects'] is original_descriptor_effects_factory
        assert row['status']=='original_factory_selected_not_qualified'
    else:
        assert 'descriptor_results' not in options
        assert row['status']=='deferred'
