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


def test_saved_composition_reopens_actual_candidate_and_original_owner_instead_of_flat_join(tmp_path):
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    module, groups = source()
    package = tmp_path / "candidate"
    package.mkdir()
    preference = package / "source_composition.json"
    preference.write_text(json.dumps({"schema": "merlin.source-composition-preferences.v1",
                                      "shared_input_siblings": True, "prepack_weights": False}))
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
    preference.write_text(preference.read_text() + " ")
    with pytest.raises(ValueError, match="compiled snapshot"):
        audit()
