"""Multiple independent source clusters retain unmatched work and observers."""
import numpy as np
import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.sibling_components import prepare_sibling_components
from merlin.runtime.linalg_numpy import evaluate
from merlin.xdsl_dialects._common import text


def multi_source(*, effect=False, transpose_weights=False, dense_views=False):
    declarations = "func.func private @opaque()" if effect else ""
    parts = [f"module {{{declarations} func.func @forward(" +
             ",".join(f"%{n}:tensor<{('3x2' if dense_views and n in 'adg' else '2x2')}xi8>" for n in "abcdefgh") +
             ")->(" + ",".join(["tensor<2x2xi32>"] * 5) + "){%z=arith.constant 0:i32"]
    if dense_views:
        for name in "adg":
            parts.append(f"%view_{name}=tensor.extract_slice %{name} [1,0] [2,2] [1,1]:tensor<3x2xi8> to tensor<2x2xi8>")
    if transpose_weights:
        for name in "bcefh":
            parts.append(f"%empty_{name}=tensor.empty():tensor<2x2xi8>")
            parts.append(f"%trans_{name}=linalg.transpose ins(%{name}:tensor<2x2xi8>) outs(%empty_{name}:tensor<2x2xi8>) permutation=[1,0]")
    for i, (left, right) in enumerate((('a', 'b'), ('a', 'c'), ('d', 'e'), ('d', 'f'), ('g', 'h'))):
        left = "view_" + left if dense_views else left
        right = "trans_" + right if transpose_weights else right
        if effect and i == 1:
            parts.append("func.call @opaque():()->()")
        parts.append(f'''%s{i}=tensor.splat %z:tensor<2x2xi32>
          %r{i}=linalg.generic {{indexing_maps=[affine_map<(m,n,k)->(m,k)>,
            affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
            iterator_types=["parallel","parallel","reduction"],
            prov.region_id="root{i}",prov.source_node_ids=["node{i}"]}}
          ins(%{left},%{right}:tensor<2x2xi8>,tensor<2x2xi8>) outs(%s{i}:tensor<2x2xi32>) {{
          ^bb0(%x:i8,%y:i8,%s:i32):
            %xx=arith.extsi %x:i8 to i32
            %yy=arith.extsi %y:i8 to i32
            %p=arith.muli %xx,%yy:i32
            %v=arith.addi %s,%p:i32
            linalg.yield %v:i32
          }} ->tensor<2x2xi32>''')
    parts.append("return " + ",".join(f"%r{i}" for i in range(5)) + ":" +
                 ",".join(["tensor<2x2xi32>"] * 5) + "}}")
    module = parse_mlir_text("\n".join(parts))
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    for i, root in enumerate(op for op in module.walk() if op.name == "linalg.generic"):
        root.attributes["prov.region_id"] = StringAttr(f"root{i}")
        root.attributes["prov.source_node_ids"] = ArrayAttr([StringAttr(f"node{i}")])
    return module


def effects():
    return BorrowedBufferEffects("independently-selected-source-effects", True, True, True)


def test_two_components_preserve_unmatched_original_and_exact_outputs():
    module = multi_source()
    before = text(module, generic=True)
    plan = prepare_sibling_components(module, effects=effects())
    assert len(plan.components) == 2
    assert text(module, generic=True) == before
    forward = next(o for o in plan.module.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == "forward")
    assert sum(o.name == "linalg.generic" for o in forward.walk()) == 1
    values = [np.arange(4, dtype=np.int8).reshape(2, 2) - i for i in range(8)]
    expected = evaluate(module, values)
    calls = {}
    for component in plan.components:
        # Independent original raw-product callback, never a target oracle.
        def call(args):
            lhs = args[0].astype(np.int64)
            return tuple((lhs @ rhs.astype(np.int64)).astype(np.int32) for rhs in args[1:3])
        calls[component.writer.symbol] = call
    actual = evaluate(plan.module, values, calls=calls)
    assert len(actual) == 5
    assert all(np.array_equal(a, b) for a, b in zip(expected, actual, strict=True))


def test_unknown_crossed_effect_keeps_that_cluster_but_other_cluster_is_usable():
    plan = prepare_sibling_components(multi_source(effect=True), effects=effects())
    assert len(plan.components) == 1
    assert plan.receipt["refused_proposals"]
    assert any(o.name == "func.call" and o.callee.root_reference.data == "opaque" for o in plan.module.walk())


def test_batch_receipts_keep_common_original_coordinates_and_final_host():
    from hashlib import sha256

    module = multi_source()
    operations = tuple(module.body.block.first_op.body.block.ops)
    roots = [op for op in operations if op.name == "linalg.generic"]
    plan = prepare_sibling_components(module, effects=effects())
    source_sha = sha256(text(module, generic=True).encode()).hexdigest()
    host_sha = sha256(text(plan.module, generic=True).encode()).hexdigest()
    assert plan.receipt["normalized_source_ir_sha256"] == source_sha
    assert plan.receipt["host_ir_sha256"] == host_sha
    for index, component in enumerate(plan.components):
        receipt = component.receipt["outline"]
        assert receipt["source_ir_sha256"] == source_sha
        assert receipt["host_ir_sha256"] == host_sha
        assert receipt["source_operation_coordinate_system"] == "normalized_batch_source"
        assert receipt["source_root_ordinals"] == [operations.index(op) for op in roots[index * 2:index * 2 + 2]]


@pytest.mark.parametrize("damage", [None, "source", "host", "core", "component_host", "component_source"])
def test_common_graph_custody_rechecks_mutations_and_component_identity(damage):
    import json
    from dataclasses import replace
    from hashlib import sha256
    from types import SimpleNamespace

    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.sibling_component_route import RoutedSiblingComponents, require_components
    from merlin.llvmlower.sibling_component_route import _snapshot as batch_snapshot
    from merlin.llvmlower.sibling_source_route import RoutedSiblingHost, SiblingSourceRoute
    from merlin.llvmlower.sibling_source_route import _snapshot as single_snapshot

    def digest(record):
        return sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

    source = multi_source()
    plan = prepare_sibling_components(source, effects=effects())
    selected = SiblingSourceRoute(sha256(text(source, generic=True).encode()).hexdigest(), effects(), partial=True)
    rewrite = SimpleNamespace(to_sidecar=lambda: {})
    components = []
    for component in plan.components:
        host = RoutedSiblingHost(source, plan.module, component.core, component.writer, "source_only",
                                 "no target implementation selected", rewrite, component.receipt, "")
        components.append(replace(host, integrity_sha256=digest(single_snapshot(host))))
    batch = RoutedSiblingComponents(source, plan.module, tuple(components), rewrite, plan.receipt, "")
    batch = replace(batch, integrity_sha256=digest(batch_snapshot(batch)))
    require_components(batch, selected)
    if damage is None:
        return
    if damage in {"source", "host", "core"}:
        changed = {"source": source, "host": plan.module, "core": components[0].core}[damage]
        changed.attributes["changed_after_validation"] = StringAttr("yes")
    else:
        field = "module" if damage == "component_host" else "source"
        changed = replace(components[0], **{field: getattr(components[0], field).clone()})
        batch = replace(batch, components=(changed, *components[1:]))
    with pytest.raises(ValueError):
        require_components(batch, selected)


@pytest.mark.parametrize("mode", ["shared", "distinct", "receipt_changed", "binding_changed"])
def test_weight_preparation_reuse_is_local_and_keeps_each_receipt_pin(tmp_path, monkeypatch, mode):
    import json
    from dataclasses import replace
    from hashlib import sha256
    from types import SimpleNamespace

    from merlin.llvmlower import weight_prepack
    from merlin.llvmlower.sibling_component_route import RoutedSiblingComponents, require_components
    from merlin.llvmlower.sibling_component_route import _snapshot as batch_snapshot
    from merlin.llvmlower.sibling_source_route import RoutedSiblingHost, SiblingSourceRoute
    from merlin.llvmlower.sibling_source_route import _snapshot as single_snapshot

    def digest(record):
        return sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def pin(name):
        path = tmp_path / name
        path.write_text(name)
        return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}

    preparations = [dict(pin("preparation0"), capture_binding=pin("binding0"))]
    preparations.append(dict(pin("preparation1"), capture_binding=pin("binding1"))
                        if mode == "distinct" else preparations[0])
    checked = []
    monkeypatch.setattr(weight_prepack, "revalidate_build_preparation", lambda path: checked.append(path))
    source = multi_source()
    plan = prepare_sibling_components(source, effects=effects())
    selected = SiblingSourceRoute(sha256(text(source, generic=True).encode()).hexdigest(), effects(), partial=True)
    rewrite = SimpleNamespace(to_sidecar=lambda: {})
    components = []
    for component, preparation in zip(plan.components, preparations, strict=True):
        receipt = {**component.receipt, "weight_preparation": preparation}
        host = RoutedSiblingHost(source, plan.module, component.core, component.writer, "source_only",
                                 "no target implementation selected", rewrite, receipt, "")
        components.append(replace(host, integrity_sha256=digest(single_snapshot(host))))
    batch = RoutedSiblingComponents(source, plan.module, tuple(components), rewrite, plan.receipt, "")
    batch = replace(batch, integrity_sha256=digest(batch_snapshot(batch)))
    require_components(batch, selected)
    assert checked == [row["path"] for row in preparations[:2 if mode == "distinct" else 1]]
    checked.clear()
    if mode in {"receipt_changed", "binding_changed"}:
        pin_to_change = preparations[0] if mode == "receipt_changed" else preparations[0]["capture_binding"]
        from pathlib import Path

        Path(pin_to_change["path"]).write_text("changed after earlier complete validation")
        with pytest.raises(ValueError, match="preparation changed"):
            require_components(batch, selected)
        assert checked == []
    else:
        require_components(batch, selected)
        assert checked == [row["path"] for row in preparations[:2 if mode == "distinct" else 1]]


@pytest.mark.parametrize("limits", [{"max_members": 1}, {"max_members": True}, {"max_components": 0}])
def test_invalid_component_limits_refuse(limits):
    with pytest.raises(ValueError, match="finite"):
        prepare_sibling_components(multi_source(), effects=effects(), **limits)


def test_complete_core_reader_accepts_shared_original_zero_and_refuses_unrelated_literal():
    from xdsl.dialects import arith
    from merlin.llvmlower.sibling_source_route import _command_buffer

    plan = prepare_sibling_components(multi_source(), effects=effects())
    for component in plan.components:
        cb, group = _command_buffer(component.core, "selected-target")
        assert len(group.widths) == 2
        assert len([c for c in cb["commands"] if c["opcode"] == "MATMUL_RESIDENT"]) == 2
    core = plan.components[0].core.clone()
    block = core.body.block.first_op.body.block
    block.insert_op_before(arith.ConstantOp.from_int_and_width(7, 32), block.last_op)
    with pytest.raises(ValueError, match="extra operations"):
        _command_buffer(core, "selected-target")


def component_metadata(module):
    """Source-only metadata fixture, never a target/placement qualification."""
    from types import SimpleNamespace
    from merlin.llvmlower.sibling_source_route import _command_buffer
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
    from hashlib import sha256

    roots = [o for o in module.walk() if o.name == "linalg.generic"]
    groups = [SimpleNamespace(index=i, root=r, members=(r,), placement="selected-target")
              for i, r in enumerate(roots)]
    source = source_group_roster(groups, list(range(5)), host_placement="host", module=module)
    emitted = {"device": "selected-target", "granularity": "group",
               "model_sha256": sha256(text(module, generic=True).encode()).hexdigest(),
               **{key: {} for key in ("signatures", "entries", "programs", "expected_interfaces", "call_buffers", "arg_access")},
               "routed": [], "skipped": []}
    for number, component in enumerate(prepare_sibling_components(module, effects=effects()).components):
        cb, group = _command_buffer(component.core, "selected-target")
        symbol = component.writer.symbol + "_device"
        interface = emit_interface_mlir(cb)
        buffers, _ = logical_buffers(interface)
        emitted["signatures"][symbol] = [group.m, sum(group.widths), group.k]
        emitted["entries"][symbol] = {"schema": "original_sibling_source_entry_v1", "name": symbol,
            "source_ir_sha256": emitted["model_sha256"], "core_ir_sha256": sha256(text(component.core, generic=True).encode()).hexdigest()}
        emitted["programs"][symbol] = cb
        emitted["expected_interfaces"][symbol] = {"mlir": interface, "sha256": sha256(interface.encode()).hexdigest()}
        emitted["call_buffers"][symbol] = [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers]
        emitted["arg_access"][symbol] = ["read"] * 3 + ["write"] * 2
        emitted["routed"].append({"group": number, "symbol": symbol, "parallel": [2, 4],
                                  "reduction": [2], "dtypes": ["i8", "i8", "i32"]})
    emitted["routed"].append({"group": 2, "symbol": "ordinary_unmatched", "source_region_id": "root4",
                              "source_node_ids": ["node4"], "parallel": [2, 2], "reduction": [2],
                              "dtypes": ["i8", "i8", "i32"]})
    return source, emitted


def test_independent_join_and_device_budget_keep_every_original_root():
    from merlin_experiments.phase1.feedback.private_component_provenance import join_components
    from merlin_experiments.phase1.feedback.private_device_audit import _composed_source_metrics
    from merlin.targetgen.private_sibling_dispatch import composition_members

    module = multi_source()
    source, emitted = component_metadata(module)
    joined = join_components(emitted, source, module=module, target="selected-target", effects=effects())
    assert joined["ordinary_joined_groups"] == {2: 4}
    assert sorted(joined["covered_source_groups"]) == list(range(5))
    assert len(composition_members(joined, assigned=emitted["routed"])) == 4
    groups, metrics = _composed_source_metrics(emitted["routed"], source, joined)
    assert groups == {0: (0, 1), 1: (2, 3), 2: (4,)}
    assert [metrics[i]["elements"] for i in range(3)] == [8, 8, 4]


@pytest.mark.parametrize("damage", ["missing_component", "missing_unmatched", "duplicate_unmatched",
                                    "member_identity", "output_precision", "operand", "root_geometry"])
def test_independent_component_join_refuses_missing_or_changed_original_work(damage):
    from merlin_experiments.phase1.feedback.private_component_provenance import join_components

    module = multi_source()
    source, emitted = component_metadata(module)
    symbol = emitted["routed"][0]["symbol"]
    if damage == "missing_component":
        emitted["routed"].pop(0)
    elif damage == "missing_unmatched":
        emitted["routed"].pop()
    elif damage == "duplicate_unmatched":
        emitted["routed"].append(dict(emitted["routed"][-1], group=3))
    elif damage == "member_identity":
        emitted["programs"][symbol]["commands"][1]["attributes"]["prov.region_id"] = "other"
    elif damage == "output_precision":
        emitted["call_buffers"][symbol][-1]["dtype"] = "i8"
    elif damage == "operand":
        emitted["programs"][symbol]["commands"][1]["operands"]["lhs"] = "weight_0"
    else:
        source["source_group_contractions"][0]["reduction"] = [3]
    with pytest.raises(ValueError):
        join_components(emitted, source, module=module, target="selected-target", effects=effects())


def test_ordinary_preference_selects_multiple_components_and_retains_unmatched_requirement():
    from types import SimpleNamespace
    from merlin.targetgen.owned_host_model_execution import _resolve_source_composition

    module = multi_source()
    roots = [o for o in module.walk() if o.name == "linalg.generic"]
    groups = [SimpleNamespace(index=i, root=r, members=(r,), placement="selected-target")
              for i, r in enumerate(roots)]
    function = module.body.block.first_op
    indices = {id(op): i for i, op in enumerate(function.body.block.ops)}
    obligations = [{"source_op_index": indices[id(r)], "source_group": i,
                    "region_id": f"root{i}", "semantic_family": "contraction"}
                   for i, r in enumerate(roots)]
    route, record = _resolve_source_composition(
        module, groups, set(range(5)), preferences={"shared_input_siblings": True},
        eligible_source_regions=obligations)
    assert route.partial is True
    assert record["component_count"] == 2
    assert record["composed_source_groups"] == [0, 1, 2, 3]
    assert record["unmatched_source_groups"] == [4]
    assert record["original_source_groups"] == list(range(5))
    obligations[0]["semantic_family"] = "supporting_required"
    route, record = _resolve_source_composition(
        module, groups, set(range(5)), preferences={"shared_input_siblings": True},
        eligible_source_regions=obligations)
    assert route is None and record["baseline_retained"] is True
