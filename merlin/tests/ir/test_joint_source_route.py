"""Disjoint head families and shared-input components retain all original work."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_head_batching import source as head_source
from test_sibling_components import effects, multi_source
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.head_source_route import project_heads
from merlin.llvmlower.sibling_components import prepare_sibling_components
from merlin.llvmlower.sibling_source_route import _text


def joint_source(*, size=2, transpose_weights=False, dense_views=False):
    head = parse_mlir_text(head_source(h=2, m=size, k=size, n=size))
    sibling = multi_source(transpose_weights=transpose_weights, dense_views=dense_views)
    if size != 2:
        sibling = parse_mlir_text(_text(sibling).replace("2x2", f"{size}x{size}"))
    originals = [head.body.block.first_op, sibling.body.block.first_op]
    inputs = [t for f in originals for t in f.function_type.inputs]
    outputs = [t for f in originals for t in f.function_type.outputs]
    block = Block(arg_types=inputs)
    offset, returned = 0, []
    for function in originals:
        arguments = function.body.block.args
        mapping = dict(zip(arguments, block.args[offset:offset + len(arguments)], strict=True))
        offset += len(arguments)
        for operation in function.body.block.ops:
            if type(operation) is func.ReturnOp:
                returned.extend(mapping[v] for v in operation.arguments)
            else:
                block.add_op(operation.clone(value_mapper=mapping))
    block.add_op(func.ReturnOp(*returned))
    module = builtin.ModuleOp([func.FuncOp("forward", (inputs, outputs), Region(block))])
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    for i, root in enumerate(roots):
        root.attributes.update({"prov.region_id": builtin.StringAttr(f"original_{i}"),
                                "prov.source_node_ids": builtin.ArrayAttr([builtin.StringAttr(f"node_{i}")])})
    module.verify()
    return module


def evidence(module):
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    groups = [SimpleNamespace(root=r, members=(r,), index=i, placement="selected") for i, r in enumerate(roots)]
    positions = {id(op): i for i, op in enumerate(module.body.block.first_op.body.block.ops)}
    records = [{"source_op_index": positions[id(g.root)], "source_group": g.index,
                "region_id": g.root.attributes["prov.region_id"].data, "semantic_family": "contraction"}
               for g in groups]
    return groups, records


def test_joint_source_is_disjoint_and_retains_every_output_and_unmatched_product():
    from merlin.runtime.linalg_numpy import evaluate
    from merlin.targetgen.source_head_selection import resolve_request

    module = joint_source()
    before = _text(module)
    groups, records = evidence(module)
    selected, record = resolve_request(module, groups, set(range(7)),
                                      preferences={"shared_input_siblings": True},
                                      eligible_source_regions=records)
    assert selected.shared_input_siblings is True
    assert record["kind"] == "independent_heads_and_shared_input_siblings"
    assert record["composed_source_groups"] == list(range(6))
    assert record["unmatched_source_groups"] == [6]
    assert record["deferred_requests"] == []
    projection = project_heads(module)
    projection.batching.revalidate()
    plan = prepare_sibling_components(projection.module, effects=effects())
    assert len(plan.components) == 2
    values = [(np.arange(np.prod(t.get_shape())).reshape(t.get_shape()) % 9 - 4).astype(np.int8)
              for t in module.body.block.first_op.function_type.inputs]
    calls = {}
    for item in plan.components:
        calls[item.writer.symbol] = lambda args: tuple(
            (args[0].astype(np.int64) @ rhs.astype(np.int64)).astype(np.int32) for rhs in args[1:3])
    actual, expected = evaluate(plan.module, values, calls=calls), evaluate(module, values)
    assert len(actual) == 7 and all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert _text(module) == before


def test_joint_selection_does_not_cover_unadmitted_or_supporting_operations():
    from merlin.targetgen.source_head_selection import resolve_heads, resolve_joint

    module = joint_source()
    groups, records = evidence(module)
    selected, record = resolve_heads(module, groups, set(range(7)), eligible_source_regions=records)
    for invalid in (records[:-2], [*records, records[2]],
                    [{**r, "semantic_family": "activation"} if r["source_group"] == 2 else r for r in records]):
        with pytest.raises(ValueError, match="evidence"):
            resolve_joint(module, groups, set(range(7)), selected, record, eligible_source_regions=invalid)
    with pytest.raises(ValueError, match="disjoint admitted"):
        resolve_joint(module, groups, {0, 1, 2}, selected, record, eligible_source_regions=records)


def test_no_siblings_keeps_checked_head_incumbent_and_reports_deferred_request():
    from test_head_source_route import original

    from merlin.targetgen.source_head_selection import resolve_request

    module, _ = original()
    groups, records = evidence(module)
    selected, record = resolve_request(module, groups, set(range(3)),
                                      preferences={"shared_input_siblings": True},
                                      eligible_source_regions=records)
    assert selected.shared_input_siblings is False
    assert record["deferred_requests"] == ["shared_input_siblings"]
    assert "sibling_composition_unavailable" in record


def test_mixed_source_metrics_and_dispatch_keep_original_per_product_denominator():
    from copy import deepcopy

    from merlin_experiments.phase1.feedback.private_group_provenance import source_group_roster
    from merlin_experiments.phase1.feedback.private_head_provenance import source_metrics

    from merlin.targetgen.private_sibling_dispatch import composition_members

    module = joint_source()
    groups, _ = evidence(module)
    source = source_group_roster(groups, list(range(7)), host_placement="host", module=module)
    components = []
    for i, ids in enumerate(((0, 1), (2, 3), (4, 5))):
        members = [{"source_group": g, "source_region_id": f"original_{g}", "geometry_m_k_n": [2, 2, 2],
                    "source_useful_macs": 8, "raw_output_elements": 4, "raw_output_element_bytes": 4}
                   for g in ids]
        component = {"physical_symbol": f"core{i}", "physical_group": i + 10, "members": members,
                     "covered_source_groups": list(ids), "required_source_groups": list(ids),
                     "raw_output_elements": 8, "raw_output_element_bytes": 4,
                     "original_source_denominator_unchanged": True}
        if i == 0:
            component.update(schema="merlin.original_head_family.v1", geometry_m_k_n=[2, 2, 2],
                             matrix_invocations=2, batch_abi_calls=1)
            for h, member in enumerate(members):
                member["head"] = h
        else:
            component["schema"] = "merlin.original_sibling_source_composition.v1"
        components.append(component)
    rows = [{"symbol": c["physical_symbol"], "group": c["physical_group"]} for c in components]
    rows.append({"symbol": "ordinary", "group": 13, "source_region_id": "original_6",
                 "source_node_ids": ["node_6"], "parallel": [2, 2], "reduction": [2],
                 "dtypes": ["i8", "i8", "i32"]})
    composition = {"schema": "merlin.original_head_source_composition.v1", "components": components,
                   "composed_source_groups": list(range(6)), "ordinary_joined_groups": {13: 6},
                   "required_source_groups": list(range(7)), "original_source_denominator_unchanged": True}
    joined, metrics = source_metrics(rows, source, composition)
    assert joined == {10: (0, 1), 11: (2, 3), 12: (4, 5), 13: (6,)}
    assert metrics[10]["elements"] == 4 and metrics[10]["matrix_invocations"] == 2
    assert metrics[11]["elements"] == 8 and metrics[12]["elements"] == 8
    assert len(composition_members(composition, assigned=rows)) == 6
    for mutation in ("duplicate_source", "duplicate_physical", "lost_head"):
        changed = deepcopy(composition)
        if mutation == "duplicate_source":
            changed["components"][1]["members"][0]["source_group"] = 0
        elif mutation == "duplicate_physical":
            changed["components"][1]["physical_group"] = 10
        else:
            changed["components"][0]["matrix_invocations"] = 1
        with pytest.raises(ValueError):
            source_metrics(rows, source, changed)
        with pytest.raises(ValueError):
            composition_members(changed, assigned=rows)
