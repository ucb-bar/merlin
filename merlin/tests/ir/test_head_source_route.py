"""Original source membership survives a stock per-slice device projection."""

from types import SimpleNamespace

import pytest
from test_head_batching import source
from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import GenericOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.head_source_route import project_heads
from merlin.llvmlower.sibling_source_route import _text


def original():
    module = parse_mlir_text(source(h=3))
    roots = tuple(op for op in module.walk() if type(op) is GenericOp)
    for index, op in enumerate(roots):
        op.attributes.update({"prov.region_id": builtin.StringAttr(f"original_{index}"),
                              "prov.source_node_ids": builtin.ArrayAttr([builtin.StringAttr(f"node_{index}")])})
    return module, roots


def test_snapshot_serializes_shared_graphs_once_but_reads_fresh_next_time(monkeypatch):
    from merlin.llvmlower import head_source_route as route
    from merlin.llvmlower.sibling_component_route import RoutedSiblingComponents

    source, _ = original()
    module = source.clone()
    rewrite = SimpleNamespace(to_sidecar=lambda: {"routed": []})
    component = RoutedSiblingComponents(source, module, (), rewrite, {}, "")
    host = SimpleNamespace(source=source, module=module, core=module, interface="interface",
                           rewrite=rewrite, receipt={}, component_host=component,
                           projection=SimpleNamespace(source=source, module=module, members=()))
    printed = []
    original_text = route._text

    def counted(value):
        printed.append(value)
        return original_text(value)

    monkeypatch.setattr(route, "_text", counted)
    before = route._snapshot(host)
    assert printed == [source, module]
    source.attributes["observer"] = builtin.StringAttr("new")
    after = route._snapshot(host)
    assert printed == [source, module, source, module]
    assert before["source"] != after["source"]
    assert after["projection_source"] == after["component_host"]["source"] == after["source"]


def test_exact_source_projection_has_every_original_member_and_readout():
    module, roots = original()
    before = _text(module)
    projection = project_heads(module)
    assert _text(module) == before
    assert len(projection.members) == 1
    family = projection.members[0]
    assert family["batch_extent"] == 3
    assert [member["head"] for member in family["members"]] == [0, 1, 2]
    assert [member["source_region_id"] for member in family["members"]] == [
        root.attributes["prov.region_id"].data for root in roots
    ]
    projection.batching.revalidate()
    assert len(projection.module.body.block.first_op.function_type.outputs) == 3


def test_missing_or_ambiguous_original_provenance_is_not_new_membership():
    module, roots = original()
    roots[1].attributes["prov.region_id"] = roots[0].attributes["prov.region_id"]
    with pytest.raises(ValueError, match="ambiguous"):
        project_heads(module)


def test_owner_request_joins_original_roots_and_refuses_supporting_work():
    from merlin.targetgen.source_head_selection import resolve_heads

    module, roots = original()
    groups = [SimpleNamespace(root=root, index=i, placement="selected") for i, root in enumerate(roots)]
    positions = {id(op): i for i, op in enumerate(module.body.block.first_op.body.block.ops)}
    evidence = [{"source_op_index": positions[id(g.root)], "source_group": g.index,
                 "region_id": g.root.attributes["prov.region_id"].data, "semantic_family": "contraction"}
                for g in groups]
    selected, record = resolve_heads(module, groups, {0, 1, 2}, eligible_source_regions=evidence)
    assert selected.record()["kind"] == "independent_heads"
    assert record["matrix_products"] == 3 and record["batch_abi_calls"] == 1
    assert record["accelerator_execution_count_reduced"] is False
    with pytest.raises(ValueError, match="supporting"):
        resolve_heads(module, groups, {0, 1, 2}, eligible_source_regions=[
            {"source_op_index": 1000, "source_group": 0, "region_id": "original_0",
             "semantic_family": "activation"},
        ])
    with pytest.raises(ValueError, match="outside independently admitted"):
        resolve_heads(module, groups, {0, 1}, eligible_source_regions=evidence)
    for absent in (None, [], evidence[:-1], [*evidence, evidence[0]]):
        with pytest.raises(ValueError, match="evidence"):
            resolve_heads(module, groups, {0, 1, 2}, eligible_source_regions=absent)
    from merlin.targetgen.source_head_selection import resolve_request

    _, requested = resolve_request(module, groups, {0, 1, 2},
                                   preferences={"shared_input_siblings": True},
                                   eligible_source_regions=evidence)
    assert requested["deferred_requests"] == ["shared_input_siblings"]
    assert requested["composition_order"] == "independent_heads_first_siblings_fallback_only"


def test_per_invocation_budget_preserves_every_original_product():
    from copy import deepcopy

    from merlin_experiments.phase1.feedback.private_head_provenance import source_metrics

    members = [{"head": i, "source_group": i} for i in range(3)]
    component = {"physical_symbol": "batch", "physical_group": 7,
                 "matrix_invocations": 3, "members": members,
                 "geometry_m_k_n": [2, 7, 5], "raw_output_elements": 30}
    composition = {"schema": "merlin.original_head_source_composition.v1",
                   "original_source_denominator_unchanged": True,
                   "composed_source_groups": [0, 1, 2], "components": [component],
                   "ordinary_joined_groups": {}}
    source_record = {"eligible_groups": [0, 1, 2], "eligible_group_metrics": {},
                     "eligible_group_provenance": [],
                     "source_group_contractions": {
                         i: {"parallel": [2, 5], "reduction": [7], "dtypes": ["i8", "i8", "i32"]}
                         for i in range(3)}}
    assigned = [{"symbol": "batch", "group": 7}]
    joined, metrics = source_metrics(assigned, source_record, composition)
    assert joined == {7: (0, 1, 2)}
    assert metrics[7] == {"elements": 10, "element_bytes": 4,
                          "matrix_invocations": 3, "source_total_elements": 30}
    for change in ({"matrix_invocations": 2}, {"members": members[:-1]},
                   {"members": [members[0], members[1], {"head": 2, "source_group": 1}]}):
        altered = deepcopy(composition)
        altered["components"][0].update(change)
        with pytest.raises(ValueError, match="roster|duplicate"):
            source_metrics(assigned, source_record, altered)


def test_whole_head_interface_uses_existing_ranked_batch_and_one_stock_kernel_call():
    from dataclasses import dataclass

    from merlin.llvmlower.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.llvmlower.head_source_route import whole_head_entries
    from merlin.targetgen.corpus_spec import CorpusBinding, build

    module, _ = original()
    family = project_heads(module).members[0]
    h = family["batch_extent"]
    m, k, n = family["geometry_m_k_n"]

    @dataclass(frozen=True)
    class Rewrite:
        entries: dict
        routed: list

        def to_sidecar(self):
            return {"routed": self.routed}

    entry = {"name": "heads", "op": "matmul", "kind": "op", "source_role": "compiler_generated",
             "source_reference": "independently projected source heads", "M": m, "K": k, "N": n}
    rewrite = Rewrite({"heads": entry, "unmatched": {"op": "unchanged"}}, [
        {"source_region_id": family["projected_region_id"], "symbol": "heads", "parallel": [h, m, n],
         "reduction": [k], "dtypes": ["i8", "i8", "i32"]}])
    assert whole_head_entries(rewrite, (family,), enabled=False) is rewrite
    selected = whole_head_entries(rewrite, (family,), enabled=True)
    assert rewrite.entries["heads"] == entry
    assert selected.entries["unmatched"] is rewrite.entries["unmatched"]
    binding = CorpusBinding("neutral", 2, "int8", "i32", True, [], "exact_int")
    _, interface = build(selected.entries["heads"], binding)
    buffers, reason = logical_buffers(interface)
    assert reason == ""
    assert [list(b.shape) for b in buffers] == [[h, m, k], [h, k, n], [h, m, n]]
    call = [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers]
    unit = emit_logical_translation_unit("neutral", {"heads": (buffers, call)},
                                         kernel_symbol_for=lambda _: "whole_kernel")
    assert not unit.skipped and unit.symbols == ("heads",)
    assert unit.text.count("whole_kernel(") == 2  # one declaration and one call, no head-slice loop
    assert "for (long slice" not in unit.text


def test_whole_head_object_budget_and_membership_remain_complete():
    from merlin_experiments.phase1.feedback.private_head_provenance import source_metrics

    members = [{"head": i, "source_group": i} for i in range(3)]
    component = {"physical_symbol": "batch", "physical_group": 7, "matrix_invocations": 3,
                 "members": members, "geometry_m_k_n": [2, 7, 5], "raw_output_elements": 30,
                 "whole_batch": True, "kernel_invocations": 1, "mathematical_products": 3,
                 "bridge_kind": "whole_logical_batch"}
    composition = {"schema": "merlin.original_head_source_composition.v1",
                   "original_source_denominator_unchanged": True,
                   "composed_source_groups": [0, 1, 2], "components": [component], "ordinary_joined_groups": {}}
    source_record = {"eligible_groups": [0, 1, 2], "eligible_group_metrics": {}, "eligible_group_provenance": [],
                     "source_group_contractions": {
                         i: {"parallel": [2, 5], "reduction": [7], "dtypes": ["i8", "i8", "i32"]}
                         for i in range(3)}}
    joined, metrics = source_metrics([{"symbol": "batch", "group": 7}], source_record, composition)
    assert joined == {7: (0, 1, 2)}
    assert metrics[7] == {"elements": 30, "element_bytes": 4,
                          "matrix_invocations": 1, "source_total_elements": 30}
    component["mathematical_products"] = 1
    with pytest.raises(ValueError, match="complete original product budget"):
        source_metrics([{"symbol": "batch", "group": 7}], source_record, composition)


def test_whole_head_request_is_explicit_and_does_not_grant_joint_sibling_route():
    from merlin.targetgen.source_head_selection import resolve_request

    module, roots = original()
    groups = [SimpleNamespace(root=root, index=i, placement="selected") for i, root in enumerate(roots)]
    positions = {id(op): i for i, op in enumerate(module.body.block.first_op.body.block.ops)}
    evidence = [{"source_op_index": positions[id(g.root)], "source_group": g.index,
                 "region_id": g.root.attributes["prov.region_id"].data, "semantic_family": "contraction"}
                for g in groups]
    selected, record = resolve_request(module, groups, {0, 1, 2},
                                      preferences={"shared_input_siblings": True, "whole_head_batches": True},
                                      eligible_source_regions=evidence)
    assert selected.whole_batch and not selected.shared_input_siblings
    assert selected.record()["whole_batch"] is True
    assert record["matrix_products"] == 3 and record["kernel_invocations"] == 1
    assert record["deferred_requests"] == ["shared_input_siblings"]
    assert record["effect_or_placement_grant"] is False


def test_repeated_source_families_share_only_compatible_whole_kernel_geometry():
    from dataclasses import dataclass

    from merlin.llvmlower.head_source_route import whole_head_entries

    @dataclass(frozen=True)
    class Rewrite:
        entries: dict
        routed: list

        def to_sidecar(self):
            return {"routed": self.routed}

    families = [{"projected_region_id": name, "batch_extent": 3, "geometry_m_k_n": [2, 7, 5]}
                for name in ("first", "second")]
    rows = [{"source_region_id": family["projected_region_id"], "symbol": "shared",
             "parallel": [3, 2, 5], "reduction": [7], "dtypes": ["i8", "i8", "i32"]}
            for family in families]
    rewrite = Rewrite({"shared": {"op": "matmul", "M": 2, "K": 7, "N": 5}}, rows)
    selected = whole_head_entries(rewrite, families, enabled=True)
    assert selected.entries["shared"]["B"] == 3
    assert selected.routed == rows and len(selected.routed) == 2
    assert rewrite.entries["shared"]["op"] == "matmul"
    with pytest.raises(ValueError, match="complete original"):
        whole_head_entries(rewrite, [families[0], families[0]], enabled=True)
    incompatible = {**families[1], "geometry_m_k_n": [2, 7, 6]}
    incompatible_rows = [rows[0], {**rows[1], "parallel": [3, 2, 6]}]
    with pytest.raises(ValueError, match="incompatible original"):
        whole_head_entries(Rewrite(rewrite.entries, incompatible_rows),
                           [families[0], incompatible], enabled=True)


def test_repeated_kernel_keeps_every_distinct_family_source_budget():
    from copy import deepcopy

    from merlin_experiments.phase1.feedback.private_head_provenance import source_metrics

    components = [{"physical_symbol": "shared", "physical_group": group, "matrix_invocations": 3,
                   "members": [{"head": h, "source_group": start+h} for h in range(3)],
                   "geometry_m_k_n": [2, 7, 5], "raw_output_elements": 30, "whole_batch": True,
                   "kernel_invocations": 1, "mathematical_products": 3,
                   "bridge_kind": "whole_logical_batch"} for group, start in ((7, 0), (8, 3))]
    composition = {"schema": "merlin.original_head_source_composition.v1",
                   "original_source_denominator_unchanged": True, "composed_source_groups": list(range(6)),
                   "components": components, "ordinary_joined_groups": {}}
    source_record = {"eligible_groups": list(range(6)), "eligible_group_metrics": {},
                     "eligible_group_provenance": [], "source_group_contractions": {
                         i: {"parallel": [2, 5], "reduction": [7], "dtypes": ["i8", "i8", "i32"]}
                         for i in range(6)}}
    assigned = [{"symbol": "shared", "group": group} for group in (7, 8)]
    joined, metrics = source_metrics(assigned, source_record, composition)
    assert joined == {7: (0, 1, 2), 8: (3, 4, 5)}
    assert metrics[7] == metrics[8] == {"elements": 30, "element_bytes": 4,
                                      "matrix_invocations": 1, "source_total_elements": 30}
    with pytest.raises(ValueError, match="distinct matching"):
        source_metrics(assigned[:-1], source_record, composition)
    altered = deepcopy(composition)
    altered["components"][1]["members"][0]["source_group"] = 0
    with pytest.raises(ValueError, match="omit or duplicate"):
        source_metrics(assigned, source_record, altered)
