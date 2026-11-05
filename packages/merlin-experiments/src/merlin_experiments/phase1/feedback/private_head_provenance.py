"""Reopen head membership and stock slice multiplicity from original source."""

import json
from pathlib import Path


def join_heads(emitted, source, *, module, target, capture, preparation_receipt=None,
               shared_input_siblings=False, effects=None, whole_batch=False, shared_rhs_row_views=False):
    from merlin.compile.complete_group_routing import select_complete_group
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.llvmlower.device_offload import rewrite_groups_to_device
    from merlin.llvmlower.head_source_route import project_heads, validate_head_member_ranges, whole_head_entries
    from merlin.llvmlower.sibling_source_route import _sha, _text
    from merlin.targetgen import corpus_spec as CS
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    from .private_component_provenance import source_subset
    from .private_group_provenance import join_routed_source_groups

    before = _text(module)
    if type(whole_batch) is not bool:
        raise ValueError("whole head audit requires an explicit exact batch-interface request")
    capture = Path(capture)
    if capture.is_dir():
        capture = capture / "model.mlir"
    projected_source = module
    if preparation_receipt is not None:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.llvmlower.weight_prepack import revalidate_build_preparation

        path = Path(preparation_receipt)
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("head preparation requires original direct custody")
        revalidate_build_preparation(path)
        record = json.loads(path.read_text())
        if _text(parse_mlir_file(Path(record["source_bundle"]) / "model.mlir")) != before:
            raise ValueError("head preparation lost original source ancestry")
        capture = Path(record["prepared_bundle"]) / "model.mlir"
        projected_source = parse_mlir_file(capture)
    projection = project_heads(projected_source, shared_rhs_row_views=shared_rhs_row_views, capture=capture)
    actual = projection.module.clone()
    if shared_input_siblings:
        from merlin.llvmlower.sibling_components import prepare_sibling_components

        siblings = prepare_sibling_components(
            projection.module, effects=effects, output_path=OF.declared_path(target), values=OF.Values(capture)
        )
        if not siblings.components:
            raise ValueError("joint head selection lost every checked sibling component")
        actual = siblings.module.clone()
    rewrite = rewrite_groups_to_device(actual, target, select=select_complete_group,
                                      capture=capture, require_complete_groups=True)
    rewrite = whole_head_entries(rewrite, projection.members, enabled=whole_batch)
    expected = rewrite.to_sidecar()
    rows = emitted.get("routed")
    symbols = {r["symbol"] for r in expected["routed"]}
    if (emitted.get("device") != target or emitted.get("model_sha256") != _sha(before)
            or not isinstance(rows, list)
            or [r for r in rows if r["symbol"] in symbols] != expected["routed"]
            or (not shared_input_siblings and rows != expected["routed"])):
        raise ValueError("head source changed its independently rederived ordinary routed roster")
    for key in ("signatures", "entries", "programs", "arg_access", "call_buffers"):
        actual_map = emitted.get(key)
        if (not isinstance(actual_map, dict) or set(actual_map) != {r["symbol"] for r in rows}
                or json.dumps({k: actual_map[k] for k in symbols}, sort_keys=True)
                != json.dumps(expected.get(key), sort_keys=True)):
            raise ValueError("head source changed its independently rederived " + key)
    binding = _mesh_tile_binding(target, "int8", "i32")
    interfaces = {}
    for symbol, entry in rewrite.entries.items():
        _, interface = CS.build(entry, binding)
        interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
    if (set(emitted.get("expected_interfaces", {})) != {r["symbol"] for r in rows}
            or {k: emitted.get("expected_interfaces", {}).get(k) for k in symbols} != interfaces):
        raise ValueError("head matrix interface differs from original normal group construction")
    roster = [*source["eligible_group_provenance"], *source.get("optional_group_provenance", [])]
    original = {(row["source_region_id"], tuple(row["source_node_ids"])): row for row in roster}
    if len(original) != len(roster):
        raise ValueError("head source has ambiguous original provenance")
    by_region = {row.get("source_region_id"): row for row in rows}
    components, covered = [], []
    for family in projection.members:
        row = by_region.get(family["projected_region_id"])
        m, k, n = family["geometry_m_k_n"]
        h = family["batch_extent"]
        if (row is None or row.get("parallel") != [h, m, n] or row.get("reduction") != [k]
                or row.get("dtypes") != ["i8", "i8", "i32"]):
            raise ValueError("head source lost its complete physical slice geometry")
        members = []
        products = validate_head_member_ranges(family)
        for member in family["members"]:
            identity = member["source_region_id"], tuple(member["source_node_ids"])
            owned = original.get(identity)
            if owned is None:
                raise ValueError("head source contains an unadmitted or replaced original member")
            index = owned["source_group"]
            original_m, original_k, original_n = member["geometry_m_k_n"]
            if source.get("source_group_contractions", {}).get(index) != {
                "parallel": [original_m, original_n], "reduction": [original_k], "dtypes": ["i8", "i8", "i32"],
            }:
                raise ValueError("head source changed a member's original typed matrix geometry")
            members.append({**member, "source_group": index, "raw_output_elements": original_m*original_n,
                            "raw_output_element_bytes": 4, "source_useful_macs": original_m*original_k*original_n})
        ids = [member["source_group"] for member in members]
        if set(ids).intersection(covered) or len(set(ids)) != len(ids) or len(ids) != products:
            raise ValueError("head source omits or repeats a matrix invocation")
        covered.extend(ids)
        components.append({"schema": "merlin.original_head_family.v1", "physical_symbol": row["symbol"],
                           "physical_group": row["group"], "geometry_m_k_n": [m, k, n],
                           "matrix_invocations": h, "batch_abi_calls": 1, "members": members,
                           "source_ir_sha256": _sha(before),
                           "interface_sha256": interfaces[row["symbol"]]["sha256"],
                           "raw_output_elements_per_invocation": m*n,
                           "raw_output_elements": h*m*n, "raw_output_element_bytes": 4,
                           "covered_source_groups": ids,
                           "required_source_groups": [g for g in source["eligible_groups"] if g in ids],
                           "original_source_denominator_unchanged": True})
        if whole_batch:
            components[-1].update(whole_batch=True, kernel_invocations=1,
                                  bridge_kind="whole_logical_batch", mathematical_products=products)
    if shared_input_siblings:
        from .private_component_provenance import join_components

        joined_siblings = join_components(
            emitted, source, module=module, target=target, effects=effects,
            capture=capture, preparation_receipt=preparation_receipt, checked_projection=projection.module,
            other_components=components,
        )
        components.extend(joined_siblings["components"])
        covered.extend(joined_siblings["composed_source_groups"])
        ordinary = joined_siblings["ordinary_joined_groups"]
    else:
        composed = {c["physical_group"] for c in components}
        ordinary = join_routed_source_groups([r for r in rows if r["group"] not in composed],
                                            source_subset(source, set(covered)))
    if not set(source["eligible_groups"]) <= set(covered) | set(ordinary.values()):
        raise ValueError("head source omitted unmatched required accelerator work")
    if _text(module) != before:
        raise ValueError("head source audit modified original source")
    return {"schema": "merlin.original_head_source_composition.v1", "source_ir_sha256": _sha(before),
            "components": components, "composed_source_groups": covered, "covered_source_groups": covered,
            "ordinary_joined_groups": ordinary, "required_source_groups": source["eligible_groups"],
            "original_source_denominator_unchanged": True, "unmatched_source_routed_normally": True,
            "matrix_products": sum(len(c["members"]) if c.get("schema") == "merlin.original_head_family.v1"
                                   else c.get("matrix_invocations", 0) for c in components),
            "batch_abi_calls": sum(c.get("batch_abi_calls", 0) for c in components),
            "shared_input_components": sum(c["schema"] == "merlin.original_sibling_source_composition.v1"
                                           for c in components),
            "accelerator_execution_count_reduced": False}


def source_metrics(assigned, source, composition):
    """Device-object budgets are per invocation; source totals stay separate."""
    from merlin.llvmlower.head_source_route import validate_head_member_ranges

    from .private_component_provenance import source_subset
    from .private_group_provenance import join_routed_source_groups

    if (composition.get("schema") != "merlin.original_head_source_composition.v1"
            or composition.get("original_source_denominator_unchanged") is not True):
        raise ValueError("head metrics require the independently rederived original source projection")
    covered = composition["composed_source_groups"]
    result, metrics = {}, {}
    by_group = {r["group"]: r for r in assigned}
    if len(by_group) != len(assigned):
        raise ValueError("head source has ambiguous physical call groups")
    components = composition["components"]
    for component in components:
        row = by_group.get(component["physical_group"])
        if row is None or row["symbol"] != component["physical_symbol"] or row["group"] in result:
            raise ValueError("head source has no distinct matching physical call group")
        if component.get("schema") == "merlin.original_sibling_source_composition.v1":
            from .private_device_audit import _composed_source_metrics

            ids = set(component["covered_source_groups"])
            subsource = source_subset(source, set(source["eligible_groups"] + source.get("optional_groups", [])) - ids)
            subgroups, submetrics = _composed_source_metrics([row], subsource, component)
            result.update(subgroups)
            metrics.update(submetrics)
            continue
        h = component["matrix_invocations"]
        members = component["members"]
        m, k, n = component["geometry_m_k_n"]
        products = validate_head_member_ranges(component)
        if (row is None or row["group"] != component["physical_group"]
                or component["raw_output_elements"] != h*m*n):
            raise ValueError("head source has no complete independently checked invocation roster")
        for member in members:
            original_m, original_k, original_n = member.get("geometry_m_k_n", [m, k, n])
            if source["source_group_contractions"].get(member["source_group"]) != {
                "parallel": [original_m, original_n], "reduction": [original_k], "dtypes": ["i8", "i8", "i32"],
            }:
                raise ValueError("head source per-invocation budget changed")
        ids = tuple(r["source_group"] for r in members)
        result[row["group"]] = ids
        whole = component.get("whole_batch", False)
        if type(whole) is not bool or whole and (
                component.get("kernel_invocations") != 1 or component.get("mathematical_products") != products
                or component.get("bridge_kind") != "whole_logical_batch"):
            raise ValueError("whole head object lost its complete original product budget")
        metrics[row["group"]] = {"elements": h*m*n if whole else m*n, "element_bytes": 4,
                                 "matrix_invocations": 1 if whole else h, "source_total_elements": h*m*n}
    ordinary_rows = [r for r in assigned if r["group"] not in {c["physical_group"] for c in components}]
    ordinary = join_routed_source_groups(ordinary_rows, source_subset(source, set(covered)))
    if ordinary != composition["ordinary_joined_groups"]:
        raise ValueError("head source unmatched group join changed")
    original_metrics = {**source.get("optional_group_metrics", {}), **source["eligible_group_metrics"]}
    for physical, original in ordinary.items():
        result[physical], metrics[physical] = (original,), original_metrics[original]
    ids = [index for values in result.values() for index in values]
    if (len(ids) != len(set(ids)) or set(result) != {r["group"] for r in assigned}
            or not set(source["eligible_groups"]) <= set(ids)):
        raise ValueError("head source metrics omit or duplicate required work")
    return result, metrics
