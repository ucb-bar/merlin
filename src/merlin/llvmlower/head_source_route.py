"""Original-owner head refolding through the existing per-slice device ABI.

Every unmatched group is routed normally. The default keeps a stock per-slice
descriptor shim. An explicit whole-batch request preserves the existing ranked
batched interface so the target may schedule its complete original products.
Source membership and physical execution remain independent audit obligations.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .borrowed_effects import BorrowedBufferEffects
from .sibling_source_route import _sha, _text


@dataclass(frozen=True)
class HeadBatchSourceRoute:
    source_ir_sha256: str
    effects: BorrowedBufferEffects
    shared_input_siblings: bool = False
    whole_batch: bool = False
    shared_rhs_row_views: bool = False

    def record(self):
        if (type(self.source_ir_sha256) is not str or len(self.source_ir_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
                or type(self.effects) is not BorrowedBufferEffects
                or any(type(v) is not bool for v in
                       (self.shared_input_siblings, self.whole_batch, self.shared_rhs_row_views))):
            raise ValueError("head route needs original source identity and typed synchronous effects")
        self.effects.__post_init__()
        record = {"kind": "independent_heads", **asdict(self)}
        if not self.whole_batch:
            record.pop("whole_batch")
        if not self.shared_rhs_row_views:
            record.pop("shared_rhs_row_views")
        return record


@dataclass(frozen=True)
class HeadProjection:
    source: object
    module: object
    batching: object
    members: tuple


def validate_head_member_ranges(family):
    """Check original products partition every physical matrix's rows exactly.

    Without explicit ranges the original unchanged-M one-product-per-slice
    representation remains mandatory. This grants no source or target custody.
    """
    h = family.get("batch_extent", family.get("matrix_invocations"))
    m, k, n = family["geometry_m_k_n"]
    members = family["members"]
    if (any(type(v) is not int or v <= 0 for v in (h, m, k, n)) or h < 2
            or type(members) not in (list, tuple) or not members
            or [v.get("head") for v in members] != list(range(len(members)))):
        raise ValueError("head source has no complete original member roster")
    keys = {"projected_head", "row_start", "row_count"}
    ranged = any(keys.intersection(v) for v in members)
    if not ranged:
        if len(members) != h or any(v.get("geometry_m_k_n", [m, k, n]) != [m, k, n] for v in members):
            raise ValueError("head source has no unchanged-M invocation roster")
        return len(members)
    rows = {i: [] for i in range(h)}
    for member in members:
        if not keys <= set(member):
            raise ValueError("head source has incomplete original row ranges")
        head, offset, count = (member[key] for key in ("projected_head", "row_start", "row_count"))
        if (any(type(v) is not int for v in (head, offset, count)) or head not in rows
                or offset < 0 or count <= 0 or offset + count > m
                or member.get("geometry_m_k_n") != [count, k, n]):
            raise ValueError("head source changed original row range or matrix geometry")
        rows[head].append((offset, count))
    for intervals in rows.values():
        cursor = 0
        for offset, count in sorted(intervals):
            if offset != cursor:
                raise ValueError("head source original row ranges overlap or leave a gap")
            cursor += count
        if cursor != m:
            raise ValueError("head source original row ranges omit observed output rows")
    return len(members)


@dataclass(frozen=True)
class _HeadSourceComposition:
    steps: tuple
    states: tuple
    require_binding: object

    def revalidate(self):
        from merlin.transforms.dense_row_batching import _state

        if any(_state(module) != state for module, state in self.states):
            raise ValueError("shared RHS head composition source or SSA changed; replan")
        if self.require_binding is not None:
            self.require_binding()
        for step in self.steps:
            step.revalidate()


def _row_head_source(module, capture):
    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.llvmlower.repeated_pointwise import _hoist_stored_repeated_pointwise, hoist_repeated_pointwise
    from merlin.transforms import dense_row_batching_check as check
    from merlin.transforms.dense_row_batching import DenseRowBatchingBudgets, _state, plan_source_dense_row_batching

    # Original coordinates/provenance come from the original complete families,
    # before either scalar preparation or row batching changes their shape.
    from merlin.transforms.head_batching import _groups
    from merlin.transforms.pipeline_state import PipelineContext
    from merlin.transforms.shared_rhs_views import plan_shared_rhs_row_views

    original = {}
    for group in _groups(module, max_heads=256, max_groups=128, max_candidates=16384):
        for head, op in enumerate(group.members):
            region, nodes = op.attributes.get("prov.region_id"), op.attributes.get("prov.source_node_ids")
            if type(region) is not builtin.StringAttr or type(nodes) is not builtin.ArrayAttr:
                raise ValueError("shared RHS head members need original source provenance")
            key = region.data, tuple(v.data for v in nodes if type(v) is builtin.StringAttr)
            if not key[0] or not key[1] or len(key[1]) != len(nodes) or key in original:
                raise ValueError("shared RHS head members have ambiguous original source identities")
            original[key] = {"head": head, "source_region_id": key[0], "source_node_ids": list(key[1]),
                             "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n]}
    require_binding = None
    if capture is None:
        prepared, _report = hoist_repeated_pointwise(module)
    else:
        capture = Path(capture)
        prepared, _report, require_binding = _hoist_stored_repeated_pointwise(
            module, bundle_dir=capture if capture.is_dir() else capture.parent)
    # Construction is proportional to this actual owner, rather than capped by
    # an unrelated tiny component's operation count. Search/member bounds stay
    # fixed and unsupported families remain ordinary unmatched source.
    operation_budget = max(16384, 2 * sum(1 for _ in prepared.walk()))
    views = plan_shared_rhs_row_views(prepared, max_operations=operation_budget)
    if views is None:
        raise ValueError("source has no checked repeated RHS row-view family")
    row_plan = plan_source_dense_row_batching(views.candidate, PipelineContext(), DenseRowBatchingBudgets(
        max_members=64, max_batches=sum(row[1] for row in views._records), max_candidates=16384,
        max_constructed_operations=operation_budget))
    if row_plan is None:
        raise ValueError("source repeated RHS views have no checked complete row batching")
    row_plan.revalidate()
    annotated = row_plan.candidate.clone()
    copied = dict(zip(row_plan.candidate.walk(), annotated.walk(), strict=True))
    mapping = {}
    for op in row_plan.candidate.walk():
        if type(op) is not GenericOp:
            continue
        try:
            batch = check._batch(op, row_plan.budgets.index_bits)
        except check.RowBatchingUnavailable:
            continue
        if batch is None:
            continue
        members = []
        for view, (offset, count) in sorted(batch["views"].items(), key=lambda item: item[1]):
            region, nodes = view.attributes.get("prov.region_id"), view.attributes.get("prov.source_node_ids")
            key = (region.data, tuple(v.data for v in nodes)) if (
                type(region) is builtin.StringAttr and type(nodes) is builtin.ArrayAttr) else None
            if key not in original or original[key]["geometry_m_k_n"] != [count, *batch["shape"][1:]]:
                raise ValueError("row-batched head lost its checked original result ancestry")
            members.append({**original[key], "row_start": offset, "row_count": count})
        region = "shared_rhs_rows_" + _sha(json.dumps(members, sort_keys=True))
        copied[op].attributes.update({"prov.region_id": builtin.StringAttr(region),
            "prov.family": builtin.StringAttr("contraction"), "prov.op": builtin.StringAttr("matmul"),
            "prov.source_node_ids": builtin.ArrayAttr(builtin.StringAttr(node)
                for member in members for node in member["source_node_ids"])})
        mapping[region] = members
    annotated.verify()
    states = tuple((value, _state(value)) for value in (module, prepared, annotated))
    return annotated, mapping, (views, row_plan), states, require_binding


def project_heads(module, *, shared_rhs_row_views=False, capture=None):
    """Refold exact head coordinates; retain all original member identities.

    Fresh aggregate provenance is only a handle for normal group construction.
    It grants no original source membership: the caller independently joins the
    complete original member roster and original supporting-operation demands.
    """
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    from merlin.transforms.head_batching import plan_independent_head_batching
    from merlin.transforms.tensor_views import fold_complete_matrix_slice_concats

    if type(shared_rhs_row_views) is not bool:
        raise ValueError("shared RHS row projection requires an explicit Boolean")
    row_members, steps, states, require_binding = {}, (), (), None
    source = module
    if shared_rhs_row_views:
        source, row_members, steps, states, require_binding = _row_head_source(module, capture)
    plan = plan_independent_head_batching(source)
    if not plan._replacements:
        raise ValueError("source has no complete independent head family")
    projected = plan.candidate.clone()
    copied = dict(zip(plan.candidate.walk(), projected.walk(), strict=True))
    rows = []
    for group, _init, batch, _slices in plan._replacements:
        original = []
        for head, root in enumerate(group.members):
            region = root.attributes.get("prov.region_id")
            nodes = root.attributes.get("prov.source_node_ids")
            if (type(region) is not StringAttr or not region.data or type(nodes) is not ArrayAttr
                    or not nodes or any(type(node) is not StringAttr for node in nodes)):
                raise ValueError("head members need complete original source provenance")
            if region.data in row_members:
                original.extend({**member, "projected_head": head} for member in row_members[region.data])
            else:
                original.append({"head": head, "source_region_id": region.data,
                                 "source_node_ids": [node.data for node in nodes],
                                 "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n]})
        original.sort(key=lambda member: member["head"])
        if len({r["source_region_id"] for r in original}) != len(original):
            raise ValueError("head members have ambiguous original source identities")
        region = "independent_heads_" + _sha(json.dumps(original, sort_keys=True))
        root = copied[batch]
        root.attributes.update({"prov.region_id": StringAttr(region), "prov.family": StringAttr("contraction"),
                                "prov.op": StringAttr("matmul"), "prov.source_node_ids": ArrayAttr(
                                    StringAttr(node) for row in original for node in row["source_node_ids"])})
        rows.append({"projected_region_id": region, "members": original,
                     "batch_extent": group.problem.products,
                     "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n]})
        validate_head_member_ranges(rows[-1])
    fold_complete_matrix_slice_concats(projected)
    projected.verify()
    plan.revalidate()
    if shared_rhs_row_views:
        plan = _HeadSourceComposition((*steps, plan), states, require_binding)
        plan.revalidate()
    return HeadProjection(module, projected, plan, tuple(rows))


def whole_head_entries(rewrite, families, *, enabled):
    """Select the existing batch interface from exact projected source geometry.

    No products are flattened or dropped. This declares a logical input/output
    shape; it grants no target storage, effects, placement or timing capability.
    Unmatched source entries and the default per-product interface stay intact.
    """
    if type(enabled) is not bool:
        raise ValueError("whole head interface selection requires an explicit Boolean")
    if not enabled:
        return rewrite
    rows = {row.get("source_region_id"): row for row in rewrite.to_sidecar()["routed"]}
    entries = dict(rewrite.entries)
    symbols, regions = {}, set()
    for family in families:
        row = rows.get(family["projected_region_id"])
        h = family["batch_extent"]
        m, k, n = family["geometry_m_k_n"]
        if (row is None or row.get("parallel") != [h, m, n] or row.get("reduction") != [k]
                or row.get("dtypes") != ["i8", "i8", "i32"] or type(h) is not int or h < 2
                or family["projected_region_id"] in regions):
            raise ValueError("whole head interface lost complete original batch geometry")
        symbol = row["symbol"]
        if symbol not in entries:
            raise ValueError("whole head interface has no original source entry")
        geometry = (h, m, k, n)
        if symbol in symbols and symbols[symbol] != geometry:
            raise ValueError("whole head shared symbol has incompatible original batch geometry")
        entries[symbol] = {**entries[symbol], "op": "gemv_batched", "B": h, "M": m, "H": k, "K": k, "N": n}
        symbols[symbol] = geometry
        regions.add(family["projected_region_id"])
    return replace(rewrite, entries=entries)


def whole_head_interfaces(rewrite, families, *, binding, enabled):
    """Rebuild only independently matched whole-head interfaces.

    Joint sibling writers already own their typed, possibly multi-output ABI.
    Preserve those interfaces verbatim while selecting the exact ranked batch
    interface for each complete head family. No storage or issue law is added.
    """
    from merlin.targetgen import corpus_spec as CS

    selected = whole_head_entries(rewrite, families, enabled=enabled)
    if not enabled:
        return selected
    interfaces = dict(selected.expected_interfaces)
    by_region = {row.get("source_region_id"): row for row in selected.to_sidecar()["routed"]}
    rebuilt = set()
    for family in families:
        symbol = by_region[family["projected_region_id"]]["symbol"]
        if symbol in rebuilt:
            continue
        _capsule, interface = CS.build(selected.entries[symbol], binding)
        interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
        rebuilt.add(symbol)
    return replace(selected, expected_interfaces=interfaces)


@dataclass(frozen=True)
class RoutedHeads:
    source: object
    module: object
    core: object
    interface: str
    rewrite: object
    projection: HeadProjection
    receipt: dict
    integrity_sha256: str
    component_host: object = None


def _snapshot(host):
    from .sibling_component_route import _snapshot as component_snapshot

    # Source, projection, core and component owners commonly alias the same
    # large modules. Serialize each distinct graph once in this invocation;
    # a later check always reads fresh IR rather than a cross-call hash cache.
    digests = {}

    def digest(module):
        key = id(module)
        if key not in digests:
            digests[key] = _sha(_text(module))
        return digests[key]

    return {"source": digest(host.source), "module": digest(host.module),
            "core": digest(host.core), "interface": _sha(host.interface),
            "rewrite": host.rewrite.to_sidecar(), "receipt": host.receipt,
            "projection_source": digest(host.projection.source),
            "projection": digest(host.projection.module), "members": host.projection.members,
            "component_host": component_snapshot(
                host.component_host, source_sha256=digest(host.component_host.source),
                host_sha256=digest(host.component_host.module)) if host.component_host is not None else None}


def component_selection(selected):
    from .sibling_source_route import SiblingSourceRoute

    selected.record()
    if not selected.shared_input_siblings:
        raise ValueError("head selection did not request checked sibling components")
    return SiblingSourceRoute(selected.source_ir_sha256, selected.effects, partial=True)


def require_host(host, selected):
    if type(host) is not RoutedHeads or type(selected) is not HeadBatchSourceRoute:
        raise ValueError("head source, member projection, host or device custody changed")
    snapshot = _snapshot(host)
    if (snapshot["source"] != selected.source_ir_sha256
            or _sha(json.dumps(snapshot, sort_keys=True, allow_nan=False)) != host.integrity_sha256):
        raise ValueError("head source, member projection, host or device custody changed")
    host.projection.batching.revalidate()
    if host.component_host is not None:
        from .sibling_component_route import require_components

        require_components(host.component_host, component_selection(selected))
        if (host.component_host.rewrite != host.rewrite or
                (host.component_host.module is not host.module
                 and _text(host.component_host.module) != _text(host.module))):
            raise ValueError("head and sibling routes lost their common actual host or rewrite")
    elif selected.shared_input_siblings:
        raise ValueError("joint head route lost its checked sibling components")
    host.module.verify()
    selected.record()


def prepare_host(device, selected, source, directory):
    from merlin.compile.mesh import _mesh_tile_binding
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen import corpus_spec as CS

    from .device_offload import rewrite_groups_to_device
    from .exact_offload import _package_sha256

    selected.record()
    actual = parse_mlir_file(source)
    original, preparation = actual, None
    if _sha(_text(actual)) != selected.source_ir_sha256:
        from .prepacked_device_capture import original_sibling_source

        original, preparation = original_sibling_source(device, selected, actual_source=source, work=directory.parent)
    projection = project_heads(actual, shared_rhs_row_views=selected.shared_rhs_row_views, capture=device.capture)
    module = projection.module.clone()
    # Same normal source/group owner for both matched and unmatched groups.
    # Only the checked source projection changes; no contraction falls back to
    # scalar host execution to make this route applicable.
    component_host = None
    if selected.shared_input_siblings:
        from .sibling_component_route import build_components

        component_host = build_components(device, component_selection(selected), projection.module,
                                          original=original, preparation=preparation)
        module, rewrite = component_host.module, component_host.rewrite
    else:
        rewrite = rewrite_groups_to_device(
            module, device.device, select=device.select, capture=device.capture, model=device.model,
            require_complete_groups=getattr(device, "require_complete_groups", False),
        )
    by_region = {row.get("source_region_id"): row for row in rewrite.to_sidecar()["routed"]}
    matched = []
    for family in projection.members:
        row = by_region.get(family["projected_region_id"])
        m, k, n = family["geometry_m_k_n"]
        if (row is None or row.get("parallel") != [family["batch_extent"], m, n]
                or row.get("reduction") != [k] or row.get("dtypes") != ["i8", "i8", "i32"]):
            raise ValueError("normal group route did not retain complete independent head geometry")
        symbol = row["symbol"]
        call = rewrite.call_buffers[symbol]
        if ([r["shape"] for r in call] != [[family["batch_extent"], m, k],
                                          [family["batch_extent"], k, n],
                                          [family["batch_extent"], m, n]]
                or [r["dtype"] for r in call] != ["i8", "i8", "i32"]):
            raise ValueError("normal head call lost original complete ranked operands/results")
        matched.append({**family, "physical_symbol": symbol, "physical_group": row["group"]})
    binding = _mesh_tile_binding(
        device.device, device.operand_dtype, device.accum_dtype, numeric_policy=device.numeric_policy)
    rewrite = whole_head_interfaces(rewrite, projection.members, binding=binding, enabled=selected.whole_batch)
    interfaces = {}
    for symbol, entry in rewrite.entries.items():
        if component_host is not None:
            interfaces[symbol] = rewrite.expected_interfaces[symbol]
        else:
            _capsule, interface = CS.build(entry, binding)
            interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
    rewrite = replace(rewrite, model_sha256=selected.source_ir_sha256,
                      package_sha256=_package_sha256(Path(device.package_dir)),
                      expected_interfaces=interfaces)
    if component_host is not None and selected.whole_batch:
        # The preceding exact head/member/interface checks justify this new
        # aggregate rewrite. Every individual sibling writer/core and its
        # sealed interface stays unchanged and is revalidated below.
        component_host = replace(component_host, rewrite=rewrite, integrity_sha256="")
    receipt = {"schema": "merlin.original_head_source_projection.v1",
               "source_ir_sha256": selected.source_ir_sha256, "prepared_source_ir_sha256": _sha(_text(actual)),
               "projected_source_ir_sha256": _sha(_text(projection.module)), "head_families": matched,
               "weight_preparation": preparation, "unmatched_source_routed_normally": True,
               "original_readouts_preserved": True, "matrix_products": sum(len(r["members"]) for r in matched),
               "batch_abi_calls": len(matched), "accelerator_execution_count_reduced": False,
               "shared_input_components": len(component_host.components) if component_host is not None else 0,
               "effect_or_placement_grant": False}
    if selected.whole_batch:
        receipt.update(whole_batch=True, kernel_invocations=len(matched),
                       bridge_kind="whole_logical_batch",
                       mathematical_products_reduced=False)
    interface = next(iter(interfaces.values()))["mlir"]
    host = RoutedHeads(original, module, projection.module, interface, rewrite, projection, receipt, "", component_host)
    snapshot = _snapshot(host)
    if component_host is not None and selected.whole_batch:
        # The fresh aggregate snapshot already contains the exact common
        # graphs and every original component receipt. Reuse those hashes
        # within this invocation, then independently revalidate below.
        component_host = replace(component_host, integrity_sha256=_sha(json.dumps(
            snapshot["component_host"], sort_keys=True, allow_nan=False)))
        host = replace(host, component_host=component_host)
    host = replace(host, integrity_sha256=_sha(json.dumps(snapshot, sort_keys=True, allow_nan=False)))
    require_host(host, selected)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "host.mlir"
    path.write_text(_text(module) + "\n")
    (directory / "core.mlir").write_text(_text(projection.module) + "\n")
    (directory / "interface.mlir").write_text(interface)
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return path, host
