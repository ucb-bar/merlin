"""Original-owner head refolding through the existing per-slice device ABI.

Every unmatched group is routed normally. The default keeps a stock per-slice
descriptor shim. An explicit whole-batch request preserves the existing ranked
batched interface so the target may schedule its complete original products.
Source membership and physical execution remain independent audit obligations.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import entry_points
from pathlib import Path

from merlin.common.digest import hash_tree, is_sha256, sha256_bytes, sha256_file, sha256_text
from merlin.compiler.lower.device.device_build import DeviceRouting
from merlin.compiler.lower.device.sibling_source_route import _sha, _text
from merlin.compiler.lower.host.borrowed_effects import BorrowedBufferEffects
from merlin.compiler.lower.options.weight_prepack import PrepackRefused, revalidate_build_preparation
from merlin.development.targetgen.execution.runner.model_kernel_outline import SCHEMA as OUTLINE_SCHEMA
from merlin.development.targetgen.execution.runner.model_kernel_outline import outline_integer_matmuls

# ---- head_source_route -------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadBatchSourceRoute:
    source_ir_sha256: str
    effects: BorrowedBufferEffects
    shared_input_siblings: bool = False
    whole_batch: bool = False
    shared_rhs_row_views: bool = False

    def record(self):
        if (
            type(self.source_ir_sha256) is not str
            or len(self.source_ir_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_ir_sha256)
            or type(self.effects) is not BorrowedBufferEffects
            or any(
                type(v) is not bool for v in (self.shared_input_siblings, self.whole_batch, self.shared_rhs_row_views)
            )
        ):
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
    if (
        any(type(v) is not int or v <= 0 for v in (h, m, k, n))
        or h < 2
        or type(members) not in (list, tuple)
        or not members
        or [v.get("head") for v in members] != list(range(len(members)))
    ):
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
        if (
            any(type(v) is not int for v in (head, offset, count))
            or head not in rows
            or offset < 0
            or count <= 0
            or offset + count > m
            or member.get("geometry_m_k_n") != [count, k, n]
        ):
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
        from merlin.compiler.transforms.dense.row_batching import _state

        if any(_state(module) != state for module, state in self.states):
            raise ValueError("shared RHS head composition source or SSA changed; replan")
        if self.require_binding is not None:
            self.require_binding()
        for step in self.steps:
            step.revalidate()


def _row_head_source(module, capture):
    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.compiler.lower.source.repeated_pointwise import (
        _hoist_stored_repeated_pointwise,
        hoist_repeated_pointwise,
    )
    from merlin.compiler.transforms.dense import row_batching_check as check

    # Original coordinates/provenance come from the original complete families,
    # before either scalar preparation or row batching changes their shape.
    from merlin.compiler.transforms.dense.head_batching import _groups
    from merlin.compiler.transforms.dense.row_batching import (
        DenseRowBatchingBudgets,
        _state,
        plan_source_dense_row_batching,
    )
    from merlin.compiler.transforms.dense.shared_rhs_views import plan_shared_rhs_row_views
    from merlin.compiler.transforms.pipeline_state import PipelineContext

    original = {}
    for group in _groups(module, max_heads=256, max_groups=128, max_candidates=16384):
        for head, op in enumerate(group.members):
            region, nodes = op.attributes.get("prov.region_id"), op.attributes.get("prov.source_node_ids")
            if type(region) is not builtin.StringAttr or type(nodes) is not builtin.ArrayAttr:
                raise ValueError("shared RHS head members need original source provenance")
            key = region.data, tuple(v.data for v in nodes if type(v) is builtin.StringAttr)
            if not key[0] or not key[1] or len(key[1]) != len(nodes) or key in original:
                raise ValueError("shared RHS head members have ambiguous original source identities")
            original[key] = {
                "head": head,
                "source_region_id": key[0],
                "source_node_ids": list(key[1]),
                "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n],
            }
    require_binding = None
    if capture is None:
        prepared, _report = hoist_repeated_pointwise(module)
    else:
        capture = Path(capture)
        prepared, _report, require_binding = _hoist_stored_repeated_pointwise(
            module, bundle_dir=capture if capture.is_dir() else capture.parent
        )
    # Construction is proportional to this actual owner, rather than capped by
    # an unrelated tiny component's operation count. Search/member bounds stay
    # fixed and unsupported families remain ordinary unmatched source.
    operation_budget = max(16384, 2 * sum(1 for _ in prepared.walk()))
    views = plan_shared_rhs_row_views(prepared, max_operations=operation_budget)
    if views is None:
        raise ValueError("source has no checked repeated RHS row-view family")
    row_plan = plan_source_dense_row_batching(
        views.candidate,
        PipelineContext(),
        DenseRowBatchingBudgets(
            max_members=64,
            max_batches=sum(row[1] for row in views._records),
            max_candidates=16384,
            max_constructed_operations=operation_budget,
        ),
    )
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
            key = (
                (region.data, tuple(v.data for v in nodes))
                if (type(region) is builtin.StringAttr and type(nodes) is builtin.ArrayAttr)
                else None
            )
            if key not in original or original[key]["geometry_m_k_n"] != [count, *batch["shape"][1:]]:
                raise ValueError("row-batched head lost its checked original result ancestry")
            members.append({**original[key], "row_start": offset, "row_count": count})
        region = "shared_rhs_rows_" + _sha(json.dumps(members, sort_keys=True))
        copied[op].attributes.update(
            {
                "prov.region_id": builtin.StringAttr(region),
                "prov.family": builtin.StringAttr("contraction"),
                "prov.op": builtin.StringAttr("matmul"),
                "prov.source_node_ids": builtin.ArrayAttr(
                    builtin.StringAttr(node) for member in members for node in member["source_node_ids"]
                ),
            }
        )
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

    from merlin.compiler.transforms.canonicalization import fold_complete_matrix_slice_concats
    from merlin.compiler.transforms.dense.head_batching import plan_independent_head_batching

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
            if (
                type(region) is not StringAttr
                or not region.data
                or type(nodes) is not ArrayAttr
                or not nodes
                or any(type(node) is not StringAttr for node in nodes)
            ):
                raise ValueError("head members need complete original source provenance")
            if region.data in row_members:
                original.extend({**member, "projected_head": head} for member in row_members[region.data])
            else:
                original.append(
                    {
                        "head": head,
                        "source_region_id": region.data,
                        "source_node_ids": [node.data for node in nodes],
                        "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n],
                    }
                )
        original.sort(key=lambda member: member["head"])
        if len({r["source_region_id"] for r in original}) != len(original):
            raise ValueError("head members have ambiguous original source identities")
        region = "independent_heads_" + _sha(json.dumps(original, sort_keys=True))
        root = copied[batch]
        root.attributes.update(
            {
                "prov.region_id": StringAttr(region),
                "prov.family": StringAttr("contraction"),
                "prov.op": StringAttr("matmul"),
                "prov.source_node_ids": ArrayAttr(
                    StringAttr(node) for row in original for node in row["source_node_ids"]
                ),
            }
        )
        rows.append(
            {
                "projected_region_id": region,
                "members": original,
                "batch_extent": group.problem.products,
                "geometry_m_k_n": [group.problem.m, group.problem.k, group.problem.n],
            }
        )
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
        if (
            row is None
            or row.get("parallel") != [h, m, n]
            or row.get("reduction") != [k]
            or row.get("dtypes") != ["i8", "i8", "i32"]
            or type(h) is not int
            or h < 2
            or family["projected_region_id"] in regions
        ):
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
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1

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
        _capsule, interface = _merlin_model_builders_1.build(selected.entries[symbol], binding)
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
    from merlin.compiler.lower.device.sibling_component_route import _snapshot as component_snapshot

    # Source, projection, core and component owners commonly alias the same
    # large modules. Serialize each distinct graph once in this invocation;
    # a later check always reads fresh IR rather than a cross-call hash cache.
    digests = {}

    def digest(module):
        key = id(module)
        if key not in digests:
            digests[key] = _sha(_text(module))
        return digests[key]

    return {
        "source": digest(host.source),
        "module": digest(host.module),
        "core": digest(host.core),
        "interface": _sha(host.interface),
        "rewrite": host.rewrite.to_sidecar(),
        "receipt": host.receipt,
        "projection_source": digest(host.projection.source),
        "projection": digest(host.projection.module),
        "members": host.projection.members,
        "component_host": component_snapshot(
            host.component_host,
            source_sha256=digest(host.component_host.source),
            host_sha256=digest(host.component_host.module),
        )
        if host.component_host is not None
        else None,
    }


def component_selection(selected):
    from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute

    selected.record()
    if not selected.shared_input_siblings:
        raise ValueError("head selection did not request checked sibling components")
    return SiblingSourceRoute(selected.source_ir_sha256, selected.effects, partial=True)


def require_host(host, selected):
    if type(host) is not RoutedHeads or type(selected) is not HeadBatchSourceRoute:
        raise ValueError("head source, member projection, host or device custody changed")
    snapshot = _snapshot(host)
    if (
        snapshot["source"] != selected.source_ir_sha256
        or _sha(json.dumps(snapshot, sort_keys=True, allow_nan=False)) != host.integrity_sha256
    ):
        raise ValueError("head source, member projection, host or device custody changed")
    host.projection.batching.revalidate()
    if host.component_host is not None:
        from merlin.compiler.lower.device.sibling_component_route import require_components

        require_components(host.component_host, component_selection(selected))
        if host.component_host.rewrite != host.rewrite or (
            host.component_host.module is not host.module and _text(host.component_host.module) != _text(host.module)
        ):
            raise ValueError("head and sibling routes lost their common actual host or rewrite")
    elif selected.shared_input_siblings:
        raise ValueError("joint head route lost its checked sibling components")
    host.module.verify()
    selected.record()


def prepare_host(device, selected, source, directory):
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file
    from merlin.compiler.lower.device.device_offload import rewrite_groups_to_device
    from merlin.compiler.selection.kernels import _mesh_tile_binding
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_2

    selected.record()
    actual = parse_mlir_file(source)
    original, preparation = actual, None
    if _sha(_text(actual)) != selected.source_ir_sha256:
        original, preparation = original_sibling_source(device, selected, actual_source=source, work=directory.parent)
    projection = project_heads(actual, shared_rhs_row_views=selected.shared_rhs_row_views, capture=device.capture)
    module = projection.module.clone()
    # Same normal source/group owner for both matched and unmatched groups.
    # Only the checked source projection changes; no contraction falls back to
    # scalar host execution to make this route applicable.
    component_host = None
    if selected.shared_input_siblings:
        from merlin.compiler.lower.device.sibling_component_route import build_components

        component_host = build_components(
            device, component_selection(selected), projection.module, original=original, preparation=preparation
        )
        module, rewrite = component_host.module, component_host.rewrite
    else:
        rewrite = rewrite_groups_to_device(
            module,
            device.device,
            select=device.select,
            capture=device.capture,
            model=device.model,
            require_complete_groups=getattr(device, "require_complete_groups", False),
        )
    by_region = {row.get("source_region_id"): row for row in rewrite.to_sidecar()["routed"]}
    matched = []
    for family in projection.members:
        row = by_region.get(family["projected_region_id"])
        m, k, n = family["geometry_m_k_n"]
        if (
            row is None
            or row.get("parallel") != [family["batch_extent"], m, n]
            or row.get("reduction") != [k]
            or row.get("dtypes") != ["i8", "i8", "i32"]
        ):
            raise ValueError("normal group route did not retain complete independent head geometry")
        symbol = row["symbol"]
        call = rewrite.call_buffers[symbol]
        if [r["shape"] for r in call] != [
            [family["batch_extent"], m, k],
            [family["batch_extent"], k, n],
            [family["batch_extent"], m, n],
        ] or [r["dtype"] for r in call] != ["i8", "i8", "i32"]:
            raise ValueError("normal head call lost original complete ranked operands/results")
        matched.append({**family, "physical_symbol": symbol, "physical_group": row["group"]})
    binding = _mesh_tile_binding(
        device.device, device.operand_dtype, device.accum_dtype, numeric_policy=device.numeric_policy
    )
    rewrite = whole_head_interfaces(rewrite, projection.members, binding=binding, enabled=selected.whole_batch)
    interfaces = {}
    for symbol, entry in rewrite.entries.items():
        if component_host is not None:
            interfaces[symbol] = rewrite.expected_interfaces[symbol]
        else:
            _capsule, interface = _merlin_model_builders_2.build(entry, binding)
            interfaces[symbol] = {"mlir": interface, "sha256": _sha(interface)}
    rewrite = replace(
        rewrite,
        model_sha256=selected.source_ir_sha256,
        package_sha256=_package_sha256(Path(device.package_dir)),
        expected_interfaces=interfaces,
    )
    if component_host is not None and selected.whole_batch:
        # The preceding exact head/member/interface checks justify this new
        # aggregate rewrite. Every individual sibling writer/core and its
        # sealed interface stays unchanged and is revalidated below.
        component_host = replace(component_host, rewrite=rewrite, integrity_sha256="")
    receipt = {
        "schema": "merlin.original_head_source_projection.v1",
        "source_ir_sha256": selected.source_ir_sha256,
        "prepared_source_ir_sha256": _sha(_text(actual)),
        "projected_source_ir_sha256": _sha(_text(projection.module)),
        "head_families": matched,
        "weight_preparation": preparation,
        "unmatched_source_routed_normally": True,
        "original_readouts_preserved": True,
        "matrix_products": sum(len(r["members"]) for r in matched),
        "batch_abi_calls": len(matched),
        "accelerator_execution_count_reduced": False,
        "shared_input_components": len(component_host.components) if component_host is not None else 0,
        "effect_or_placement_grant": False,
    }
    if selected.whole_batch:
        receipt.update(
            whole_batch=True,
            kernel_invocations=len(matched),
            bridge_kind="whole_logical_batch",
            mathematical_products_reduced=False,
        )
    interface = next(iter(interfaces.values()))["mlir"]
    host = RoutedHeads(original, module, projection.module, interface, rewrite, projection, receipt, "", component_host)
    snapshot = _snapshot(host)
    if component_host is not None and selected.whole_batch:
        # The fresh aggregate snapshot already contains the exact common
        # graphs and every original component receipt. Reuse those hashes
        # within this invocation, then independently revalidate below.
        component_host = replace(
            component_host,
            integrity_sha256=_sha(json.dumps(snapshot["component_host"], sort_keys=True, allow_nan=False)),
        )
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


# ---- prepacked_device_capture ------------------------------------------------------------------
# Keep ordinary complete-group manifests aligned with checked weight preparation.


def rebind(device, *, source: Path, prepared: Path, work: Path, compilation=None):
    """Rebind only the ordinary all-group selector, without moving source certificates.

    Exact selections, catalogs, provider transforms and other source-bound owners
    need their own checked composition. They refuse instead of borrowing the
    prepared bundle's manifest or silently retaining the original weights.
    """
    source, prepared = Path(source).resolve(), Path(prepared).resolve()
    if source == prepared or device is None:
        return device
    from merlin.compiler.planning.complete import select_complete_group

    if (
        type(device) is not DeviceRouting
        or device.select is not select_complete_group
        or not device.require_complete_groups
        or device.granularity != "group"
        or any(
            getattr(device, key) is not None
            for key in (
                "exact_selection",
                "catalog_manifest",
                "catalog_object",
                "catalog_builder",
                "prepared_transform",
                "post_offload_transform",
                "offload_selection",
                "llvm_transform_chain",
                "target_schedule",
                "target_configuration",
                "connected_spatial",
            )
        )
        or device.descriptor_results
        and not callable(device.descriptor_effects)
        or not device.descriptor_results
        and device.descriptor_effects is not None
    ):
        raise PrepackRefused("weight preparation has no checked composition with this source-bound device route")
    # The descriptor owner is an unissued factory here. It must independently
    # qualify the actual prepared source, generated shim and observed objects
    # later; rebinding the complete selector carries no old effects certificate.
    capture = None if device.capture is None else Path(device.capture).resolve()
    if capture not in {source, source / "model.mlir"}:
        raise PrepackRefused("complete-group capture differs from the original weight-preparation source")
    receipt = Path(work) / "bundle_preparation.json"
    revalidate_build_preparation(receipt)
    record = json.loads(receipt.read_bytes())
    if record["source_bundle"] != str(source) or record["prepared_bundle"] != str(prepared):
        raise PrepackRefused("device manifest preparation differs from the checked bundle selection")
    sibling = getattr(device, "sibling_source", None)
    if sibling is not None:
        import hashlib

        from merlin.compiler.dialects.xdsl_context import text
        from merlin.compiler.ir import mlir_query as mq
        from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute

        if type(sibling) not in (SiblingSourceRoute, HeadBatchSourceRoute):
            raise PrepackRefused("weight preparation needs the original typed sibling selection")
        sibling.record()
        original = mq.parse(source / "model.mlir")
        if hashlib.sha256(text(original, generic=True).encode()).hexdigest() != sibling.source_ir_sha256:
            raise PrepackRefused("weight preparation source differs from the original sibling selection")
    selected = prepared / "model.mlir" if capture.name == "model.mlir" else prepared
    bound = Path(work) / "device_capture_preparation.json"
    bound.write_text(
        json.dumps(
            {
                "schema": "merlin.prepacked_device_capture.v1",
                "source_capture": str(capture),
                "prepared_capture": str(selected),
                "preparation_sha256": sha256_file(receipt),
                "scope": "ordinary complete-group selector and checked weight manifest; no source-certificate transfer",
            },
            indent=2,
        )
        + "\n"
    )
    if compilation is not None:
        compilation.bind_preparation("device_capture_preparation", bound)
    return replace(device, capture=selected)


def original_sibling_source(device, selected, *, actual_source, work):
    """Reopen an exact checked layout derivation without moving source authority.

    The sibling selection continues to bind the original SSA. Its prepared host
    and runtime consume the same independently verified layout/manifest rewrite.
    This function neither accepts candidate preparation receipts nor changes
    effects, numerical policy or source eligibility.
    """
    import hashlib

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.ir import mlir_query as mq

    receipt = Path(work) / "bundle_preparation.json"
    capture_receipt = Path(work) / "device_capture_preparation.json"
    revalidate_build_preparation(receipt)
    record = json.loads(receipt.read_bytes())
    binding = json.loads(capture_receipt.read_bytes())
    source = Path(record["source_bundle"])
    prepared = Path(record["prepared_bundle"])
    actual_source = Path(actual_source).resolve()
    capture = Path(device.capture).resolve()
    if (
        actual_source != prepared / "model.mlir"
        or capture not in (prepared, actual_source)
        or binding.get("preparation_sha256") != sha256_file(receipt)
        or Path(binding.get("prepared_capture", "")).resolve() != capture
        or Path(binding.get("source_capture", "")).resolve() not in (source, source / "model.mlir")
    ):
        raise PrepackRefused("sibling source does not consume its original checked weight preparation")
    original = mq.parse(source / "model.mlir").clone()
    if hashlib.sha256(text(original, generic=True).encode()).hexdigest() != selected.source_ir_sha256:
        raise PrepackRefused("prepared sibling source lost its original source selection")
    return original, {
        "path": str(receipt.resolve()),
        "sha256": sha256_file(receipt),
        "capture_binding": {"path": str(capture_receipt.resolve()), "sha256": sha256_file(capture_receipt)},
        "scope": "checked bit-preserving immutable weight layout; original source and effects retained",
    }


# ---- exact_offload -----------------------------------------------------------------------------
# Exact captured-model selection for the whole-model device rewrite.
#
# This is a placement *identity*, not a claim that a diagnostic outline has passed
# Phase 1. It binds one immutable model, its selected operation IDs, the actual
# interface bytes, one OOT compiler tree, and a reviewed Phase 0 release checked by
# the installed host verifier. The rewrite and build recheck these identities; a changed
# preparation pass or package must not turn a selection for one operation into a
# shape-based choice of another. The release verifier belongs to the optional
# experiment owner, so core does not read experiment paths. This is a trusted-host
# evidence gate, not isolation against arbitrary Python running in that host process.


@dataclass(frozen=True)
class SelectedKernel:
    operation_id: str
    interface_mlir: str
    interface_sha256: str


@dataclass(frozen=True)
class ReleaseBinding:
    """Data selected by the trusted host; no caller-supplied verifier code."""

    seal_path: Path
    descriptor: Path
    application: str
    review_digest: str


@dataclass(frozen=True)
class ExactOffloadSelection:
    target: str
    model_sha256: str
    package_sha256: str
    transport: str
    abi_sha256: str
    kernels: tuple[SelectedKernel, ...]
    software_spec_sha256: str
    capability_contract_sha256: str
    certification_sha256: tuple[str, ...] = ()
    release_binding: ReleaseBinding | None = None

    @classmethod
    def from_outline(
        cls,
        outline: Mapping,
        *,
        model: bytes,
        target: str,
        software_spec: bytes,
        capability_contract: bytes,
        package_dir: str | Path,
        operation_ids: tuple[str, ...],
        interface_root: str | Path | None = None,
    ) -> ExactOffloadSelection:
        """Bind explicit IDs from one outline; never promote unknown support.

        Re-derivation from the exact source bytes is intentional. Changing a
        diagnostic JSON status cannot admit an operation, and the returned
        selection still cannot execute until :meth:`certify` runs independently.
        """
        regenerated = outline_integer_matmuls(
            model, target=target, software_spec=software_spec, capability_contract=capability_contract
        )
        digest = outline.get("model_sha256")
        if (
            outline.get("schema") != OUTLINE_SCHEMA
            or outline.get("target") != target
            or not is_sha256(digest)
            or digest != regenerated["model_sha256"]
            or outline.get("software_spec_sha256") != regenerated["software_spec_sha256"]
            or outline.get("capability_contract_sha256") != regenerated["capability_contract_sha256"]
        ):
            raise ValueError("outline, exact model, software spec, or capability bytes disagree")
        if not operation_ids or len(set(operation_ids)) != len(operation_ids):
            raise ValueError("exact selection needs distinct, explicitly named operation IDs")
        actual_candidates = []
        for row in outline.get("candidates", ()):
            candidate = dict(row)
            if "interface_file" in candidate:
                name = candidate.pop("interface_file")
                if interface_root is None or not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("materialized outline needs a safe interface_root")
                path = Path(interface_root) / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"outline interface {name!r} is missing or a symlink")
                candidate["interface_mlir"] = path.read_text(encoding="utf-8")
            actual_candidates.append(candidate)
        if actual_candidates != regenerated["candidates"]:
            raise ValueError("outline candidates differ from deterministic re-derivation")
        by_id = {row["operation_id"]: row for row in regenerated["candidates"]}
        if len(by_id) != len(regenerated["candidates"]):
            raise ValueError("outline has duplicate candidate operation IDs")
        kernels = []
        for operation_id in operation_ids:
            candidate = by_id.get(operation_id)
            if candidate is None or not operation_id.startswith(f"mlir:{digest}:"):
                raise ValueError(f"{operation_id!r} is not a candidate of this exact model")
            if candidate.get("software_admission", {}).get("status") != "admitted":
                raise ValueError(f"{operation_id!r} has no reviewed SW admission")
            interface = candidate.get("interface_mlir")
            interface_sha = candidate.get("interface_sha256")
            if (
                not isinstance(interface, str)
                or not is_sha256(interface_sha)
                or sha256_text(interface) != interface_sha
            ):
                raise ValueError(f"{operation_id!r} has changed or unreadable interface bytes")
            kernels.append(SelectedKernel(operation_id, interface, interface_sha))
        package_sha256 = _package_sha256(Path(package_dir))
        transport, abi_sha256 = _backend_identity(target)
        return cls(
            str(target),
            str(digest),
            package_sha256,
            transport,
            abi_sha256,
            tuple(kernels),
            sha256_bytes(software_spec),
            sha256_bytes(capability_contract),
        )

    def certify(
        self, package_dir: str | Path, *, runs_root: str | Path, simulator: str, timeout: int
    ) -> ExactOffloadSelection:
        """Run the selected *exact* interfaces through the existing OOT numerical oracle.

        A pass flag copied into JSON is never accepted here. This method invokes
        the evaluator with accelerator trace required, retains its run artifacts,
        and refuses any skipped/unavailable oracle or source-tree mutation.
        """
        if not simulator or type(timeout) is not int or timeout <= 0:
            raise ValueError("certification needs an explicit simulator and positive timeout")
        from merlin.development.targetgen.packaging import loading as oot_runner

        self.check_release()
        self.check_package(package_dir)
        self.check_backend_contract()
        root = Path(runs_root)
        root.mkdir(parents=True, exist_ok=True)
        receipts = []
        for index, kernel in enumerate(self.kernels):
            with tempfile.TemporaryDirectory(prefix="merlin_exact_iface_", dir=root) as td:
                interface = Path(td) / "selected.interface.mlir"
                interface.write_text(kernel.interface_mlir, encoding="utf-8")
                result = oot_runner.certify(
                    str(package_dir),
                    interface,
                    runs_root=str(root),
                    run_id=f"exact_{self.model_sha256[:12]}_{index}",
                    simulator=simulator,
                    target=self.target,
                    timeout=timeout,
                    require_accelerator_trace=True,
                )
            self.check_package(package_dir)
            self.check_backend_contract()
            trace = result.get("trace_check") or {}
            if (
                result.get("status") != "pass"
                or (result.get("oracle") or {}).get("result") != "pass"
                or trace.get("status") != "pass"
                or trace.get("drives_accelerator") is not True
            ):
                raise ValueError(f"selected {kernel.operation_id} did not pass a running accelerator oracle")
            receipts.append(sha256_text(json.dumps(result, sort_keys=True, default=str)))
        return ExactOffloadSelection(
            self.target,
            self.model_sha256,
            self.package_sha256,
            self.transport,
            self.abi_sha256,
            self.kernels,
            self.software_spec_sha256,
            self.capability_contract_sha256,
            tuple(receipts),
            self.release_binding,
        )

    @property
    def certified(self) -> bool:
        return (
            bool(self.kernels)
            and len(self.certification_sha256) == len(self.kernels)
            and all(is_sha256(receipt) for receipt in self.certification_sha256)
        )

    def check_package(self, package_dir: str | Path) -> None:
        observed = _package_sha256(Path(package_dir))
        if observed != self.package_sha256:
            raise ValueError("selected OOT compiler package tree changed after placement")

    def check_backend_contract(self) -> None:
        if _backend_identity(self.target) != (self.transport, self.abi_sha256):
            raise ValueError("selected transport or pointer ABI changed after exact placement")

    def check_release(self) -> None:
        binding = self.release_binding
        if (
            type(binding) is not ReleaseBinding
            or not is_sha256(binding.review_digest)
            or not isinstance(binding.seal_path, Path)
            or not isinstance(binding.descriptor, Path)
            or not isinstance(binding.application, str)
            or not binding.application
        ):
            raise ValueError("exact offload needs a verified reviewed Phase 0 release")
        if _release_verifier()(binding, self) != binding.review_digest:
            raise ValueError("exact offload release verifier did not return the selected review identity")

    @property
    def by_operation_id(self) -> dict[str, SelectedKernel]:
        return {kernel.operation_id: kernel for kernel in self.kernels}


def _release_verifier():
    """Resolve the installed host owner, never executable code carried by a selection.

    This is a trusted-process API, not a sandbox against arbitrary Python running
    in the host process. An absent or ambiguous owner refuses exact offload.
    """
    providers = tuple(entry_points(group="merlin.exact_offload_release"))
    if len(providers) != 1 or providers[0].name != "reviewed_phase0":
        raise ValueError("exact offload requires one installed reviewed Phase 0 release verifier")
    return providers[0].load()


def _backend_identity(target: str) -> tuple[str, str]:
    from merlin.compiler.lower.device.build import objects_buildable
    from merlin.compiler.lower.device.shim import kernel_abi_for
    from merlin.target.registry.experiment_descriptor import load_capability_manifest
    from merlin.target.system import link_for

    unavailable = objects_buildable(target)
    if unavailable:
        raise ValueError(f"selected target has no linkable whole-model path: {unavailable}")
    manifest = load_capability_manifest(target)
    link = link_for(target, manifest.endpoint_kind)
    abi = kernel_abi_for(target)
    if abi is None or not link.command_transport:
        raise ValueError("selected target has no derived command transport or kernel pointer ABI")
    return str(link.command_transport), sha256_text(repr(abi))


def _package_sha256(root: Path) -> str:
    generated = {"build", "__pycache__", ".git"}
    if (
        root.is_symlink()
        or not root.is_dir()
        or any(
            path.is_symlink() and not generated.intersection(path.relative_to(root).parts) for path in root.rglob("*")
        )
    ):
        raise ValueError("selected OOT compiler package must be a regular, symlink-free tree")
    package = hash_tree(root)
    if not package["present"] or not package["n_files"] or not is_sha256(package["sha256"]):
        raise ValueError("selected OOT compiler package has no readable source tree")
    return str(package["sha256"])
