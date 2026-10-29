"""Exact source-to-emission identity join for operator-private whole-model builds.

Group indices describe one compiler stage and may change during preparation. The
capture frontend's typed region and contributing-node IDs are the join authority;
missing or ambiguous IDs are never replaced by matching counts, shapes or names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def eligible_source_identity(root: Any) -> tuple[str, tuple[str, ...]]:
    """Read one complete captured root identity, independent of group numbering."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    attributes = getattr(root, "attributes", {})
    region = attributes.get("prov.region_id")
    nodes = attributes.get("prov.source_node_ids")
    if not isinstance(region, StringAttr) or not region.data or not isinstance(nodes, ArrayAttr) or not nodes.data:
        raise ValueError("eligible source root lacks complete provenance identity")
    if any(not isinstance(node, StringAttr) or not node.data for node in nodes.data):
        raise ValueError("eligible source root has malformed provenance nodes")
    identities = tuple(sorted(node.data for node in nodes.data))
    if len(identities) != len(set(identities)):
        raise ValueError("eligible source root repeats a provenance node")
    return region.data, identities


def source_group_roster(groups, required, *, host_placement, module=None) -> dict[str, Any]:
    """Bind required and optional device groups to original source, before emission.

    Optional groups remain outside the mandatory coverage denominator. An optional
    root without a complete source identity/extent cannot gain admission by being
    emitted; record its refusal without preventing the ordinary host route.
    """
    from merlin.common import mlir_query as mq

    source: dict[str, Any] = {
        "eligible_groups": [], "eligible_group_provenance": [], "eligible_group_metrics": {},
        "optional_groups": [], "optional_group_provenance": [], "optional_group_metrics": {},
        "optional_group_refusals": [],
    }
    required = set(required)
    for group in groups:
        mandatory = group.index in required
        if group.placement == host_placement:
            if mandatory:
                raise ValueError("eligible source group was assigned to host")
            continue
        try:
            region, nodes = eligible_source_identity(group.root)
            metric = None
            for member in reversed(group.members):
                for value in member.results:
                    shape, _ = mq.type_shape_dtype(value.type)
                    elements = 1
                    for extent in shape:
                        elements *= extent
                    size = mq.value_bytes(value.type)
                    if shape and elements > 0 and size > 0 and size % elements == 0:
                        metric = {"elements": elements, "element_bytes": size // elements}
                        break
                if metric is not None:
                    break
            if metric is None:
                raise ValueError("source-bound device group has no static output extent")
        except ValueError as error:
            if mandatory:
                raise
            source["optional_group_refusals"].append({"source_group": group.index, "reason": str(error)})
            continue
        prefix = "eligible" if mandatory else "optional"
        source[f"{prefix}_groups"].append(group.index)
        source[f"{prefix}_group_provenance"].append(
            {"source_group": group.index, "source_region_id": region, "source_node_ids": list(nodes)}
        )
        source[f"{prefix}_group_metrics"][group.index] = metric
    if set(source["eligible_groups"]) != required:
        raise ValueError("source roster omits independently eligible source computation")
    for prefix in ("eligible", "optional"):
        source[f"{prefix}_groups"].sort()
    # Validate even unused optional identities: ambiguity is never resolved by
    # whether this particular compilation happened to select the group.
    join_routed_source_groups([
        {"group": row["source_group"], **row} for row in source["eligible_group_provenance"]
    ], source)
    if module is not None:
        from merlin.compile.analysis.contraction_shapes import observe_contractions

        shapes = {id(op): shape for op, shape in observe_contractions(module)}
        admitted = set(source["eligible_groups"]) | set(source["optional_groups"])
        source["source_group_contractions"] = {
            group.index: {"parallel": list(shape.parallel), "reduction": list(shape.reduction),
                          "dtypes": list(shape.dtypes)}
            for group in groups
            if group.index in admitted and (shape := shapes.get(id(group.root))) is not None
        }
    return source


def join_routed_source_groups(assigned: Any, source: Mapping[str, Any]) -> dict[int, int]:
    """Join every emitted call to a required or optional original source root.

    A region ID must be unique within one captured program; a repeated ID has no
    unambiguous source owner, regardless of whether this model happened to route it.
    Preparation may renumber groups, so those integers are never join keys.
    Every required root must be emitted; any subset of the optional roster may be.
    """
    expected = source.get("eligible_group_provenance")
    eligible = source.get("eligible_groups")
    optional = source.get("optional_groups", [])
    optional_expected = source.get("optional_group_provenance", [])
    if any(not isinstance(value, list) for value in (expected, eligible, optional, optional_expected, assigned)):
        raise ValueError("routed group source roster is incomplete")
    for indices in (eligible, optional):
        if any(type(group) is not int or group < 0 for group in indices) or indices != sorted(set(indices)):
            raise ValueError("eligible or optional source group indices are ambiguous")
    if set(eligible) & set(optional):
        raise ValueError("required and optional source group indices are ambiguous")

    def identity(row: Any) -> tuple[str, tuple[str, ...]]:
        if not isinstance(row, Mapping):
            raise ValueError("routed group has no provenance record")
        region, nodes = row.get("source_region_id"), row.get("source_node_ids")
        if (
            type(region) is not str
            or not region
            or not isinstance(nodes, list)
            or not nodes
            or any(type(node) is not str or not node for node in nodes)
            or nodes != sorted(set(nodes))
        ):
            raise ValueError("routed group has missing, malformed, or duplicate source provenance")
        return region, tuple(nodes)

    source_by_region: dict[str, tuple[tuple[str, ...], int]] = {}
    source_groups = set()
    for roster, indices in ((expected, eligible), (optional_expected, optional)):
        roster_groups = set()
        for row in roster:
            region, nodes = identity(row)
            group = row.get("source_group")
            if type(group) is not int or group < 0 or group in source_groups or region in source_by_region:
                raise ValueError("eligible or optional source group provenance is ambiguous")
            source_groups.add(group)
            roster_groups.add(group)
            source_by_region[region] = nodes, group
        if len(roster) != len(indices) or sorted(roster_groups) != indices:
            raise ValueError("source group provenance differs from source eligibility")

    joined: dict[int, int] = {}
    emitted_regions = set()
    for row in assigned:
        region, nodes = identity(row)
        group = row.get("group")
        if type(group) is not int or group < 0 or group in joined or region in emitted_regions:
            raise ValueError("routed group provenance or emitted group index is ambiguous")
        emitted_regions.add(region)
        original = source_by_region.get(region)
        if original is None or original[0] != nodes:
            raise ValueError("routed group provenance differs from independently bound source")
        joined[group] = original[1]
    if not set(eligible) <= set(joined.values()):
        raise ValueError("routed groups omit independently eligible source computation")
    return joined


def require_intended_host_selection(skipped: Any) -> None:
    """Accept ordinary host selection only after required-source coverage is checked."""
    if not isinstance(skipped, (list, tuple)) or any(
        not isinstance(row, (list, tuple)) or len(row) != 2
        or not isinstance(row[0], str) or not row[0]
        or row[1] != "the placement did not select this group's root"
        for row in skipped
    ):
        raise ValueError("linked build retained a device refusal outside intended host selection")


def verify_routed_source_coverage(emitted: Mapping[str, Any], source: Mapping[str, Any]) -> dict[int, int]:
    """Require every independently eligible root, allowing intended host selection.

    ``source`` remains the original source-admission witness, never a roster
    reconstructed from this sidecar. Host selection cannot excuse a missing,
    duplicate or changed eligible identity. Other emitter refusals stay fatal.
    This grants no linked-symbol, body, host-compute or numerical admission.
    """
    assigned = emitted.get("routed")
    if not isinstance(assigned, list) or not assigned:
        raise ValueError("linked roster differs from eligible source groups")
    demand = source.get("accelerator_source_demand")
    if (not isinstance(demand, Mapping) or demand.get("status") != "source_demand_derived"
        or demand.get("n_unknown_host_groups") != 0 or demand.get("n_missing_direct_groups") != 0
        or demand.get("n_unresolved_raw_regions") != 0
        or demand.get("demanded_group_provenance") != source.get("eligible_group_provenance")
        or demand.get("n_demanded_groups") != len(source.get("eligible_groups", []))):
        raise ValueError("linked source coverage lacks complete independent accelerator demand")
    joined = join_routed_source_groups(assigned, source)
    require_intended_host_selection(emitted.get("skipped", []))
    return joined


def _conv_entry_signature_matches(emitted, symbol, row):
    """Join a logical NHWC convolution ABI to its captured matrix contraction."""
    entry = emitted.get("entries", {}).get(symbol)
    buffers = emitted.get("call_buffers", {}).get(symbol)
    if emitted.get("granularity") != "group" or not isinstance(entry, Mapping) or entry.get("op") != "conv2d":
        return False
    epilogue = entry.get("epilogue", [])
    if epilogue not in ([], ["acc_scale"], ["bias_add"], ["bias_add", "acc_scale"],
                        ["acc_scale", "relu"], ["bias_add", "acc_scale", "relu"]):
        return False
    if entry.get("layout", "NHWC") != "NHWC":
        return False
    if entry.get("groups", 1) != 1 or type(entry.get("groups", 1)) is not int:
        return False
    biased = "bias_add" in epilogue
    if not isinstance(buffers, list) or len(buffers) != (4 if biased else 3):
        return False
    if any(not isinstance(b, Mapping) for b in buffers):
        return False
    if [b.get("role") for b in buffers] != (
        ["input_0", "weight_0", "bias_0", "out_0"] if biased else ["input_0", "weight_0", "out_0"]
    ):
        return False
    program = emitted.get("programs", {}).get(symbol)
    stated = program.get("entry") if isinstance(program, Mapping) else None
    if not isinstance(stated, Mapping):
        return False
    for key in ("op", "Himg", "Wimg", "ci", "kh", "kw", "N", "stride", "padding", "dilation", "epilogue", "acc_scale",
                "acc_scale_by_column", "output_dtype"):
        default = [1, 1] if key == "dilation" else [] if key == "epilogue" else None
        if entry.get(key, default) != stated.get(key, default):
            return False
    image = buffers[0].get("shape")
    if not isinstance(image, list) or len(image) != 4 or any(type(x) is not int or x <= 0 for x in image):
        return False
    batch, h, w, ci = image
    names = ("Himg", "Wimg", "ci", "kh", "kw", "N")
    if any(type(entry.get(k)) is not int or entry[k] <= 0 for k in names):
        return False
    if entry.get("output_dtype", buffers[-1].get("dtype")) != buffers[-1].get("dtype"):
        return False
    if "acc_scale_by_column" in entry:
        from merlin.runtime.commandbuffer import declared_acc_scale_by_column

        if "acc_scale" not in epilogue:
            return False
        try:
            declared_acc_scale_by_column(dict(entry), entry["N"], op="source-bound convolution")
        except (TypeError, ValueError):
            return False
    if (h, w, ci) != (entry["Himg"], entry["Wimg"], entry["ci"]):
        return False
    stride, padding, dilation = entry.get("stride"), entry.get("padding"), entry.get("dilation", [1, 1])
    for values, length, minimum in ((stride, 2, 1), (padding, 4, 0), (dilation, 2, 1)):
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(type(x) is not int or x < minimum for x in values)
        ):
            return False
    oh = (h + padding[0] + padding[1] - dilation[0] * (entry["kh"] - 1) - 1) // stride[0] + 1
    ow = (w + padding[2] + padding[3] - dilation[1] * (entry["kw"] - 1) - 1) // stride[1] + 1
    if oh <= 0 or ow <= 0:
        return False
    m, n, k = batch * oh * ow, entry["N"], entry["kh"] * entry["kw"] * ci
    # This supported logical ABI has an image, flattened stored weights,
    # optional i32 bias, and a flattened committed output. Match the fused
    # readout to the same source-stated epilogue rather than assuming a bare
    # contraction whenever a convolution uses the logical NHWC interface.
    return (
        row.get("parallel") == [m, n]
        and row.get("reduction") == [k]
        and buffers[1].get("shape") == [k, n]
        and (not biased or (buffers[2].get("shape") == [n] and buffers[2].get("dtype") == "i32"))
        and buffers[-1].get("shape") == [m, n]
        and [buffers[i].get("dtype") for i in (0, 1, len(buffers) - 1)] == row.get("dtypes")
    )


def convolution_input_preparations(emitted, source):
    """Bind ordinary logical convolution copies to original source contractions.

    Source formation owns the original dimensions/types; the trusted host rewrite
    owns the logical call ABI. Neither candidate output nor a matching count can
    substitute for that join. Arithmetic and all numerical checks remain separate.
    """
    joined = join_routed_source_groups(emitted.get("routed"), source)
    preparations = {}
    for row in emitted["routed"]:
        symbol = row["symbol"]
        entry = emitted.get("entries", {}).get(symbol, {})
        if entry.get("op") != "conv2d":
            continue
        # Only a bare contraction receives the ordinary copy exemption. A
        # source-bound fused readout changes the accumulator's output format;
        # it remains subject to the normal source/signature and host audits.
        if emitted.get("signatures", {}).get(symbol) != [] or entry.get("epilogue"):
            continue
        original_group = joined[row["group"]]
        original = source.get("source_group_contractions", {}).get(original_group)
        if not isinstance(original, Mapping) or any(
            row.get(key) != original.get(key) for key in ("parallel", "reduction", "dtypes")
        ):
            raise ValueError("convolution entry differs from original source contraction")
        if not _conv_entry_signature_matches(emitted, symbol, row):
            raise ValueError("logical convolution input preparation differs from source-bound call ABI")
        preparations[symbol] = {
            "entry": entry, "call_buffers": emitted["call_buffers"][symbol],
            "source_group": original_group, "argument_alignment": 1,
        }
    return preparations


def routed_kernel_symbols(emitted: Mapping[str, Any]) -> set[str]:
    """Validate unique compiled kernels, independently of the source call roster.

    The trusted emitter may reuse one monomorphic kernel for several source calls.
    Source identities remain one-to-one; sharing never permits a missing call or a
    different dimension/type signature to borrow the same linked symbol.
    """
    rows, signatures = emitted.get("routed"), emitted.get("signatures")
    if not isinstance(rows, list) or not rows or not isinstance(signatures, Mapping):
        raise ValueError("routed kernel signature roster is incomplete")
    by_symbol = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("routed kernel record is malformed")
        symbol = row.get("symbol")
        if (
            type(symbol) is not str
            or not symbol
            or not symbol.isascii()
            or not (symbol[0].isalpha() or symbol[0] in "_.$")
            or any(not (char.isalnum() or char in "_.$") for char in symbol)
        ):
            raise ValueError("routed kernel symbol is missing or malformed")
        parallel, reduction, dtypes = row.get("parallel"), row.get("reduction"), row.get("dtypes")
        if (
            not isinstance(parallel, list)
            or not parallel
            or not isinstance(reduction, list)
            or any(type(size) is not int or size < 0 for size in parallel + reduction)
            or not isinstance(dtypes, list)
            or not dtypes
            or any(type(dtype) is not str or not dtype for dtype in dtypes)
        ):
            raise ValueError("routed kernel has malformed dimension/type signature")
        signature = (tuple(parallel), tuple(reduction), tuple(dtypes))
        if symbol in by_symbol and by_symbol[symbol] != signature:
            raise ValueError("reused kernel symbol has inconsistent dimension/type signature")
        extents = signatures.get(symbol)
        if (
            not isinstance(extents, list)
            or any(type(size) is not int for size in extents)
            or (
                extents != parallel + reduction
                and not (extents == [] and _conv_entry_signature_matches(emitted, symbol, row))
            )
        ):
            raise ValueError("routed kernel extents differ from compiled signature")
        by_symbol[symbol] = signature
    if set(signatures) != set(by_symbol):
        raise ValueError("compiled kernel signature roster differs from routed symbols")
    return set(by_symbol)


def join_sibling_source_composition(emitted, source, *, module, target, preparation_receipt=None):
    """Re-derive a many-to-one raw projection from original typed source.

    The original owner must independently select this route. This source join
    does not authorize it, qualify borrowed effects or replace per-member actual
    device-work attribution. Original required/optional denominators stay intact.
    """
    import hashlib
    import json

    from xdsl.dialects import func

    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.llvmlower.sibling_source_route import _command_buffer
    from merlin.llvmlower.source_region_host import outline_source_siblings
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir
    from merlin.xdsl_dialects._common import text

    original = text(module, generic=True)
    source_sha = hashlib.sha256(original.encode()).hexdigest()
    projection_module = module
    preparation = None
    if preparation_receipt is not None:
        from pathlib import Path

        from merlin.compile.model_execution_inputs import file_sha256
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.llvmlower.weight_prepack import revalidate_build_preparation

        receipt_path = Path(preparation_receipt)
        if (not receipt_path.is_absolute() or not receipt_path.is_file()
            or any(path.is_symlink() for path in (receipt_path, *receipt_path.parents))):
            raise ValueError("sibling preparation needs its original direct owned receipt")
        receipt_sha = file_sha256(receipt_path)
        revalidate_build_preparation(receipt_path)
        record = json.loads(receipt_path.read_text())
        source_path = Path(record["source_bundle"]) / "model.mlir"
        prepared_path = Path(record["prepared_bundle"]) / "model.mlir"
        if text(parse_mlir_file(source_path), generic=True) != original:
            raise ValueError("sibling preparation ancestry differs from original eligible source")
        projection_module = parse_mlir_file(prepared_path)
        preparation = {
            "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
            "schema": record["schema"], "identity_sha256": record["identity_sha256"],
            "source_model_sha256": file_sha256(source_path), "prepared_model_sha256": file_sha256(prepared_path),
            "immutable_arguments": record["immutable_arguments"],
        }
    projection_sha = hashlib.sha256(text(projection_module, generic=True).encode()).hexdigest()
    outlined, _ = outline_source_siblings(projection_module)
    cb, group = _command_buffer(outlined.core, target)
    functions = [op for op in outlined.core.body.block.ops if isinstance(op, func.FuncOp)]
    function = functions[0]
    operations = tuple(function.body.block.ops)
    roots = tuple(operations[index] for index in group.contraction_ordinals)
    symbol = function.sym_name.data + "_device"
    interface = emit_interface_mlir(cb)
    interface_sha = hashlib.sha256(interface.encode()).hexdigest()
    buffers, reason = logical_buffers(interface)
    if buffers is None:
        raise ValueError("source sibling composition has no complete logical interface: " + reason)
    # Validate the complete independent roster, including unused optional IDs.
    roster = [*source.get("eligible_group_provenance", []), *source.get("optional_group_provenance", [])]
    join_routed_source_groups([{"group": row["source_group"], **row} for row in roster], source)
    by_region = {row["source_region_id"]: row for row in roster}
    members = []
    for index, (root, width) in enumerate(zip(roots, group.widths, strict=True)):
        region, nodes = eligible_source_identity(root)
        original_group = by_region.get(region)
        if original_group is None or list(nodes) != original_group["source_node_ids"]:
            raise ValueError("sibling member differs from independently admitted original source provenance")
        source_group = original_group["source_group"]
        geometry = source.get("source_group_contractions", {}).get(source_group)
        if geometry != {"parallel": [group.m, width], "reduction": [group.k], "dtypes": ["i8", "i8", "i32"]}:
            raise ValueError("sibling raw product differs from original contraction precision or extents")
        members.append({
            "source_group": source_group, "source_region_id": region,
            # The independent roster canonicalizes the complete identity set;
            # actual typed-call metadata must retain the source's exact order.
            "source_node_ids": [node.data for node in root.attributes["prov.source_node_ids"]],
            "output_role": "out_" + str(index), "geometry_m_k_n": [group.m, group.k, width],
            "raw_output_elements": group.m * width, "raw_output_element_bytes": 4,
            "source_useful_macs": group.m * group.k * width,
        })
    covered = [row["source_group"] for row in members]
    if len(set(covered)) != len(covered) or not set(source["eligible_groups"]) <= set(covered):
        raise ValueError("sibling composition duplicates or omits independently eligible original source groups")
    expected_entry = {
        "schema": "original_sibling_source_entry_v1", "name": symbol, "source_ir_sha256": source_sha,
        "core_ir_sha256": hashlib.sha256(text(outlined.core, generic=True).encode()).hexdigest(),
    }
    def same(actual, expected):
        try:
            return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
                expected, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return False

    assigned = emitted.get("routed")
    if (
        emitted.get("device") != target
        or emitted.get("granularity") != "group"
        or emitted.get("model_sha256") != source_sha
        or not same(emitted.get("signatures"), {symbol: [group.m, sum(group.widths), group.k]})
        or not same(emitted.get("entries"), {symbol: expected_entry})
        or not same(emitted.get("programs"), {symbol: cb})
        or not same(emitted.get("expected_interfaces"), {symbol: {"mlir": interface, "sha256": interface_sha}})
        or not same(emitted.get("call_buffers"), {symbol: [
            {"role": buffer.role, "shape": list(buffer.shape), "dtype": buffer.dtype} for buffer in buffers
        ]})
        or emitted.get("arg_access") != {symbol: ["read"] * (len(members) + 1) + ["write"] * len(members)}
        or not isinstance(assigned, list) or len(assigned) != 1 or not isinstance(assigned[0], Mapping)
        or assigned[0].get("symbol") != symbol
        or not (assigned[0].get("group") is None or type(assigned[0].get("group")) is int
                and assigned[0]["group"] == 0)
        or not same(assigned[0].get("parallel"), [group.m, sum(group.widths)])
        or not same(assigned[0].get("reduction"), [group.k])
        or assigned[0].get("dtypes") != ["i8", "i8", "i32"]
        or assigned[0].get("source_region_id") is not None or assigned[0].get("source_node_ids")
    ):
        raise ValueError("emitted sibling composition differs from independently rederived source and complete ABI")
    require_intended_host_selection(emitted.get("skipped", []))
    if text(module, generic=True) != original:
        raise ValueError("sibling source composition join changed original source")
    if preparation is not None:
        revalidate_build_preparation(receipt_path)
        if (file_sha256(receipt_path) != receipt_sha
            or file_sha256(source_path) != preparation["source_model_sha256"]
            or file_sha256(prepared_path) != preparation["prepared_model_sha256"]):
            raise ValueError("sibling original preparation changed during source join")
    return {
        "schema": "merlin.original_sibling_source_composition.v1",
        "source_ir_sha256": source_sha, "prepared_source_ir_sha256": projection_sha,
        "preparation": preparation, "interface_sha256": interface_sha,
        "physical_symbol": symbol, "physical_group": assigned[0].get("group"), "members": members,
        "host_entry_symbol": function.sym_name.data,
        "borrowed_entry_symbol": function.sym_name.data + "_borrowed",
        "required_source_groups": list(source["eligible_groups"]),
        "covered_source_groups": covered,
        "raw_output_elements": sum(row["raw_output_elements"] for row in members),
        "raw_output_element_bytes": 4,
        "source_useful_macs": sum(row["source_useful_macs"] for row in members),
        "original_source_denominator_unchanged": True,
        "scope": "original source identities, ordered raw products and complete logical interface only; "
                 "actual member work, host calls, effects, objects, execution and required tiers remain separate",
    }
