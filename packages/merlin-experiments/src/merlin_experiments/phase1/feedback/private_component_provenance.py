"""Independently rederive each composed core and every unmatched source join."""

import hashlib
import json
from pathlib import Path


def source_subset(source, removed):
    """Keep a disjoint remaining roster; never alter the original denominator."""
    result = dict(source)
    for kind in ("eligible", "optional"):
        result[kind + "_groups"] = [g for g in source.get(kind + "_groups", []) if g not in removed]
        result[kind + "_group_provenance"] = [
            r for r in source.get(kind + "_group_provenance", []) if r["source_group"] not in removed
        ]
    return result


def join_components(emitted, source, *, module, target, effects, capture=None, preparation_receipt=None,
                    checked_projection=None, other_components=()):
    from xdsl.dialects import func

    from merlin.compile.model_execution_inputs import file_sha256
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.llvmlower.sibling_components import prepare_sibling_components
    from merlin.llvmlower.sibling_readout_commands import require_writer_interface
    from merlin.llvmlower.sibling_source_route import _command_buffer
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir
    from merlin.xdsl_dialects._common import text
    from merlin.xdsl_dialects.lowering import output_fusion as OF

    from .private_group_provenance import eligible_source_identity, join_routed_source_groups

    def sha(value):
        return hashlib.sha256(value.encode()).hexdigest()

    before = text(module, generic=True)
    projection, preparation = module, None
    if preparation_receipt is not None:
        from merlin.llvmlower.weight_prepack import revalidate_build_preparation

        path = Path(preparation_receipt)
        if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("component preparation requires original direct custody")
        revalidate_build_preparation(path)
        digest = file_sha256(path)
        record = json.loads(path.read_text())
        if text(parse_mlir_file(Path(record["source_bundle"]) / "model.mlir"), generic=True) != before:
            raise ValueError("component preparation lost original source ancestry")
        projection = parse_mlir_file(Path(record["prepared_bundle"]) / "model.mlir")
        capture = Path(record["prepared_bundle"])
        preparation = {"path": str(path), "sha256": digest}
    if checked_projection is not None:
        # The original head owner supplies its independently rederived projection,
        # never candidate metadata. Its inverse/source checks remain mandatory.
        projection = checked_projection
    # Independently derive the same numerical frontier from current original
    # source and its checked capture, never from the candidate's COMMIT attrs.
    plan = prepare_sibling_components(
        projection, effects=effects, output_path=OF.declared_path(target), values=OF.Values(capture)
    )
    if not plan.components or emitted.get("device") != target or emitted.get("model_sha256") != sha(before):
        raise ValueError("component rewrite lost original source or selected target")
    roster = [*source["eligible_group_provenance"], *source.get("optional_group_provenance", [])]
    join_routed_source_groups([{"group": r["source_group"], **r} for r in roster], source)
    by_region = {r["source_region_id"]: r for r in roster}
    assigned = emitted.get("routed")
    if not isinstance(assigned, list) or len({r.get("group") for r in assigned}) != len(assigned):
        raise ValueError("component physical group roster is missing or ambiguous")
    rows = {r["symbol"]: r for r in assigned}
    if len(rows) != len(assigned):
        raise ValueError("component physical call symbols are ambiguous")
    components, covered = [], []
    for item in plan.components:
        cb, group = _command_buffer(item.core, target, readouts=item.receipt.get("readouts"))
        interface = emit_interface_mlir(cb)
        buffers, reason = logical_buffers(interface)
        if buffers is None:
            raise ValueError("component complete interface refused: " + reason)
        require_writer_interface(plan.module, item.writer, buffers)
        symbol = item.writer.symbol + "_device"
        row = rows.get(symbol)
        fn = next(op for op in item.core.body.block.ops if isinstance(op, func.FuncOp))
        ops, members = tuple(fn.body.block.ops), []
        for index, width in zip(group.contraction_ordinals, group.widths, strict=True):
            root = ops[index]
            region, nodes = eligible_source_identity(root)
            original = by_region.get(region)
            if original is None or original["source_node_ids"] != list(nodes):
                raise ValueError("component member lacks exact original provenance")
            number = original["source_group"]
            if source.get("source_group_contractions", {}).get(number) != {
                "parallel": [group.m, width],
                "reduction": [group.k],
                "dtypes": ["i8", "i8", "i32"],
            }:
                raise ValueError("component changed original product precision or geometry")
            members.append(
                {
                    "source_group": number,
                    "source_region_id": region,
                    "source_node_ids": [v.data for v in root.attributes["prov.source_node_ids"]],
                    "output_role": "out_" + str(len(members)),
                    "geometry_m_k_n": [group.m, group.k, width],
                    "raw_output_elements": group.m * width,
                    "raw_output_element_bytes": 4,
                    "source_useful_macs": group.m * group.k * width,
                }
            )
        expected = {
            "signatures": [group.m, sum(group.widths), group.k],
            "entries": {
                "schema": "original_sibling_source_entry_v1",
                "name": symbol,
                "source_ir_sha256": sha(before),
                "core_ir_sha256": sha(text(item.core, generic=True)),
            },
            "programs": cb,
            "expected_interfaces": {"mlir": interface, "sha256": sha(interface)},
            "call_buffers": [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in buffers],
            "arg_access": ["write" if b.role.startswith("out_") else "read" for b in buffers],
        }
        if (
            row is None
            or row.get("parallel") != [group.m, sum(group.widths)]
            or row.get("reduction") != [group.k]
            or row.get("dtypes") != ["i8", "i8", "i32"]
            or row.get("source_region_id") is not None
            or row.get("source_node_ids")
            or any(
                json.dumps(emitted.get(key, {}).get(symbol), sort_keys=True) != json.dumps(value, sort_keys=True)
                for key, value in expected.items()
            )
        ):
            raise ValueError("actual component interface, products or logical ABI differ from original source")
        indices = [r["source_group"] for r in members]
        if set(indices).intersection(covered):
            raise ValueError("source components duplicate an original root")
        covered.extend(indices)
        components.append(
            {
                "schema": "merlin.original_sibling_source_composition.v1",
                "source_ir_sha256": sha(before),
                "prepared_source_ir_sha256": sha(text(projection, generic=True)),
                "interface_sha256": sha(interface),
                "preparation": preparation,
                "physical_symbol": symbol,
                "physical_group": row["group"],
                "members": members,
                "host_entry_symbol": item.writer.symbol,
                "borrowed_entry_symbol": item.writer.borrowed_symbol,
                "descriptor_arguments": len(buffers),
                "required_source_groups": [g for g in source["eligible_groups"] if g in indices],
                "covered_source_groups": indices,
                "raw_output_element_bytes": 4,
                "raw_output_elements": sum(m["raw_output_elements"] for m in members),
                "original_source_denominator_unchanged": True,
            }
        )
    symbols = {r["physical_symbol"] for r in components}
    other_symbols = {c["physical_symbol"] for c in other_components}
    other_covered = {g for c in other_components for g in c["covered_source_groups"]}
    if symbols.intersection(other_symbols) or set(covered).intersection(other_covered):
        raise ValueError("joint source compositions overlap original or physical work")
    ordinary = [r for r in assigned if r["symbol"] not in symbols | other_symbols]
    joined = join_routed_source_groups(ordinary, source_subset(source, set(covered) | other_covered))
    if set(joined.values()).intersection(covered) or not set(source["eligible_groups"]) <= set(covered) | set(
        joined.values()
    ) | other_covered:
        raise ValueError("mixed source composition duplicates or omits original work")
    if preparation is not None:
        revalidate_build_preparation(path)
        if file_sha256(path) != digest:
            raise ValueError("component preparation changed during original source join")
    if text(module, generic=True) != before:
        raise ValueError("component audit changed original source")
    return {
        "schema": "merlin.original_sibling_source_components.v1",
        "components": components,
        "ordinary_joined_groups": joined,
        "required_source_groups": source["eligible_groups"],
        "composed_source_groups": covered,
        "covered_source_groups": [*covered, *joined.values()],
        "original_source_denominator_unchanged": True,
    }
