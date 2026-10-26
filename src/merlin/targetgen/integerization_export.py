"""Structural integer-contraction observations for capture export checks."""

from dataclasses import dataclass, replace

from merlin.common import mlir_query as query

from .capture_refusals import M2MUnavailable


def worker_export_complete(result, partition, metadata, *, func_name="forward"):
    """Admit a new worker export with its live module and independent site observation."""
    import hashlib

    from ._m2m_capture_artifacts import frontend_trace_bytes

    program_sha256 = hashlib.sha256(result.mlir_text.encode("utf-8")).hexdigest()
    return receipt_export_complete(
        metadata.get("integerization_receipt"), partition, result.module,
        func_name=func_name, require_census=True, program_sha256=program_sha256,
        dynamic_evidence=lambda: capture_dynamic_evidence(
            metadata, frontend_trace_bytes(getattr(result, "capture_trace", None) or {}),
            program_sha256=program_sha256,
        ),
    )


def receipt_export_complete(receipt: dict | None, partition: dict | None, module=None, **evidence) -> bool:
    """Worker admission, including contraction-free programs, uses the shared census policy."""
    if receipt is None:
        return True
    verdict = admit_integerization_census(receipt, module, **evidence)
    if not verdict.accepted:
        return False
    from merlin.capture.integerization import non_contraction_program

    return bool(
        partition is not None
        and (non_contraction_program(receipt) or (
            receipt.get("accumulator_bound_checked") is True
            and receipt.get("exported_integer_mm_count", 0) + receipt.get("exported_integer_bmm_count", 0) > 0
        ))
        and isinstance(receipt.get("golden_agreement"), dict)
        and receipt["golden_agreement"].get("status") == "passed"
    )


def _entry(module, func_name):
    for op in query.walk(module, "func.func"):
        if getattr(getattr(op, "sym_name", None), "data", None) == func_name:
            return op
    raise ValueError(f"no @{func_name} function")


def _integer_contraction_rank(op) -> int | None:
    if query.op_name(op) == "linalg.generic" and query.attr_str(op, "prov.op") != "int_matmul":
        return None
    if len(op.operands) < 2 or not op.results:
        return None
    lhs, rhs = (query.type_shape_dtype(value.type)[1] for value in op.operands[:2])
    shape, result = query.type_shape_dtype(op.results[0].type)
    return len(shape) if (lhs, rhs, result) == ("i8", "i8", "i32") else None


def exported_integer_counts(program: str, *, func_name: str = "forward") -> tuple[int, int]:
    """Count matrix and batched matrix contractions in the exported entry."""
    try:
        forward = _entry(query.parse(program), func_name)
        structural = structural_bmm = 0
        for op in query.walk(forward, "linalg.matmul", "linalg.generic"):
            rank = _integer_contraction_rank(op)
            if rank is not None:
                structural += rank == 2
                structural_bmm += rank > 2
        return structural, structural_bmm
    except Exception as exc:
        raise M2MUnavailable(f"cannot verify integerized model export: {exc}") from exc


def _contractions(module, *, func_name: str = "forward"):
    """The entry function's complete byte/byte -> i32 contraction inventory."""
    for ordinal, op in enumerate(_entry(module, func_name).walk()):
        if query.op_name(op) in {"linalg.matmul", "linalg.generic"}:
            rank = _integer_contraction_rank(op)
            if rank is not None:
                yield ordinal, op, rank


def _source_ids(op):
    from xdsl.dialects.builtin import ArrayAttr

    value = op.attributes.get("prov.source_node_ids")
    return [item.data for item in value] if isinstance(value, ArrayAttr) else []


def contraction_census(module, prepared: dict, static_calls: int, *, func_name: str = "forward") -> dict:
    """Bind every prepared integer call to exactly one exported contraction."""
    from merlin.capture.integerization import INTEGER_MM_TARGETS

    expected = []
    for node in prepared.get("nodes", []):
        target = node.get("target")
        results = node.get("results") or []
        if target not in INTEGER_MM_TARGETS | {"aten.bmm.default"}:
            continue
        if not results or results[0].get("dtype") != "int32":
            continue
        transform = node.get("transformation")
        if target == "m2m_q.int_mm_seeded.default" and transform == "pt2e_integerization":
            provenance = "seeded"
        elif transform in {"pt2e_integerization", "static_attention"}:
            provenance = transform
        else:
            provenance = "unaccounted"
        expected.append({"source_node_id": node["id"], "provenance": provenance,
                         "rank": len(results[0].get("shape") or []),
                         "origin_node_ids": node.get("origin_node_ids") or []})
    census = {"schema": "merlin.integer_contraction_census.v1", "calls": expected,
              "static_attention_device_calls": static_calls}
    census["export_bindings"] = _export_bindings(module, expected, func_name=func_name)
    return census


def _export_bindings(module, calls, *, func_name: str = "forward"):
    identities = {row["source_node_id"] for row in calls}
    return [{"ordinal": ordinal, "rank": rank,
             "source_node_ids": _source_ids(op),
             "contraction_source_ids": sorted(identities.intersection(_source_ids(op)))}
            for ordinal, op, rank in _contractions(module, func_name=func_name)]


@dataclass(frozen=True)
class CensusAdmission:
    """A fail-closed census decision, separate from numerical/partition admission."""

    census_present: bool
    census_valid: bool = False
    structure_valid: bool = False
    counts_agree: bool = False
    dynamic_required: bool = False
    dynamic_validated: bool = False
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.reason is None and self.structure_valid and self.counts_agree


def admit_integerization_census(
    receipt: dict, module=None, *, func_name: str = "forward", dynamic_evidence=None,
    program_sha256: str | None = None, require_census: bool = False,
) -> CensusAdmission:
    """Authoritative admission for saved and live integer exports.

    Dynamic evidence may be a saved projection or a lazy reader of receipt-bound raw
    evidence. It is never inferred from a structural census or matching totals.
    Legacy receipts without a census retain exact per-rank emission/export equality.
    New workers require a census even for contraction-free programs.
    """
    present = isinstance(receipt, dict) and "integer_contraction_census" in receipt
    verdict = CensusAdmission(census_present=present)
    try:
        if not isinstance(receipt, dict):
            raise ValueError("invalid integerization receipt")
        census = receipt.get("integer_contraction_census")
        if present and (not isinstance(census, dict) or census.get("schema") != "merlin.integer_contraction_census.v1"):
            raise ValueError("invalid or null integer contraction census")
        if require_census and not present:
            raise ValueError("integer contraction census is required")
        verdict = replace(verdict, census_valid=present)
        emitted, exported = _receipt_counts(receipt)
        static_calls = 0
        if present:
            counts = _reconcile_census(census, module, func_name=func_name)
            verdict = replace(verdict, structure_valid=True)
            static_calls = counts["static_attention"][0]
            expected_emitted = tuple(a + b for a, b in zip(counts["pt2e_integerization"], counts["seeded"]))
            expected_exported = (expected_emitted[0] + static_calls, expected_emitted[1])
            if emitted != expected_emitted or exported != expected_exported:
                raise ValueError("integer contraction census counts disagree with emissions or exports")
        else:
            verdict = replace(verdict, structure_valid=True)
            if emitted != exported:
                kind = "batched exports" if emitted[1] != exported[1] else "exports"
                raise ValueError(f"legacy integer {kind} must exactly equal emissions")
        if module is not None:
            ranks = [rank for _, _, rank in _contractions(module, func_name=func_name)]
            if exported != (sum(rank == 2 for rank in ranks), sum(rank > 2 for rank in ranks)):
                kind = "batched export" if exported[1] != sum(rank > 2 for rank in ranks) else "export"
                raise ValueError(f"integer {kind} counts disagree with parsed program")
        required = sum(exported) > sum(emitted) or static_calls > 0
        verdict = replace(verdict, counts_agree=True, dynamic_required=required)
        evidence = dynamic_evidence() if callable(dynamic_evidence) else dynamic_evidence
        if required and evidence is None:
            raise ValueError("independent dynamic integerization evidence is required")
        if evidence is not None:
            from merlin.capture.integerization import dynamic_integerization_count

            calls = dynamic_integerization_count(
                evidence, emitted=emitted[0], exported=exported[0], program_sha256=program_sha256,
            )
            if calls != static_calls:
                raise ValueError("dynamic observation disagrees with static-attention census")
            if present and _dynamic_owners(census) != {
                tuple(sorted(row["lineage"])): row["calls"] for row in evidence["dynamic_owners"]
            }:
                raise ValueError("dynamic observation disagrees with census owners")
            verdict = replace(verdict, dynamic_validated=True)
        return verdict
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
        return replace(verdict, reason=str(exc))


def census_export_complete(receipt: dict, module=None, **evidence) -> bool:
    """Compatibility boolean adapter; all admission policy lives in one verdict."""
    return admit_integerization_census(receipt, module, **evidence).accepted


def _receipt_counts(receipt):
    from merlin.capture.integerization import non_contraction_program

    default = 0 if non_contraction_program(receipt) else None
    emitted = (receipt.get("integer_mm_emitted", default), receipt.get("integer_bmm_emitted", 0))
    exported = (receipt.get("exported_integer_mm_count", default), receipt.get("exported_integer_bmm_count", 0))
    if any(type(value) is not int or value < 0 for value in (*emitted, *exported)):
        raise ValueError("invalid integer emission or export counts")
    return emitted, exported


def capture_dynamic_evidence(meta: dict, raw_trace: bytes, *, program_sha256: str) -> dict | None:
    """Project raw dynamic observations after the caller binds trace and program bytes."""
    from hashlib import sha256

    from merlin.capture.integerization import dynamic_integerization_projection

    from .frontend_trace import loads_frontend_trace

    stats = meta.get("quantization_stats")
    if not isinstance(stats, dict) or stats.get("dynamic_contractions") is None:
        return None
    receipt = meta["integerization_receipt"]
    return dynamic_integerization_projection(
        meta, stats, trace=loads_frontend_trace(raw_trace),
        trace_identity={"sha256": sha256(raw_trace).hexdigest(), "bytes": len(raw_trace)},
        program_sha256=program_sha256,
        emitted=receipt["integer_mm_emitted"], exported=receipt["exported_integer_mm_count"],
    )


def read_capture_dynamic_evidence(meta: dict, capture_root, *, program_sha256: str) -> dict | None:
    """Capsule adapter: reopen the local trace and check the worker's recorded digest."""
    from hashlib import sha256

    stats = meta.get("quantization_stats")
    if not isinstance(stats, dict) or stats.get("dynamic_contractions") is None:
        return None
    if capture_root is None:
        raise ValueError("dynamic integerization evidence has no capture directory")
    path = capture_root / "frontend-trace.json"
    identity = meta.get("frontend_trace")
    if path.is_symlink() or not isinstance(identity, dict):
        raise ValueError("dynamic integerization evidence lacks a bound frontend trace")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != identity.get("sha256"):
        raise ValueError("dynamic frontend trace does not match its digest")
    return capture_dynamic_evidence(meta, raw, program_sha256=program_sha256)


def _valid_origins(origins) -> bool:
    """Node IDs stamped by snapshot_exported_program: g:<stage>:<scope>:n<ordinal>."""
    if not isinstance(origins, list) or not origins:
        return False
    for identity in origins:
        if not isinstance(identity, str):
            return False
        parts = identity.split(":")
        if len(parts) != 4:
            return False
        graph, stage, scope, node = parts
        ordinal = node.removeprefix("n")
        if (
            graph != "g" or not stage or not scope or not node.startswith("n")
            or not ordinal.isascii() or not ordinal.isdecimal()
            or (len(ordinal) > 1 and ordinal.startswith("0"))
        ):
            return False
    return True


def _reconcile_census(census, module, *, func_name):
    """Recompute source intersections before trusting any saved ownership claim."""
    calls, bindings = census.get("calls"), census.get("export_bindings")
    if not isinstance(calls, list) or not isinstance(bindings, list):
        raise ValueError("invalid census calls or export bindings")
    identities = {}
    counts = {"pt2e_integerization": [0, 0], "seeded": [0, 0], "static_attention": [0, 0]}
    for row in calls:
        if not isinstance(row, dict):
            raise ValueError("invalid census call")
        identity, rank, provenance = row.get("source_node_id"), row.get("rank"), row.get("provenance")
        if (not isinstance(identity, str) or not identity or identity in identities
                or type(rank) is not int or rank < 2 or provenance not in counts
                or not _valid_origins(row.get("origin_node_ids"))):
            raise ValueError("invalid or duplicate census call identity, rank, producer or origins")
        identities[identity] = row
        counts[provenance][rank > 2] += 1
    seen, ordinals = set(), set()
    for binding in bindings:
        identity = _binding_identity(binding, identities, ordinals)
        if identity in seen or binding["rank"] != identities[identity]["rank"]:
            raise ValueError("duplicate or wrong-rank export binding")
        seen.add(identity)
    if seen != set(identities):
        raise ValueError("integer contraction calls and exports are not a complete union")
    if module is not None and bindings != _export_bindings(module, calls, func_name=func_name):
        raise ValueError("saved export bindings disagree with parsed program")
    static = census.get("static_attention_device_calls")
    if type(static) is not int or static < 0 or counts["static_attention"] != [static, 0]:
        raise ValueError("static-attention count disagrees with census calls")
    return counts


def _binding_identity(binding, identities, ordinals):
    if not isinstance(binding, dict):
        raise ValueError("invalid export binding")
    ordinal, rank = binding.get("ordinal"), binding.get("rank")
    sources, claimed = binding.get("source_node_ids"), binding.get("contraction_source_ids")
    if (type(ordinal) is not int or ordinal < 0 or ordinal in ordinals
            or type(rank) is not int or rank < 2
            or not isinstance(sources, list)
            or any(not isinstance(source, str) or not source for source in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("invalid export ordinal, rank or source identities")
    ordinals.add(ordinal)
    intersection = sorted(set(identities).intersection(sources))
    if not isinstance(claimed, list) or claimed != intersection or len(intersection) != 1:
        raise ValueError("export binding source intersection is not exactly one census call")
    return intersection[0]


def _dynamic_owners(census):
    from collections import Counter

    owners = Counter()
    for row in census["calls"]:
        if row["provenance"] == "static_attention":
            lineage = tuple(sorted(v for v in row["origin_node_ids"] if v.startswith("g:original:")))
            owners[lineage] += 1
    return dict(owners)
