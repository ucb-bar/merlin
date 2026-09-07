"""Dialect-neutral operation capability contracts.

An operation is identified by ``{domain, dialect, operation}`` (for example an MLIR dialect
operation or a machine instruction).  This is deliberately broader than an ISA table: RVV, host
dialects, and out-of-tree target dialects can all use the same envelope.  Target-specific semantics
are data nested under ``semantics`` rather than branches in consumers.

The first adapter records operations discovered from a self-hosted ISA taxonomy.  A scalar memory
operation is one semantic facet of that generic record.  ISA presence establishes only that an
operation is declared, so its initial status is ``unknown``; behavioral observations may refine it
to ``supported`` or ``unsupported`` while preserving both sources of evidence.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any


_STATUSES = frozenset({"unknown", "supported", "unsupported"})


def operation_contract_for_target(
        target: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the common operation registry for ``target`` and its capability manifest.

    This is the public seam for preflight and prompt consumers that start from a loaded target
    contract.  It reuses the existing dialect-plan and ISA-taxonomy adapters, then applies any
    observations already carried by the contract.  Callers therefore do not need to know which
    instruction dialect, compute-unit kind, or target family they are inspecting.
    """
    from .capability_manifests import dialect_plan_from_manifest
    from .isa_taxonomy import taxonomy_for_target

    plan = dialect_plan_from_manifest(dict(manifest))
    return merge_operation_contracts(
        derive_dialect_operation_contract(plan),
        derive_isa_operation_contract(
            taxonomy_for_target(str(target)), dialect=str(plan["dialect_name"])),
        manifest.get("operation_capabilities") or {},
    )


def _evidence(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("operation capability evidence must be a mapping")
    kind = str(value.get("kind") or "").strip()
    detail = str(value.get("detail") or "").strip()
    if not kind or not detail:
        raise ValueError("operation capability evidence requires non-empty kind and detail")
    return {"kind": kind, "detail": detail}


def _identity(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("operation capability record must be a mapping")
    domain = str(value.get("domain") or "").strip()
    dialect = str(value.get("dialect") or "").strip()
    operation = str(value.get("operation") or "").strip()
    if not domain or not dialect or not operation:
        raise ValueError("operation identity requires non-empty domain, dialect, and operation")
    return domain, dialect, operation


def derive_isa_operation_contract(
        taxonomy: Mapping[str, Any], *, dialect: str) -> dict[str, Any]:
    """Adapt a discovered ISA taxonomy to the generic operation-capability contract.

    No mnemonic or target table is consulted.  Every operation comes from ``by_mnemonic``; semantic
    facets, when available, were derived from that operation's own implementation.
    """
    dialect = str(dialect).strip()
    if not dialect:
        raise ValueError("operation dialect must be non-empty")
    operations: list[dict[str, Any]] = []
    for mnemonic, entry in sorted((taxonomy.get("by_mnemonic") or {}).items()):
        if not isinstance(entry, Mapping):
            continue
        scalar_memory = entry.get("scalar_memory")
        semantics: dict[str, Any]
        effects: list[str] = []
        detail = str(entry.get("class") or mnemonic)
        if isinstance(scalar_memory, Mapping):
            direction = str(scalar_memory.get("direction") or "")
            if direction not in ("load", "store"):
                raise ValueError(f"{mnemonic}: scalar-memory direction must be load or store")
            address_space = str(scalar_memory.get("address_space") or "").strip()
            if not address_space:
                raise ValueError(f"{mnemonic}: scalar-memory address_space is missing")
            semantics = {
                "kind": "memory", "scope": "scalar", "direction": direction,
                "address_space": address_space,
            }
            width = scalar_memory.get("width_bytes")
            if width is not None:
                if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                    raise ValueError(f"{mnemonic}: scalar-memory width_bytes must be positive")
                semantics["width_bytes"] = width
            address_unit = scalar_memory.get("address_unit_bytes")
            if address_unit is not None:
                if isinstance(address_unit, bool) or not isinstance(address_unit, int) \
                        or address_unit <= 0:
                    raise ValueError(
                        f"{mnemonic}: scalar-memory address_unit_bytes must be positive")
                semantics["address_unit_bytes"] = address_unit
            addressing = scalar_memory.get("addressing")
            if isinstance(addressing, Mapping) and addressing:
                semantics["addressing"] = copy.deepcopy(dict(addressing))
            effects = ["movement"]  # shared Phase-2 optimization-effect vocabulary
            detail = str(scalar_memory.get("effect_method") or detail)
        else:
            # A role is already a target-agnostic semantic classification derived from typed operands.
            # Preserve it without translating it into a guessed effect.
            semantics = {"kind": str(entry.get("role") or "unknown")}
        operations.append({
            "domain": "instruction",
            "dialect": dialect,
            "operation": str(mnemonic),
            "effects": effects,
            "semantics": semantics,
            "status": "unknown",
            "evidence": [{"kind": "isa_definition", "detail": detail}],
        })
    return {"version": 1, "operations": operations}


def derive_dialect_operation_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt an existing target dialect plan to the common operation contract.

    The plan is already the canonical compute-unit -> dialect projection.  Reusing its ``dialect_name``
    and ``ops`` keeps this registry aligned with generated xDSL/MLIR targets; it does not re-derive a
    second set of operation names.
    """
    dialect = str(plan.get("dialect_name") or "").strip()
    if not dialect:
        raise ValueError("dialect operation contract requires dialect_plan.dialect_name")
    operations = []
    for raw in plan.get("ops") or ():
        name = str(raw.get("name") or "").strip() if isinstance(raw, Mapping) else ""
        if not name:
            raise ValueError("dialect operation contract contains an op without a name")
        semantics = raw.get("semantics")
        if semantics is not None and not isinstance(semantics, Mapping):
            raise ValueError(f"dialect operation {name!r} semantics must be a mapping")
        effects = raw.get("effects") or ()
        if not isinstance(effects, (list, tuple)) or any(not isinstance(item, str) for item in effects):
            raise ValueError(f"dialect operation {name!r} effects must be a string sequence")
        operations.append({
            "domain": "dialect", "dialect": dialect, "operation": name,
            "effects": list(effects),
            "semantics": copy.deepcopy(dict(semantics or {"kind": "compute"})),
            "status": "unknown",
            "evidence": [{"kind": "dialect_plan", "detail": "derived from compute_units.ops"}],
        })
    return {"version": 1, "operations": operations}


def merge_operation_contracts(*contracts: Mapping[str, Any]) -> dict[str, Any]:
    """Join operation declarations from existing domains into one stable registry.

    Domain is part of identity, so ``dialect/foo`` and ``instruction/foo`` remain distinct stages of
    one lowering.  A repeated identity must agree on its semantic payload; disagreement is surfaced
    instead of choosing whichever declaration happened to be merged last.
    """
    operations: list[dict[str, Any]] = []
    by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    for contract in contracts:
        if not isinstance(contract, Mapping):
            raise ValueError("operation capability contract must be a mapping")
        for raw in contract.get("operations") or ():
            if not isinstance(raw, Mapping):
                raise ValueError("operation capability entries must be mappings")
            op = copy.deepcopy(dict(raw))
            ident = _identity(op)
            if ident in by_id:
                prior = by_id[ident]
                comparable = ("effects", "semantics")
                if any(prior.get(key) != op.get(key) for key in comparable):
                    raise ValueError(f"conflicting declarations for operation {ident!r}")
                for evidence in op.get("evidence") or ():
                    item = _evidence(evidence)
                    if item not in prior.setdefault("evidence", []):
                        prior["evidence"].append(item)
                continue
            status = str(op.get("status") or "unknown")
            if status not in _STATUSES:
                raise ValueError(f"operation status must be one of {sorted(_STATUSES)}, got {status!r}")
            op["status"] = status
            op["evidence"] = [_evidence(item) for item in (op.get("evidence") or ())]
            by_id[ident] = op
            operations.append(op)
        raw_observations = contract.get("observations") or ()
        if not isinstance(raw_observations, (list, tuple)):
            raise ValueError("operation capability observations must be a sequence")
        observations.extend(copy.deepcopy(list(raw_observations)))
    result: dict[str, Any] = {"version": 1, "operations": operations}
    if observations:
        result["observations"] = observations
    return merge_operation_observations(result, observations) if observations else result


def merge_operation_observations(
        declared: Mapping[str, Any], observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a capability contract refined by measured operation observations.

    Observations must identify an already-declared operation.  This fail-closed join prevents a typo
    or a probe aimed at the wrong dialect from manufacturing a capability or prohibition.
    """
    result = copy.deepcopy(dict(declared))
    operations = result.get("operations")
    if not isinstance(operations, list):
        raise ValueError("operation capability contract operations must be a list")
    by_id = {_identity(op): op for op in operations if isinstance(op, dict)}
    for observation in observations:
        ident = _identity(observation)
        if ident not in by_id:
            raise ValueError(f"operation observation {ident!r} is not declared")
        status = str(observation.get("status") or "").strip()
        if status not in _STATUSES:
            raise ValueError(f"operation status must be one of {sorted(_STATUSES)}, got {status!r}")
        op = by_id[ident]
        op["status"] = status
        if observation.get("evidence") is not None:
            op.setdefault("evidence", []).append(_evidence(observation["evidence"]))
    return result


def _addressing_text(addressing: Any, address_unit_bytes: Any = None) -> str:
    if not isinstance(addressing, Mapping):
        return "declared addressing"
    if addressing.get("mode") == "base_plus_immediate":
        bits = addressing.get("offset_bits")
        signed = "signed " if addressing.get("offset_signed", True) else "unsigned "
        width = f"{bits}-bit " if isinstance(bits, int) and bits > 0 else ""
        scale = addressing.get("offset_scale", 1)
        scaled = ""
        if isinstance(scale, int) and not isinstance(scale, bool) and scale > 1:
            scaled = f" × {scale} address units"
            if isinstance(address_unit_bytes, int) and not isinstance(address_unit_bytes, bool) \
                    and address_unit_bytes > 0:
                scaled += f" ({scale * address_unit_bytes} bytes)"
        return f"base + {signed}{width}immediate{scaled}"
    return str(addressing.get("mode") or "declared addressing").replace("_", " ")


def _memory_prompt_block(contract: Mapping[str, Any], *, scope: str | None) -> str:
    capability = contract.get("operation_capabilities") or {}
    operations = capability.get("operations") if isinstance(capability, Mapping) else None
    if not isinstance(operations, list):
        return ""
    memory = [op for op in operations if isinstance(op, Mapping)
              and (op.get("semantics") or {}).get("kind") == "memory"
              and (scope is None or (op.get("semantics") or {}).get("scope") == scope)]
    if not memory:
        return ""
    lines = ["## Memory operation contract (derived, behaviorally refinable)"]
    has_unsupported = False
    for op in memory:
        _domain, dialect, operation = _identity(op)
        sem = op["semantics"]
        status = str(op.get("status") or "unknown")
        if status not in _STATUSES:
            status = "unknown"
        has_unsupported |= status == "unsupported"
        state = "UNSUPPORTED" if status == "unsupported" else (
            "verified supported" if status == "supported" else "declared, unverified")
        width = sem.get("width_bytes")
        width_text = f", {width} byte" + ("s" if width != 1 else "") if width else ""
        address_unit = sem.get("address_unit_bytes")
        unit_text = (f", address unit: {address_unit} byte"
                     + ("s" if address_unit != 1 else "")) if address_unit else ""
        lines.append(
            f"- `{dialect}.{operation}`: {sem.get('direction', 'memory')} "
            f"{sem.get('address_space', 'memory')}{width_text}{unit_text}, "
            f"{_addressing_text(sem.get('addressing'), address_unit)} — **{state}**")
    if has_unsupported:
        lines.append("The backend must not emit operations marked **UNSUPPORTED**; choose a declared "
                     "supported path or decline the affected lowering explicitly.")
    else:
        lines.append("Treat unverified operations as hypotheses until the behavioral preflight confirms "
                     "them; do not infer support merely because an opcode decodes.")
    return "\n".join(lines) + "\n\n"


def memory_operation_prompt_block(contract: Mapping[str, Any]) -> str:
    """Render every memory facet, irrespective of scalar/vector/target dialect scope."""
    return _memory_prompt_block(contract, scope=None)


def scalar_memory_prompt_block(contract: Mapping[str, Any]) -> str:
    """Backward-compatible focused rendering of scalar-memory facets."""
    return _memory_prompt_block(contract, scope="scalar")
