"""Check selected trace evidence against the complete declared capsule requirements.

This is a static input check before paid authoring. It runs no compiler or
simulator and never invents an operand layout or turns unavailable evidence
into a hardware capability.
"""

from __future__ import annotations

import json
from pathlib import Path

from merlin.targetgen.rocc.semantics import BoundSemantics


def _observable_fields(program: dict) -> set[str]:
    available = set()
    for step in program.get("fields", []):
        kind = step["as"]
        if kind in {"const", "operand", "raw"}:
            known = True
        elif kind == "equals":
            known = step.get("from") in available
        elif kind == "all_equals":
            known = all(row.get("from") in available for row in step.get("predicates", []))
        else:
            known = type(step.get("offset")) is int and type(step.get("width")) is int
        if known:
            available.add(step["name"])
    return available


def inspect(capsules: list[dict], *, semantics, contract: dict) -> dict:
    """Report whether the original selected decoder can observe required modes.

    Availability is not a claim that a future compiler emits the right commands.
    Those commands and their numerics still need actual execution/trace checks.
    Legacy non-bound backends keep their existing decoder admission path.
    """
    record = {
        "schema": "merlin.phase1.trace_readiness.v1",
        "status": "ready",
        "n_capsules": len(capsules),
        "requirements": {},
        "blockers": [],
        "scope": "selected evidence availability only; no compiler or execution qualification",
    }
    if type(semantics) is not BoundSemantics:
        record["applicability"] = "legacy_decoder"
        return record
    observer = semantics.operand_observer
    isa = observer.isa_constants(semantics._target) if observer is not None else {}
    fields = {name: _observable_fields(program) for name, program in isa.get("OPERAND_FIELDS", {}).items()}
    classes = set(isa.get("FUNCT_CLASS", {}).values()) | set(isa.get("CONFIG_SUBTYPE", {}).values())
    roles = (contract.get("corpus_instruction_roles") or {}).get("by_role") or {}
    requirements = {}
    for capsule in capsules:
        if capsule.get("kind") == "model" or capsule.get("expected", {}).get("refusal"):
            continue
        for mode, required in (capsule.get("expected", {}).get("modes") or {}).items():
            if required:
                requirements.setdefault(mode, []).append(capsule["name"])
    record["requirements"] = {mode: sorted(names) for mode, names in sorted(requirements.items())}
    for mode, names in record["requirements"].items():
        reason = None
        if observer is None:
            if mode != "movement":
                reason = "missing_selected_operand_protocol"
            elif not roles.get("memory") or not roles.get("acc_readout"):
                reason = "missing_selected_movement_roles"
        else:
            required_fields = {
                "i8": {"MVOUT": {"readout"}},
                "relu": {"CONFIG_ST": {"relu"}},
                "acc_scale": {"CONFIG_ST": {"acc_scale"}},
                "k_accumulate": {"PRELOAD": {"accumulate"}},
            }.get(mode, {})
            if any(not needed <= fields.get(label, set()) for label, needed in required_fields.items()):
                reason = "unresolved_selected_operand_fields"
            if (
                reason is None
                and mode in {"relu", "acc_scale", "load_state_resident"}
                and isa.get("CONFIG_SELECTOR") is None
            ):
                reason = "unresolved_selected_config_selector"
            if mode == "resident_reuse":
                loads = {label for label in classes if label in {"MVIN", "MVIN2", "MVIN3"}}
                if not loads or any(
                    not {"spad_addr", "dram", "rows", "cols"} <= fields.get(label, set()) for label in loads
                ):
                    reason = "unresolved_selected_load_identity"
            if mode == "movement" and (not classes & {"MVIN", "MVIN2", "MVIN3"} or "MVOUT" not in classes):
                reason = "missing_selected_movement_roles"
            if mode == "load_state_resident" and observer.load_states is None:
                reason = "missing_selected_load_state_evidence"
        if reason:
            record["blockers"].append({"mode": mode, "reason": reason, "capsules": names})
    if record["blockers"]:
        record["status"] = "blocked"
    return record


def require_ready(
    roots, *, target: str, contract: Path, output: Path, hidden_roots=None, expected_hidden_capsules: int | None = None
) -> dict:
    """Use the same selected backend and capsule discovery as grading, once per admission."""
    from merlin.runtime.backends import base
    from merlin.targetgen.capsule_runner import discover_capsules

    backend = base.get_backend(target)
    capsules = discover_capsules(roots, labels={"public", "dev", "hidden"}, contract=contract)
    hidden = []
    if hidden_roots is not None:
        hidden = discover_capsules(hidden_roots, labels={"hidden"}, contract=contract)
        capsules += hidden
    if expected_hidden_capsules is not None and (
        type(expected_hidden_capsules) is not int or expected_hidden_capsules < 0
        or len(hidden) != expected_hidden_capsules
    ):
        raise ValueError("trace readiness hidden cohort differs from the verified hidden snapshot")
    names = [capsule["name"] for capsule in capsules]
    if len(names) != len(set(names)):
        raise ValueError("trace readiness cohort has duplicate capsule ownership")
    semantics = getattr(backend, "rocc_semantics", None)
    data = backend._data()["contract"] if type(semantics) is BoundSemantics else {}
    record = inspect(capsules, semantics=semantics, contract=data)
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    if record["status"] != "ready":
        modes = ", ".join(row["mode"] for row in record["blockers"])
        raise ValueError(f"Phase 1 trace evidence is unavailable for declared modes: {modes}; see {output}")
    return record
