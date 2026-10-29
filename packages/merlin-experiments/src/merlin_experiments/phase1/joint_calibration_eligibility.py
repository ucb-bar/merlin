"""Shared correctness prerequisite for calibration registration and replay.

This is an admission screen, not provenance or timing authority. Existing native
product reopening, selected source checks and independent physical gates remain.
"""
from __future__ import annotations


def eligible(result, capsule=None, *, compiler_sha256=None):
    from merlin.targetgen.trace_check import requirements_complete
    from .feedback.joint_curriculum import _structural_prerequisites
    from .feedback.joint_physical_correctness import _split_l3_prerequisites

    if not isinstance(result, dict):
        return False
    numeric, tiers, trace = (result.get(k) for k in ("numeric", "tiers", "trace_check"))
    if not all(isinstance(v, dict) for v in (numeric, tiers, trace)):
        return False
    l2 = tiers.get("L2")
    if (numeric.get("status") != "pass" or not isinstance(l2, dict)
            or l2.get("status") != "pass" or l2.get("engine") != "spike"
            or not requirements_complete(trace) or trace.get("violations")
            or trace.get("unavailable_requirements")
            or not all(isinstance(v, dict) for v in tiers.values())
            or any(v.get("status") == "fail" for v in tiers.values())
            or not _structural_prerequisites(result)
            or not (result.get("compiler_source") or result.get("candidate_native_execution"))):
        return False
    # Named source-placement audit failures must not hide behind an unrelated
    # numerical PASS. Descriptive placement counts without a status are not a
    # new authority; ordinary source/trace owners still establish applicability.
    for owner in (result, result.get("model_execution_check") or {}):
        if not isinstance(owner, dict):
            return False
        for key in ("placement", "source_placement", "candidate_source_placement"):
            if key not in owner or owner[key] is None:
                continue
            audit = owner[key]
            if (not isinstance(audit, dict) or audit.get("violations") or audit.get("failure")
                    or (audit.get("status") is not None
                        and audit["status"] not in {"pass", "clean", "verified", "admitted", "satisfied"})):
                return False
    source = result.get("compiler_source") or {}
    if not isinstance(source, dict):
        return False
    if compiler_sha256 is not None and source and source.get("candidate_sha256") != compiler_sha256:
        return False
    pending_l3 = False
    if result.get("failure") or result.get("status") != "pass":
        # Keep the exact existing missing-L3 shape, including its original
        # unavailable reason, mandatory marker, source and complete output check.
        pending_l3 = _split_l3_prerequisites(result, compiler_sha256 or source.get("candidate_sha256"))
        if not pending_l3:
            return False
    required = {name for name, tier in tiers.items() if tier.get("mandatory") is True}
    if capsule is not None:
        if not isinstance(capsule, dict) or capsule.get("kind") == "model":
            return False
        declared = capsule.get("required_oracle_tiers", [])
        if (not isinstance(declared, list) or any(type(t) is not str for t in declared)
                or len(declared) != len(set(declared))):
            return False
        required.update(declared)
        if pending_l3 and "L3" not in declared:
            return False
    for tier in required:
        if tier == "L3" and pending_l3:
            continue
        observed = tiers.get(tier, {})
        if observed.get("status") == "pass":
            continue
        if tier in {"L0", "L1"} and observed.get("not_applicable") is True:
            # Paired whole-program nonapplicability was independently checked
            # by _structural_prerequisites above, not inferred from this flag.
            continue
        return False
    return True


def require(result, capsule, *, compiler_sha256=None):
    if not eligible(result, capsule, compiler_sha256=compiler_sha256):
        raise ValueError("calibration requires structurally admitted complete Spike numerics; "
                         "only explicit pending required L3 is allowed")
