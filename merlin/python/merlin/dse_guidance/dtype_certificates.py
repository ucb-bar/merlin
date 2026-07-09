"""Accuracy-gated dtype candidate certificates — which low-bit formats are DSE-legal, and why.

The numerical contract surfaces structural low-bit candidates (resident packed weights / native
low-bit compute). This module turns each into a per-format **certificate**: for every candidate
dtype it states the accuracy status (measured-pass where the gate exists, otherwise
blocked-by-missing-accuracy — never assumed), the resident capacity at that format, the compiler
proofs and HW/SW abstractions it would require, and what a DSE engine should explore. The point is
to tell a DSE engine which formats are accuracy-legal design points and which are attractive but
blocked — without claiming any speedup.

Accuracy is the one measurable-now leg: int8 (W8A8) is measured (`accuracy_gate`); fp8/int4/fp4/fp6
are `unavailable` until a gate is run. Capacity is the analytical dtype-scaled resident size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import accuracy_gate as AG
from merlin.dse_guidance.contract import ABSTRACTION_MAP, _NOT_CLAIMED
from merlin.dse_guidance.design_envelope import E_DERIVED, E_MEASURED, E_NA

# The low-bit format ladder we certify (storage-format labels; accuracy gate maps them to families).
_FORMAT_LADDER = ("int8_w8a8", "fp8_w8a8", "int4_weight_only", "int4_weight_fp8_activation")
# Axes that are dtype candidates (only these get format certificates).
_DTYPE_AXES = ("resident_packed_lowbit_weights", "native_lowbit_compute")
# Resident-capacity key per format (the dtype-scaled capacity computed by the design envelope).
_CAP_KEY = {"int8_w8a8": "int8", "fp8_w8a8": "fp8", "int4_weight_only": "int4",
            "int4_weight_fp8_activation": "int4"}
# Compiler proofs each axis would require (from the P4-d certificate spec; not invented per-run).
_COMPILER_PROOFS = {
    "resident_packed_lowbit_weights": [
        "weights invariant across K", "packed layout persists across repeated_head dispatches",
        "scale metadata preserved", "dequant/requant placement known"],
    "native_lowbit_compute": [
        "low-bit operands legal for the matmul", "accumulator width (i32/fp32) sufficient",
        "requant placement known", "scale metadata preserved"],
}


@dataclass
class DtypeCertificate:
    workload: str
    region: str
    candidate_axis: str
    dtype: str
    accuracy_status: str               # measured_pass | measured_fail | unavailable
    accuracy_source: str | None
    dse_status: str                    # accuracy_legal_structural_candidate | blocked_by_*
    resident_capacity_at_format_B: float | None
    required_compiler_proofs: list = field(default_factory=list)
    required_hw_abstractions: str = ""
    what_dse_should_explore: list = field(default_factory=list)
    what_is_not_claimed: list = field(default_factory=lambda: list(_NOT_CLAIMED))


def _status_for(workload: str, fmt: str) -> tuple[str, str | None, str]:
    s = AG.status_for(workload, fmt)
    if s == "pass":
        return "measured_pass", "docs/results.md (W8A8 accuracy table)", \
               "accuracy_legal_structural_candidate"
    if s == "fail":
        return "measured_fail", "docs/results.md (W8A8 accuracy table)", "blocked_by_accuracy"
    return "unavailable", None, "blocked_by_missing_accuracy"


def certificates(nc, env, workload: str, region: str = "repeated_head") -> list[DtypeCertificate]:
    """Per (dtype candidate, format) certificate; accuracy never assumed (unavailable -> blocked)."""
    out: list[DtypeCertificate] = []
    cap = env.capacity_by_dtype_B if env is not None else {}
    for cand in (nc.candidates if nc is not None else []):
        if cand.axis not in _DTYPE_AXES:
            continue
        abstraction = ABSTRACTION_MAP.get(cand.axis, {})
        for fmt in _FORMAT_LADDER:
            acc_status, acc_src, dse_status = _status_for(workload, fmt)
            out.append(DtypeCertificate(
                workload=workload, region=region, candidate_axis=cand.axis, dtype=fmt,
                accuracy_status=acc_status, accuracy_source=acc_src, dse_status=dse_status,
                resident_capacity_at_format_B=cap.get(_CAP_KEY[fmt]),
                required_compiler_proofs=list(_COMPILER_PROOFS.get(cand.axis, [])),
                required_hw_abstractions=abstraction.get("system_abstraction",
                                                         getattr(cand, "required_hw_support", "")),
                what_dse_should_explore=list(abstraction.get("dse_knobs", []))))
    return out


def to_yaml_obj(certs: list[DtypeCertificate], workload: str) -> dict:
    return {"numerical_candidate_certificates": {
        "workload": workload,
        "note": "accuracy is the only measurable-now leg: int8 (W8A8) measured; fp8/int4 unavailable "
                "until a gate is run (NOT assumed). Capacity is dtype-scaled resident size "
                "(analytical). No speedup/cycle/area claimed.",
        "certificates": [
            {"region": c.region, "candidate": c.candidate_axis, "dtype": c.dtype,
             "accuracy": {"status": c.accuracy_status, "source": c.accuracy_source,
                          "evidence": (E_MEASURED if c.accuracy_status.startswith("measured")
                                       else E_NA)},
             "resident_capacity_at_format": {"bytes": c.resident_capacity_at_format_B,
                                             "evidence": (E_DERIVED if c.resident_capacity_at_format_B
                                                          is not None else E_NA)},
             "required_compiler_proofs": c.required_compiler_proofs,
             "required_hw_abstractions": c.required_hw_abstractions,
             "what_dse_should_explore": c.what_dse_should_explore,
             "dse_status": c.dse_status,
             "what_is_not_claimed": c.what_is_not_claimed}
            for c in certs],
    }}


def gated_csv(packages) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for p in packages:
        for c in p.get("certs", []):
            rows.append({
                "workload": c.workload, "region": c.region, "candidate_axis": c.candidate_axis,
                "dtype": c.dtype, "accuracy_status": c.accuracy_status, "dse_status": c.dse_status,
                "resident_capacity_at_format_B": (round(c.resident_capacity_at_format_B)
                                                  if c.resident_capacity_at_format_B is not None
                                                  else "unavailable"),
            })
    return _csv(rows, ["workload", "region", "candidate_axis", "dtype", "accuracy_status",
                       "dse_status", "resident_capacity_at_format_B"])
