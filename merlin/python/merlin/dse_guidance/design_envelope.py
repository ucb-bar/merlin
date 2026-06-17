"""Design-envelope derivation — requirements and theoretical bounds, NOT calibration.

For DSE of a not-yet-existent accelerator, fitting a cost model to an existing hardware instance
does not transfer to the proposed design. The useful, honest computation is the other direction:
from the recovered workload contract (temporal + numerical) and the real-time deadline, derive the
**requirements** the workload imposes — required compute throughput, memory bandwidth, resident
capacity, command rate — which any candidate design must satisfy. Given an optional candidate
design, also report **roofline feasibility / theoretical lower bounds**.

This module claims NO speedup and NO gap_closure. Every field is labelled with where it came from:

  recovered_from_ir   — counted from the captured IR (MACs, bytes)
  recovered_from_prov_fqn — region role recovered from the capture
  assumed_reference   — K / H / control_rate (architecture reference values)
  derived_requirement — computed from the workload + an assumed deadline
  design_assumption   — feasibility computed against a hypothetical candidate design
  measured            — a real measurement (e.g. dispatch count)
  unavailable         — not computable without an input that does not exist yet
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Element widths in BYTES (fractional for sub-byte formats) — for dtype-scaled capacity.
ELEMENT_BYTES: dict[str, float] = {
    "f32": 4.0, "float32": 4.0, "fp32": 4.0,
    "bf16": 2.0, "f16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "f8": 1.0, "int8": 1.0, "i8": 1.0,
    "int4": 0.5, "i4": 0.5, "fp4": 0.5,
    "fp6": 0.75,
}
# Candidate storage formats we report capacity for (the numerical-contract low-bit ladder).
CAPACITY_FORMATS = ("bf16", "fp8", "int8", "int4", "fp6")

# Evidence labels.
E_IR = "recovered_from_ir"
E_FQN = "recovered_from_prov_fqn"
E_CONFIG = "recovered_from_model_config"   # published architecture constant (model card / registry)
E_ASSUMED = "assumed_reference"
E_DERIVED = "derived_requirement"
E_DESIGN = "design_assumption"
E_MEASURED = "measured"
E_NA = "unavailable"


def _elem_bytes(dtype: str | None) -> float:
    return ELEMENT_BYTES.get(str(dtype or "f32").strip().lower(), 4.0)


@dataclass
class Requirement:
    name: str
    value: float | None
    unit: str
    evidence: str
    note: str = ""


@dataclass
class DesignEnvelope:
    workload: str
    region: str
    K: int
    deadline_s: float | None
    deadline_evidence: str
    captured_dtype: str
    requirements: list[Requirement] = field(default_factory=list)
    capacity_by_dtype_B: dict[str, float] = field(default_factory=dict)
    feasibility: dict | None = None
    candidate_axes: list[dict] = field(default_factory=list)

    def req(self, name: str) -> Requirement | None:
        return next((r for r in self.requirements if r.name == name), None)


def derive(workload: str, *, K: int, deadline_s: float | None, deadline_evidence: str,
           macs_per_step: int, weight_bytes: int, activation_bytes_per_step: int,
           dispatches_per_step: int, dispatches_evidence: str,
           captured_dtype: str, region: str = "repeated_head",
           design: dict | None = None) -> DesignEnvelope:
    """Derive workload requirements (and optional candidate feasibility) for one repeated region.

    All inputs are recovered/assumed facts; this function only does the arithmetic and labels it.
    """
    K = max(int(K), 1)
    macs_per_replan = int(macs_per_step) * K
    weight_reload_per_replan = int(weight_bytes) * K
    avoidable_reload = int(weight_bytes) * max(K - 1, 0)
    dispatches_per_replan = int(dispatches_per_step) * K

    reqs: list[Requirement] = [
        Requirement("macs_per_replan", float(macs_per_replan), "MAC", E_IR,
                    f"macs_per_step x K={K}"),
        Requirement("resident_capacity_required", float(weight_bytes), "B", E_IR,
                    f"action-head weights ({captured_dtype}); resident set to avoid reloads"),
        Requirement("weight_reload_bytes_per_replan", float(weight_reload_per_replan), "B", E_IR,
                    f"naive reload: weight_bytes x K={K}"),
        Requirement("avoidable_weight_reload_bytes", float(avoidable_reload), "B", E_IR,
                    "removed by residency (keep 1 load, avoid K-1)"),
        Requirement("dispatches_per_replan", float(dispatches_per_replan), "dispatch",
                    dispatches_evidence, f"dispatches_per_step x K={K}"),
    ]
    # Deadline-dependent rate requirements (derived from an assumed deadline) — else unavailable.
    if deadline_s and deadline_s > 0:
        reqs += [
            Requirement("required_compute_rate", macs_per_replan / deadline_s, "MAC/s", E_DERIVED,
                        "macs_per_replan / deadline"),
            Requirement("required_weight_bandwidth", weight_reload_per_replan / deadline_s, "B/s",
                        E_DERIVED, "naive reload bandwidth if non-resident"),
            Requirement("required_activation_bandwidth",
                        int(activation_bytes_per_step) * K / deadline_s, "B/s", E_DERIVED,
                        "activation traffic / deadline"),
            Requirement("required_command_rate", dispatches_per_replan / deadline_s, "dispatch/s",
                        E_DERIVED, "host command issue rate to meet deadline"),
        ]
    else:
        for n, u in (("required_compute_rate", "MAC/s"), ("required_weight_bandwidth", "B/s"),
                     ("required_activation_bandwidth", "B/s"), ("required_command_rate",
                                                                "dispatch/s")):
            reqs.append(Requirement(n, None, u, E_NA, "no deadline (control_rate) supplied"))

    # dtype-scaled resident capacity (element count from the captured dtype).
    n_elem = int(weight_bytes) / _elem_bytes(captured_dtype)
    capacity = {fmt: n_elem * ELEMENT_BYTES[fmt] for fmt in CAPACITY_FORMATS}

    env = DesignEnvelope(
        workload=workload, region=region, K=K, deadline_s=deadline_s,
        deadline_evidence=deadline_evidence, captured_dtype=captured_dtype,
        requirements=reqs, capacity_by_dtype_B=capacity,
        candidate_axes=_candidate_axes(K, weight_bytes, avoidable_reload, capacity,
                                       dispatches_per_replan),
    )
    if design:
        env.feasibility = _feasibility(env, design, macs_per_replan, weight_reload_per_replan,
                                       dispatches_per_replan)
    return env


def _candidate_axes(K, weight_bytes, avoidable_reload, capacity, dispatches_per_replan) -> list[dict]:
    """Structural design axes the requirements imply (no magnitude/speedup)."""
    axes = []
    if K > 1:
        for fmt in ("bf16", "int8", "int4"):
            axes.append({
                "axis": f"resident_{fmt}_head_weights",
                "structural_evidence": {"K": K, "weight_bytes": weight_bytes},
                "requirement": {"resident_capacity_B": round(capacity[fmt])},
                "theoretical_effect": {"removes_avoidable_weight_reload_B": avoidable_reload},
                "needed_to_quantify": ["target memory hierarchy + bandwidth model",
                                       "packed-layout support", "quantization accuracy"],
            })
        axes.append({
            "axis": "autonomous_K_loop",
            "structural_evidence": {"K": K, "dispatches_per_replan": dispatches_per_replan},
            "requirement": {"bounded_loop": True, "loop_carried_state": True,
                            "device_dependency_tracking": True},
            "theoretical_effect": {"removes_host_submissions_inside_K_loop": True},
            "needed_to_quantify": ["actual host/device submit latency", "runtime model"],
        })
        axes.append({
            "axis": "command_buffer_per_replan",
            "structural_evidence": {"dispatches_per_replan": dispatches_per_replan},
            "requirement": {"static_dependency_graph": True},
            "theoretical_effect": {"collapses_per_dispatch_host_submits": True},
            "needed_to_quantify": ["measured per-dispatch host submit cost"],
        })
    return axes


def _feasibility(env: DesignEnvelope, design: dict, macs_per_replan: int,
                 weight_reload_per_replan: int, dispatches_per_replan: int) -> dict:
    """Roofline feasibility vs a hypothetical candidate design (all design_assumption)."""
    out: dict = {"design": design.get("name", "candidate"), "evidence": E_DESIGN}
    d = env.deadline_s
    clock = design.get("clock_ghz")
    mpc = design.get("macs_per_cycle")
    bw = design.get("dram_bandwidth_gb_s")
    cap = design.get("local_memory_mb")
    submit_ns = design.get("command_submit_ns")
    dtypes = [s.lower() for s in (design.get("supported_dtypes") or [])]

    if clock and mpc:
        peak = mpc * clock * 1e9
        cb = macs_per_replan / peak
        out["compute_bound_s"] = cb
        out["compute_feasible"] = (cb <= d) if d else None
    if bw:
        mb = weight_reload_per_replan / (bw * 1e9)
        out["memory_bound_s"] = mb
        out["memory_feasible"] = (mb <= d) if d else None
    if "compute_bound_s" in out or "memory_bound_s" in out:
        out["latency_lower_bound_s"] = max(out.get("compute_bound_s", 0.0),
                                           out.get("memory_bound_s", 0.0))
    if cap:
        # Feasible if SOME candidate dtype's resident set fits; report the smallest that fits.
        cap_B = cap * 1e6
        fits = {f: env.capacity_by_dtype_B[f] for f in env.capacity_by_dtype_B
                if env.capacity_by_dtype_B[f] <= cap_B}
        out["local_memory_B"] = cap_B
        out["capacity_feasible"] = bool(fits)
        out["fits_dtypes"] = sorted(fits, key=lambda f: env.capacity_by_dtype_B[f])
    if dtypes:
        out["dtype_feasible"] = {f: (f in dtypes) for f in CAPACITY_FORMATS}
    out["command_feasible"] = (
        (dispatches_per_replan * submit_ns * 1e-9 <= d) if (submit_ns and d) else E_NA)
    return out


def from_recovered(topology, attribution, *, captured_dtype: str = "f32",
                   dispatches_per_step: int | None = None,
                   dispatches_evidence: str = E_DERIVED,
                   design: dict | None = None) -> DesignEnvelope | None:
    """Build an envelope from a recovered topology + attribution (the repeated head).

    Returns None if no repeated head was attributed. ``dispatches_per_step`` should be the
    MEASURED dispatch count when available (pass ``dispatches_evidence='measured'``); otherwise it
    falls back to the head's matmul count, which under-counts real dispatches (~13x, see the
    dispatch-coupling measurement) — labelled accordingly.
    """
    head = attribution.role("repeated_head")
    if head is None or head.attribution_status != "attributed":
        return None
    f = head.facts
    if dispatches_per_step is None:
        dispatches_per_step = int(f.get("matmul_count", 0))
        dispatches_evidence = E_DERIVED   # matmul-count proxy, not the true dispatch count
    deadline_s = (topology.replan_deadline_ms / 1000.0
                  if topology.replan_deadline_ms else None)
    return derive(
        workload=topology.workload, K=topology.K, deadline_s=deadline_s,
        deadline_evidence=E_ASSUMED, macs_per_step=int(f.get("macs_per_invocation", 0)),
        weight_bytes=int(f.get("weight_bytes", 0)),
        activation_bytes_per_step=int(f.get("activation_bytes_per_invocation", 0)),
        dispatches_per_step=int(dispatches_per_step), dispatches_evidence=dispatches_evidence,
        captured_dtype=captured_dtype, design=design)


# ------------------------------------------------------------------ emitters

def to_yaml_obj(env: DesignEnvelope) -> dict:
    return {
        "design_envelope": {
            "workload": env.workload, "region": env.region, "K": env.K,
            "deadline": {"seconds": env.deadline_s, "evidence": env.deadline_evidence},
            "captured_dtype": env.captured_dtype,
            "requirements": [
                {"name": r.name, "value": r.value, "unit": r.unit, "evidence": r.evidence,
                 "note": r.note} for r in env.requirements],
            "resident_capacity_by_dtype_B": {k: round(v) for k, v in env.capacity_by_dtype_B.items()},
            "candidate_axes": env.candidate_axes,
            "feasibility": env.feasibility,
            "status": {"quantitative_speedup": "not_claimed"},
        }
    }


def _human_bytes(b: float | None) -> str:
    if b is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024


def markdown(env: DesignEnvelope) -> str:
    L = [f"# Design envelope — {env.workload} / {env.region}\n"]
    L.append("> Requirements derived from the workload contract — NOT calibrated to any hardware. "
             "No speedup is claimed.\n")
    d = "n/a" if env.deadline_s is None else f"{env.deadline_s*1e3:.1f} ms"
    L.append(f"- K = {env.K}  ·  replan deadline = {d} ({env.deadline_evidence})  ·  "
             f"captured dtype = {env.captured_dtype}\n")
    L.append("## Requirements\n")
    L.append("| requirement | value | unit | evidence |")
    L.append("|-------------|-------|------|----------|")
    for r in env.requirements:
        val = "n/a" if r.value is None else (f"{r.value:.3e}" if r.unit != "B"
                                             else _human_bytes(r.value))
        L.append(f"| {r.name} | {val} | {r.unit} | {r.evidence} |")
    L.append("")
    L.append("## Resident capacity by storage format\n")
    L.append("| format | resident set |")
    L.append("|--------|--------------|")
    for f in CAPACITY_FORMATS:
        L.append(f"| {f} | {_human_bytes(env.capacity_by_dtype_B[f])} |")
    L.append("")
    if env.feasibility:
        L.append("## Feasibility vs candidate design (roofline; design_assumption)\n")
        for k, v in env.feasibility.items():
            L.append(f"- **{k}**: {v}")
        L.append("")
    L.append("## Candidate design axes (structural; quantification gated)\n")
    for a in env.candidate_axes:
        L.append(f"- **{a['axis']}** — needs: {', '.join(a['needed_to_quantify'])}")
    L.append("")
    return "\n".join(L)


_CSV_COLUMNS = ["workload", "region", "requirement", "value", "unit", "evidence", "note"]


def requirements_rows(env: DesignEnvelope) -> list[dict]:
    rows = [{"workload": env.workload, "region": env.region, "requirement": r.name,
             "value": ("" if r.value is None else r.value), "unit": r.unit,
             "evidence": r.evidence, "note": r.note} for r in env.requirements]
    for f in CAPACITY_FORMATS:
        rows.append({"workload": env.workload, "region": env.region,
                     "requirement": f"resident_capacity_{f}", "value": round(env.capacity_by_dtype_B[f]),
                     "unit": "B", "evidence": E_IR, "note": "dtype-scaled from captured element count"})
    return rows


def requirements_csv(envs: list[DesignEnvelope]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    w.writeheader()
    for env in envs:
        for row in requirements_rows(env):
            w.writerow(row)
    return buf.getvalue()
