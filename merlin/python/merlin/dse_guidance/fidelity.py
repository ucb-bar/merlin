"""Capture fidelity report — does a flat capture preserve the VLA DSE unit?

A model2MLIR capture is a single flattened forward (or single step): per ``docs/results.md`` the
whole-model captures "use each weight once" and emit 0 contract facts. That means the loop / rate
/ deadline structure the VLA actually has is **gone**. Rather than just assert "flat is bad", this
module audits a specific capture against its (recovered) runtime topology and reports exactly what
structure was lost and which DSE axes that loss hides.

The headline claim this enables:

    Merlin can tell when a capture is not a faithful DSE unit.

Severity follows the workload class: flow/diffusion (Class A) and autoregressive decode (Class C)
lose their inner loop and are high-risk; a regression/parallel head (Class B) has no inner loop to
lose and is lower-risk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import topology as TOP
from merlin.dse_guidance.topology import VlaRuntimeTopology

# Each lost structural element hides these DSE axes (see candidates.py for the certificates).
HIDDEN_AXES_BY_STRUCTURE: dict[str, list[str]] = {
    "denoise_loop": ["resident_action_head_weights", "autonomous_K_loop", "command_batching"],
    "token_decode_loop": ["decode_kv_cache_path", "autonomous_K_loop", "command_batching"],
    "kv_cache_growth": ["decode_kv_cache_path", "resident_prefix_kv"],
    "prefix_kv_reuse": ["resident_prefix_kv"],
    "async_backbone_head_overlap": ["backbone_head_partition", "async_chunk_overlap"],
    "action_chunk_horizon": ["async_chunk_overlap"],
    "replan_deadline": ["backbone_head_partition"],
}


@dataclass
class CaptureFidelity:
    workload: str
    workload_class: str
    capture_unit: str
    parsed: bool
    preserved_structure: list[str]
    missing_structure: list[str]
    hidden_axes: list[str]
    severity: str
    reasons: list[str] = field(default_factory=list)
    # P21-S1: recovered-from-IR loop structure (None for a flat capture)
    recovered_structure: list[str] = field(default_factory=list)
    loop_recovery: dict | None = None


def _missing_structure(topo: VlaRuntimeTopology) -> list[str]:
    """The structural elements a flat single-pass capture cannot preserve for this workload."""
    missing: list[str] = []
    is_ar = topo.workload_class == TOP.CLASS_AUTOREGRESSIVE
    if topo.has_repeated_head():
        missing.append("token_decode_loop" if is_ar else "denoise_loop")
    if is_ar:
        missing.append("kv_cache_growth")
    if topo.state_crossing_boundaries():
        missing.append("prefix_kv_reuse")
    if (topo.H or 0) > 1:
        missing.append("action_chunk_horizon")
    if topo.control_rate_hz:
        missing.append("replan_deadline")
    if topo.backbone_phases() and topo.head_phases():
        missing.append("async_backbone_head_overlap")
    # de-dup, preserve order
    seen: set[str] = set()
    return [m for m in missing if not (m in seen or seen.add(m))]


def assess(topo: VlaRuntimeTopology, capture_facts=None, loop_recovery=None) -> CaptureFidelity:
    """Audit a capture (optional :class:`models.CaptureFacts`) against its runtime topology.

    ``loop_recovery`` (a :class:`loop_recovery.LoopRecovery`, optional) is the
    P21-S1 IR evidence from a loop-preserving capture: when present, K, the
    loop-carried state (latent / KV cache) and the repeated region are recovered
    from ``scf.for`` and the corresponding structure flips missing -> recovered.
    """
    preserved: list[str] = []
    parsed = bool(capture_facts and getattr(capture_facts, "parsed", False))
    if parsed:
        preserved = ["matmul_shapes", "dtype_information", "op_mix", "weight_sizes"]

    missing = _missing_structure(topo)

    # --- P21-S1: a loop-preserving capture recovers loop/K/KV/region from IR ---
    recovered: list[str] = []
    lr_present = bool(loop_recovery and getattr(loop_recovery, "present", False))
    if lr_present:
        # K + the repeated region are recovered structurally (scf.for body)
        for tag in ("denoise_loop", "token_decode_loop"):
            if tag in missing:
                missing.remove(tag)
        recovered.append(f"K_loop(recovered K={loop_recovery.K}, source=IR)")
        recovered.append(
            f"repeated_region({loop_recovery.repeated_region_op_count} ops, structural)")
        roles = [c.role for c in loop_recovery.carried_state]
        if any(r in ("latent", "kv_cache", "token_buffer") for r in roles):
            recovered.append(
                f"loop_carried_state({loop_recovery.n_iter_args} iter_args: "
                f"{','.join(sorted(set(roles)))})")
        if "kv_cache" in roles:
            for tag in ("kv_cache_growth", "prefix_kv_reuse"):
                if tag in missing:
                    missing.remove(tag)
            recovered.append(
                f"kv_cache_state(recovered, {loop_recovery.kv_cache_bytes} bytes)")

    hidden: list[str] = []
    for m in missing:
        for ax in HIDDEN_AXES_BY_STRUCTURE.get(m, []):
            if ax not in hidden:
                hidden.append(ax)

    severity = TOP.CLASS_SEVERITY.get(topo.workload_class, "medium")
    # A workload with no repeated head and no deadline loses little — downgrade.
    if not topo.has_repeated_head() and not topo.control_rate_hz:
        severity = "low"

    reasons: list[str] = []
    if "denoise_loop" in missing or "token_decode_loop" in missing:
        reasons.append(
            f"the capture hides the K={topo.K}-step action-head loop, which is exactly the "
            "signal needed to evaluate resident weights, command batching, and autonomous-loop "
            "interfaces")
    if "prefix_kv_reuse" in missing:
        reasons.append("prefix/KV produced once by the backbone and reused across the head is "
                       "flattened to a single use, hiding the resident-prefix/KV axis")
    if "replan_deadline" in missing:
        reasons.append("the real-time replan deadline is not represented, so backbone/head "
                       "partition and async overlap cannot be reasoned about from the capture")
    if not parsed and capture_facts is not None:
        reasons.append("the capture did not parse with stock xDSL, so even op-level structure "
                       "is unavailable (head/backbone attribution impossible)")
    if lr_present:
        severity = "low"
        reasons.append(
            f"loop-preserving capture: K={loop_recovery.K}, the loop-carried state "
            f"({', '.join(sorted({c.role for c in loop_recovery.carried_state}))}) and the "
            f"{loop_recovery.repeated_region_op_count}-op repeated region are recovered directly "
            "from scf.for in the IR — the K-loop / KV-state / region-role caveats are closed "
            "for this capture (no assumed-K, no fqn heuristic)")
    if not missing and not lr_present:
        reasons.append("no multi-rate structure is implied by this workload; the flat capture is "
                       "an adequate DSE unit")

    return CaptureFidelity(
        workload=topo.workload,
        workload_class=topo.workload_class,
        capture_unit="loop_preserving_while_loop" if lr_present else "flat_forward_or_single_step",
        parsed=parsed,
        preserved_structure=preserved,
        missing_structure=missing,
        hidden_axes=hidden,
        severity=severity,
        reasons=reasons,
        recovered_structure=recovered,
        loop_recovery=(loop_recovery.to_dict() if lr_present else None),
    )


def to_report_dict(f: CaptureFidelity) -> dict:
    return {
        "capture_fidelity": {
            "workload": f.workload,
            "workload_class": f.workload_class,
            "capture_unit": f.capture_unit,
            "preserved_structure": f.preserved_structure,
            "recovered_structure": f.recovered_structure,
            "missing_structure": f.missing_structure,
            "hidden_dse_axes": f.hidden_axes,
            "dse_risk": {"severity": f.severity, "reasons": f.reasons},
            "loop_recovery": f.loop_recovery,
        }
    }


def markdown(f: CaptureFidelity) -> str:
    L = [f"# Capture fidelity — {f.workload}\n"]
    L.append(f"- workload class: **{f.workload_class}**")
    L.append(f"- capture unit: `{f.capture_unit}`")
    L.append(f"- DSE risk severity: **{f.severity}**\n")
    L.append("> Merlin can tell when a capture is not a faithful DSE unit.\n")
    L.append("**Preserved by the flat capture:** "
             + (", ".join(f.preserved_structure) if f.preserved_structure
                else "_nothing usable (capture did not parse)_"))
    L.append("")
    if f.recovered_structure:
        L.append("**Recovered from IR (loop-preserving capture):** "
                 + ", ".join(f.recovered_structure))
        L.append("")
    L.append("**Lost to flattening:** "
             + (", ".join(f.missing_structure) if f.missing_structure else "_none_"))
    L.append("")
    if f.hidden_axes:
        L.append("**DSE axes hidden by this loss:** " + ", ".join(f.hidden_axes))
        L.append("")
    if f.reasons:
        L.append("**Why it matters:**")
        for r in f.reasons:
            L.append(f"- {r}")
        L.append("")
    return "\n".join(L)
