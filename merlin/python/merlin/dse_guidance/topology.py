"""VLA runtime topology — the workload contract a flat capture erases.

For a VLA, the DSE unit is not ``model.mlir`` (one flattened forward). It is::

    slow backbone / prefix phase (once per replan)
      -> repeated action-head loop (K denoise / decode steps)
        -> action chunk (H actions)
          -> robot executes the chunk while the next replan computes (async)

This module is the structural front-end: it lifts temporal metadata into a
:class:`VlaRuntimeTopology` with an explicit workload class, phase roles, timing, and the
loop-invariant / loop-carried / boundary-crossing state. Capture-fidelity (:mod:`.fidelity`)
and candidate discovery (:mod:`.candidates`) consume this — *before* any quantitative ranking.

It is built on :class:`merlin.dse_guidance.temporal.TemporalMetadata` (which already carries the
region roles and timing), adding the VLA workload-class taxonomy and the derived state-flow view.
Hand-authored sidecar metadata is the first-class input (Level 0 reconstruction); IR-based
region attribution (Level 1) and loop-preserving capture (Level 2) are future work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import temporal as T

# Workload classes (how the action head is structured) — drives capture-fidelity severity.
CLASS_FLOW_MATCHING = "flow_matching_action_head"      # Class A: iterative denoise/flow head
CLASS_REGRESSION_PARALLEL = "regression_parallel_head"  # Class B: single-shot regression head
CLASS_AUTOREGRESSIVE = "autoregressive_decode"          # Class C: token-by-token decode
CLASS_UNKNOWN = "unknown"

# How badly flattening hurts each class (the loop/rate structure it destroys).
CLASS_SEVERITY = {
    CLASS_FLOW_MATCHING: "high",
    CLASS_AUTOREGRESSIVE: "high",
    CLASS_REGRESSION_PARALLEL: "low",
    CLASS_UNKNOWN: "medium",
}


def classify(class_hint: str | None) -> str:
    """Map a free-form ``class`` hint (or ``family/loop_kind``) onto a workload class."""
    h = (class_hint or "").lower()
    if any(k in h for k in ("flow", "diffusion", "denoise")):
        return CLASS_FLOW_MATCHING
    if any(k in h for k in ("autoregress", "token_decode", "action_token", "decode", "llm")):
        return CLASS_AUTOREGRESSIVE
    if any(k in h for k in ("regression", "parallel", "single_shot")):
        return CLASS_REGRESSION_PARALLEL
    return CLASS_UNKNOWN


@dataclass
class VlaRuntimeTopology:
    workload: str
    workload_class: str
    temporal: T.TemporalMetadata

    # Convenience timing accessors (mirror the temporal model).
    @property
    def K(self) -> int:
        return self.temporal.K

    @property
    def H(self) -> int:
        return self.temporal.H

    @property
    def control_rate_hz(self) -> float:
        return self.temporal.control_rate_hz

    @property
    def replan_deadline_ms(self) -> float:
        return self.temporal.replan_deadline_ms

    def backbone_phases(self) -> list[T.Region]:
        return self.temporal.backbone_regions()

    def head_phases(self) -> list[T.Region]:
        return self.temporal.repeated_head_regions()

    def has_repeated_head(self) -> bool:
        return self.temporal.has_repeated_head()

    def loop_invariant_state(self) -> set[str]:
        return self.temporal.loop_invariant_state()

    def loop_carried_state(self) -> set[str]:
        out: set[str] = set()
        for r in self.temporal.regions:
            out.update(r.loop_carried_state)
        return out

    def state_crossing_boundaries(self) -> list[dict]:
        """State produced by a once-per-replan phase and consumed by the repeated head.

        These are the prefix/KV/feature tensors that live across the K-loop — the residency and
        partition candidates hinge on them.
        """
        produced_by_backbone: dict[str, str] = {}
        for r in self.backbone_phases():
            for s in r.produces:
                produced_by_backbone[s] = r.name
        crossings: list[dict] = []
        for r in self.head_phases():
            for s in r.consumes:
                if s in produced_by_backbone:
                    crossings.append({"state": s, "produced_by": produced_by_backbone[s],
                                      "consumed_by": r.name, "reused_times": self.K})
            # Loop-invariant state listed on the head also crosses into the loop.
            for s in r.loop_invariant_state:
                if s in produced_by_backbone and not any(c["state"] == s for c in crossings):
                    crossings.append({"state": s, "produced_by": produced_by_backbone[s],
                                      "consumed_by": r.name, "reused_times": self.K})
        return crossings

    def deadline_equation(self) -> str:
        return "t_backbone + K * t_head_step <= H / control_rate_hz"


def from_temporal(temporal: T.TemporalMetadata) -> VlaRuntimeTopology:
    return VlaRuntimeTopology(
        workload=temporal.workload,
        workload_class=classify(temporal.cls),
        temporal=temporal,
    )


def load(path) -> VlaRuntimeTopology:
    return from_temporal(T.load(path))


def to_report_dict(topo: VlaRuntimeTopology) -> dict:
    """Schema-friendly mapping for a vla_runtime_topology artifact."""
    return {
        "workload": topo.workload,
        "topology": {
            "class": topo.workload_class,
            "timing": {
                "K": topo.K, "H": topo.H, "control_rate_hz": topo.control_rate_hz,
                "replan_deadline_ms": topo.replan_deadline_ms,
                "equation": topo.deadline_equation(),
            },
            "phases": [
                {"id": r.name, "role": r.role, "cadence": r.cadence,
                 "invocation_count": r.invocation_count or r.loop_trip_count,
                 "loop_invariant_state": r.loop_invariant_state,
                 "loop_carried_state": r.loop_carried_state,
                 "produces": r.produces, "consumes": r.consumes}
                for r in topo.temporal.regions
            ],
            "state_crossing_boundaries": topo.state_crossing_boundaries(),
        },
    }
