"""Multi-rate phase model + pipeline-overlap analysis.

This consumes the P6 contract graph (phases with recovered cadences + the data/state edges) and the
P5/P7 operator facts to ask: which phases run at different cadences, and which could overlap? It
emits a per-phase multi-rate model and a set of *candidate* overlaps, each with the buffering /
event / queue abstraction it would require and the evidence that is still missing. It does **not**
schedule, pick buffer counts as a design, or claim a speedup — every overlap is a structural
candidate ("structurally suggests", "candidate overlap", "blocked by missing timing evidence").

Phases / overlaps that need structure the flat capture does not carry (sensor input, control-tick
consumer, safety/postprocess, KV movement) are reported ``unavailable`` / ``unknown`` with the
missing evidence named, never invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from merlin.dse_guidance import resource_hierarchy as RH
from merlin.dse_guidance import topology as TOP
from merlin.dse_guidance.design_envelope import E_DERIVED, E_FQN, E_NA

# P8-a phase classes.
PC_SENSOR = "sensor_or_input"
PC_BACKBONE = "backbone_or_encoder"
PC_PREFIX = "prefix_or_context_builder"
PC_ACTION_HEAD = "repeated_action_head"
PC_DECODER = "decoder_token_step"
PC_FINAL = "final_projection"
PC_CHUNK = "action_chunk_output"
PC_CONTROL = "control_tick_consumer"
PC_SAFETY = "safety_or_postprocess"
PC_UNKNOWN = "unknown"

# phase classes that are runtime/IO and never appear in a model-forward capture (named for
# completeness, reported as not-in-capture rather than dropped).
_NOT_IN_CAPTURE = [PC_SENSOR, PC_CHUNK, PC_CONTROL, PC_SAFETY]

# allowed pipeline/overlap abstraction vocabulary (the verifier checks membership).
ALLOWED_ABSTRACTIONS = {"region_level_dispatch", "async_queue", "event_token",
                        "producer_consumer_queue", "double_buffered_action_chunk",
                        "prefix_state_object", "loop_carried_state_handle", "bounded_loop_command",
                        "resident_weight_object"}


def _phase_class(role: str | None, workload_class: str) -> str:
    if role in ("backbone_once",):
        return PC_BACKBONE
    if role == "prefix_builder":
        return PC_PREFIX
    if role == "repeated_head":
        return PC_DECODER if workload_class == TOP.CLASS_AUTOREGRESSIVE else PC_ACTION_HEAD
    if role == "control_loop":
        return PC_CONTROL
    return PC_UNKNOWN


@dataclass
class Phase:
    workload: str
    phase_id: str
    phase_class: str
    role: str
    cadence: str
    invocations: object
    period_s: object
    deadline_s: object
    state_inputs: list
    state_outputs: list
    resource_class: str
    evidence: str
    missing: list = field(default_factory=list)


@dataclass
class OverlapCandidate:
    source_phase: str
    target_phase: str
    dependency_type: str
    can_overlap: str                 # yes | no | unknown
    why: str
    required_abstractions: list
    required_buffer_count: object    # int >= 1 or "unavailable"
    event_queue_support: list
    missing_evidence: str
    evidence: str


def _dominant_resource_class(shapes, role: str) -> str:
    from collections import Counter
    c = Counter(RH._op_rc(s) for s in shapes if s.region_role == role)
    return c.most_common(1)[0][0] if c else "unavailable"


def phase_model(graph, shapes) -> list[Phase]:
    """Per-phase multi-rate model for one workload (from the graph's phase nodes)."""
    wl = graph.workload
    deadline_s = graph.rate_model.get("replan_deadline_s", {}).get("value")
    phases = []
    for n in graph.nodes:
        if n.kind != "phase":
            continue
        rate = n.rate or {}
        st = n.state_summary or {}
        role = n.region_role or "unknown"
        rc = _dominant_resource_class(shapes, role)
        missing = []
        if rate.get("period_s") is None:
            missing.append("per-phase wall-clock period (runtime measurement)")
        phases.append(Phase(
            workload=wl, phase_id=n.id.split(":")[-1],
            phase_class=_phase_class(role, graph.workload_class), role=role,
            cadence=rate.get("cadence", "unknown"), invocations=rate.get("invocations"),
            period_s=rate.get("period_s"), deadline_s=deadline_s,
            state_inputs=list(st.get("consumes", [])) or list(st.get("loop_invariant_state", [])),
            state_outputs=list(st.get("produces", [])),
            resource_class=rc, evidence=E_FQN, missing=missing))
    return phases


def overlap_candidates(graph, phases, has_control_loop: bool = False) -> list[OverlapCandidate]:
    """Candidate phase overlaps, each evaluated against the recovered structure (no scheduling).

    Two candidates are gated on per-workload recovered structure, not asserted uniformly: the
    backbone/action overlap requires *recovered backbone compute* (a backbone phase with attributed
    ops — absent when the capture is the head only), and the control-tick overlap requires a real
    control loop (a VLA action chunk consumed at a control rate — absent for a plain LLM).
    """
    # backbone overlap is only meaningful if the capture actually contains backbone COMPUTE
    bb = next((p for p in phases if p.phase_class == PC_BACKBONE), None)
    has_backbone_compute = bb is not None and bb.resource_class != "unavailable"
    head = next((p for p in phases if p.phase_class in (PC_ACTION_HEAD, PC_DECODER)), None)
    head_id = head.phase_id if head else "head"
    out: list[OverlapCandidate] = []

    # 1) next replan's backbone overlaps the current action chunk's execution
    out.append(OverlapCandidate(
        source_phase="backbone(next replan)", target_phase="action_execution(current chunk)",
        dependency_type="cross_replan_no_shared_state",
        can_overlap=("yes" if has_backbone_compute else "unknown"),
        why=("backbone of replan n+1 shares no loop-carried state with replan n's head, so the "
             "rates decouple" if has_backbone_compute else
             "no backbone compute is recovered in this capture (the capture is the head only) — "
             "the overlap target is not present"),
        required_abstractions=["double_buffered_action_chunk", "async_queue", "event_token",
                               "prefix_state_object"],
        required_buffer_count=(2 if has_backbone_compute else "unavailable"),
        event_queue_support=["async_queue", "event_token"],
        missing_evidence=("per-phase timing (to size the overlap)" if has_backbone_compute else
                          "backbone compute is not in this capture"),
        evidence=(E_DERIVED if has_backbone_compute else E_NA)))

    # 2) DMA prefetch of resident weights overlaps compute
    out.append(OverlapCandidate(
        source_phase="dma_prefetch(resident weights)", target_phase=head_id,
        dependency_type="read_only_resident_state",
        can_overlap="yes", why="loop-invariant weights are read-only across the K-loop, so the next "
                               "tile can be prefetched while the current step computes",
        required_abstractions=["resident_weight_object", "async_queue"],
        required_buffer_count=2, event_queue_support=["async_queue"],
        missing_evidence="DRAM bandwidth (to size prefetch depth)", evidence=E_FQN))

    # 3) the action-head K-loop represented as a bounded device-side loop
    out.append(OverlapCandidate(
        source_phase=head_id, target_phase=head_id,
        dependency_type="bounded_loop", can_overlap="yes",
        why="the repeated head is a bounded K-loop with loop-invariant weights — representable as a "
            "device-side bounded loop command, removing host re-dispatch per step",
        required_abstractions=["bounded_loop_command", "loop_carried_state_handle",
                               "resident_weight_object"],
        required_buffer_count=1, event_queue_support=["region_level_dispatch"],
        missing_evidence="host dispatch/sync latency (to value the saved re-dispatches)",
        evidence=E_FQN))

    # 4) control-tick consumption decoupled from replan inference (the VLA async loop) — only for
    #    workloads that actually have a real-time control loop (a VLA action chunk + control rate)
    out.append(OverlapCandidate(
        source_phase="control_tick_consumer", target_phase="replan_inference(next)",
        dependency_type="different_cadence_producer_consumer",
        can_overlap=("yes" if has_control_loop else "unknown"),
        why=("the control loop consumes H actions at the control rate while the next replan "
             "computes — different cadences, no data dependency" if has_control_loop else
             "this workload has no real-time control loop (not a VLA action head) — no control "
             "tick to decouple"),
        required_abstractions=["double_buffered_action_chunk", "producer_consumer_queue",
                               "event_token"],
        required_buffer_count=(2 if has_control_loop else "unavailable"),
        event_queue_support=["producer_consumer_queue", "event_token"],
        missing_evidence=("the control-tick consumer is a runtime phase not in the model-forward "
                          "capture; its timing is unavailable" if has_control_loop else
                          "no control loop in this workload's architecture"),
        evidence=(E_DERIVED if has_control_loop else E_NA)))

    # 5) decode token steps pipelined with KV movement — needs attention structure (not visible)
    out.append(OverlapCandidate(
        source_phase="decode_token_step", target_phase="kv_cache_movement",
        dependency_type="kv_dependency", can_overlap="unknown",
        why="attention / KV structure is lowered into the matmul projections and not recoverable",
        required_abstractions=["producer_consumer_queue", "loop_carried_state_handle"],
        required_buffer_count="unavailable", event_queue_support=[],
        missing_evidence="attention/KV structure (a loop-preserving capture would expose it)",
        evidence=E_NA))
    return out


# --------------------------------------------------------------------------- emitters

def pipeline_envelope_yaml(phase_by_workload: dict) -> dict:
    return {"pipeline_envelope": {
        "note": "multi-rate phase model from the recovered contract graph. Cadences/roles are "
                "recovered_from_prov_fqn; per-phase wall-clock periods are a runtime measurement "
                "(unavailable). Phase classes that are runtime/IO (sensor, control-tick consumer, "
                "action-chunk output, safety/postprocess) never appear in a model-forward capture "
                "and are listed below, not invented. No speedup/scheduling claim.",
        "phase_classes_not_in_capture": _NOT_IN_CAPTURE,
        "workloads": [
            {"workload": wl,
             "phases": [
                 {"phase": p.phase_id, "phase_class": p.phase_class, "cadence": p.cadence,
                  "invocations": p.invocations,
                  "period_s": p.period_s if p.period_s is not None else "unavailable",
                  "deadline_s": p.deadline_s if p.deadline_s is not None else "unavailable",
                  "state_inputs": p.state_inputs, "state_outputs": p.state_outputs,
                  "candidate_resource_class": p.resource_class, "evidence": p.evidence,
                  "missing": p.missing}
                 for p in phases]}
            for wl, phases in phase_by_workload.items()]}}


def pipeline_stage_csv(phase_by_workload: dict) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for wl, phases in phase_by_workload.items():
        for p in phases:
            rows.append({"workload": wl, "phase": p.phase_id, "phase_class": p.phase_class,
                         "cadence": p.cadence, "invocations": p.invocations,
                         "period_s": p.period_s if p.period_s is not None else "unavailable",
                         "candidate_resource_class": p.resource_class, "evidence": p.evidence})
    return _csv(rows, ["workload", "phase", "phase_class", "cadence", "invocations", "period_s",
                       "candidate_resource_class", "evidence"])


def pipeline_candidates_yaml(overlap_by_workload: dict) -> dict:
    return {"pipeline_candidates": {
        "note": "candidate phase overlaps. can_overlap is a STRUCTURAL judgement (yes/no/unknown) "
                "from the recovered dependencies + cadences — NOT a schedule and NOT a speedup. "
                "Each carries the buffering/event/queue abstraction it would require and the "
                "evidence still missing. 'yes' means structurally permitted, not realized.",
        "workloads": [
            {"workload": wl,
             "candidates": [
                 {"source_phase": c.source_phase, "target_phase": c.target_phase,
                  "dependency_type": c.dependency_type, "can_overlap": c.can_overlap,
                  "why": c.why, "required_abstractions": c.required_abstractions,
                  "required_buffer_count": c.required_buffer_count,
                  "event_queue_support": c.event_queue_support,
                  "missing_evidence": c.missing_evidence,
                  "what_is_not_claimed": "no speedup, no schedule, no deadline met; structural "
                                         "overlap candidate only",
                  "evidence": c.evidence}
                 for c in cands]}
            for wl, cands in overlap_by_workload.items()]}}


def buffering_requirement_csv(overlap_by_workload: dict) -> str:
    from merlin.dse_guidance.corpus import _csv
    rows = []
    for wl, cands in overlap_by_workload.items():
        for c in cands:
            rows.append({
                "workload": wl, "source_phase": c.source_phase, "target_phase": c.target_phase,
                "can_overlap": c.can_overlap,
                "required_buffer_count": (c.required_buffer_count
                                          if c.required_buffer_count != "unavailable"
                                          else "unavailable"),
                "event_queue_support": "; ".join(c.event_queue_support) or "—",
                "required_abstractions": "; ".join(c.required_abstractions)})
    return _csv(rows, ["workload", "source_phase", "target_phase", "can_overlap",
                       "required_buffer_count", "event_queue_support", "required_abstractions"])


def overlap_report_md(phase_by_workload: dict, overlap_by_workload: dict) -> str:
    from collections import Counter
    L = ["# Pipeline overlap opportunities\n",
         "> Which phases run at different cadences and could structurally overlap. **Structural "
         "candidates only** — `can_overlap: yes` means *structurally permitted by the recovered "
         "dependencies and rates*, not scheduled and not a speedup.\n"]
    L.append("## Multi-rate structure\n")
    L.append("| workload | phases | cadences |")
    L.append("|---|---|---|")
    for wl, phases in phase_by_workload.items():
        cads = ", ".join(sorted({p.cadence for p in phases}))
        L.append(f"| {wl} | {len(phases)} | {cads} |")
    L.append("")
    L.append("## Candidate overlaps (common across workloads)\n")
    L.append("| candidate | can_overlap | required abstractions | buffers |")
    L.append("|---|---|---|---|")
    any_cands = next(iter(overlap_by_workload.values()), [])
    for c in any_cands:
        L.append(f"| {c.source_phase} ‖ {c.target_phase} | {c.can_overlap} | "
                 f"{', '.join(c.required_abstractions)} | {c.required_buffer_count} |")
    L.append("")
    # which abstractions are repeatedly required
    req = Counter()
    for cands in overlap_by_workload.values():
        for c in cands:
            if c.can_overlap == "yes":
                req.update(c.required_abstractions)
    L.append("## Abstractions repeatedly required for overlap\n")
    for ab, n in req.most_common():
        L.append(f"- `{ab}` — required by {n} candidate overlaps")
    L.append("\n## Findings\n")
    L.append("- **Backbone/head and control/inference decouple by cadence** — the once-per-replan "
             "backbone and the K-times head run at different rates, and the control loop consumes "
             "actions at yet another rate; these are candidate overlaps a future DSE should consider.")
    L.append("- **The K-loop is representable as a bounded device-side loop** (loop-invariant "
             "weights + bounded trip count) — `requires bounded_loop_command`.")
    L.append("- **`double_buffered_action_chunk` / `async_queue` recur** as the abstractions overlap "
             "needs — `requires event/queue abstraction`.")
    L.append("- **Blocked by missing timing/dependency evidence:** per-phase wall-clock periods, "
             "host dispatch/sync latency, DRAM bandwidth, and (for KV pipelining) attention "
             "structure are all `unavailable` — they block quantitative scheduling, not the "
             "structural overlap candidates.")
    L.append("\n**Caveat (structural, not realized):** these are candidate overlaps the structure "
             "permits. **No speedup**, schedule, or deadline-met claim is made.\n")
    return "\n".join(L)
