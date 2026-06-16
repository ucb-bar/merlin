"""Orchestration core: one workload -> the full set of DSE-guidance artifacts.

Shared by the single-workload CLI and the exhaustive study so the two never diverge. Given a
region (optional), temporal metadata, a baseline cost, and optional measured CPU coupling, it
builds the flat and multi-rate representations, runs the axis triage under both, ingests the
CPU-coupling result (or records that it was unavailable), and produces the calibration anchor.

:func:`write_artifacts` lays the results out under an output directory exactly as the plan's
acceptance criteria specify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.artifacts import Artifact, yaml_artifact
from merlin.dse_guidance import calibration, report
from merlin.dse_guidance.aet_ingest import UNAVAILABLE_MESSAGE, CpuCoupling
from merlin.dse_guidance.baseline_cost import BaselineCost
from merlin.dse_guidance.representation import Representation, build_representations, to_report_dict
from merlin.dse_guidance.temporal import TemporalMetadata
from merlin.dse_guidance.triage import triage
from merlin.dse_guidance import candidates as CAND
from merlin.dse_guidance import fidelity as FID
from merlin.dse_guidance import topology as TOP


@dataclass
class GuidanceResult:
    workload: str
    flat: Representation
    multirate: Representation
    # Structural front-end (valid without calibration).
    topology: TOP.VlaRuntimeTopology
    fidelity: FID.CaptureFidelity
    candidate_axes: list
    # Quantitative back-end (only when a baseline cost is supplied; uncalibrated by default).
    triage_flat: dict | None
    triage_multirate: dict | None
    baseline: BaselineCost | None
    coupling: CpuCoupling | None
    coupling_per_replan: dict | None
    calibration_rows: list[dict]
    deadline: dict | None
    is_negative_control: bool
    warnings: list[str] = field(default_factory=list)
    attribution: object | None = None    # Level-1 RegionAttribution, when a capture is available


def _deadline_feasibility(multirate: Representation, baseline: BaselineCost) -> dict:
    """Surface the timing budget t_backbone + K*t_head_step <= H/control_rate_hz.

    We do not have the backbone/head split unless a region provides it, so we report the
    coarse fact we can ground: whether the baseline total fits the replan deadline (only
    meaningful when both are in ms).
    """
    out: dict = {
        "equation": "t_backbone + K * t_head_step <= H / control_rate_hz",
        "replan_deadline_ms": multirate.replan_deadline_ms,
        "baseline_total": baseline.baseline_total_ms,
        "baseline_unit": baseline.unit,
    }
    if multirate.replan_deadline_ms is not None and baseline.unit == "ms":
        slack = multirate.replan_deadline_ms - baseline.baseline_total_ms
        out["slack_ms"] = round(slack, 4)
        out["deadline_met"] = slack >= 0
        out["evidence_type"] = "structural_bound"
    else:
        out["deadline_met"] = None
        out["note"] = ("deadline feasibility not evaluated (baseline unit is not ms or no "
                       "deadline available)")
    return out


def run_guidance(temporal: TemporalMetadata, baseline: BaselineCost | None = None,
                 region: dict | None = None,
                 coupling: CpuCoupling | None = None,
                 overrides: dict | None = None,
                 capture_facts=None,
                 attribution=None) -> GuidanceResult:
    """Run the guidance pipeline for one workload.

    The structural front-end (topology, capture fidelity, candidate axes) is always produced —
    it is valid without calibration. The quantitative triage runs only when a ``baseline`` cost
    is supplied, and is uncalibrated unless its components are measured/calibrated.
    """
    reps = build_representations(temporal, region, overrides=overrides)
    flat, multirate = reps["flat"], reps["multirate"]
    workload = (baseline.workload if baseline is not None else temporal.workload)

    # --- structural front-end (no calibration needed) ---
    topo = TOP.from_temporal(temporal)
    capture_fidelity = FID.assess(topo, capture_facts)
    candidate_axes = CAND.detect(topo, capture_facts, attribution=attribution)

    warnings = list(temporal.warnings)

    # --- quantitative back-end (only with a baseline) ---
    triage_flat = triage_multirate = None
    coupling_per_replan = None
    calib: list[dict] = []
    deadline = None
    if baseline is not None:
        warnings += list(baseline.warnings)
        if coupling is not None:
            coupling_per_replan = coupling.per_replan(
                dispatches_per_step=multirate.dispatches_per_step, K=multirate.K,
                num_regions=max(len(temporal.regions), 1))
        triage_flat = triage(flat, baseline, coupling_per_replan)
        triage_multirate = triage(multirate, baseline, coupling_per_replan)
        calib = calibration.anchor_rows(
            baseline.workload, coupling,
            dispatches_per_step=multirate.dispatches_per_step, K=multirate.K,
            num_regions=max(len(temporal.regions), 1))
        deadline = _deadline_feasibility(multirate, baseline)

    return GuidanceResult(
        workload=workload,
        flat=flat, multirate=multirate,
        topology=topo, fidelity=capture_fidelity, candidate_axes=candidate_axes,
        triage_flat=triage_flat, triage_multirate=triage_multirate,
        baseline=baseline, coupling=coupling, coupling_per_replan=coupling_per_replan,
        calibration_rows=calib,
        deadline=deadline,
        is_negative_control=not multirate.facts.get("has_k_loop", False),
        warnings=warnings,
        attribution=attribution,
    )


def write_artifacts(result: GuidanceResult, out_dir: str | Path) -> list[Path]:
    """Write all guidance artifacts (and optional figures) under ``out_dir``.

    Structural artifacts (topology, capture fidelity, candidate axes, flat-vs-multirate) are
    always written. Quantitative artifacts (axis triage, bottleneck, deadline, calibration) are
    written only when a baseline cost was supplied, and are labelled uncalibrated.
    """
    out = Path(out_dir)
    # --- structural front-end (always) ---
    artifacts: list[Artifact] = [
        yaml_artifact("vla_runtime_topology.yaml", TOP.to_report_dict(result.topology),
                      header=f"vla_runtime_topology: {result.workload}"),
        Artifact("capture_fidelity_report.md", FID.markdown(result.fidelity)),
        yaml_artifact("capture_fidelity.yaml", FID.to_report_dict(result.fidelity)),
        Artifact("dse_candidate_axes.md", CAND.markdown(result.topology, result.candidate_axes)),
        yaml_artifact("dse_candidate_axes.yaml", CAND.to_yaml_obj(result.candidate_axes),
                      header=f"dse_candidate_axes: {result.workload} (structural; no calibration)"),
        yaml_artifact("flat_report.yaml", to_report_dict(result.flat),
                      header=f"flat_report: {result.workload}"),
        yaml_artifact("multirate_report.yaml", to_report_dict(result.multirate),
                      header=f"multirate_report: {result.workload}"),
        Artifact("flat_vs_multirate_diff.csv",
                 report.flat_vs_multirate_csv(result.flat, result.multirate)),
    ]
    # Level-1 region attribution (when a real capture was available).
    if result.attribution is not None:
        from merlin.dse_guidance import attribution as ATTR
        artifacts.append(yaml_artifact("region_attribution.yaml",
                                       ATTR.to_yaml_obj(result.attribution),
                                       header=f"region_attribution (Level-1): {result.workload}"))

    # --- quantitative back-end (only with a baseline; uncalibrated by default) ---
    if result.baseline is not None:
        artifacts += [
            Artifact("axis_triage.csv", report.triage_csv(result.triage_multirate)),
            Artifact("axis_triage_flat.csv", report.triage_csv(result.triage_flat)),
            Artifact("axis_triage.md",
                     report.triage_md(result.triage_multirate, result.triage_flat,
                                      result.baseline)),
            Artifact("bottleneck_breakdown.csv",
                     report.bottleneck_breakdown_csv(result.baseline)),
            yaml_artifact("deadline_feasibility.yaml", result.deadline,
                          header=f"deadline_feasibility: {result.workload}"),
        ]
        if result.coupling_per_replan is not None:
            artifacts.append(yaml_artifact("cpu_coupling_result.yaml",
                                           result.coupling_per_replan,
                                           header=f"cpu_coupling_result: {result.workload}"))
        else:
            artifacts.append(Artifact("cpu_coupling_result.txt", UNAVAILABLE_MESSAGE + "\n"))
        if result.calibration_rows:
            artifacts.append(Artifact("calibration_anchor.csv",
                                      calibration.anchor_csv(result.calibration_rows)))
        if result.is_negative_control:
            artifacts.append(Artifact("negative_control_report.md",
                                      report.negative_control_md(result.triage_multirate,
                                                                 result.baseline)))

    paths = [a.write(out) for a in artifacts]
    paths += _maybe_figures(result, out / "figures")
    return paths


def _maybe_figures(result: GuidanceResult, fig_dir: Path) -> list[Path]:
    from merlin.dse_guidance import plots
    written: list[Path] = []
    if result.baseline is None:
        return written  # structural-only run: no quantitative plots
    fig_dir.mkdir(parents=True, exist_ok=True)
    if plots.axis_triage_plot(result.triage_multirate, fig_dir / "axis_triage.png"):
        written.append(fig_dir / "axis_triage.png")
    if plots.bottleneck_plot(result.baseline, fig_dir / "bottleneck_breakdown.png"):
        written.append(fig_dir / "bottleneck_breakdown.png")
    if plots.flat_vs_multirate_plot(result.flat, result.multirate,
                                    fig_dir / "flat_vs_multirate.png"):
        written.append(fig_dir / "flat_vs_multirate.png")
    return written
