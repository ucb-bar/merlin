"""Exhaustive cross-workload study: run DSE guidance over every supported workload.

Discovers the ``semantic_memory`` benchmark regions (plus any explicitly configured workloads),
synthesizes an analytical baseline cost and temporal view for each region that lacks a measured
fixture (tagged ``analytical`` — the weakest grounding, see :mod:`merlin.dse_guidance.synth`),
runs the full guidance pipeline per workload, and aggregates a cross-workload axis ranking.

The aggregate answers, across the whole supported workload set: *which DSE axis closes the most
target gap, for which workloads, and on what evidence* — never a subjective verdict.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

from merlin.common import paths
from merlin.common.artifacts import Artifact
from merlin.common.yaml import load_yaml
from merlin.dse_guidance import baseline_cost as BC
from merlin.dse_guidance import models as M
from merlin.dse_guidance import synth, temporal as T
from merlin.dse_guidance.aet_ingest import CpuCoupling
from merlin.dse_guidance.pipeline import GuidanceResult, run_guidance, write_artifacts


@dataclass
class WorkloadSpec:
    name: str
    temporal: T.TemporalMetadata
    baseline: BC.BaselineCost
    region: dict | None = None
    coupling: CpuCoupling | None = None
    overrides: dict | None = None
    capture_facts: object | None = None
    source: str = "synthesized"   # "synthesized" | "measured_fixture" | "model_capture"


def discover_semantic_memory() -> list[str]:
    """Names of all ``semantic_memory`` benchmark regions (sorted, deterministic)."""
    d = paths.merlin_dir() / "benchmarks" / "semantic_memory"
    return sorted(p.stem for p in d.glob("*.yaml"))


def spec_from_region(name: str, region: dict, control_rate_hz: float = 30.0,
                     target_fraction: float | None = 0.5) -> WorkloadSpec:
    """Build a fully-synthesized (analytical) study spec from a region."""
    temporal = T.parse(synth.synth_temporal(region, control_rate_hz=control_rate_hz))
    baseline = BC.parse(synth.analytical_baseline_cost(region, target_fraction=target_fraction))
    return WorkloadSpec(name=name, temporal=temporal, baseline=baseline,
                        region=region, source="synthesized")


def discover_specs() -> list[WorkloadSpec]:
    """The design-pressure semantic_memory regions as synthesized study specs."""
    specs: list[WorkloadSpec] = []
    for name in discover_semantic_memory():
        region = load_yaml(paths.merlin_dir() / "benchmarks" / "semantic_memory" / f"{name}.yaml")
        specs.append(spec_from_region(region.get("name", name), region))
    return specs


def spec_from_model(base_model: str, capture_dirs: list[str]) -> WorkloadSpec:
    """Build a study spec for a real captured model from its arch + model.mlir aggregate."""
    arch = M.MODEL_ARCH[base_model]
    capture = M._prefer_capture(capture_dirs)
    facts = M.capture_facts(capture)
    temporal = T.parse(M.temporal_doc(arch))
    baseline = BC.parse(M.baseline_doc(arch, facts))
    source = "model_capture" if facts.parsed else "model_capture(unparsed)"
    overrides = {
        "has_epilogue": facts.has_epilogue,
        "dispatches_per_step": max(facts.n_matmuls, 1),
        "has_dependency_chain": facts.n_matmuls > 1 or facts.has_epilogue,
        "macs_per_step": facts.total_macs or None,
        # The head reuses immutable weights across the decode loop. But this flat whole-model
        # capture does NOT separate action-head cost from the backbone, so we deliberately do
        # not supply a dram_reducible_fraction: residency is structurally legal but its benefit
        # is reported unquantified (no fabricated whole-model K-reuse). Quantifying it requires a
        # backbone/head cost split, which a single-pass capture cannot provide.
        "weights_immutable": True,
    }
    return WorkloadSpec(name=base_model, temporal=temporal, baseline=baseline,
                        region=None, overrides=overrides, source=source,
                        coupling=None, capture_facts=facts)


def discover_model_specs() -> list[WorkloadSpec]:
    """All real captured models (the supported VLA/LM zoo) as study specs."""
    captures = M.discover_model_captures()
    specs: list[WorkloadSpec] = []
    for base_model in sorted(captures):
        specs.append(spec_from_model(base_model, captures[base_model]))
    return specs


def run_model_study(out_dir: str | Path) -> dict:
    """Run the exhaustive study over the real model zoo, plus the measured calibration anchor."""
    out = Path(out_dir)
    summary = run_study(discover_model_specs(), out)
    _write_calibration_anchor(out)
    return summary


def _write_calibration_anchor(out: Path) -> None:
    """Emit prediction_vs_measurement.csv + calibration_anchor.md from real measured cycles."""
    captures = M.discover_model_captures()
    rows: list[dict] = []
    for base_model, dirs in sorted(captures.items()):
        arch = M.MODEL_ARCH.get(base_model)
        if arch is None or arch.measured_cycles is None:
            continue
        rows.extend(M.calibration_rows(arch, M.capture_facts(M._prefer_capture(dirs))))

    cols = ["workload", "quantity", "predicted", "measured", "error_pct", "evidence_type",
            "interpretation"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    Artifact("prediction_vs_measurement.csv", buf.getvalue()).write(out)

    L = ["# Calibration anchor — prediction vs measurement\n"]
    if not rows:
        L.append("No real measured cycle total is available in the repo, so no calibration "
                 "anchor was produced. (No fabricated values.)\n")
    else:
        L.append("The only durable real hardware measurement in the repo is the xr0 fp32 "
                 "FireSim total (`docs/results.md`). We compare it against the analytical cost "
                 "model's prediction — **honestly, including a large mismatch**.\n")
        L.append("| workload | quantity | predicted | measured | error % | evidence |")
        L.append("|----------|----------|-----------|----------|---------|----------|")
        for r in rows:
            meas = r["measured"]
            meas_s = f"{meas:.3e}" if isinstance(meas, (int, float)) else str(meas)
            pred = r["predicted"]
            pred_s = f"{pred:.3e}" if isinstance(pred, (int, float)) else str(pred)
            err = "n/a" if r["error_pct"] is None else f"{r['error_pct']}"
            L.append(f"| {r['workload']} | {r['quantity']} | {pred_s} | {meas_s} | {err} | "
                     f"{r['evidence_type']} |")
        L.append("")
        for r in rows:
            L.append(f"- **{r['workload']}/{r['quantity']}**: {r['interpretation']}")
        L.append("\n**Conclusion:** the analytical cost model is not calibrated to real cycles. "
                 "The structural/legality results (which axes the flat capture hides) stand on "
                 "their own; the quantitative gap_closure magnitudes are analytical and must not "
                 "be read as measured until the cost model is calibrated.\n")
    Artifact("calibration_anchor.md", "\n".join(L)).write(out)


def run_study(specs: list[WorkloadSpec], out_dir: str | Path) -> dict:
    """Run guidance for every spec, write per-workload artifacts + an aggregate summary."""
    out = Path(out_dir)
    results: list[GuidanceResult] = []
    for spec in specs:
        res = run_guidance(spec.temporal, spec.baseline, region=spec.region,
                           coupling=spec.coupling, overrides=spec.overrides,
                           capture_facts=spec.capture_facts)
        write_artifacts(res, out / spec.name)
        results.append(res)

    summary_csv = _summary_csv(specs, results)
    summary_md = _summary_md(specs, results)
    Artifact("study_summary.csv", summary_csv).write(out)
    Artifact("study_summary.md", summary_md).write(out)
    return {"workloads": [s.name for s in specs], "out": str(out),
            "n_workloads": len(specs)}


_SUMMARY_COLUMNS = [
    "workload", "source", "representation", "axis", "family",
    "gap_closure", "priority_score", "evidence_type", "confidence",
    "legality", "cost_tier", "benefit", "baseline_unit",
]


def _summary_csv(specs: list[WorkloadSpec], results: list[GuidanceResult]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_SUMMARY_COLUMNS)
    w.writeheader()
    for spec, res in zip(specs, results):
        for rep_name, tr in (("multirate", res.triage_multirate), ("flat", res.triage_flat)):
            for r in tr["axes"]:
                w.writerow({
                    "workload": spec.name, "source": spec.source,
                    "representation": rep_name, "axis": r["axis"], "family": r["family"],
                    "gap_closure": "" if r["gap_closure"] is None else r["gap_closure"],
                    "priority_score": "" if r["priority_score"] is None else r["priority_score"],
                    "evidence_type": r["evidence_type"], "confidence": r["confidence"],
                    "legality": r["legality"], "cost_tier": r["cost_tier"],
                    "benefit": r["benefit_ms"], "baseline_unit": res.baseline.unit,
                })
    return buf.getvalue()


def _summary_md(specs: list[WorkloadSpec], results: list[GuidanceResult]) -> str:
    L: list[str] = []
    L.append("# DSE guidance — exhaustive cross-workload study\n")
    L.append("> Merlin does not perform DSE. Merlin prevents DSE from optimizing the wrong "
             "abstraction.\n")
    L.append(f"Workloads studied: **{len(specs)}**. Baselines marked _synthesized_ use the "
             "analytical cost model (evidence tag `analytical`); measured fixtures override "
             "them where available.\n")

    # Per-workload top axis (multi-rate).
    L.append("## Top axis per workload (multi-rate)\n")
    L.append("| workload | source | top axis | gap_closure | priority | evidence | unit |")
    L.append("|----------|--------|----------|-------------|----------|----------|------|")
    for spec, res in zip(specs, results):
        top = _top(res.triage_multirate)
        if top is None:
            why = ("capture did not parse; structural legality only"
                   if "unparsed" in spec.source else "no axis with a positive gap")
            L.append(f"| {spec.name} | {spec.source} | _{why}_ | n/a | n/a | n/a | "
                     f"{res.baseline.unit} |")
            continue
        gc = "n/a" if top["gap_closure"] is None else f"{top['gap_closure']:.3f}"
        ps = "n/a" if top["priority_score"] is None else f"{top['priority_score']:.4f}"
        L.append(f"| {spec.name} | {spec.source} | {top['axis']} | {gc} | {ps} | "
                 f"{top['evidence_type']} | {res.baseline.unit} |")
    L.append("")

    # Cross-workload axis ranking.
    L.append("## Axis ranking across all workloads (multi-rate)\n")
    L.append("For each axis: number of workloads where it is legal with a positive priority, "
             "and the mean / max priority across those workloads.\n")
    agg: dict[str, list[float]] = {}
    families: dict[str, str] = {}
    for res in results:
        for r in res.triage_multirate["axes"]:
            families[r["axis"]] = r["family"]
            if r["legality"] and r["priority_score"]:
                agg.setdefault(r["axis"], []).append(r["priority_score"])
    L.append("| axis | family | #workloads | mean priority | max priority |")
    L.append("|------|--------|------------|---------------|--------------|")
    for axis in sorted(agg, key=lambda a: -(sum(agg[a]) / len(agg[a]))):
        vals = agg[axis]
        L.append(f"| {axis} | {families.get(axis,'')} | {len(vals)} | "
                 f"{sum(vals)/len(vals):.4f} | {max(vals):.4f} |")
    if not agg:
        L.append("| _none_ | | 0 | | |")
    L.append("")

    # Flat-vs-multirate legality flips: which axes BECOME legal once the loop is visible. This
    # is the structural thesis and holds even when magnitudes are unavailable (unparsed capture).
    L.append("## Representation flips (axes the flat capture hides)\n")
    L.append("A flat whole-model capture reuses each weight once (0 contract facts, per "
             "results.md). The multi-rate view re-exposes the decode/denoise loop, making these "
             "axes legal:\n")
    L.append("| workload | becomes legal under multi-rate |")
    L.append("|----------|--------------------------------|")
    for spec, res in zip(specs, results):
        flat_legal = {r["axis"] for r in res.triage_flat["axes"] if r["legality"]}
        multi_legal = {r["axis"] for r in res.triage_multirate["axes"] if r["legality"]}
        gained = sorted(multi_legal - flat_legal)
        L.append(f"| {spec.name} | {', '.join(gained) if gained else '—'} |")
    L.append("")
    return "\n".join(L)


def _top(tr: dict) -> dict | None:
    ranked = [r for r in tr["axes"] if r["priority_score"] is not None and r["legality"]]
    return ranked[0] if ranked else None
