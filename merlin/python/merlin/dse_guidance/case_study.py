"""Cross-workload case study over real `prov.fqn` captures.

The value is breadth, not one model: this runs the provenance-aware pipeline over every real
recaptured model under ``merlin/benchmarks/dse_guidance/recaptures/<workload>/model.mlir`` and
produces a cross-workload analysis with **explicit per-field provenance** — so a reader can see,
for each number, whether it was recovered from IR, recovered from `prov.fqn`, an assumed
reference, calibrated, or unavailable.

It deliberately produces **no uncalibrated speedup**: residency/loop candidates report real
attributed facts and stay `blocked_by: missing_calibration`.

Reference captures are real architectures via the `prov.fqn`-enabled model2MLIR, at small/random
configs (e.g. depth 2) — the *structure and provenance are real*; magnitudes are small instances.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from merlin.common import paths
from merlin.common.artifacts import Artifact
from merlin.dse_guidance import attribution as ATTR
from merlin.dse_guidance import candidates as CAND
from merlin.dse_guidance import contract as CON
from merlin.dse_guidance import design_envelope as DE
from merlin.dse_guidance import numerical_contract as NC
from merlin.dse_guidance import temporal as T
from merlin.dse_guidance import topology as TOP

# Provenance labels (what kind of evidence backs each field).
P_IR = "recovered_from_ir"
P_FQN = "recovered_from_prov_fqn"
P_ASSUMED = "assumed_reference"
P_CALIB = "calibrated"
P_UNCAL = "uncalibrated"
P_NA = "unavailable"

# Recaptured real workloads (class + reference loop count K). K is assumed/reference, not measured.
RECAP_MODELS: dict[str, dict] = {
    "rdt": {"class": "diffusion/denoise_steps", "K": 5,
            "note": "RDT denoise step (depth 2, random init)"},
    "openvla": {"class": "autoregressive_vla/action_token_decode", "K": 7,
                "note": "OpenVLA: fused ViT vision backbone + Llama decode head (small config)"},
    "small_llama": {"class": "llm/token_decode", "K": 32,
                    "note": "small Llama decoder (2 layers)"},
    "tiny_llama": {"class": "llm/token_decode", "K": 32,
                   "note": "tiny Llama decoder (2 layers)"},
}


def _recap_dir(workload: str):
    return paths.merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures" / workload


def available_models() -> list[str]:
    return [w for w in RECAP_MODELS if (_recap_dir(w) / "model.mlir").is_file()]


@dataclass
class WorkloadCase:
    workload: str
    cls: str
    K: int
    note: str
    n_matmuls: int
    topo: TOP.VlaRuntimeTopology
    attribution: ATTR.RegionAttribution
    candidates: list


def analyze(workload: str) -> WorkloadCase:
    from merlin.dse_guidance import models as M
    spec = RECAP_MODELS[workload]
    K = int(spec["K"])
    # Action horizon H and control rate are architecture facts (not the denoise/decode count K);
    # pull them from the model registry so the replan deadline is correct (H != K).
    arch = M.MODEL_ARCH.get(workload)
    H = int(arch.action_horizon) if (arch and arch.action_horizon) else K
    control = float(arch.control_rate_hz) if (arch and arch.control_rate_hz) else 30.0
    temporal = T.parse({
        "workload": workload, "class": spec["class"],
        "timing": {"K": K, "H": H, "control_rate_hz": control},
        "regions": [
            {"name": "backbone", "role": "backbone_once", "invocation_count": 1},
            {"name": "head", "role": "repeated_head", "invocation_count": K,
             "loop_invariant_state": ["weights"]},
        ],
    })
    topo = TOP.from_temporal(temporal)
    cap = str(_recap_dir(workload))
    attr = ATTR.attribute(cap, topo)                 # roles auto-recovered from prov.fqn
    cands = CAND.detect(topo, attribution=attr)
    return WorkloadCase(workload=workload, cls=spec["class"], K=K, note=spec["note"],
                        n_matmuls=len(ATTR.extract_matmuls(cap)),
                        topo=topo, attribution=attr, candidates=cands)


def _role(case: WorkloadCase, role: str):
    return case.attribution.role(role)


# --------------------------------------------------------------- workshop provenance CSV

_CSV_COLUMNS = ["workload", "item", "flat_view", "recovered_view", "evidence_source",
                "dse_implication", "quantification_status"]


def _rows_for(case: WorkloadCase) -> list[dict]:
    rows: list[dict] = []
    head = _role(case, "repeated_head")
    bb = _role(case, "backbone_once")

    if head and head.attribution_status == "attributed":
        f = head.facts
        rows.append({
            "workload": case.workload, "item": "action-head weight reuse",
            "flat_view": "weights used once (0 contract facts)",
            "recovered_view": (f"repeated_head x{case.K}: {f['matmul_count']} matmuls, "
                               f"{f['weight_bytes']/1e6:.1f} MB weights, "
                               f"{f['macs_per_invocation']/1e9:.2f} GMAC/step"),
            "evidence_source": f"{P_FQN} (role) + {P_IR} (facts); K={case.K} {P_ASSUMED}",
            "dse_implication": "resident_action_head_weights",
            "quantification_status": "blocked: missing_calibration",
        })
        rows.append({
            "workload": case.workload, "item": "K-step loop",
            "flat_view": "absent (loop unrolled by torch.export)",
            "recovered_view": f"bounded repeated head, K={case.K}",
            "evidence_source": f"{P_ASSUMED} (K) + {P_FQN} (head exists)",
            "dse_implication": "autonomous_K_loop",
            "quantification_status": "blocked: missing_calibration",
        })
    if bb and bb.attribution_status == "attributed":
        rows.append({
            "workload": case.workload, "item": "backbone vs head split",
            "flat_view": "undifferentiated single forward",
            "recovered_view": (f"backbone_once: {bb.facts['matmul_count']} matmuls; "
                               f"repeated_head: {head.facts['matmul_count'] if head else 0} matmuls"),
            "evidence_source": f"{P_FQN}",
            "dse_implication": "backbone_head_partition",
            "quantification_status": "structural (no magnitude claimed)",
        })
    # CPU coupling row — uniformly unavailable.
    rows.append({
        "workload": case.workload, "item": "host dispatch / sync coupling",
        "flat_view": "not represented",
        "recovered_view": "not measured",
        "evidence_source": P_NA,
        "dse_implication": "command_batching / autonomous_K_loop",
        "quantification_status": "blocked: measurement required",
    })
    return rows


def provenance_csv(cases: list[WorkloadCase]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_CSV_COLUMNS)
    w.writeheader()
    for case in cases:
        for row in _rows_for(case):
            w.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------- case study markdown

def case_study_md(cases: list[WorkloadCase]) -> str:
    L = ["# Cross-workload provenance case study\n"]
    L.append("> Flat model captures are insufficient DSE units. With `prov.fqn` provenance, Merlin "
             "recovers region roles from real IR, attaches real compute/memory facts to the "
             "repeated action head, and emits structural DSE candidates — while refusing "
             "quantitative benefit until calibration exists.\n")
    L.append(f"Real recaptured workloads: **{len(cases)}** "
             f"({', '.join(c.workload for c in cases)}). Captures are real architectures via the "
             "`prov.fqn`-enabled model2MLIR at small/random configs — structure & provenance real, "
             "magnitudes are small instances.\n")
    # Headline result table: one row per workload (flat vs recovered vs facts vs implication).
    L.append("## Headline: flat capture vs recovered contract\n")
    L.append("| workload | flat view | recovered view | real facts | DSE implication | quantification |")
    L.append("|----------|-----------|----------------|------------|-----------------|----------------|")
    for c in cases:
        head = _role(c, "repeated_head")
        bb = _role(c, "backbone_once")
        recovered = "repeated_head" + (" + backbone_once split" if bb else "")
        if head and head.attribution_status == "attributed":
            facts = (f"{head.facts['matmul_count']} mm, "
                     f"{head.facts['weight_bytes']/1e6:.0f} MB, "
                     f"{head.facts['macs_per_invocation']/1e9:.1f} GMAC/step xK={c.K}")
        else:
            facts = "—"
        impl = "resident_action_head_weights" + (", backbone_head_partition" if bb else "")
        L.append(f"| {c.workload} | weights once, no K-loop | {recovered} | {facts} | {impl} | "
                 f"blocked: missing_calibration |")
    L.append("")
    L.append("## Per-workload recovery\n")
    L.append("| workload | class | matmuls | roles recovered (from prov.fqn) | repeated_head facts | quant |")
    L.append("|----------|-------|---------|---------------------------------|---------------------|-------|")
    for c in cases:
        roles = {}
        for r in c.attribution.regions:
            roles[r.role] = len(r.matmul_indices)
        head = _role(c, "repeated_head")
        hf = (f"{head.facts['matmul_count']} mm, {head.facts['weight_bytes']/1e6:.0f} MB, "
              f"{head.facts['macs_per_invocation']/1e9:.1f} GMAC/step xK={c.K}"
              if head and head.attribution_status == "attributed" else "—")
        L.append(f"| {c.workload} | {c.cls} | {c.n_matmuls} | "
                 f"{', '.join(f'{k}:{v}' for k, v in roles.items())} | {hf} | "
                 f"blocked: missing_calibration |")
    L.append("")
    L.append("## What flattening hides vs what provenance recovers\n")
    L.append("- **Flat view:** weights used once, no K-loop, no backbone/head split, no deadline — "
             "so residency / autonomous-loop / partition axes are invisible or illegal.")
    L.append("- **Recovered view:** roles from `prov.fqn`, real per-region MACs/bytes, the repeated "
             "head and (for OpenVLA) the vision-backbone/LM split made explicit.")
    L.append("- **Honest gate:** every candidate carries real facts but stays "
             "`blocked_by: missing_calibration`; no speedup is claimed.\n")
    L.append("## Evidence provenance legend\n")
    L.append(f"`{P_IR}` · `{P_FQN}` · `{P_ASSUMED}` · `{P_CALIB}` · `{P_UNCAL}` · `{P_NA}`\n")
    L.append("See `cross_workload_provenance.csv` for the per-item flat-vs-recovered table, and "
             "`numerical_contract_fidelity_report.md` for the precision/quantization contract "
             "(the orthogonal axis: every int8/fp8 zoo capture stores weights low-bit but runs "
             "f32 matmuls — native low-bit compute and the packed layout are absent).\n")
    return "\n".join(L)


def _head_weight_bytes(case: WorkloadCase) -> int | None:
    head = case.attribution.role("repeated_head")
    return head.facts.get("weight_bytes") if head and head.attribution_status == "attributed" else None


def zoo_numerical_audit() -> list:
    """Audit the numerical contract of the existing quantized zoo captures (int8/fp8/fp32).

    These predate prov.fqn but still carry prov.quantization + dtypes — enough to show the
    cross-zoo finding that low-bit weights are dequantized to f32 before compute.
    """
    root = paths.repo_root() / "output"
    contracts = []
    seen = set()
    for d in sorted(root.glob("*_consistent")):
        if not (d / "model.mlir").is_file():
            continue
        # one capture per (base, dtype) is plenty; keep it readable
        if d.name in seen:
            continue
        seen.add(d.name)
        contracts.append(NC.audit(str(d), workload=d.name))
    return [c for c in contracts if c.declared_quantization != "none" or c.low_bit_storage] \
        or contracts


def run_case_study(out_dir) -> dict:
    """Analyze all available recaptured workloads; write the cross-workload artifacts."""
    from pathlib import Path
    from merlin.common.artifacts import yaml_artifact
    out = Path(out_dir)
    models = available_models()
    cases = [analyze(w) for w in models]
    Artifact("cross_workload_provenance.csv", provenance_csv(cases)).write(out)
    Artifact("case_study.md", case_study_md(cases)).write(out)
    envelopes = []
    for c in cases:
        cap = str(_recap_dir(c.workload))
        dtype = NC.extract_numerical_facts(cap).get("compute_dtype", "f32")
        yaml_artifact(f"{c.workload}/region_attribution.yaml", ATTR.to_yaml_obj(c.attribution),
                      header=f"region_attribution (Level-1, prov.fqn): {c.workload}").write(out)
        Artifact(f"{c.workload}/dse_candidate_axes.md",
                 CAND.markdown(c.topo, c.candidates)).write(out)
        # numerical contract per recaptured workload (fp32 -> low-bit opportunity)
        nc = NC.audit(cap, workload=c.workload, workload_class=c.cls,
                      repeated_head_weight_bytes=_head_weight_bytes(c),
                      has_epilogue=bool(c.attribution.role("repeated_head")))
        yaml_artifact(f"{c.workload}/numerical_contract.yaml", NC.to_yaml_obj(nc),
                      header=f"numerical_contract: {c.workload}").write(out)
        # design envelope (hardware-independent requirements)
        env = DE.from_recovered(c.topo, c.attribution, captured_dtype=dtype)
        if env is not None:
            envelopes.append(env)
            yaml_artifact(f"{c.workload}/design_envelope.yaml", DE.to_yaml_obj(env),
                          header=f"design_envelope (requirements, not calibration): {c.workload}").write(out)
            Artifact(f"{c.workload}/design_envelope.md", DE.markdown(env)).write(out)
        # workload-contract package: abstraction candidates + readiness + measurement plan + report
        cands = CON.abstraction_candidates(c.candidates, nc)
        readiness = CON.dse_readiness(c.topo, c.attribution, nc, cpu_coupling_available=False)
        plan = CON.measurement_plan(cands)
        yaml_artifact(f"{c.workload}/abstraction_candidates.yaml", CON.abstraction_yaml(cands),
                      header=f"abstraction_candidates: {c.workload}").write(out)
        yaml_artifact(f"{c.workload}/dse_readiness.yaml", CON.readiness_yaml(readiness),
                      header=f"dse_readiness: {c.workload}").write(out)
        yaml_artifact(f"{c.workload}/measurement_plan.yaml", {"measurement_plan": plan},
                      header=f"measurement_plan: {c.workload}").write(out)
        Artifact(f"{c.workload}/workload_contract_report.md",
                 CON.workload_contract_report_md(c.topo, c.attribution, nc, env, cands,
                                                 readiness, plan)).write(out)
    if envelopes:
        Artifact("requirements_table.csv", DE.requirements_csv(envelopes)).write(out)

    # cross-zoo numerical-contract report (shows low-bit compute lost across the quantized zoo)
    zoo = zoo_numerical_audit()
    Artifact("numerical_contract_fidelity_report.md", NC.fidelity_report_md(zoo)).write(out)
    # measured dispatch coupling (the one measured runtime leg) — committed measured data
    try:
        from merlin.dse_guidance import dispatch_measure as DM
        dm = DM.load_measured()
        if dm:
            Artifact("dispatch_coupling_report.md", DM.report_md(dm)).write(out)
            Artifact("dispatch_coupling.csv", DM.to_csv(DM.calibration_rows(dm))).write(out)
    except FileNotFoundError:
        pass
    return {"workloads": models, "out": str(out), "numerical_captures": len(zoo)}
