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
from merlin.dse_guidance import accuracy_gate as AG
from merlin.dse_guidance import attribution as ATTR
from merlin.dse_guidance import candidates as CAND
from merlin.dse_guidance import command_graph as CG
from merlin.dse_guidance import compiler_proof as CP
from merlin.dse_guidance import contract_graph as CGRAPH
from merlin.dse_guidance import contract as CON
from merlin.dse_guidance import design_envelope as DE
from merlin.dse_guidance import dtype_certificates as DC
from merlin.dse_guidance import memory_envelope as ME
from merlin.dse_guidance import numerical_contract as NC
from merlin.dse_guidance import operator_geometry as OG
from merlin.dse_guidance import primitive_coverage as PC
from merlin.dse_guidance import search_space as SS
from merlin.dse_guidance import state_lifetime as SL
from merlin.dse_guidance import temporal as T
from merlin.dse_guidance import topology as TOP
from merlin.dse_guidance import workload_family as WF

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


def _csv(rows: list[dict], cols: list[str]) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    return buf.getvalue()


def _envval(env, name):
    if env is None:
        return None
    r = env.req(name)
    return None if (r is None or r.value is None) else r.value


def _roles_str(case) -> str:
    return ", ".join(f"{r.role}:{len(r.matmul_indices)}" for r in case.attribution.regions)


def _workload_contract_table(packages) -> str:
    rows = []
    for p in packages:
        env = p["env"]
        rows.append({
            "workload": p["case"].workload, "class": p["case"].cls, "roles": _roles_str(p["case"]),
            "macs_per_replan": _envval(env, "macs_per_replan"),
            "avoidable_reload_bytes": _envval(env, "avoidable_weight_reload_bytes"),
            "resident_bf16_B": (round(env.capacity_by_dtype_B["bf16"]) if env else None),
            "top_abstraction": (p["cands"][0].system_abstraction if p["cands"] else ""),
            "accuracy_int8": p["acc"], "ready_structural": True,
            "ready_quantitative": p["readiness"].ready,
            "missing_before_dse": "; ".join(p["readiness"].missing),
        })
    return _csv(rows, ["workload", "class", "roles", "macs_per_replan", "avoidable_reload_bytes",
                       "resident_bf16_B", "top_abstraction", "accuracy_int8", "ready_structural",
                       "ready_quantitative", "missing_before_dse"])


def _abstraction_pressure_table(packages) -> str:
    rows = []
    for p in packages:
        for c in p["cands"]:
            rows.append({"workload": p["case"].workload, "axis": c.axis,
                         "system_abstraction": c.system_abstraction,
                         "dse_knobs": "; ".join(c.dse_knobs_exposed),
                         "blocked_by": c.quantification_blocked_by})
    return _csv(rows, ["workload", "axis", "system_abstraction", "dse_knobs", "blocked_by"])


def _abstraction_pressure_ranking(packages) -> str:
    """Across-workload pressure ranking — how many workloads/regions suggest each abstraction.

    This is *structural pressure*, a count — NOT a speedup or a performance ranking. evidence_strength
    is a qualitative label (strong/partial/structural_only) based on how many suggesting workloads
    carry real attributed IR facts for the axis.
    """
    from collections import Counter
    agg: dict[str, dict] = {}
    for p in packages:
        for c in p["cands"]:
            a = agg.setdefault(c.axis, {"sys": c.system_abstraction, "workloads": set(),
                                        "n_regions": 0, "blocked": Counter()})
            a["workloads"].add(p["case"].workload)
            if (c.why_this_exists or {}).get("attributed_facts"):
                a["n_regions"] += 1            # an attributed region with real IR facts backs it
            a["blocked"][c.quantification_blocked_by] += 1
    rows = []
    for axis, a in agg.items():
        nw, nr = len(a["workloads"]), a["n_regions"]
        strength = "strong" if (nr >= nw and nr > 0) else "partial" if nr > 0 else "structural_only"
        rows.append({"axis": axis, "system_abstraction": a["sys"], "n_workloads": nw,
                     "n_regions": nr, "evidence_strength": strength,
                     "blocked_by": (a["blocked"].most_common(1)[0][0] if a["blocked"] else "")})
    rows.sort(key=lambda r: (-r["n_workloads"], -r["n_regions"], r["axis"]))
    return _csv(rows, ["axis", "system_abstraction", "n_workloads", "n_regions",
                       "evidence_strength", "blocked_by"])


def _measurement_priority_table(packages) -> str:
    """Aggregate the per-workload measurement plans into a priority table (what to measure next).

    Categories come from contract.measurement_plan: accuracy_measurable_now / proxy_measured /
    target_measured. n_candidates_unblocked counts abstraction candidates that need the measurement.
    """
    from collections import defaultdict
    cat_of: dict[str, str] = {}
    wls: dict[str, set] = defaultdict(set)
    for p in packages:
        plan = p.get("plan") or {}
        mn = plan.get("measurable_now", {})
        for m in mn.get("accuracy", []):
            cat_of[m] = "accuracy_measurable_now"; wls[m].add(p["case"].workload)
        for m in mn.get("runtime_proxy", []):
            cat_of[m] = "proxy_measured"; wls[m].add(p["case"].workload)
        for m in plan.get("needs_target_design", []):
            cat_of[m] = "target_measured"; wls[m].add(p["case"].workload)
    unblocks: dict[str, int] = defaultdict(int)
    for p in packages:
        for c in p["cands"]:
            for m in set(c.measurements_needed):
                if m in cat_of:
                    unblocks[m] += 1
    order = {"accuracy_measurable_now": 0, "proxy_measured": 1, "target_measured": 2}
    rows = [{"measurement": m, "category": cat, "n_candidates_unblocked": unblocks.get(m, 0),
             "workloads": "; ".join(sorted(wls[m]))} for m, cat in cat_of.items()]
    rows.sort(key=lambda r: (order[r["category"]], -r["n_candidates_unblocked"], r["measurement"]))
    return _csv(rows, ["measurement", "category", "n_candidates_unblocked", "workloads"])


def _dse_readiness_table(packages) -> str:
    rows = []
    for p in packages:
        f = p["readiness"].fields
        rows.append({
            "workload": p["case"].workload,
            "topology_recovered": f["topology_recovered"]["available"],
            "role_attribution": f["role_attribution"]["available"],
            "K_source": f["K_source"]["source"],
            "deadline_source": f["deadline_source"]["source"],
            "dtype_contract": f["dtype_contract"]["available"],
            "accuracy_status": f["accuracy_constraints"].get("int8_status", "unavailable"),
            "cpu_coupling": f["cpu_coupling"]["source"],
            "ready_structural_DSE": True,
            "ready_quantitative_DSE": p["readiness"].ready,
            "missing_before_ranking": "; ".join(p["readiness"].missing),
        })
    return _csv(rows, ["workload", "topology_recovered", "role_attribution", "K_source",
                       "deadline_source", "dtype_contract", "accuracy_status", "cpu_coupling",
                       "ready_structural_DSE", "ready_quantitative_DSE", "missing_before_ranking"])


def _dtype_capacity_table(packages) -> str:
    rows = []
    for p in packages:
        env = p["env"]
        if env is None:
            continue
        cap = env.capacity_by_dtype_B
        rows.append({
            "workload": p["case"].workload, "captured_dtype": env.captured_dtype,
            "bf16_B": round(cap["bf16"]), "fp8_B": round(cap["fp8"]), "int8_B": round(cap["int8"]),
            "fp6_B": round(cap["fp6"]), "int4_B": round(cap["int4"]),
            "avoidable_reload_B": _envval(env, "avoidable_weight_reload_bytes"),
        })
    return _csv(rows, ["workload", "captured_dtype", "bf16_B", "fp8_B", "int8_B", "fp6_B",
                       "int4_B", "avoidable_reload_B"])


def _case_study_summary_md(packages) -> str:
    L = ["# Case study summary — workload-contract analysis\n"]
    L.append("> Flat captures are not DSE-ready workload descriptions. Merlin recovers a temporal/"
             "numerical workload contract from provenance-rich captures and emits HW/SW abstraction "
             "candidates + hardware-independent requirements for a future DSE engine — no speedup "
             "claimed.\n")
    L.append("| workload | recovered roles | real IR facts | derived requirement | implied abstraction | missing before DSE |")
    L.append("|----------|-----------------|---------------|---------------------|---------------------|--------------------|")
    for p in packages:
        env = p["env"]
        macs = _envval(env, "macs_per_replan")
        avoid = _envval(env, "avoidable_weight_reload_bytes")
        facts = (f"{macs/1e9:.0f} GMAC/replan, {avoid/1e9:.2f} GB avoidable reload"
                 if macs and avoid else "—")
        bf16 = f"resident bf16 {env.capacity_by_dtype_B['bf16']/1e6:.0f} MB" if env else "—"
        impl = p["cands"][0].system_abstraction if p["cands"] else "—"
        L.append(f"| {p['case'].workload} | {_roles_str(p['case'])} | {facts} | {bf16} | {impl} | "
                 f"{'; '.join(p['readiness'].missing)} |")
    L.append("")
    L.append("See per-workload `<workload>/workload_contract_report.md` for the full package, "
             "`requirements_table.csv` / `dtype_capacity_table.csv` for requirements, "
             "`abstraction_pressure_table.csv` for the HW/SW abstractions, "
             "`dse_readiness_summary.csv` for readiness, and `accuracy_gate_report.md` for the "
             "measured int8 accuracy leg.\n")
    return "\n".join(L)


def _readme_md(packages) -> str:
    names = ", ".join(p["case"].workload for p in packages)
    return (
        "# Merlin workload-contract analysis — case study\n\n"
        "Merlin is a compiler-based **workload-contract analysis** tool for accelerator DSE. It "
        "does not pick a design and does not calibrate against existing hardware. It recovers the "
        "temporal + numerical workload contract a flat capture erases and emits a DSE-ready "
        "package: region facts, hardware-independent requirements, HW/SW abstraction candidates, a "
        "measurement plan, and a readiness report.\n\n"
        f"Workloads (real `prov.fqn` recaptures): **{names}**.\n\n"
        "## Read this folder\n"
        "- `current_state_audit.md` — V0 freeze audit (standalone); `claim_evidence_matrix.csv`, "
        "`known_limitations.md`, `reproducibility_check.log` are its companions.\n"
        "- `case_study_summary.md` — start here (the central table).\n"
        "- `<workload>/workload_contract_report.md` — full per-workload package.\n"
        "- `requirements_table.csv`, `dtype_capacity_table.csv` — design requirements (hw-independent).\n"
        "- `abstraction_pressure_table.csv` — implied HW/SW abstractions + DSE knobs (per workload).\n"
        "- `abstraction_pressure_ranking.csv` — across-workload pressure ranking (a count, not a "
        "speedup).\n"
        "- `resident_state_table.csv` — state lifetimes (loop-invariant / carried / boundary-crossing) "
        "+ the abstraction each implies.\n"
        "- `compiler_proof_matrix.csv` — the compiler proof each abstraction needs + its status "
        "(proven_for_workload / assumed / unknown).\n"
        "- `workload_family_table.csv` — workloads clustered into families (iterative_denoise / "
        "token_decode / single_shot).\n"
        "- `<workload>/dse_search_space_template.yaml`, `dse_search_space_template_<family>.yaml` — "
        "the **DSE search-space template** (the bridge a DSE engine consumes: enabled axes + knobs).\n"
        "- `measurement_priority_table.csv` — what to measure next, ranked by candidates unblocked.\n"
        "- `operator_shape_table.csv`, `operator_geometry.yaml` — per-operator geometry (M/N/K, "
        "MACs, aspect, shape_class + semantic role). `shape_summary_by_workload.csv`, "
        "`shape_summary_by_region.csv`, `operator_cluster_table.csv`, `operator_geometry_report.md` "
        "summarise it (structural geometry only).\n"
        "- `tile_waste_table.csv`, `primitive_coverage_matrix.csv`, `primitive_regret_table.csv` — "
        "candidate compute-primitive (tile / GEMV-lane) structural coverage + cross-workload regret; "
        "`primitive_coverage_report.md`, `cross_workload_coverage_report.md` read them (no speedup).\n"
        "- `workload_contract_graph.yaml`, `workload_contract_graph_summary.md` — the **multi-rate "
        "workload contract graph** (the central IR later phases consume: phase/region/operator/"
        "state nodes + typed edges). `phase_rate_table.csv`, `multi_rate_contract.yaml`, "
        "`rate_mismatch_report.md` expose the per-phase cadence + rate model (structural only).\n"
        "- `traffic_table.csv` — per-region byte traffic + avoidable reload (memory/reuse envelope).\n"
        "- `dispatch_granularity_table.csv` — command-graph view (honest: loop unrolled, syncs "
        "unavailable).\n"
        "- `accuracy_gated_dtype_candidates.csv` — which low-bit formats are accuracy-legal vs "
        "blocked (int8 measured; fp8/int4 unavailable).\n"
        "- `torchao_integration_plan.md` — plan (not a sweep) for wiring low-bit formats to the "
        "numerical candidates.\n"
        "- `dse_readiness_summary.csv` — what a DSE engine can consume today + what's missing.\n"
        "- `accuracy_gate_report.md` — measured int8 accuracy (the measurable-now leg).\n"
        "- `numerical_contract_fidelity_report.md`, `dispatch_coupling_report.md`, "
        "`cost_calibration.md` — supporting evidence (calibration is a demoted existing-target anchor).\n\n"
        "## Regenerate\n```\nmerlin-dse-guidance --case-study \\\n"
        "  --out merlin/benchmarks/dse_guidance/case_study\n```\n\n"
        "Every number carries an evidence label (`recovered_from_ir` / `recovered_from_prov_fqn` / "
        "`assumed_reference` / `derived_requirement` / `design_assumption` / `measured` / "
        "`proxy_measured` / `unavailable`). No file claims a speedup for unbuilt hardware.\n")


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
    packages = []                       # accumulate per-workload data for cross-workload tables
    geom_by_workload: dict = {}         # P5: per-workload operator geometry
    all_shapes: list = []
    conv_visible = False
    graphs: list = []                   # P6: per-workload multi-rate contract graphs
    for c in cases:
        cap = str(_recap_dir(c.workload))
        dtype = NC.extract_numerical_facts(cap).get("compute_dtype", "f32")
        records = ATTR.extract_matmuls(cap)
        # P5 operator geometry: per-op shapes for this workload (role from attribution/prov.fqn)
        shapes = OG.operator_shapes(records, c.workload, c.attribution)
        geom_by_workload[c.workload] = shapes
        all_shapes.extend(shapes)
        conv_visible = conv_visible or OG.conv_ops_present(
            (_recap_dir(c.workload) / "model.mlir").read_text(errors="ignore"))
        yaml_artifact(f"{c.workload}/region_attribution.yaml", ATTR.to_yaml_obj(c.attribution),
                      header=f"region_attribution (Level-1, prov.fqn): {c.workload}").write(out)
        Artifact(f"{c.workload}/dse_candidate_axes.md",
                 CAND.markdown(c.topo, c.candidates)).write(out)
        # state-lifetime / residency view (which tensors persist + the abstraction they imply)
        recs = SL.state_records(c.topo, c.attribution)
        yaml_artifact(f"{c.workload}/state_lifetime.yaml", SL.to_yaml_obj(recs, c.workload),
                      header=f"state_lifetime: {c.workload}").write(out)
        # memory-traffic / reuse envelope + (honest) command-graph view
        mem = ME.region_traffic(c.attribution)
        yaml_artifact(f"{c.workload}/memory_envelope.yaml", ME.to_yaml_obj(mem, c.workload),
                      header=f"memory_envelope: {c.workload}").write(out)
        cmd = CG.command_graph(c.topo, c.attribution)
        yaml_artifact(f"{c.workload}/command_graph.yaml", CG.to_yaml_obj(cmd),
                      header=f"command_graph (honest; loop unrolled): {c.workload}").write(out)
        # numerical contract per recaptured workload (fp32 -> low-bit opportunity)
        nc = NC.audit(cap, workload=c.workload, workload_class=c.cls,
                      repeated_head_weight_bytes=_head_weight_bytes(c),
                      has_epilogue=bool(c.attribution.role("repeated_head")),
                      records=records, attribution=c.attribution)
        yaml_artifact(f"{c.workload}/numerical_contract.yaml", NC.to_yaml_obj(nc),
                      header=f"numerical_contract: {c.workload}").write(out)
        # P6: multi-rate workload contract graph (joins topo/attribution/nc/geometry/state)
        graphs.append(CGRAPH.build_graph(c, shapes, nc, recs))
        # design envelope (hardware-independent requirements)
        env = DE.from_recovered(c.topo, c.attribution, captured_dtype=dtype)
        if env is not None:
            envelopes.append(env)
            yaml_artifact(f"{c.workload}/design_envelope.yaml", DE.to_yaml_obj(env),
                          header=f"design_envelope (requirements, not calibration): {c.workload}").write(out)
            Artifact(f"{c.workload}/design_envelope.md", DE.markdown(env)).write(out)
        # accuracy-gated dtype candidate certificates (int8 measured; others blocked, never assumed)
        certs = DC.certificates(nc, env, c.workload)
        yaml_artifact(f"{c.workload}/numerical_candidate_certificates.yaml",
                      DC.to_yaml_obj(certs, c.workload),
                      header=f"numerical_candidate_certificates: {c.workload}").write(out)
        # workload-contract package: abstraction candidates + readiness + measurement plan + report
        cands = CON.abstraction_candidates(c.candidates, nc)
        acc = AG.status_for(c.workload, "int8")          # measured int8 accuracy (else unavailable)
        readiness = CON.dse_readiness(c.topo, c.attribution, nc, cpu_coupling_available=False,
                                      accuracy_status=acc)
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
        pkg = {"case": c, "env": env, "nc": nc, "cands": cands, "readiness": readiness,
               "acc": acc, "state": recs, "plan": plan, "mem": mem, "cmd": cmd, "certs": certs}
        # DSE search-space template (the bridge artifact) — per workload
        yaml_artifact(f"{c.workload}/dse_search_space_template.yaml",
                      SS.to_yaml_obj(SS.template_for_workload(pkg)),
                      header=f"dse_search_space_template: {c.workload}").write(out)
        packages.append(pkg)
    if envelopes:
        Artifact("requirements_table.csv", DE.requirements_csv(envelopes)).write(out)

    # cross-workload presentation tables (P1/P2)
    Artifact("workload_contract_table.csv", _workload_contract_table(packages)).write(out)
    Artifact("abstraction_pressure_table.csv", _abstraction_pressure_table(packages)).write(out)
    Artifact("dse_readiness_summary.csv", _dse_readiness_table(packages)).write(out)
    Artifact("dtype_capacity_table.csv", _dtype_capacity_table(packages)).write(out)
    Artifact("case_study_summary.md", _case_study_summary_md(packages)).write(out)

    # contract-completeness package (state lifetime, compiler proofs, pressure ranking,
    # workload families, search-space templates, measurement priority) — the DSE-ready hand-off
    Artifact("resident_state_table.csv", SL.resident_state_csv(packages)).write(out)
    Artifact("compiler_proof_matrix.csv", CP.compiler_proof_csv(packages)).write(out)
    Artifact("abstraction_pressure_ranking.csv",
             _abstraction_pressure_ranking(packages)).write(out)
    Artifact("workload_family_table.csv", WF.workload_family_csv(packages)).write(out)
    Artifact("measurement_priority_table.csv",
             _measurement_priority_table(packages)).write(out)
    # memory-traffic envelope, command-graph granularity, accuracy-gated dtype certificates
    Artifact("traffic_table.csv", ME.traffic_csv(packages)).write(out)
    Artifact("dispatch_granularity_table.csv", CG.dispatch_granularity_csv(packages)).write(out)
    Artifact("accuracy_gated_dtype_candidates.csv", DC.gated_csv(packages)).write(out)
    # per-family search-space templates (union of member-workload axes)
    for fam, axis_set in sorted(WF.family_axis_sets(packages).items()):
        yaml_artifact(f"dse_search_space_template_{fam}.yaml",
                      SS.to_yaml_obj(SS.template_for_family(fam, axis_set)),
                      header=f"dse_search_space_template (family): {fam}").write(out)
    # P5 operator-geometry + primitive-coverage (search-space formation; structural geometry only)
    Artifact("operator_shape_table.csv", OG.operator_shape_csv(all_shapes)).write(out)
    yaml_artifact("operator_geometry.yaml", OG.to_yaml_obj(geom_by_workload, conv_visible),
                  header="operator_geometry (structural; no speedup)").write(out)
    Artifact("shape_summary_by_workload.csv",
             OG.shape_summary_by_workload_csv(geom_by_workload)).write(out)
    Artifact("shape_summary_by_region.csv",
             OG.shape_summary_by_region_csv(geom_by_workload)).write(out)
    Artifact("operator_cluster_table.csv", OG.operator_cluster_csv(all_shapes)).write(out)
    Artifact("operator_geometry_report.md",
             OG.report_md(geom_by_workload, all_shapes)).write(out)
    cov = PC.all_coverage(all_shapes)
    per_wl = PC.aggregate_by_primitive_workload(cov)
    regret = PC.aggregate_regret(cov, per_wl)
    Artifact("tile_waste_table.csv", PC.tile_waste_csv(cov)).write(out)
    Artifact("primitive_coverage_matrix.csv",
             PC.primitive_coverage_matrix_csv(per_wl)).write(out)
    Artifact("primitive_coverage_report.md",
             PC.coverage_report_md(per_wl, regret)).write(out)
    Artifact("primitive_regret_table.csv", PC.primitive_regret_csv(regret)).write(out)
    Artifact("cross_workload_coverage_report.md",
             PC.cross_workload_report_md(regret, per_wl)).write(out)
    # P6 multi-rate workload contract graph (the central IR for later phases; structural only)
    yaml_artifact("workload_contract_graph.yaml", CGRAPH.to_yaml_obj(graphs),
                  header="workload_contract_graph (multi-rate; no speedup)").write(out)
    Artifact("workload_contract_graph_summary.md", CGRAPH.summary_md(graphs)).write(out)
    Artifact("phase_rate_table.csv", CGRAPH.phase_rate_csv(graphs)).write(out)
    yaml_artifact("multi_rate_contract.yaml", CGRAPH.multi_rate_contract_yaml(graphs),
                  header="multi_rate_contract (rate model + phases)").write(out)
    Artifact("rate_mismatch_report.md", CGRAPH.rate_mismatch_report_md(graphs)).write(out)
    # TorchAO integration plan (a plan, not a sweep)
    Artifact("torchao_integration_plan.md", NC.torchao_integration_plan_md()).write(out)
    # accuracy gate (measurable-now real leg)
    Artifact("accuracy_gate_report.md", AG.report_md()).write(out)
    Artifact("accuracy_gate_results.csv", AG.to_csv()).write(out)
    Artifact("README.md", _readme_md(packages)).write(out)

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
