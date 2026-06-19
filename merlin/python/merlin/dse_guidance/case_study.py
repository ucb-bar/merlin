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
from merlin.dse_guidance import boundary_placement as BP
from merlin.dse_guidance import candidates as CAND
from merlin.dse_guidance import command_graph as CG
from merlin.dse_guidance import compiler_proof as CP
from merlin.dse_guidance import contract_graph as CGRAPH
from merlin.dse_guidance import contract as CON
from merlin.dse_guidance import design_envelope as DE
from merlin.dse_guidance import dma_buffer_analysis as DMA
from merlin.dse_guidance import dtype_certificates as DC
from merlin.dse_guidance import fusion_epilogue as FE
from merlin.dse_guidance import mapspace as MS
from merlin.dse_guidance import memory_envelope as ME
from merlin.dse_guidance import operand_locality as OL
from merlin.dse_guidance import quant_metadata as QM
from merlin.dse_guidance import numerical_contract as NC
from merlin.dse_guidance import operator_geometry as OG
from merlin.dse_guidance import parallelism as PAR
from merlin.dse_guidance import pipeline_envelope as PE
from merlin.dse_guidance import primitive_coverage as PC
from merlin.dse_guidance import processing_unit_guidance as PUG
from merlin.dse_guidance import resource_hierarchy as RH
from merlin.dse_guidance import sharding as SH
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
    # full-corpus recaptures (prov.fqn via model2MLIR; small/random configs, structure real).
    # Studyable = parses with the ingest xDSL (shared `} -> (T1,T2)` normalizer) AND has linear-layer
    # GEMMs with prov.fqn roles. xr0's linears are batched (3D/4D activation x 2D weight) and bitvla's
    # are plain 2D -- both handled by extract_matmuls' leading-dim fold (attention bmms stay uncounted,
    # uniformly with the rest of the corpus, which counts linear-layer GEMMs).
    "rdt2": {"class": "diffusion/denoise_steps", "K": 5,
             "note": "RDT2 diffusion denoise step (depth 2, random init)"},
    "groot_n1d7": {"class": "diffusion/denoise_steps", "K": 4,
                   "note": "GR00T N1.5 flow-matching action head (2 layers, random init)"},
    "molmoact": {"class": "autoregressive_vla/action_token_decode", "K": 8,
                 "note": "MolmoAct causal LM forward (4 layers, random init)"},
    "smolvla": {"class": "flow_matching/denoise_steps", "K": 10,
                "note": "SmolVLA: SmolVLM2 backbone + action expert, denoise step (2 vlm layers)"},
    "pi05": {"class": "flow_matching/denoise_steps", "K": 10,
             "note": "pi0.5: PaliGemma backbone + gemma action expert, flow-matching step"},
    "xr0": {"class": "diffusion/denoise_steps", "K": 5,
            "note": "XR-0 batched-attention DiT denoise step (2 dit layers, random init); "
                    "K=5 from source num_steps (P19 config-drift fix; was 10)"},
    "bitvla": {"class": "autoregressive_vla/action_token_decode", "K": 7,
               "note": "BitVLA: BitNet ternary LM decode (2 layers, fp32 fake-quant capture)"},
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
    # K, H, control rate are the model's published architecture constants — source them from the
    # model registry (the model card / config), not a bare assumption (recovered_from_model_config).
    arch = M.MODEL_ARCH.get(workload)
    K = int(arch.loop_count) if (arch and arch.loop_count) else int(spec["K"])
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


# curated topic -> artifact map (the entry points a consumer actually wants).
_ARTIFACT_INDEX = {
    "readme": "README.md",
    "summary": "case_study_summary.md",
    "workload_contract": "workload_contract_table.csv",
    "requirements": "requirements_table.csv",
    "dse_readiness": "dse_readiness_summary.csv",
    "consolidated_knobs": "dse_search_space_knobs.yaml",
    "operator_geometry": "operator_shape_table.csv",
    "primitive_coverage": "primitive_coverage_matrix.csv",
    "contract_graph": "workload_contract_graph.yaml",
    "parallelism": "critical_path_table.csv",
    "sharding": "sharding_table.csv",
    "memory_envelope": "data_movement_table.csv",
    "dma_streams": "dma_stream_table.csv",
    "fusion_epilogue": "epilogue_pattern_table.csv",
    "boundary_matrix": "hw_sw_boundary_matrix.csv",
    "boundary_contracts": "boundary_candidate_contracts.yaml",
    "responsibility_matrix": "responsibility_split_matrix.csv",
    "measurement_priority": "measurement_priority_table.csv",
    "accuracy_gate": "accuracy_gate_report.md",
}


def _dse_contract_manifest(packages, knob_catalog, boundary_certs) -> dict:
    """One machine-readable object a DSE engine (or a human) loads to consume the whole package:
    per-workload readiness + facts, the consolidated knob groups, the boundary-placement top, the
    measurements still needed, and a topic->artifact index. Pointers, not duplication."""
    per_wl = {}
    missing_union: set = set()
    for p in packages:
        c = p["case"]
        r = p["readiness"]
        per_wl[c.workload] = {
            "class": c.cls, "K": c.K, "K_source": "recovered_from_model_config",
            "roles": sorted({reg.role for reg in c.attribution.regions}),
            "head_weight_bytes": _head_weight_bytes(c),
            "accuracy_int8": p["acc"], "ready_structural_dse": True,
            "ready_quantitative_dse": bool(r.ready),
            "missing_before_quantitative_dse": list(r.missing)}
        missing_union.update(r.missing)
    groups = knob_catalog["dse_search_space_knobs"]["knob_groups"]
    top_boundary = sorted(boundary_certs, key=lambda c: -c.boundary_pressure_score)[:8]
    return {
        "schema_version": "1",
        "generator": "merlin.dse_guidance case study (P1-P12)",
        "what_is_not_claimed": "no speedup, cycles, area, energy, or a chosen/ranked design; this "
                               "is a DSE search space + workload contract, not a selection. Magnitudes "
                               "are small random-init capture instances.",
        "workloads": sorted(per_wl),
        "per_workload": per_wl,
        "search_space_knob_groups": [
            {"group": g["group"], "source_phase": g["source_phase"], "enabled": g["enabled"],
             "n_knobs": len(g["knobs"])} for g in groups],
        "boundary_placement": {
            "n_abstractions": len(boundary_certs),
            "levels": list(BP.LEVELS),
            "score_is": "evidence_breadth (not performance/priority)",
            "top_by_evidence_breadth": [
                {"abstraction": c.abstraction, "boundary_pressure_score": c.boundary_pressure_score,
                 "strong_levels": [b["level"] for b in c.boundary_levels
                                   if b["status"] == "strong_candidate"],
                 "supporting_workloads": c.supporting_workloads} for c in top_boundary]},
        "measurements_needed_before_quantitative_dse": sorted(missing_union),
        "artifacts_index": dict(_ARTIFACT_INDEX),
    }


def _dse_search_space_knobs(all_shapes, all_axes, dags, units, overlap_by_wl, pat_by_wl,
                            region_mem_by_wl, boundary_certs=None) -> dict:
    """Consolidate the structural search-space knobs discovered across P5-P10 into one catalog —
    the bridge artifact a DSE engine consumes alongside the per-workload abstraction-axis template.
    Each knob group is grounded in the computed results, evidence-labeled; no speedup is claimed."""
    prim = [n for n, _, _ in PC.TILE_PRIMITIVES] + [n for n, _ in PC.GEMV_PRIMITIVES]
    clean_mn = sum(1 for a in all_axes if a.axis in ("M", "N") and a.shardable[8]
                   and not a.has_tail[8])
    avg_par = round(sum(d.available_parallelism for d in dags) / len(dags), 3) if dags else 0.0
    units_ir = sorted({u.unit for u in units if u.workloads})
    pipe_abs = sorted({a for cands in overlap_by_wl.values() for c in cands
                       if c.can_overlap == "yes" for a in c.required_abstractions})
    epi = {"bias": False, "activation": False, "scale": False, "clamp": False, "cast": False}
    for pats in pat_by_wl.values():
        for p in pats:
            epi["bias"] |= p.has_bias
            epi["activation"] |= p.has_activation
            epi["scale"] |= p.has_scale
            epi["clamp"] |= p.has_clamp
            epi["cast"] |= p.has_cast
    epi_ops = [k for k, v in epi.items() if v]
    groups = [
        {"group": "compute_primitive_shape", "source_phase": "P5", "knobs": prim, "enabled": True,
         "evidence": DE.E_DERIVED,
         "gated_by": "structural tile/lane coverage of the real operator geometry (no perf)"},
        {"group": "intra_op_sharding", "source_phase": "P7",
         "knobs": [f"shard_axis in {{M,N,K}}", f"shard_count in {list(SH.UNIT_COUNTS)}"],
         "enabled": clean_mn > 0, "evidence": DE.E_DERIVED,
         "gated_by": f"{clean_mn} reduction-free M/N shards available; K-sharding needs "
                     "partial-sum reduction"},
        {"group": "inter_op_parallelism", "source_phase": "P7", "knobs": ["num_engines"],
         "enabled": avg_par >= 1.5, "evidence": DE.E_DERIVED,
         "gated_by": f"avg inter-op parallelism {avg_par}x (low -> limited; favors intra-op "
                     "sharding)"},
        {"group": "processing_unit_set", "source_phase": "P7/P8", "knobs": units_ir,
         "enabled": True, "evidence": DE.E_IR,
         "gated_by": "distinct operator families (dense GEMM + skinny/GEMV) + epilogue + DMA"},
        {"group": "pipeline_overlap", "source_phase": "P8", "knobs": pipe_abs,
         "enabled": bool(pipe_abs), "evidence": DE.E_DERIVED,
         "gated_by": "candidate overlaps gated on recovered structure (backbone compute / control "
                     "loop); per-phase timing needed to schedule"},
        {"group": "memory_residency", "source_phase": "P9",
         "knobs": ["resident_weight_object", f"weight_dtype in {list(DE.CAPACITY_FORMATS)}",
                   "prefetch_depth"], "enabled": True, "evidence": DE.E_IR,
         "gated_by": "weight-dominated memory pressure; bandwidth needs a design YAML"},
        {"group": "dma_streams", "source_phase": "P9",
         "knobs": ["multi_stream_dma_descriptor", "double_buffered_activation_tile",
                   "prefetch_weight_once"], "enabled": True, "evidence": DE.E_IR,
         "gated_by": "3 byte-carrying streams/region (weight/activation/output)"},
        {"group": "epilogue_fusion", "source_phase": "P10",
         "knobs": [f"epilogue_op_set subset of {epi_ops}", "accumulator_dtype",
                   "requant_in_epilogue"], "enabled": bool(epi_ops), "evidence": DE.E_IR,
         "gated_by": "directly-fused epilogue slot proven (addmm bias); low-bit/scale gated by a "
                     "low-bit capture + accuracy"},
    ]
    if boundary_certs:
        top = sorted(boundary_certs, key=lambda c: -c.boundary_pressure_score)[:6]
        groups.append({
            "group": "hw_sw_boundary_placement", "source_phase": "P12",
            "knobs": [f"{c.abstraction}@{{compiler/runtime/command/isa/microcode/datapath}}"
                      for c in top],
            "enabled": True, "evidence": DE.E_DERIVED,
            "gated_by": "boundary placement is a search-space axis (Merlin does not choose); see "
                        "boundary_candidate_contracts.yaml + boundary_dse_knobs.yaml"})
    return {"dse_search_space_knobs": {
        "note": "consolidated STRUCTURAL search-space knobs discovered across P5-P12, each with its "
                "source phase, enabled status, evidence, and what gates it. This is the bridge a "
                "future DSE engine consumes together with the per-workload abstraction-axis "
                "template (dse_search_space_template.yaml). Structural only — no speedup, cycle, or "
                "area claim.",
        "knob_groups": groups,
        "what_is_not_claimed": "no speedup, no schedule, no chosen design; these are search-space "
                               "dimensions a DSE engine would explore, with the measurements each "
                               "needs named in the per-phase artifacts"}}


def _dse_search_space_knobs_md(catalog: dict) -> str:
    groups = catalog["dse_search_space_knobs"]["knob_groups"]
    L = ["# DSE search-space knobs (consolidated P5-P10)\n",
         "> The structural search-space dimensions the workload-contract analysis discovered, "
         "consolidated as the bridge a future DSE engine consumes (alongside the per-workload "
         "abstraction-axis `dse_search_space_template.yaml`). **Structural only — no speedup, no "
         "chosen design.**\n"]
    L.append("| knob group | phase | enabled | knobs | gated by |")
    L.append("|---|---|---|---|---|")
    for g in groups:
        knobs = ", ".join(str(k) for k in g["knobs"]) or "—"
        L.append(f"| {g['group']} | {g['source_phase']} | {g['enabled']} | {knobs} | "
                 f"{g['gated_by']} |")
    L.append("\nEach knob group is evidence-labeled and grounded in the per-phase artifacts "
             "(P5 geometry/coverage, P7 sharding/hierarchy, P8 pipeline, P9 memory/DMA, P10 "
             "epilogue). The measurements each knob needs before a *quantitative* DSE decision "
             "(per-unit throughput, bandwidth, accuracy for low-bit, per-phase timing) are named "
             "in those artifacts. **No speedup is claimed.**\n")
    return "\n".join(L)


def _readme_md(packages) -> str:
    names = ", ".join(p["case"].workload for p in packages)
    return (
        "# Merlin workload-contract analysis — case study\n\n"
        "**Consume this package as one object:** `dse_contract.json` is the machine-readable "
        "manifest (per-workload readiness + facts, the search-space knob groups, the boundary-"
        "placement top, the measurements still needed, and a topic→artifact index). Query it with "
        "`merlin-dse-guidance --query {summary,knobs,boundary[:abstraction],missing,index}`.\n\n"
        "**Meta-analysis (P13):** `merlin-dse-guidance --insight-mining` mines this package (per "
        "network + combined), scores DSE-usefulness, and extracts evidence-tiered presentation "
        "findings + plots into a regeneratable, **non-committed** timestamped run under `results/` "
        "(`<scope>_<UTCstamp>_dse_analysis/`).\n\n"
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
        "the **DSE search-space template** (per-workload abstraction axes + knobs).\n"
        "- `hw_sw_boundary_matrix.csv`, `boundary_candidate_contracts.yaml`, "
        "`boundary_placement_report.md`, `responsibility_split_matrix.csv`, "
        "`interface_contract_sketches.md`, `isa_candidate_primitives.yaml`, "
        "`runtime_object_candidates.yaml`, `command_isa_candidates.yaml`, `boundary_dse_knobs.yaml` "
        "— the **HW/SW boundary search space**: where each abstraction could live "
        "(compiler/runtime/command/ISA/microcode/datapath), what each side manages, and the "
        "compiler proof + DSE knobs each placement creates (Merlin generates options, does not "
        "choose; no speedup).\n"
        "- `dse_search_space_knobs.yaml`, `dse_search_space_knobs.md` — the **consolidated "
        "structural search-space knobs** discovered across P5-P12 (primitive shapes, sharding, "
        "processing units, pipeline overlap, memory/DMA, epilogue fusion, boundary placement) — the "
        "capstone bridge a DSE engine consumes.\n"
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
        "- `dag_parallelism_report.md`, `critical_path_table.csv`, `concurrency_windows.csv`, "
        "`parallel_region_candidates.yaml` — inter-op DAG concurrency (work/span, not a speedup).\n"
        "- `sharding_table.csv`, `sharding_opportunities.yaml`, `intra_op_sharding_report.md` — "
        "per-matmul M/N/K sharding geometry + required reduction/broadcast abstractions.\n"
        "- `operator_cluster_to_hierarchy.csv`, `parallel_hierarchy_hints.yaml`, "
        "`resource_pressure_table.csv`, `processing_unit_candidates.yaml`, "
        "`processing_unit_parallelism_report.md` — hierarchical resource analysis: which "
        "processing-unit shapes the workloads imply (one bigger / many identical / specialized).\n"
        "- `pipeline_envelope.yaml`, `pipeline_stage_table.csv` — multi-rate phase model "
        "(cadence per phase). `pipeline_candidates.yaml`, `buffering_requirement_table.csv`, "
        "`overlap_opportunities.md` — candidate phase overlaps + the buffer/event/queue "
        "abstractions each requires (structural, not scheduled).\n"
        "- `processing_unit_guidance.yaml`, `heterogeneity_report.md` — monolithic vs. replicated "
        "vs. heterogeneous evidence + the search-space implication (evidence only, no selection).\n"
        "- `memory_hierarchy_envelope.yaml`, `data_movement_table.csv`, `reuse_lifetime_table.csv`, "
        "`memory_abstraction_candidates.yaml`, `memory_envelope_report.md` — per-region byte "
        "envelope (weight/activation/output + dtype-scaled resident set), reuse/residency, and the "
        "memory abstractions implied (no bandwidth claim).\n"
        "- `dma_stream_table.csv`, `buffer_requirement_table.csv`, `dma_pressure_report.md` — "
        "structural data-movement streams + minimum buffering per region (no bandwidth/deadline "
        "claim).\n"
        "- `epilogue_pattern_table.csv`, `accumulator_contract_table.csv`, "
        "`numerical_epilogue_candidates.yaml`, `lost_numerical_contracts.csv`, "
        "`fusion_opportunity_report.md` — fusion/epilogue/accumulator placement: detected "
        "matmul-epilogue patterns, accumulator + dequant/requant contract, fused abstraction "
        "certificates, and the numerical contracts the flat capture erased (no low-bit perf claim).\n"
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


_FULL_INV_COLS = ["workload", "source", "op_index", "op_class", "prov_op", "prov_fqn", "role",
                  "M", "K", "N", "batch", "macs"]
_WORK_COV_COLS = ["workload", "n_linear_matmul", "linear_gemm_macs", "n_attention_ops",
                  "attention_macs", "n_batched_matmul", "batched_matmul_macs",
                  "total_recovered_macs", "visible_linear_fraction", "n_softmax",
                  "n_normalization", "n_conv", "n_activation", "n_elementwise", "n_reduction",
                  "n_layout", "n_other"]


def operator_full_inventory_csv(cases: list[WorkloadCase]) -> str:
    """Every recovered operator (linear-GEMM `linalg.matmul` + the `linalg.generic` ops the flat
    capture lowered but kept: attention/softmax/norm/conv/elementwise) with class + MACs where
    recoverable. The complete op graph the named-matmul view (operator_shape_table) is a subset of."""
    rows: list[dict] = []
    for c in cases:
        cap = str(_recap_dir(c.workload))
        for r in ATTR.extract_matmuls(cap):
            rows.append({"workload": c.workload, "source": "linalg.matmul", "op_index": r.index,
                         "op_class": "linear_gemm", "prov_op": r.op or "", "prov_fqn": r.fqn or "",
                         "role": ATTR.role_from_fqn(r.fqn) or "", "M": r.M, "K": r.K, "N": r.N,
                         "batch": 1, "macs": r.macs})
        for r in ATTR.extract_non_gemm_ops(cap):
            rows.append({"workload": c.workload, "source": "linalg.generic", "op_index": r.index,
                         "op_class": r.op_class, "prov_op": r.prov_op or "", "prov_fqn": r.fqn or "",
                         "role": r.role or "", "M": r.M, "K": r.K, "N": r.N, "batch": r.batch,
                         "macs": r.macs})
    return _csv(rows, _FULL_INV_COLS)


def work_coverage_csv(cases: list[WorkloadCase]) -> str:
    """Per-workload MAC accounting: linear-GEMM vs attention MAC mass (both recovered from IR shapes)
    + op-class counts. `visible_linear_fraction = linear / (linear+attention)` answers "how much of the
    compute is the linear-GEMM geometry this study analyzes" — attention MACs are real, not estimated;
    softmax/norm/conv/elementwise are counted (their MACs are memory/elementwise-bound, not reported)."""
    from collections import Counter
    rows: list[dict] = []
    for c in cases:
        cap = str(_recap_dir(c.workload))
        mm, ng = ATTR.extract_matmuls(cap), ATTR.extract_non_gemm_ops(cap)
        lin = sum(r.macs for r in mm)
        attn = sum(r.macs for r in ng if r.op_class == ATTR.OPC_ATTENTION)
        bmm = sum(r.macs for r in ng if r.op_class == ATTR.OPC_BATCHED_MATMUL)
        cls = Counter(r.op_class for r in ng)
        tot = lin + attn + bmm
        rows.append({"workload": c.workload, "n_linear_matmul": len(mm), "linear_gemm_macs": lin,
                     "n_attention_ops": cls.get(ATTR.OPC_ATTENTION, 0), "attention_macs": attn,
                     "n_batched_matmul": cls.get(ATTR.OPC_BATCHED_MATMUL, 0), "batched_matmul_macs": bmm,
                     "total_recovered_macs": tot,
                     "visible_linear_fraction": round(lin / tot, 4) if tot else 1.0,
                     "n_softmax": cls.get(ATTR.OPC_SOFTMAX, 0),
                     "n_normalization": cls.get(ATTR.OPC_NORM, 0), "n_conv": cls.get(ATTR.OPC_CONV, 0),
                     "n_activation": cls.get(ATTR.OPC_ACTIVATION, 0),
                     "n_elementwise": cls.get(ATTR.OPC_ELEMENTWISE, 0),
                     "n_reduction": cls.get(ATTR.OPC_REDUCTION, 0),
                     "n_layout": cls.get(ATTR.OPC_LAYOUT, 0), "n_other": cls.get(ATTR.OPC_OTHER, 0)})
    return _csv(rows, _WORK_COV_COLS)


# P18 Stage B: capture-level ablation. The raw multi-level recaptures (recaptures_levels/, ~18 MB)
# are regenerable via m2m flags and are gitignored; only this SMALL op-count summary is committed.
_ABLATION_OPS = ["linalg.matmul", "linalg.generic", "linalg_ext.softmax", "linalg_ext.layer_norm",
                 "quant_ext.dequantize", "scf.for"]
_ABLATION_LEVELS = [("flat", "recaptures", "model.mlir"),
                    ("high_level", "recaptures_levels", "model_highlevel.mlir"),
                    ("quant_qdq", "recaptures_levels", "model_qdq.mlir")]
_ABLATION_CSV_COLS = (["workload", "level", "available"]
                      + [o.replace(".", "_") for o in _ABLATION_OPS])


def capture_level_ablation_csv() -> str:
    """Op-vocabulary per (workload, capture level), read from the local recaptures_levels/ (which are
    gitignored + regenerable via `workloads/capture.py <wl> --level high-level` and `--formats int8`).
    Returns '' if no multi-level recaptures are present (then the committed summary is left as-is)."""
    bench = paths.merlin_dir() / "benchmarks" / "dse_guidance"
    lvl = bench / "recaptures_levels"
    wls = sorted(p.name for p in lvl.glob("*")
                 if (p / "model_highlevel.mlir").is_file() or (p / "model_qdq.mlir").is_file()) \
        if lvl.is_dir() else []
    if not wls:
        return ""
    rows = []
    for w in wls:
        for level, sub, fname in _ABLATION_LEVELS:
            p = bench / sub / w / fname
            row = {"workload": w, "level": level, "available": p.is_file()}
            txt = p.read_text(errors="ignore") if p.is_file() else ""
            for o in _ABLATION_OPS:
                row[o.replace(".", "_")] = txt.count(o)
            rows.append(row)
    return _csv(rows, _ABLATION_CSV_COLS)


# P21-S1: loop-preserving recovery (scf.for from torch.while_loop). Reads the
# loop-preserving captures (recaptures_loop/<w>/model.mlir) and reports K, the
# loop-carried state and the KV cache RECOVERED FROM IR — the before/after that
# closes the assumed-K / KV-state / loop-carried / region-role caveats.
_LOOP_RECOVERY_COLS = ["workload", "loop_preserved", "K", "K_source", "n_iter_args",
                       "loop_carried_roles", "kv_cache_bytes", "repeated_region_op_count",
                       "flat_view", "recovered_view", "evidence"]


def loop_recovery_csv() -> str:
    """'' if no loop-preserving captures are present (committed summary left as-is)."""
    from merlin.dse_guidance.loop_recovery import recover_loop
    bench = paths.merlin_dir() / "benchmarks" / "dse_guidance"
    loop_dir = bench / "recaptures_loop"
    if not loop_dir.is_dir():
        return ""
    wls = sorted(p.name for p in loop_dir.glob("*") if (p / "model.mlir").is_file())
    if not wls:
        return ""
    rows = []
    for w in wls:
        lr = recover_loop(loop_dir / w / "model.mlir", w)
        if not lr.present:
            continue
        roles = ",".join(c.role for c in lr.carried_state)
        rows.append({
            "workload": w, "loop_preserved": True, "K": lr.K, "K_source": lr.K_source,
            "n_iter_args": lr.n_iter_args, "loop_carried_roles": roles,
            "kv_cache_bytes": lr.kv_cache_bytes if lr.kv_cache_bytes else "n/a",
            "repeated_region_op_count": lr.repeated_region_op_count,
            "flat_view": "loop unrolled by torch.export; K assumed (config); KV/latent erased",
            "recovered_view": f"scf.for(0,{lr.K},1); {lr.n_iter_args} iter_args carried "
                              f"({roles}); repeated region = {lr.repeated_region_op_count} ops",
            "evidence": "recovered_from_ir (scf.for, prov.op=while_loop)",
        })
    return _csv(rows, _LOOP_RECOVERY_COLS) if rows else ""


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
        # P6: multi-rate workload contract graph (joins topo/attribution/nc/geometry/state +
        # real operator data dependencies recovered from the SSA use-def graph)
        graphs.append(CGRAPH.build_graph(c, shapes, nc, recs,
                                         dependencies=ATTR.matmul_dependencies(cap)))
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
    # P18: the FULL recovered op graph (linear GEMM + attention/softmax/norm/conv/elementwise lowered
    # to linalg.generic but kept) + the per-workload linear-vs-attention MAC accounting. Additive —
    # operator_shape_table (the named-matmul subset used by P5-P16) is unchanged.
    Artifact("operator_full_inventory.csv", operator_full_inventory_csv(cases)).write(out)
    Artifact("work_coverage_table.csv", work_coverage_csv(cases)).write(out)
    abl = capture_level_ablation_csv()       # only written when the local level recaptures exist
    if abl:
        Artifact("capture_level_ablation.csv", abl).write(out)
    lrcsv = loop_recovery_csv()              # P21-S1: only when loop-preserving captures exist
    if lrcsv:
        Artifact("loop_preserving_recovery.csv", lrcsv).write(out)
    # P21 S2/S3: deployment-real magnitudes (depth x n_layers, config-exact) + KV sizing
    from merlin.dse_guidance import real_config as RC
    _loopdir = paths.merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures_loop"
    Artifact("real_config_magnitudes.csv", RC.magnitudes_csv()).write(out)
    Artifact("kv_cache_sizing.csv", RC.kv_sizing_csv(_loopdir)).write(out)
    # P20 Tool A: Timeloop-native mapspace seeds from the recovered contraction ops (structural; no perf)
    _wl_caps = [(c.workload, str(_recap_dir(c.workload))) for c in cases]
    Artifact("dataflow_candidate_table.csv",
             _csv(MS.dataflow_rows(_wl_caps), MS._DATAFLOW_COL)).write(out)
    yaml_artifact("timeloop_problem_shapes.yaml", MS.problem_shapes(_wl_caps),
                  header="timeloop_problem_shapes (structural search-space seeds; no perf claim)").write(out)
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
    # P7 parallelism / sharding / hierarchical resource analysis (structural; no speedup)
    dags = [PAR.analyze_graph(g) for g in graphs]
    Artifact("dag_parallelism_report.md", PAR.report_md(dags)).write(out)
    Artifact("critical_path_table.csv", PAR.critical_path_csv(dags)).write(out)
    Artifact("concurrency_windows.csv", PAR.concurrency_windows_csv(dags)).write(out)
    yaml_artifact("parallel_region_candidates.yaml",
                  PAR.parallel_region_candidates_yaml(dags),
                  header="parallel_region_candidates (structural; no speedup)").write(out)
    shard_by_workload = {w: SH.all_shard_axes(s) for w, s in geom_by_workload.items()}
    all_axes = SH.all_shard_axes(all_shapes)
    Artifact("sharding_table.csv", SH.sharding_csv(all_axes)).write(out)
    yaml_artifact("sharding_opportunities.yaml",
                  SH.sharding_opportunities_yaml(shard_by_workload),
                  header="sharding_opportunities (structural; no speedup)").write(out)
    Artifact("intra_op_sharding_report.md", SH.report_md(shard_by_workload, all_axes)).write(out)
    clusters = RH.cluster_hierarchy(all_shapes)
    pressure = RH.resource_pressure(all_shapes, dags)
    units = RH.processing_unit_candidates(all_shapes, pressure)
    Artifact("operator_cluster_to_hierarchy.csv",
             RH.cluster_to_hierarchy_csv(clusters)).write(out)
    structural_hints = RH.structural_hierarchy_hints(all_shapes, all_axes, dags)
    yaml_artifact("parallel_hierarchy_hints.yaml",
                  RH.hierarchy_hints_yaml(clusters, structural_hints),
                  header="parallel_hierarchy_hints (structural; no speedup)").write(out)
    Artifact("resource_pressure_table.csv", RH.resource_pressure_csv(pressure)).write(out)
    yaml_artifact("processing_unit_candidates.yaml",
                  RH.processing_unit_candidates_yaml(units),
                  header="processing_unit_candidates (structural; no speedup)").write(out)
    Artifact("processing_unit_parallelism_report.md",
             RH.processing_unit_report_md(units, pressure, clusters, dags)).write(out)
    # P8 pipeline / multi-rate overlap / processing-unit multiplicity guidance (structural)
    from merlin.dse_guidance import models as _M
    _VLA_FAMILIES = {"flow_matching", "diffusion", "autoregressive_vla"}

    def _has_control_loop(w: str) -> bool:
        a = _M.MODEL_ARCH.get(w)
        return bool(a and a.family in _VLA_FAMILIES and a.control_rate_hz and a.action_horizon)

    graph_by_wl = {g.workload: g for g in graphs}
    phase_by_wl = {w: PE.phase_model(graph_by_wl[w], geom_by_workload[w])
                   for w in geom_by_workload}
    overlap_by_wl = {w: PE.overlap_candidates(graph_by_wl[w], phase_by_wl[w],
                                              has_control_loop=_has_control_loop(w))
                     for w in phase_by_wl}
    yaml_artifact("pipeline_envelope.yaml", PE.pipeline_envelope_yaml(phase_by_wl),
                  header="pipeline_envelope (multi-rate phase model; no speedup)").write(out)
    Artifact("pipeline_stage_table.csv", PE.pipeline_stage_csv(phase_by_wl)).write(out)
    yaml_artifact("pipeline_candidates.yaml", PE.pipeline_candidates_yaml(overlap_by_wl),
                  header="pipeline_candidates (structural overlap; no speedup)").write(out)
    Artifact("buffering_requirement_table.csv",
             PE.buffering_requirement_csv(overlap_by_wl)).write(out)
    Artifact("overlap_opportunities.md",
             PE.overlap_report_md(phase_by_wl, overlap_by_wl)).write(out)
    pug = PUG.guidance(all_shapes, dags, pressure, all_axes)
    yaml_artifact("processing_unit_guidance.yaml", PUG.guidance_yaml(pug),
                  header="processing_unit_guidance (evidence only; no selection)").write(out)
    Artifact("heterogeneity_report.md", PUG.heterogeneity_report_md(pug)).write(out)
    # P9 memory / DMA / buffer / data-movement envelope (structural; no bandwidth/speedup)
    region_mem_by_wl = {c.workload: ME.region_memory(c.attribution, geom_by_workload[c.workload])
                        for c in cases}
    reuse_by_wl = {w: ME.reuse_lifetime(rm) for w, rm in region_mem_by_wl.items()}
    stream_by_wl = {w: DMA.all_streams(rm) for w, rm in region_mem_by_wl.items()}
    buf_by_wl = {w: DMA.buffer_requirements(rm) for w, rm in region_mem_by_wl.items()}
    yaml_artifact("memory_hierarchy_envelope.yaml",
                  ME.memory_hierarchy_yaml(region_mem_by_wl),
                  header="memory_hierarchy_envelope (structural; no bandwidth)").write(out)
    Artifact("data_movement_table.csv", ME.data_movement_csv(region_mem_by_wl)).write(out)
    # P20 Tool B: per-operand locality + resident-capacity-by-dtype (reads the data_movement just written)
    Artifact("operand_locality_table.csv", OL.locality_csv(out)).write(out)
    Artifact("capacity_requirement_table.csv", OL.capacity_csv(out)).write(out)
    Artifact("cache_vs_scratchpad_boundary.md", OL.boundary_md(out)).write(out)
    # P20 Tool E: low-bit quant metadata from the qdq recaptures (only when present); committed summary
    _qm = QM.quant_csv(out)
    if _qm:
        Artifact("quant_metadata_visibility.csv", _qm).write(out)
        Artifact("lowbit_capture_requirements.md", QM.requirements_md(out)).write(out)
    # P21-S4: native low-bit datapath (bitvla packed-int2 ternary), when the native capture exists
    _nat = QM.native_csv(out)
    if _nat:
        Artifact("native_lowbit_datapath.csv", _nat).write(out)
    Artifact("reuse_lifetime_table.csv", ME.reuse_lifetime_csv(reuse_by_wl)).write(out)
    yaml_artifact("memory_abstraction_candidates.yaml",
                  ME.memory_abstraction_candidates_yaml(reuse_by_wl),
                  header="memory_abstraction_candidates (structural)").write(out)
    Artifact("memory_envelope_report.md",
             ME.memory_envelope_report_md(region_mem_by_wl)).write(out)
    Artifact("dma_stream_table.csv", DMA.dma_stream_csv(stream_by_wl)).write(out)
    Artifact("buffer_requirement_table.csv", DMA.buffer_requirement_csv(buf_by_wl)).write(out)
    Artifact("dma_pressure_report.md",
             DMA.dma_pressure_report_md(stream_by_wl, region_mem_by_wl)).write(out)
    # P10 fusion / epilogue / accumulator / numerical-contract integration (structural; no speedup)
    pat_by_wl = {c.workload: FE.epilogue_patterns(str(_recap_dir(c.workload))) for c in cases}
    acc_by_wl: dict = {}
    cert_by_wl: dict = {}
    for p in packages:
        c = p["case"]
        cap = str(_recap_dir(c.workload))
        pats = pat_by_wl[c.workload]
        accs = FE.accumulator_contract(ATTR.extract_matmuls(cap), c.attribution, p["nc"], pats)
        acc_by_wl[c.workload] = accs
        cert_by_wl[c.workload] = FE.certificates(pats, accs)
    Artifact("epilogue_pattern_table.csv", FE.epilogue_pattern_csv(pat_by_wl)).write(out)
    Artifact("accumulator_contract_table.csv",
             FE.accumulator_contract_csv(acc_by_wl)).write(out)
    yaml_artifact("numerical_epilogue_candidates.yaml",
                  FE.epilogue_candidates_yaml(cert_by_wl),
                  header="numerical_epilogue_candidates (structural; no low-bit perf)").write(out)
    Artifact("lost_numerical_contracts.csv", FE.lost_contracts_csv(acc_by_wl)).write(out)
    Artifact("fusion_opportunity_report.md",
             FE.fusion_report_md(pat_by_wl, acc_by_wl, cert_by_wl)).write(out)
    # P12 HW/SW boundary-placement analysis (the boundary search space; Merlin does not choose)
    ev_ctx = {}
    for c in cases:
        sh = geom_by_workload[c.workload]
        ev_ctx[c.workload] = {
            "dense": any(s.shape_class == "squareish_gemm" for s in sh),
            "gemv": any(s.shape_class in ("gemv_like", "wide_skinny", "tall_skinny") for s in sh),
            "backbone": any(s.region_role == "backbone_once" for s in sh),
            "epilogue": any(p.has_bias for p in pat_by_wl[c.workload]),
            "k_loop": c.K > 1, "control_loop": _has_control_loop(c.workload),
            "decode": c.topo.workload_class == TOP.CLASS_AUTOREGRESSIVE}
    cp_proofs = {r.axis: (r.compiler_proof_needed, r.status) for r in CP.proof_matrix(packages)}
    boundary_certs = BP.build_certificates(ev_ctx, cp_proofs)
    resp_rows = BP.responsibility_rows()
    Artifact("hw_sw_boundary_matrix.csv",
             BP.hw_sw_boundary_matrix_csv(boundary_certs)).write(out)
    yaml_artifact("boundary_candidate_contracts.yaml",
                  BP.boundary_candidate_contracts_yaml(boundary_certs),
                  header="boundary_candidate_contracts (search space; no speedup)").write(out)
    Artifact("boundary_placement_report.md",
             BP.boundary_report_md(boundary_certs, resp_rows)).write(out)
    Artifact("responsibility_split_matrix.csv",
             BP.responsibility_split_csv(resp_rows)).write(out)
    Artifact("interface_contract_sketches.md",
             BP.interface_contract_sketches_md(boundary_certs)).write(out)
    yaml_artifact("isa_candidate_primitives.yaml",
                  BP.isa_candidate_primitives_yaml(boundary_certs),
                  header="isa_candidate_primitives (sketch; no speedup)").write(out)
    yaml_artifact("runtime_object_candidates.yaml",
                  BP.runtime_object_candidates_yaml(boundary_certs),
                  header="runtime_object_candidates (sketch; no speedup)").write(out)
    yaml_artifact("command_isa_candidates.yaml",
                  BP.command_isa_candidates_yaml(boundary_certs),
                  header="command_isa_candidates (sketch; no speedup)").write(out)
    yaml_artifact("boundary_dse_knobs.yaml", BP.boundary_dse_knobs_yaml(boundary_certs),
                  header="boundary_dse_knobs (search-space knobs; no speedup)").write(out)
    # Consolidated DSE search-space knobs across P5-P12 (the bridge a DSE engine consumes)
    knob_catalog = _dse_search_space_knobs(all_shapes, all_axes, dags, units, overlap_by_wl,
                                           pat_by_wl, region_mem_by_wl, boundary_certs)
    yaml_artifact("dse_search_space_knobs.yaml", knob_catalog,
                  header="dse_search_space_knobs (consolidated P5-P10; no speedup)").write(out)
    Artifact("dse_search_space_knobs.md", _dse_search_space_knobs_md(knob_catalog)).write(out)
    # Single machine-readable manifest a DSE engine / human loads to consume the whole package
    import json as _json
    manifest = _dse_contract_manifest(packages, knob_catalog, boundary_certs)
    Artifact("dse_contract.json",
             _json.dumps(manifest, indent=2, sort_keys=True) + "\n").write(out)
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
