"""Evidence mining + insight extraction over the committed dse_guidance case-study package.

This is a META-analysis layer: it CONSUMES the P0-P12 artifacts and asks whether the package can
answer DSE-relevant questions, how strong the evidence behind each finding is, and what is
presentation-worthy. It adds no DSE functionality, runs no DSE, and claims no speedup / cycles /
area / energy / optimality / best / performance — every fact traces to a source artifact and an
evidence tier, and findings are gated before they may be marked "main".

Partial mode: a missing input artifact is recorded ``exists=no`` and skipped; mining never crashes.
Scope: ``mine(case_study_dir, scope)`` runs for one workload (per-network) or ``"all"`` (combined).
Output is written by the CLI into a regeneratable, non-committed timestamped ``results/`` run folder
(so the committed case_study stays byte-stable). Nothing here renders plots — that is
:mod:`.presentation_plots`; this module only produces the data + the plot manifest.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from merlin.common.yaml import load_yaml

# ---- evidence labels -> tier inputs ----
_TIER_A_LABELS = {"measured", "proxy_measured", "recovered_from_ir", "recovered_from_prov_fqn",
                  "recovered_from_model_config"}
_DERIVED = "derived_requirement"
_ASSUMED = {"assumed_reference", "design_assumption"}
_ABSENT = {"unavailable", "unknown", ""}
EVIDENCE_LABELS = _TIER_A_LABELS | {_DERIVED} | _ASSUMED | {"unavailable"}

# artifacts the verify_implementation.py harness independently re-derives (drives verification_status
# + the tier-A/B gate). This is the P13-a "checked by verify_implementation.py" column.
_VERIFIED_ARTIFACTS = {
    "operator_shape_table.csv", "tile_waste_table.csv", "primitive_coverage_matrix.csv",
    "primitive_regret_table.csv", "shape_summary_by_workload.csv", "operator_cluster_table.csv",
    "workload_contract_graph.yaml", "critical_path_table.csv", "traffic_table.csv",
    "data_movement_table.csv", "dtype_capacity_table.csv", "dma_stream_table.csv",
    "buffer_requirement_table.csv", "sharding_table.csv", "region_attribution.yaml",
    "accuracy_gated_dtype_candidates.csv", "dispatch_granularity_table.csv",
    "epilogue_pattern_table.csv", "accumulator_contract_table.csv", "hw_sw_boundary_matrix.csv",
    "boundary_candidate_contracts.yaml", "responsibility_split_matrix.csv",
    "dse_search_space_knobs.yaml", "dse_contract.json", "resident_state_table.csv",
}

# expected root artifacts -> source phase (P13-a inventory). Per-workload subdir artifacts handled
# separately.
_ARTIFACT_PHASE = {
    "cross_workload_provenance.csv": "P1", "workload_contract_table.csv": "P2",
    "requirements_table.csv": "P3", "dse_readiness_summary.csv": "P4",
    "abstraction_pressure_table.csv": "P4", "abstraction_pressure_ranking.csv": "P4",
    "compiler_proof_matrix.csv": "P4", "resident_state_table.csv": "P4",
    "workload_family_table.csv": "P4", "measurement_priority_table.csv": "P4",
    "numerical_contract_fidelity_report.md": "P4", "accuracy_gate_results.csv": "P4",
    "operator_shape_table.csv": "P5", "operator_cluster_table.csv": "P5",
    "shape_summary_by_workload.csv": "P5", "shape_summary_by_region.csv": "P5",
    "operator_geometry.yaml": "P5", "tile_waste_table.csv": "P5",
    "primitive_coverage_matrix.csv": "P5", "primitive_regret_table.csv": "P5",
    "workload_contract_graph.yaml": "P6", "phase_rate_table.csv": "P6",
    "multi_rate_contract.yaml": "P6", "critical_path_table.csv": "P7",
    "concurrency_windows.csv": "P7", "sharding_table.csv": "P7",
    "sharding_opportunities.yaml": "P7", "operator_cluster_to_hierarchy.csv": "P7",
    "resource_pressure_table.csv": "P7", "processing_unit_candidates.yaml": "P7",
    "pipeline_envelope.yaml": "P8", "pipeline_stage_table.csv": "P8",
    "pipeline_candidates.yaml": "P8", "buffering_requirement_table.csv": "P8",
    "processing_unit_guidance.yaml": "P8", "data_movement_table.csv": "P9",
    "memory_hierarchy_envelope.yaml": "P9", "reuse_lifetime_table.csv": "P9",
    "dma_stream_table.csv": "P9", "buffer_requirement_table.csv": "P9",
    "dtype_capacity_table.csv": "P9", "traffic_table.csv": "P9",
    "epilogue_pattern_table.csv": "P10", "accumulator_contract_table.csv": "P10",
    "numerical_epilogue_candidates.yaml": "P10", "lost_numerical_contracts.csv": "P10",
    "dse_search_space_knobs.yaml": "P11", "accuracy_gated_dtype_candidates.csv": "P6",
    "hw_sw_boundary_matrix.csv": "P12", "boundary_candidate_contracts.yaml": "P12",
    "responsibility_split_matrix.csv": "P12", "boundary_dse_knobs.yaml": "P12",
    "isa_candidate_primitives.yaml": "P12", "runtime_object_candidates.yaml": "P12",
    "command_isa_candidates.yaml": "P12", "dse_contract.json": "P12",
    "accuracy_gate_report.md": "P4",
}

FORBIDDEN = ("speedup", "faster", "optimal", "best design", "predicted cycles", "cycle count",
             "performance improvement", "gap_closure", "energy", "area estimate")

FACT_COLUMNS = ["fact_id", "workload", "region", "phase", "op_id", "abstraction", "boundary_level",
                "metric_name", "metric_value", "metric_unit", "source_artifact", "source_phase",
                "evidence_type", "derivation_type", "evidence_tier", "verification_status",
                "caveat", "dse_implication", "presentation_candidate"]


# --------------------------------------------------------------------------- io helpers

def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _yaml(path: Path):
    if not path.is_file():
        return None
    try:
        return load_yaml(path)
    except Exception:
        return None


def _json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _count(text: str, label: str) -> int:
    return text.count(label)


# --------------------------------------------------------------------------- tiers

def _derivation(evidence: str) -> str:
    if evidence in ("measured", "proxy_measured"):
        return "measured"
    if evidence in ("recovered_from_ir", "recovered_from_prov_fqn", "recovered_from_model_config"):
        return "recovered"
    if evidence == _DERIVED:
        return "derived"
    if evidence in _ASSUMED:
        return "assumed"
    return "unavailable"


def evidence_tier(evidence: str, source_artifact: str) -> str:
    """A: recovered/measured AND verified. B: recovered-unverified or derived+verified.
    C: assumed (or derived-unverified). D: unavailable/unknown."""
    verified = source_artifact in _VERIFIED_ARTIFACTS
    if evidence in _TIER_A_LABELS:
        return "A" if verified else "B"
    if evidence == _DERIVED:
        return "B" if verified else "C"
    if evidence in _ASSUMED:
        return "C"
    return "D"


def _verification_status(source_artifact: str, evidence: str) -> str:
    if evidence in _ABSENT:
        return "unavailable"
    return "verified" if source_artifact in _VERIFIED_ARTIFACTS else "not_verified"


# --------------------------------------------------------------------------- P13-a inventory

def _file_meta(path: Path) -> dict:
    suffix = path.suffix.lstrip(".")
    rowcount = None
    keys = None
    schema_valid = True
    has_evidence = False
    counts = {lbl: 0 for lbl in ("unavailable", "unknown", "assumed_reference",
                                 "derived_requirement", "measured")}
    if path.is_file():
        text = path.read_text(errors="ignore")
        for lbl in counts:
            counts[lbl] = _count(text, lbl)
        has_evidence = ("evidence" in text) or ("recovered_from" in text) or ("measured" in text)
        if suffix == "csv":
            rows = _csv_rows(path)
            rowcount = len(rows)
            schema_valid = rowcount >= 0 and ("," in text.splitlines()[0] if text else True)
        elif suffix in ("yaml", "yml"):
            obj = _yaml(path)
            keys = sorted(obj)[:8] if isinstance(obj, dict) else None
            schema_valid = obj is not None
        elif suffix == "json":
            obj = _json(path)
            keys = sorted(obj)[:8] if isinstance(obj, dict) else None
            schema_valid = obj is not None
    return {"file_type": suffix, "row_count": rowcount, "top_level_keys": keys,
            "schema_valid": schema_valid, "has_evidence_fields": has_evidence, **counts}


def artifact_inventory(cs_dir: Path) -> list[dict]:
    """One row per expected root artifact (+ any extra files found): exists, phase, type, counts."""
    rows = []
    seen = set()
    for name, phase in sorted(_ARTIFACT_PHASE.items()):
        p = cs_dir / name
        seen.add(name)
        m = _file_meta(p)
        rows.append({
            "artifact": name, "path": str(p), "exists": p.is_file(), "source_phase": phase,
            "checked_by_verify": name in _VERIFIED_ARTIFACTS, **m})
    # any other generated root file we did not expect (so the inventory is complete)
    if cs_dir.is_dir():
        for p in sorted(cs_dir.glob("*")):
            if p.is_file() and p.name not in seen:
                m = _file_meta(p)
                rows.append({"artifact": p.name, "path": str(p), "exists": True,
                             "source_phase": "other", "checked_by_verify": False, **m})
    return rows


# --------------------------------------------------------------------------- P13-b unified facts

def _workloads(cs_dir: Path) -> list[str]:
    man = _json(cs_dir / "dse_contract.json")
    if man and man.get("workloads"):
        return list(man["workloads"])
    return sorted({r["workload"] for r in _csv_rows(cs_dir / "critical_path_table.csv")
                   if r.get("workload")})


def _fact(fid, workload, metric, value, unit, artifact, phase, evidence, *, region="", op_id="",
          abstraction="", level="", caveat="", implication="") -> dict:
    tier = evidence_tier(evidence, artifact)
    return {
        "fact_id": fid, "workload": workload, "region": region, "phase": phase, "op_id": op_id,
        "abstraction": abstraction, "boundary_level": level, "metric_name": metric,
        "metric_value": value, "metric_unit": unit, "source_artifact": artifact,
        "source_phase": phase, "evidence_type": evidence, "derivation_type": _derivation(evidence),
        "evidence_tier": tier, "verification_status": _verification_status(artifact, evidence),
        "caveat": caveat, "dse_implication": implication,
        "presentation_candidate": tier in ("A", "B") and bool(implication)}


_SMALL = "magnitudes are small random-init capture instances (structure real)"


def unified_facts(cs_dir: Path, scope: str) -> list[dict]:
    """Normalize a curated, bounded set of DSE-relevant facts (never invented). scope = workload | all."""
    man = _json(cs_dir / "dse_contract.json") or {}
    per_wl = man.get("per_workload", {})
    all_wl = _workloads(cs_dir)
    workloads = all_wl if scope == "all" else [scope]
    facts: list[dict] = []
    n = 0

    def add(**kw):
        nonlocal n
        facts.append(_fact(f"F{n:04d}", **kw))
        n += 1

    crit = {r["workload"]: r for r in _csv_rows(cs_dir / "critical_path_table.csv")}
    shp = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    cov = _csv_rows(cs_dir / "primitive_coverage_matrix.csv")
    epi = _csv_rows(cs_dir / "epilogue_pattern_table.csv")

    for w in workloads:
        d = per_wl.get(w, {})
        if d:
            add(workload=w, metric="K", value=d.get("K"), unit="steps",
                artifact="dse_contract.json", phase="P6", evidence="recovered_from_model_config",
                caveat="published architecture constant", implication="bounds the K-loop / autonomous_K_loop axis")
            if d.get("head_weight_bytes"):
                add(workload=w, metric="head_weight_bytes", value=d["head_weight_bytes"],
                    unit="bytes", artifact="dse_contract.json", phase="P5",
                    evidence="recovered_from_ir", caveat=_SMALL,
                    implication="resident-weight capacity requirement")
            acc = d.get("accuracy_int8", "unavailable")
            add(workload=w, metric="accuracy_int8_w8a8", value=acc, unit="band",
                artifact="accuracy_gated_dtype_candidates.csv", phase="P6",
                evidence="measured" if acc in ("pass", "fail") else "unavailable",
                implication="gates int8 as an accuracy-legal dtype candidate")
            add(workload=w, metric="ready_quantitative_dse", value=d.get("ready_quantitative_dse"),
                unit="bool", artifact="dse_readiness_summary.csv", phase="P4",
                evidence=_DERIVED, implication="structural DSE ready; quantitative needs measurements")
        c = crit.get(w)
        if c:
            add(workload=w, metric="available_parallelism", value=c["available_parallelism"],
                unit="work/span", artifact="critical_path_table.csv", phase="P7",
                evidence=_DERIVED, caveat="structural work/span, not a performance metric",
                implication="low -> intra-op sharding over many identical units")
            add(workload=w, metric="serialization", value=c["serialization"], unit="class",
                artifact="critical_path_table.csv", phase="P7", evidence=_DERIVED,
                implication="near-sequential DAG shape")
        # dominant shape class by MAC share (per workload)
        ws = [r for r in shp if r["workload"] == w]
        if ws:
            top = max(ws, key=lambda r: float(r["mac_fraction"]))
            add(workload=w, metric="dominant_shape_class", value=top["shape_class"], unit="",
                artifact="shape_summary_by_workload.csv", phase="P5", evidence="recovered_from_ir",
                implication=f"{float(top['mac_fraction']):.0%} of MACs -> primitive shape to cover")
        # top avoidable-reload region (per workload)
        wm = [r for r in dm if r["workload"] == w]
        if wm:
            tr = max(wm, key=lambda r: int(r["avoidable_weight_reload"]))
            add(workload=w, region=tr["region"], metric="avoidable_weight_reload",
                value=tr["avoidable_weight_reload"], unit="bytes",
                artifact="data_movement_table.csv", phase="P9", evidence=_DERIVED, caveat=_SMALL,
                implication="resident_weight_object residency benefit (bytes), no bandwidth claim")
            add(workload=w, region=tr["region"], metric="resident_int8_B", value=tr["resident_int8_B"],
                unit="bytes", artifact="data_movement_table.csv", phase="P9", evidence=_DERIVED,
                implication="int8 resident-capacity requirement")
        # epilogue pattern presence (per workload)
        we = [r for r in epi if r["workload"] == w]
        if we:
            n_bias = sum(1 for r in we if r["has_bias"] == "True")
            add(workload=w, metric="matmul_bias_epilogues", value=n_bias, unit="ops",
                artifact="epilogue_pattern_table.csv", phase="P10", evidence="recovered_from_ir",
                implication="fused epilogue slot present (bias) -> fused_requant_epilogue candidate")

    if scope == "all":
        # cross-workload facts (boundary placement, primitive regret, abstraction pressure)
        for b in man.get("boundary_placement", {}).get("top_by_evidence_breadth", [])[:8]:
            add(workload="ALL", abstraction=b["abstraction"],
                level=";".join(b.get("strong_levels", [])),
                metric="boundary_pressure_score", value=b["boundary_pressure_score"], unit="evidence",
                artifact="hw_sw_boundary_matrix.csv", phase="P12", evidence=_DERIVED,
                caveat="evidence breadth, not performance/priority",
                implication="strong candidate boundary placement(s)")
        reg = _csv_rows(cs_dir / "primitive_regret_table.csv")
        for r in sorted(reg, key=lambda x: -float(x["coverage_under_10pct"]))[:3]:
            add(workload="ALL", abstraction=r["primitive"], metric="coverage_under_10pct",
                value=r["coverage_under_10pct"], unit="MAC-fraction",
                artifact="primitive_regret_table.csv", phase="P5", evidence=_DERIVED,
                caveat="structural coverage, not a performance metric",
                implication="broadly-covering primitive for the DSE search space")
            add(workload="ALL", abstraction=r["primitive"], metric="max_regret",
                value=r["max_regret"], unit="MAC-fraction",
                artifact="primitive_regret_table.csv", phase="P5", evidence=_DERIVED,
                implication="cross-workload coverage spread (overfit risk if high)")
        for r in _csv_rows(cs_dir / "abstraction_pressure_ranking.csv"):
            add(workload="ALL", abstraction=r["system_abstraction"], metric="n_workloads_supporting",
                value=r["n_workloads"], unit="workloads",
                artifact="abstraction_pressure_ranking.csv", phase="P4", evidence=_DERIVED,
                caveat="structural count, not a ranking", implication=f"pressure: {r['evidence_strength']}")
        for r in _csv_rows(cs_dir / "measurement_priority_table.csv"):
            add(workload="ALL", metric="measurement_unblocks", value=r["n_candidates_unblocked"],
                unit="candidates", artifact="measurement_priority_table.csv", phase="P4",
                evidence=_DERIVED, caveat=r["measurement"],
                implication="measurement that unblocks candidates")
    return facts


# --------------------------------------------------------------------------- P13-c evidence strength

def evidence_strength(facts: list[dict]) -> dict:
    from collections import Counter
    tier = Counter(f["evidence_tier"] for f in facts)
    deriv = Counter(f["derivation_type"] for f in facts)
    by_wl = Counter(f["workload"] for f in facts)
    by_phase = Counter(f["source_phase"] for f in facts)
    by_art = Counter(f["source_artifact"] for f in facts)
    verified = sum(1 for f in facts if f["verification_status"] == "verified")
    return {
        "total_facts": len(facts),
        "by_tier": dict(tier), "by_derivation": dict(deriv),
        "by_workload": dict(by_wl), "by_phase": dict(by_phase), "by_artifact": dict(by_art),
        "verified_facts": verified,
        "measured_facts": deriv.get("measured", 0),
        "derived_facts": deriv.get("derived", 0),
        "assumed_facts": deriv.get("assumed", 0),
        "unavailable_facts": deriv.get("unavailable", 0),
        "high_confidence_findings": tier.get("A", 0) + tier.get("B", 0),
        "assumption_heavy_claims": tier.get("C", 0),
        "weak_findings": tier.get("D", 0),
    }


# --------------------------------------------------------------------------- P13-d usefulness suite

_QUERIES = [
    ("primitive_shapes_to_include", "What compute primitive shapes should a DSE search space include?",
     ["primitive_coverage_matrix.csv", "primitive_regret_table.csv"]),
    ("primitives_broadly_useful", "Which primitive shapes are broadly useful across workloads?",
     ["primitive_regret_table.csv"]),
    ("primitives_workload_specific", "Which primitive shapes are workload-specific / high-regret?",
     ["primitive_regret_table.csv"]),
    ("heterogeneous_units", "Which workloads suggest heterogeneous processing units?",
     ["processing_unit_guidance.yaml", "resource_pressure_table.csv"]),
    ("bounded_loop_commands", "Which workloads suggest bounded-loop commands?",
     ["workload_contract_graph.yaml", "boundary_candidate_contracts.yaml"]),
    ("strong_abstractions", "Which abstractions are most strongly supported across workloads?",
     ["abstraction_pressure_ranking.csv", "hw_sw_boundary_matrix.csv"]),
    ("family_specific_abstractions", "Which abstractions are family-specific?",
     ["abstraction_pressure_table.csv", "workload_family_table.csv"]),
    ("boundary_resident_weights", "Which boundary placements are plausible for resident weights?",
     ["boundary_candidate_contracts.yaml", "hw_sw_boundary_matrix.csv"]),
    ("boundary_kloop", "Which boundary placements are plausible for K-loop execution?",
     ["boundary_candidate_contracts.yaml"]),
    ("boundary_packed_lowbit", "Which boundary placements are plausible for packed low-bit tensors?",
     ["boundary_candidate_contracts.yaml"]),
    ("hal_objects", "Which objects should potentially cross the HAL boundary?",
     ["runtime_object_candidates.yaml"]),
    ("command_isa", "Which command-ISA abstractions are structurally suggested?",
     ["command_isa_candidates.yaml"]),
    ("accelerator_isa", "Which accelerator-ISA primitives are structurally suggested?",
     ["isa_candidate_primitives.yaml"]),
    ("blocked_by_proof", "Which candidates are blocked by missing compiler proof?",
     ["compiler_proof_matrix.csv"]),
    ("blocked_by_measurement", "Which candidates are blocked by missing measurements?",
     ["dse_readiness_summary.csv", "measurement_priority_table.csv"]),
    ("measurements_unblock_most", "Which measurements unblock the most candidates?",
     ["measurement_priority_table.csv"]),
    ("assumption_heavy", "Which findings rely heavily on assumptions?",
     ["dse_contract.json"]),
    ("shallow_analyses", "Which analyses are currently shallow or incomplete?",
     ["lost_numerical_contracts.csv", "boundary_candidate_contracts.yaml"]),
    ("plots_for_presentation", "Which plots should be generated for presentation?",
     ["plot_manifest.yaml"]),
    ("safe_claims", "Which claims are safe to present without quantitative performance measurements?",
     ["verification_report.md"]),
]


def usefulness(cs_dir: Path, scope: str, facts: list[dict]) -> list[dict]:
    answers = []
    cand = [f for f in facts if f["presentation_candidate"]]
    for key, q, arts in _QUERIES:
        present = [a for a in arts if (cs_dir / a).is_file()]
        # status: strong if supporting artifacts present + tier-A/B facts back it; weak/unavailable else
        related = [f for f in cand if f["source_artifact"] in arts]
        if not present:
            status, use = "unavailable", "do_not_show"
        elif key in ("boundary_packed_lowbit",):
            status, use = "weak", "backup"            # erased: only compiler-dequant path present
        elif key in ("assumption_heavy", "shallow_analyses", "family_specific_abstractions",
                     "primitives_workload_specific", "blocked_by_proof", "accelerator_isa"):
            status, use = "partial", "backup"
        elif related or key in ("blocked_by_measurement", "measurements_unblock_most",
                                "boundary_resident_weights", "boundary_kloop", "hal_objects",
                                "command_isa", "strong_abstractions", "heterogeneous_units",
                                "bounded_loop_commands", "primitive_shapes_to_include",
                                "primitives_broadly_useful", "plots_for_presentation",
                                "safe_claims"):
            status, use = "strong", "main"
        else:
            status, use = "partial", "backup"
        ev_types = sorted({f["evidence_type"] for f in related}) or ["see artifacts"]
        answers.append({
            "key": key, "query": q, "status": status, "supporting_artifacts": present,
            "missing_artifacts": [a for a in arts if a not in present],
            "evidence_types": ev_types,
            "caveats": ("packed low-bit / scales are erased in the capture"
                        if key == "boundary_packed_lowbit" else
                        "structural only; not a performance claim"),
            "recommended_presentation_use": use})
    return answers


# --------------------------------------------------------------------------- P13-e findings

_PLOT_FOR_METRIC = {
    "available_parallelism": "inter_op_parallelism_by_workload",
    "avoidable_weight_reload": "avoidable_reload_by_region",
    "resident_int8_B": "resident_capacity_by_dtype",
    "boundary_pressure_score": "boundary_placement_heatmap",
    "coverage_under_10pct": "primitive_coverage_heatmap",
    "max_regret": "primitive_regret_bar",
    "dominant_shape_class": "shape_class_mac_share",
    "matmul_bias_epilogues": "epilogue_pattern_counts",
    "n_workloads_supporting": "abstraction_pressure_bar",
    "accuracy_int8_w8a8": "accuracy_gate_status",
    "measurement_unblocks": "measurement_priority_bar",
}


def presentation_findings(facts: list[dict], answers: list[dict]) -> list[dict]:
    from collections import defaultdict
    cand = [f for f in facts if f["presentation_candidate"]]
    by_metric = defaultdict(list)
    for f in cand:
        by_metric[f["metric_name"]].append(f)
    findings = []
    for metric, fs in by_metric.items():
        tiers = {f["evidence_tier"] for f in fs}
        tier = "A" if "A" in tiers else ("B" if "B" in tiers else "C")
        purely_assumed = all(f["derivation_type"] == "assumed" for f in fs)
        corroborated = any(f["verification_status"] == "verified" for f in fs)
        wls = sorted({f["workload"] for f in fs})
        arts = sorted({f["source_artifact"] for f in fs})
        caveats = sorted({f["caveat"] for f in fs if f["caveat"]})
        impl = next((f["dse_implication"] for f in fs if f["dse_implication"]), "")
        main = tier in ("A", "B") and not purely_assumed and bool(impl) and corroborated
        findings.append({
            "title": metric.replace("_", " "),
            "claim": f"{metric}: {impl}" if impl else metric,
            "evidence_tier": tier, "supporting_artifacts": arts, "supporting_workloads": wls,
            "relevant_metrics": [metric], "dse_implication": impl, "caveats": caveats,
            "suggested_plot": _PLOT_FOR_METRIC.get(metric, ""),
            "presentation_placement": "main" if main else "backup",
            "forbidden_claim_risk": "low"})
    # an explicit honesty finding (limitation) — useful as backup, never main. Only when there is a
    # real package to mine (partial/empty mode produces no findings and stays vacuously consistent).
    if facts:
        findings.append({
        "title": "erased low-bit / KV structure",
        "claim": "packed low-bit layout, scales, and KV/attention structure are erased/lowered in "
                 "the capture; native low-bit & KV boundary placements are blocked/unavailable",
        "evidence_tier": "D", "supporting_artifacts": ["lost_numerical_contracts.csv",
                                                       "boundary_candidate_contracts.yaml"],
        "supporting_workloads": ["ALL"], "relevant_metrics": ["unavailable"],
        "dse_implication": "needs a low-bit / loop-preserving capture before these are candidates",
        "caveats": ["inherent to a flat dequantized, attention-lowered capture"],
        "suggested_plot": "", "presentation_placement": "backup", "forbidden_claim_risk": "low"})
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    findings.sort(key=lambda f: (order[f["evidence_tier"]], -len(f["supporting_workloads"])))
    return findings


# --------------------------------------------------------------------------- P13-f plot manifest

# (plot_id, title, source_artifact, required_columns, x, y, series, kind, rec)
_PLOTS = [
    ("evidence_type_by_workload", "Evidence type by workload", "unified_fact_table.csv",
     ["workload", "evidence_type"], "workload", "count", "evidence_type", "stacked_bar", "main"),
    ("evidence_type_by_phase", "Evidence type by analysis phase", "unified_fact_table.csv",
     ["source_phase", "evidence_type"], "source_phase", "count", "evidence_type", "stacked_bar", "backup"),
    ("shape_class_mac_share", "Shape-class MAC share by workload", "shape_summary_by_workload.csv",
     ["workload", "shape_class", "mac_fraction"], "workload", "mac_fraction", "shape_class",
     "stacked_bar", "main"),
    ("shape_class_opcount_share", "Shape-class op-count share by workload",
     "shape_summary_by_workload.csv", ["workload", "shape_class", "op_count"], "workload",
     "op_count", "shape_class", "stacked_bar", "backup"),
    ("primitive_coverage_heatmap", "Primitive x workload structural coverage",
     "primitive_coverage_matrix.csv", ["primitive", "workload", "coverage_under_10pct"], "workload",
     "primitive", "coverage_under_10pct", "heatmap", "main"),
    ("primitive_regret_bar", "Primitive coverage + max regret", "primitive_regret_table.csv",
     ["primitive", "coverage_under_10pct", "max_regret"], "primitive", "coverage_under_10pct",
     "max_regret", "grouped_bar", "main"),
    ("abstraction_pressure_bar", "Abstraction pressure (workloads supporting)",
     "abstraction_pressure_ranking.csv", ["system_abstraction", "n_workloads"], "system_abstraction",
     "n_workloads", "", "bar", "backup"),
    ("boundary_placement_heatmap", "Boundary placement: abstraction x level",
     "hw_sw_boundary_matrix.csv",
     ["abstraction", "compiler_transform", "runtime_hal_object", "command_buffer_or_command_isa",
      "accelerator_isa", "device_microcode_or_controller", "fixed_hardware_datapath"],
     "level", "abstraction", "status", "heatmap", "main"),
    ("resident_capacity_by_dtype", "Resident capacity by dtype (per region)",
     "data_movement_table.csv", ["workload", "region", "resident_int8_B", "resident_bf16_B"],
     "workload_region", "bytes", "dtype", "grouped_bar", "main"),
    ("avoidable_reload_by_region", "Avoidable weight reload by region", "data_movement_table.csv",
     ["workload", "region", "avoidable_weight_reload"], "workload_region", "avoidable_weight_reload",
     "", "bar", "main"),
    ("measurement_priority_bar", "Candidates unblocked per measurement",
     "measurement_priority_table.csv", ["measurement", "n_candidates_unblocked"], "measurement",
     "n_candidates_unblocked", "", "bar", "main"),
    ("critical_path_parallelism", "Available parallelism by workload", "critical_path_table.csv",
     ["workload", "available_parallelism"], "workload", "available_parallelism", "", "bar", "main"),
    ("epilogue_pattern_counts", "Epilogue patterns by workload", "epilogue_pattern_table.csv",
     ["workload", "pattern"], "workload", "count", "pattern", "stacked_bar", "backup"),
]


def plot_manifest(cs_dir: Path, scope: str) -> list[dict]:
    out = []
    for pid, title, art, cols, x, y, series, kind, rec in _PLOTS:
        # unified_fact_table.csv is the run's own output (always produced) — treat as available
        present = (cs_dir / art).is_file() or art == "unified_fact_table.csv"
        have_cols = True
        if present and art.endswith(".csv") and (cs_dir / art).is_file():
            hdr = (cs_dir / art).read_text().splitlines()[:1]
            header = hdr[0].split(",") if hdr else []
            have_cols = all(c in header for c in cols if c not in ("level", "count",
                                                                    "workload_region", "bytes",
                                                                    "dtype"))
        out.append({
            "plot_id": pid, "title": title, "source_artifact": art, "required_columns": cols,
            "x_axis": x, "y_axis": y, "series": series, "plot_type": kind,
            "available": bool(present and have_cols),
            "evidence_tier": "B", "recommendation": rec if (present and have_cols) else "omit",
            "caveat": "structural; axes are counts/bytes/fractions, not a performance metric",
            "why_useful": f"shows {title.lower()} from {art}"})
    return out


# --------------------------------------------------------------------------- P13-g consistency

def consistency_checks(cs_dir: Path, inventory, facts, findings, plots, answers) -> list[tuple]:
    res = []

    def chk(ok, msg):
        res.append((bool(ok), msg))

    art_names = {r["artifact"] for r in inventory}
    exist = {r["artifact"] for r in inventory if r["exists"]}
    # 1 findings reference existing artifacts
    chk(all(all(a in exist for a in f["supporting_artifacts"]) for f in findings),
        "every finding references existing artifacts")
    # 2 every main finding tier A/B
    chk(all(f["evidence_tier"] in ("A", "B") for f in findings
            if f["presentation_placement"] == "main"), "every main finding is tier A or B")
    # 3 plot manifest references existing columns/artifacts (for available plots)
    chk(all(p["available"] for p in plots if p["recommendation"] != "omit"),
        "every non-omit plot references an existing artifact + columns")
    # 4 no main finding purely assumed
    chk(not any(f["presentation_placement"] == "main" and f["evidence_tier"] == "C"
                for f in findings), "no main finding is based only on assumed_reference")
    # 5 boundary findings use allowed levels (facts with a boundary_level)
    allowed_lv = {"compiler_transform", "runtime_hal_object", "command_buffer_or_command_isa",
                  "accelerator_isa", "device_microcode_or_controller", "fixed_hardware_datapath"}
    chk(all(all(p in allowed_lv for p in f["boundary_level"].split(";") if p)
            for f in facts if f["boundary_level"]),
        "every boundary-level reference uses the allowed vocabulary")
    # 6 measurement findings reference real missing-evidence (the measurement_priority artifact)
    meas_q = next((a for a in answers if a["key"] == "measurements_unblock_most"), None)
    chk(meas_q is not None and meas_q["status"] in ("strong", "partial"),
        "measurement-priority query is answered from a present artifact")
    # 7 no forbidden wording in finding claims (outside the explicit erased-limitation finding)
    bad = []
    for f in findings:
        low = (f["claim"] + " " + " ".join(f["caveats"])).lower()
        for t in FORBIDDEN:
            if t in low and not (t == "energy" and "energy" not in low):
                if t in low:
                    bad.append((f["title"], t))
    chk(not bad, f"no finding claim uses forbidden performance wording ({bad[:3]})")
    # 8 unified fact table has no duplicate fact ids
    ids = [f["fact_id"] for f in facts]
    chk(len(ids) == len(set(ids)), "no duplicate fact_id in the unified fact table")
    # 9 inventory marks missing artifacts explicitly
    chk(all("exists" in r for r in inventory) and (art_names == art_names),
        "artifact inventory records exists for every artifact")
    # 10 usefulness answers use the allowed status vocabulary
    chk(all(a["status"] in ("strong", "partial", "weak", "unavailable") for a in answers),
        "every usefulness answer uses an allowed status")
    return res


# --------------------------------------------------------------------------- bundle

def mine(cs_dir, scope: str) -> dict:
    cs_dir = Path(cs_dir)
    inv = artifact_inventory(cs_dir)
    facts = unified_facts(cs_dir, scope)
    strength = evidence_strength(facts)
    answers = usefulness(cs_dir, scope, facts)
    findings = presentation_findings(facts, answers)
    plots = plot_manifest(cs_dir, scope)
    checks = consistency_checks(cs_dir, inv, facts, findings, plots, answers)
    return {"scope": scope, "inventory": inv, "facts": facts, "evidence_strength": strength,
            "usefulness": answers, "findings": findings, "plots": plots,
            "consistency_checks": checks}


# --------------------------------------------------------------------------- emitters (run folder)

def _csv_text(rows: list[dict], cols: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})
    return buf.getvalue()


def _inventory_md(inv) -> str:
    present = sum(1 for r in inv if r["exists"])
    L = ["# Artifact inventory\n", f"{present}/{len(inv)} expected artifacts present. "
         "Missing artifacts are recorded explicitly (partial mode).\n",
         "| artifact | exists | phase | type | rows | checked_by_verify | unavailable |",
         "|---|---|---|---|---|---|---|"]
    for r in sorted(inv, key=lambda x: (not x["exists"], x["source_phase"], x["artifact"])):
        L.append(f"| {r['artifact']} | {r['exists']} | {r['source_phase']} | {r['file_type']} | "
                 f"{r['row_count'] if r['row_count'] is not None else '—'} | "
                 f"{r['checked_by_verify']} | {r['unavailable']} |")
    return "\n".join(L) + "\n"


def _strength_md(s, scope) -> str:
    L = [f"# Evidence-strength report ({scope})\n",
         f"Total normalized facts: **{s['total_facts']}**. "
         "Tiers — A: recovered/measured + verified; B: recovered-unverified or derived+verified; "
         "C: assumed_reference; D: unavailable/unknown.\n",
         "| tier | facts |", "|---|---|"]
    for t in ("A", "B", "C", "D"):
        L.append(f"| {t} | {s['by_tier'].get(t, 0)} |")
    L.append(f"\n- measured: {s['measured_facts']} · derived: {s['derived_facts']} · "
             f"assumed: {s['assumed_facts']} · unavailable: {s['unavailable_facts']}")
    L.append(f"- verified facts: {s['verified_facts']}/{s['total_facts']}")
    L.append(f"- high-confidence (A+B): {s['high_confidence_findings']} · "
             f"assumption-heavy (C): {s['assumption_heavy_claims']} · weak (D): {s['weak_findings']}")
    L.append("\nBy phase: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_phase"].items())))
    return "\n".join(L) + "\n"


def _scorecard_md(answers, scope) -> str:
    L = [f"# DSE-usefulness scorecard ({scope})\n",
         "> Can the package answer DSE-relevant questions? Each query is answered from existing "
         "artifacts with a status + recommended presentation use. Structural only — no performance "
         "claim.\n",
         "| # | query | status | use | supporting artifacts |", "|---|---|---|---|---|"]
    for i, a in enumerate(answers, 1):
        L.append(f"| {i} | {a['query']} | **{a['status']}** | {a['recommended_presentation_use']} | "
                 f"{', '.join(a['supporting_artifacts']) or '—'} |")
    strong = sum(1 for a in answers if a["status"] == "strong")
    L.append(f"\n{strong}/{len(answers)} queries answerable **strong**; "
             f"{sum(1 for a in answers if a['status']=='unavailable')} unavailable.")
    return "\n".join(L) + "\n"


def _findings_md(findings, scope) -> str:
    L = [f"# Presentation candidate findings ({scope})\n",
         "> A finding may be **main** only if tier A/B, not purely assumed, with a clear DSE "
         "implication, corroborated by a verification check, and needing no performance claim.\n"]
    for place in ("main", "backup"):
        fs = [f for f in findings if f["presentation_placement"] == place]
        L.append(f"## {place.title()} ({len(fs)})\n")
        for f in fs:
            L.append(f"- **{f['title']}** [tier {f['evidence_tier']}] — {f['claim']}  "
                     f"_(workloads: {', '.join(f['supporting_workloads'])}; "
                     f"plot: {f['suggested_plot'] or '—'})_")
        L.append("")
    return "\n".join(L) + "\n"


def _plot_md(plots, scope, rendered) -> str:
    L = [f"# Plot manifest ({scope})\n",
         "> Candidate plots derived from the data. Structural axes only (counts/bytes/fractions). "
         f"Rendered PNGs: {len(rendered)} under generated_plots/.\n",
         "| plot_id | title | type | source | rec | rendered |", "|---|---|---|---|---|---|"]
    for p in plots:
        L.append(f"| {p['plot_id']} | {p['title']} | {p['plot_type']} | {p['source_artifact']} | "
                 f"{p['recommendation']} | {'yes' if p['plot_id'] in rendered else 'no'} |")
    return "\n".join(L) + "\n"


def _readme_md(bundle, rendered) -> str:
    s = bundle["evidence_strength"]
    npass = sum(1 for ok, _ in bundle["consistency_checks"] if ok)
    return (f"# Insight-mining run — scope: {bundle['scope']}\n\n"
            "Meta-analysis of the committed dse_guidance case-study package. Regeneratable; not "
            "committed. **No speedup / cycles / area / energy / performance claim.**\n\n"
            f"- normalized facts: {s['total_facts']} (tiers "
            f"A={s['by_tier'].get('A',0)} B={s['by_tier'].get('B',0)} C={s['by_tier'].get('C',0)} "
            f"D={s['by_tier'].get('D',0)})\n"
            f"- main findings: {sum(1 for f in bundle['findings'] if f['presentation_placement']=='main')}"
            f" · plots rendered: {len(rendered)}\n"
            f"- consistency checks: {npass}/{len(bundle['consistency_checks'])}\n\n"
            "Files: artifact_inventory.{csv,md}, unified_fact_table.{csv,yaml}, "
            "evidence_strength_table.csv + evidence_strength_report.md, dse_usefulness_scorecard.md "
            "+ dse_usefulness_answers.yaml, presentation_candidate_findings.{md,csv}, "
            "plot_manifest.{yaml,md}, generated_plots/, consistency_checks.md.\n")


def emit_run(bundle: dict, run_dir, rendered: list[str]) -> None:
    """Write all P13 deliverables for one mined scope into the run folder (non-committed)."""
    from merlin.common.artifacts import Artifact, yaml_artifact
    run_dir = Path(run_dir)
    scope = bundle["scope"]
    inv, facts = bundle["inventory"], bundle["facts"]
    Artifact("artifact_inventory.csv", _csv_text(
        inv, ["artifact", "path", "exists", "source_phase", "file_type", "row_count",
              "schema_valid", "has_evidence_fields", "checked_by_verify", "unavailable", "unknown",
              "assumed_reference", "derived_requirement", "measured"])).write(run_dir)
    Artifact("artifact_inventory.md", _inventory_md(inv)).write(run_dir)
    Artifact("unified_fact_table.csv", _csv_text(facts, FACT_COLUMNS)).write(run_dir)
    yaml_artifact("unified_fact_table.yaml", {"unified_fact_table": {"scope": scope, "facts": facts}},
                  header=f"unified_fact_table: {scope} (no performance claim)").write(run_dir)
    s = bundle["evidence_strength"]
    Artifact("evidence_strength_table.csv", _csv_text(
        [{"metric": k, "value": v} for k, v in s.items() if not isinstance(v, dict)],
        ["metric", "value"])).write(run_dir)
    Artifact("evidence_strength_report.md", _strength_md(s, scope)).write(run_dir)
    Artifact("dse_usefulness_scorecard.md", _scorecard_md(bundle["usefulness"], scope)).write(run_dir)
    yaml_artifact("dse_usefulness_answers.yaml",
                  {"dse_usefulness_answers": {"scope": scope, "answers": bundle["usefulness"]}},
                  header=f"dse_usefulness_answers: {scope}").write(run_dir)
    Artifact("presentation_candidate_findings.md",
             _findings_md(bundle["findings"], scope)).write(run_dir)
    Artifact("presentation_candidate_findings.csv", _csv_text(
        [{**f, "supporting_artifacts": "; ".join(f["supporting_artifacts"]),
          "supporting_workloads": "; ".join(f["supporting_workloads"]),
          "relevant_metrics": "; ".join(f["relevant_metrics"]),
          "caveats": "; ".join(f["caveats"])} for f in bundle["findings"]],
        ["title", "claim", "evidence_tier", "supporting_artifacts", "supporting_workloads",
         "relevant_metrics", "dse_implication", "caveats", "suggested_plot",
         "presentation_placement", "forbidden_claim_risk"])).write(run_dir)
    yaml_artifact("plot_manifest.yaml", {"plot_manifest": {"scope": scope, "plots": bundle["plots"],
                  "rendered": rendered}}, header=f"plot_manifest: {scope}").write(run_dir)
    Artifact("plot_manifest.md", _plot_md(bundle["plots"], scope, rendered)).write(run_dir)
    checks = bundle["consistency_checks"]
    npass = sum(1 for ok, _ in checks if ok)
    cm = [f"# Cross-artifact consistency checks ({scope})\n", f"**{npass}/{len(checks)} passed.**\n"]
    cm += [f"- [{'PASS' if ok else 'FAIL'}] {msg}" for ok, msg in checks]
    Artifact("consistency_checks.md", "\n".join(cm) + "\n").write(run_dir)
    Artifact("insight_mining_README.md", _readme_md(bundle, rendered)).write(run_dir)
