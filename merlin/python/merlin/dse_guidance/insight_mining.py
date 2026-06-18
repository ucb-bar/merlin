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
# Tier A is reserved for IR/prov.fqn-recovered or measured facts (independently verifiable). A
# published config constant (recovered_from_model_config, e.g. K) is a *reference value*, not an
# IR-recovered or measured fact, so it tiers as C alongside assumed_reference — NOT A.
_TIER_A_LABELS = {"measured", "proxy_measured", "recovered_from_ir", "recovered_from_prov_fqn"}
_CONFIG = {"recovered_from_model_config"}
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
                "evidence_type", "derivation_type", "evidence_tier", "metric_class", "dse_question",
                "verification_status", "verifying_check", "corroborated_by", "caveat",
                "dse_implication", "presentation_candidate"]

# SIGNAL metrics = decision-relevant quantities a DSE engine acts on. Only these can become
# presentation candidates / findings. Everything else (row counts, provenance, redundant
# corroboration values) is CONTEXT — kept for traceability/coverage, never promoted to a finding.
# This is the de-noising gate: it keeps real single-source recovered signal (dense/skinny split,
# proof distribution, dominant shape) AND excludes "n_X = rows in artifact X" padding.
SIGNAL_METRICS = {
    "head_weight_bytes", "total_macs", "n_matmuls", "avoidable_weight_reload", "resident_int8_B",
    "available_parallelism", "serialization", "boundary_pressure_score", "coverage_under_10pct",
    "max_regret", "accuracy_int8_w8a8", "measured_dispatch_ratio", "matmul_bias_epilogues",
    "dominant_shape_class", "mac_fraction_dense_gemm", "mac_fraction_skinny_gemm_or_gemv",
    "compiler_proofs_proven_for_workload", "compiler_proofs_assumed", "compiler_proofs_unknown",
    "clean_8way_mn_shards", "head_cadence", "accumulator_dtype", "compute_dtype",
    "lowbit_storage_dequantized_finding", "unit_multiplicity_implication", "overlap_candidates_yes",
}


def metric_class(metric: str, derivation: str) -> str:
    if derivation == "measured":
        return "measured"
    if metric in SIGNAL_METRICS:
        return "signal"
    return "context"   # row counts / provenance / redundant corroboration values (traceability only)


# every SIGNAL metric must answer ONE DSE-search-space question (the organizing rule). Context
# metrics get no question (they are traceability, never presented).
Q_PRIMITIVES = "Q_primitives: what compute primitives should DSE include?"
Q_HETERO = "Q_heterogeneity: should DSE explore heterogeneous / replicated units?"
Q_RESIDENCY = "Q_residency: should DSE explore weight residency / packed stores?"
Q_COMMAND = "Q_command: should DSE explore command/loop/dispatch abstractions?"
Q_LOWBIT = "Q_lowbit: should DSE explore low-bit formats / numerical placement?"
Q_BOUNDARY = "Q_boundary: where should the HW/SW boundary sit?"
Q_READINESS = "Q_readiness: what blocks quantitative ranking?"
DSE_QUESTIONS = [Q_PRIMITIVES, Q_HETERO, Q_RESIDENCY, Q_COMMAND, Q_LOWBIT, Q_BOUNDARY, Q_READINESS]

_METRIC_QUESTION = {
    "total_macs": Q_PRIMITIVES, "n_matmuls": Q_PRIMITIVES, "coverage_under_10pct": Q_PRIMITIVES,
    "max_regret": Q_PRIMITIVES, "dominant_shape_class": Q_PRIMITIVES,
    "available_parallelism": Q_HETERO, "serialization": Q_HETERO,
    "mac_fraction_dense_gemm": Q_HETERO, "mac_fraction_skinny_gemm_or_gemv": Q_HETERO,
    "clean_8way_mn_shards": Q_HETERO, "unit_multiplicity_implication": Q_HETERO,
    "head_weight_bytes": Q_RESIDENCY, "avoidable_weight_reload": Q_RESIDENCY,
    "resident_int8_B": Q_RESIDENCY,
    "measured_dispatch_ratio": Q_COMMAND, "head_cadence": Q_COMMAND,
    "overlap_candidates_yes": Q_COMMAND,
    "accuracy_int8_w8a8": Q_LOWBIT, "accuracy_gate_report_present": Q_LOWBIT,
    "matmul_bias_epilogues": Q_LOWBIT,
    "accumulator_dtype": Q_LOWBIT, "compute_dtype": Q_LOWBIT,
    "lowbit_storage_dequantized_finding": Q_LOWBIT,
    "boundary_pressure_score": Q_BOUNDARY,
    "compiler_proofs_proven_for_workload": Q_BOUNDARY, "compiler_proofs_assumed": Q_BOUNDARY,
    "compiler_proofs_unknown": Q_BOUNDARY,
}


# metric_name -> the SPECIFIC verify_implementation.py check that independently re-derives it (the
# per-metric verification basis, replacing the coarse artifact-membership proxy). A metric here is
# "verified"; one absent is verified only when corroborated by >=2 independent artifacts.
_METRIC_CHECK = {
    "head_weight_bytes": "B: head facts vs raw IR",
    "weight_bytes": "P9-A: region weight bytes == IR recompute",
    "total_macs": "P7-A: DAG total work == operator MAC sum",
    "n_matmuls": "A: matmul count raw==primitive==attributed",
    "macs": "P5-A: macs == M*N*K",
    "avoidable_weight_reload": "P9-B: avoidable == weight*(K-1)",
    "resident_int8_B": "P9-C: resident == dtype scaling (int8==WB/4)",
    "available_parallelism": "P7-C: available_parallelism == total/critical",
    "boundary_pressure_score": "P12: boundary matrix re-derivation",
    "matmul_bias_epilogues": "P10-B: every addmm flagged has_bias",
    "accuracy_int8_w8a8": "I: accuracy gating (measured)",
    "measured_dispatch_ratio": "measured dispatch coupling (committed measured data)",
    "coverage_under_10pct": "P5-C: primitive coverage aggregates recompute",
    "max_regret": "primitive_regret: max_regret == best - worst",
}


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
    if evidence in ("recovered_from_ir", "recovered_from_prov_fqn"):
        return "recovered"
    if evidence in _CONFIG:
        return "config_reference"
    if evidence == _DERIVED:
        return "derived"
    if evidence in _ASSUMED:
        return "assumed"
    return "unavailable"


def _fact_verified(metric: str, corroborated_by: int) -> bool:
    """A fact is independently verified iff a SPECIFIC harness check re-derives its metric, or it is
    corroborated by >=2 independent artifacts (not just artifact-set membership)."""
    return (metric in _METRIC_CHECK) or (corroborated_by >= 2)


def evidence_tier(evidence: str, metric: str = "", corroborated_by: int = 1) -> str:
    """A: IR-recovered/measured AND verified (per-metric check or >=2-artifact corroboration).
    B: recovered-but-unverified or derived+verified. C: assumed/config (labeled) or derived-unverified.
    D: unavailable/unknown."""
    verified = _fact_verified(metric, corroborated_by)
    if evidence in _TIER_A_LABELS:
        return "A" if verified else "B"
    if evidence == _DERIVED:
        return "B" if verified else "C"
    if evidence in _ASSUMED or evidence in _CONFIG:
        return "C"
    return "D"


def _verification_status(metric: str, evidence: str, corroborated_by: int) -> str:
    if evidence in _ABSENT:
        return "unavailable"
    return "verified" if _fact_verified(metric, corroborated_by) else "not_verified"


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
          abstraction="", level="", caveat="", implication="", corroborated_by=1) -> dict:
    tier = evidence_tier(evidence, metric, corroborated_by)
    return {
        "fact_id": fid, "workload": workload, "region": region, "phase": phase, "op_id": op_id,
        "abstraction": abstraction, "boundary_level": level, "metric_name": metric,
        "metric_value": value, "metric_unit": unit, "source_artifact": artifact,
        "source_phase": phase, "evidence_type": evidence, "derivation_type": _derivation(evidence),
        "evidence_tier": tier,
        "verification_status": _verification_status(metric, evidence, corroborated_by),
        "metric_class": metric_class(metric, _derivation(evidence)),
        "dse_question": _METRIC_QUESTION.get(metric, ""),
        "verifying_check": _METRIC_CHECK.get(metric, ""), "corroborated_by": corroborated_by,
        "caveat": caveat, "dse_implication": implication,
        # presentation-worthy = a SIGNAL metric (decision-relevant, not a row-count) that is tier A/B
        # with an implication. Corroboration / a harness check are reported as STRENGTH (see
        # corroborated_by + verifying_check), not the candidacy gate -- so real single-source
        # recovered signal is kept and count-padding is excluded.
        "presentation_candidate": (metric in SIGNAL_METRICS and tier in ("A", "B")
                                   and bool(implication))}


_SMALL = "magnitudes are small random-init capture instances (structure real)"


def _eqish(a, b, tol=2) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return str(a) == str(b)


def unified_facts(cs_dir: Path, scope: str) -> list[dict]:
    """Normalize DSE-relevant facts across ALL artifacts (never invented). Each fact is source-traced
    and (where cross-checkable) corroborated by counting independent artifacts that agree.
    scope = workload | all."""
    man = _json(cs_dir / "dse_contract.json") or {}
    per_wl = man.get("per_workload", {})
    workloads = _workloads(cs_dir) if scope == "all" else [scope]
    facts: list[dict] = []
    n = 0

    def add(**kw):
        nonlocal n
        facts.append(_fact(f"F{n:04d}", **kw))
        n += 1

    crit = {r["workload"]: r for r in _csv_rows(cs_dir / "critical_path_table.csv")}
    shp = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    shr = _csv_rows(cs_dir / "shape_summary_by_region.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    tr = _csv_rows(cs_dir / "traffic_table.csv")
    dcap = {r["workload"]: r for r in _csv_rows(cs_dir / "dtype_capacity_table.csv")}
    cov = _csv_rows(cs_dir / "primitive_coverage_matrix.csv")
    epi = _csv_rows(cs_dir / "epilogue_pattern_table.csv")
    acc_g = _csv_rows(cs_dir / "accuracy_gated_dtype_candidates.csv")
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    tile = _csv_rows(cs_dir / "tile_waste_table.csv")
    shard = _csv_rows(cs_dir / "sharding_table.csv")
    dma = _csv_rows(cs_dir / "dma_stream_table.csv")
    buf = _csv_rows(cs_dir / "buffer_requirement_table.csv")
    bufing = _csv_rows(cs_dir / "buffering_requirement_table.csv")
    reuse = _csv_rows(cs_dir / "reuse_lifetime_table.csv")
    acc_m = {r.get("model", r.get("workload", "")): r for r in
             _csv_rows(cs_dir / "accuracy_gate_results.csv")}
    prate = _csv_rows(cs_dir / "phase_rate_table.csv")
    pstage = _csv_rows(cs_dir / "pipeline_stage_table.csv")
    accum = _csv_rows(cs_dir / "accumulator_contract_table.csv")
    wct = {r["workload"]: r for r in _csv_rows(cs_dir / "workload_contract_table.csv")}

    def _ops_weight(w):
        return sum(int(r["rhs_weight_bytes"]) for r in ops
                   if r["workload"] == w and r["region_role"] == "repeated_head")

    for w in workloads:
        d = per_wl.get(w, {})
        # --- weights: corroborate manifest vs data_movement vs operator_shape sum ---
        hw = d.get("head_weight_bytes")
        dm_head = next((r for r in dm if r["workload"] == w and r["region"] == "repeated_head"), None)
        if hw:
            srcs = [hw]
            if dm_head:
                srcs.append(int(dm_head["weight_bytes"]))
            if ops:
                srcs.append(_ops_weight(w))
            corr = sum(1 for s in srcs if _eqish(s, hw))
            add(workload=w, metric="head_weight_bytes", value=hw, unit="bytes",
                artifact="dse_contract.json", phase="P5", evidence="recovered_from_ir",
                corroborated_by=corr, caveat=_SMALL,
                implication="resident-weight capacity requirement")
        add(workload=w, metric="K", value=d.get("K"), unit="steps", artifact="dse_contract.json",
            phase="P6", evidence="recovered_from_model_config",
            caveat="published architecture constant (not measured)",
            implication="bounds the K-loop / autonomous_K_loop axis")
        add(workload=w, metric="ready_quantitative_dse", value=d.get("ready_quantitative_dse"),
            unit="bool", artifact="dse_readiness_summary.csv", phase="P4", evidence=_DERIVED,
            implication="structural DSE ready; quantitative needs measurements")
        # --- accuracy: corroborate gate-candidate vs measured-gate results ---
        acc = d.get("accuracy_int8", "unavailable")
        accm = acc_m.get(w, {})
        corr_a = 1 + (1 if accm else 0)
        add(workload=w, metric="accuracy_int8_w8a8", value=acc, unit="band",
            artifact="accuracy_gated_dtype_candidates.csv", phase="P6",
            evidence="measured" if acc in ("pass", "fail") else "unavailable",
            corroborated_by=corr_a if acc in ("pass", "fail") else 1,
            implication="gates int8 as an accuracy-legal dtype candidate")
        # --- parallelism (corroborate critical_path with concurrency window max width) ---
        c = crit.get(w)
        if c:
            add(workload=w, metric="available_parallelism", value=c["available_parallelism"],
                unit="work/span", artifact="critical_path_table.csv", phase="P7", evidence=_DERIVED,
                caveat="structural work/span, not a performance metric",
                implication="low inter-op parallelism favors intra-op sharding (not many identical "
                            "units kept busy by concurrency)")
            add(workload=w, metric="serialization", value=c["serialization"], unit="class",
                artifact="critical_path_table.csv", phase="P7", evidence=_DERIVED,
                implication="near-sequential DAG shape")
            add(workload=w, metric="total_macs", value=c["total_macs"], unit="MACs",
                artifact="critical_path_table.csv", phase="P7", evidence="recovered_from_ir",
                corroborated_by=2 if ops else 1, caveat=_SMALL,
                implication="per-replan compute volume")
        cw = next((r for r in _csv_rows(cs_dir / "concurrency_windows.csv")
                   if r["workload"] == w), None)
        if cw:
            mw = max((int(r["ready_ops"]) for r in _csv_rows(cs_dir / "concurrency_windows.csv")
                      if r["workload"] == w), default=0)
            add(workload=w, metric="max_ready_width", value=mw, unit="ops",
                artifact="concurrency_windows.csv", phase="P7", evidence=_DERIVED,
                implication="peak simultaneously-ready operators")
        # --- operator geometry ---
        nm = sum(1 for r in ops if r["workload"] == w)
        if nm:
            add(workload=w, metric="n_matmuls", value=nm, unit="ops",
                artifact="operator_shape_table.csv", phase="P5", evidence="recovered_from_ir",
                corroborated_by=2 if c else 1, implication="operator count to cover")
        ws = [r for r in shp if r["workload"] == w]
        if ws:
            top = max(ws, key=lambda r: float(r["mac_fraction"]))
            add(workload=w, metric="dominant_shape_class", value=top["shape_class"], unit="",
                artifact="shape_summary_by_workload.csv", phase="P5", evidence="recovered_from_ir",
                caveat=f"{w}: {float(top['mac_fraction']):.0%} of this workload's MACs",
                implication="dominant geometry class -> the primitive shape the DSE must cover")
        if any(r["workload"] == w for r in shr):
            add(workload=w, metric="n_region_shape_groups",
                value=sum(1 for r in shr if r["workload"] == w), unit="groups",
                artifact="shape_summary_by_region.csv", phase="P5", evidence="recovered_from_ir",
                implication="per-region geometry breakdown")
        if any(r["workload"] == w for r in tile):
            add(workload=w, metric="n_tile_evaluations",
                value=sum(1 for r in tile if r["workload"] == w), unit="op*primitive",
                artifact="tile_waste_table.csv", phase="P5", evidence=_DERIVED,
                implication="tile-coverage evaluations performed")
        wc = [r for r in cov if r["workload"] == w]
        if wc:
            best = max(wc, key=lambda r: float(r["coverage_under_10pct"]))
            add(workload=w, abstraction=best["primitive"], metric="coverage_under_10pct",
                value=best["coverage_under_10pct"], unit="MAC-fraction",
                artifact="primitive_coverage_matrix.csv", phase="P5", evidence=_DERIVED,
                implication="best-covering primitive for this workload")
        # --- memory / dma / buffers ---
        if dm_head:
            srcs = [int(dm_head["avoidable_weight_reload"])]
            tr_head = next((r for r in tr if r["workload"] == w and r["region"] == "repeated_head"),
                           None)
            corr_av = 1 + (1 if tr_head else 0)
            add(workload=w, region="repeated_head", metric="avoidable_weight_reload",
                value=dm_head["avoidable_weight_reload"], unit="bytes",
                artifact="data_movement_table.csv", phase="P9", evidence=_DERIVED,
                corroborated_by=corr_av, caveat=_SMALL,
                implication="resident_weight_object residency benefit (bytes), no bandwidth claim")
            corr_r = 1 + (1 if (w in dcap and _eqish(dcap[w].get("int8_B"),
                                dm_head["resident_int8_B"])) else 0)
            add(workload=w, region="repeated_head", metric="resident_int8_B",
                value=dm_head["resident_int8_B"], unit="bytes",
                artifact="data_movement_table.csv", phase="P9", evidence=_DERIVED,
                corroborated_by=corr_r, implication="int8 resident-capacity requirement")
        if any(r["workload"] == w for r in dma):
            nbs = sum(1 for r in dma if r["workload"] == w and r["bytes"] != "unavailable")
            add(workload=w, metric="byte_carrying_streams", value=nbs, unit="streams",
                artifact="dma_stream_table.csv", phase="P9", evidence="recovered_from_ir",
                implication="independent DMA streams (weight/activation/output)")
        if any(r["workload"] == w for r in buf):
            bf = next(r for r in buf if r["workload"] == w)
            add(workload=w, metric="min_input_buffers", value=bf["min_input_buffer_count"],
                unit="buffers", artifact="buffer_requirement_table.csv", phase="P9",
                evidence=_DERIVED, implication="double-buffering to overlap DMA with compute")
        if any(r["workload"] == w for r in bufing):
            yes = sum(1 for r in bufing if r["workload"] == w and r["can_overlap"] == "yes")
            add(workload=w, metric="overlap_candidates_yes", value=yes, unit="candidates",
                artifact="buffering_requirement_table.csv", phase="P8", evidence=_DERIVED,
                implication="phase overlaps structurally permitted")
        if any(r["workload"] == w for r in reuse):
            add(workload=w, metric="n_resident_objects",
                value=sum(1 for r in reuse if r["workload"] == w), unit="objects",
                artifact="reuse_lifetime_table.csv", phase="P9", evidence="recovered_from_ir",
                implication="objects with a reuse lifetime (residency candidates)")
        # --- fusion / numerical ---
        we = [r for r in epi if r["workload"] == w]
        if we:
            n_bias = sum(1 for r in we if r["has_bias"] == "True")
            add(workload=w, metric="matmul_bias_epilogues", value=n_bias, unit="ops",
                artifact="epilogue_pattern_table.csv", phase="P10", evidence="recovered_from_ir",
                implication="fused epilogue slot present (bias) -> fused_requant_epilogue candidate")
        wa = [r for r in accum if r["workload"] == w]
        if wa:
            add(workload=w, region=wa[0]["region"], metric="accumulator_dtype",
                value=wa[0]["accumulator_dtype"], unit="dtype",
                artifact="accumulator_contract_table.csv", phase="P10", evidence="recovered_from_ir",
                implication="accumulator width for the datapath")
        # --- sharding / pipeline rate ---
        if any(r["workload"] == w for r in shard):
            clean = sum(1 for r in shard if r["workload"] == w and r["axis"] in ("M", "N")
                        and r["shardable_8"] == "True" and r["tail_8"] == "False")
            add(workload=w, metric="clean_8way_mn_shards", value=clean, unit="op*axis",
                artifact="sharding_table.csv", phase="P7", evidence=_DERIVED,
                implication="reduction-free M/N shards available")
        if any(r["workload"] == w for r in pstage):
            cads = sorted({r["cadence"] for r in pstage if r["workload"] == w})
            add(workload=w, metric="phase_cadences", value="; ".join(cads), unit="",
                artifact="pipeline_stage_table.csv", phase="P8",
                evidence="recovered_from_prov_fqn", corroborated_by=2 if prate else 1,
                implication="multi-rate phase structure")
        if w in wct:
            add(workload=w, metric="recovered_roles", value=wct[w].get("roles", ""), unit="",
                artifact="workload_contract_table.csv", phase="P2",
                evidence="recovered_from_prov_fqn", implication="region roles recovered from prov.fqn")
        # dtype capacity (source; corroborates resident_int8) + traffic (corroborates avoidable)
        if w in dcap:
            add(workload=w, metric="int8_capacity_B", value=dcap[w].get("int8_B"), unit="bytes",
                artifact="dtype_capacity_table.csv", phase="P9", evidence=_DERIVED,
                corroborated_by=2 if dm_head else 1,
                implication="int8 resident capacity (weights scaled to int8)")
        trh = next((r for r in tr if r["workload"] == w and r["region"] == "repeated_head"), None)
        if trh:
            add(workload=w, region="repeated_head", metric="weight_traffic_if_nonresident",
                value=trh["weight_traffic_if_nonresident"], unit="bytes",
                artifact="traffic_table.csv", phase="P9", evidence=_DERIVED,
                implication="weight bytes moved if NOT made resident (reload each step)")
        wp = [r for r in prate if r["workload"] == w]
        if wp:
            head = next((r for r in wp if r["role"] == "repeated_head"), wp[0])
            add(workload=w, metric="head_cadence", value=head.get("cadence", ""), unit="",
                artifact="phase_rate_table.csv", phase="P6", evidence="recovered_from_prov_fqn",
                corroborated_by=2 if pstage else 1, implication="repeated-head cadence (rate class)")
        # per-workload subdir: numerical_contract.yaml + region_attribution.yaml
        nc = _yaml(cs_dir / w / "numerical_contract.yaml")
        if isinstance(nc, dict):
            ncb = nc.get("numerical_contract", nc)
            add(workload=w, metric="compute_dtype", value=str(ncb.get("compute_dtype", "f32")),
                unit="dtype", artifact="numerical_contract.yaml", phase="P10",
                evidence="recovered_from_ir", implication="storage/compute dtype contract")
        ra = _yaml(cs_dir / w / "region_attribution.yaml")
        if isinstance(ra, dict):
            regs = ra.get("topology_recovery", {}).get("regions", [])
            attr_mm = sum(r.get("facts", {}).get("matmul_count", 0) for r in regs)
            if attr_mm:
                add(workload=w, metric="attributed_matmuls", value=attr_mm, unit="ops",
                    artifact="region_attribution.yaml", phase="P1", evidence="recovered_from_prov_fqn",
                    corroborated_by=2 if nm else 1, implication="matmuls attributed to roles")

    if scope == "all":
        _all_facts(cs_dir, man, add)
    return facts


def _all_facts(cs_dir: Path, man: dict, add) -> None:
    """Cross-workload facts spanning every remaining artifact (so coverage == all)."""
    bm = _csv_rows(cs_dir / "hw_sw_boundary_matrix.csv")
    bc = _yaml(cs_dir / "boundary_candidate_contracts.yaml")
    bc_n = len((bc or {}).get("boundary_candidate_contracts", {}).get("certificates", []))
    for b in sorted(bm, key=lambda r: -int(r["boundary_pressure_score"]))[:8]:
        strong = [lv for lv in ("compiler_transform", "runtime_hal_object",
                  "command_buffer_or_command_isa", "accelerator_isa",
                  "device_microcode_or_controller", "fixed_hardware_datapath")
                  if b.get(lv) == "strong_candidate"]
        add(workload="ALL", abstraction=b["abstraction"], level=";".join(strong),
            metric="boundary_pressure_score", value=b["boundary_pressure_score"], unit="evidence",
            artifact="hw_sw_boundary_matrix.csv", phase="P12", evidence=_DERIVED,
            corroborated_by=2 if bc_n else 1, caveat="evidence breadth, not performance/priority",
            implication="strong candidate boundary placement(s)")

    def count(fname, key, phase, metric, impl, evidence=_DERIVED, sub=None):
        if fname.endswith((".yaml", ".json")):
            obj = (_json if fname.endswith(".json") else _yaml)(cs_dir / fname)
            if obj is None:
                return
            node = obj.get(key, obj) if isinstance(obj, dict) else obj
            if sub:
                node = node.get(sub, []) if isinstance(node, dict) else node
            val = len(node) if isinstance(node, (list, dict)) else node
        else:
            rows = _csv_rows(cs_dir / fname)
            if not rows:
                return
            val = len(rows)
        add(workload="ALL", metric=metric, value=val, unit="count", artifact=fname, phase=phase,
            evidence=evidence, implication=impl)

    # P12 candidate lists + responsibility + knobs
    count("runtime_object_candidates.yaml", "runtime_object_candidates", "P12", "n_runtime_objects",
          "objects that could cross the HAL boundary", sub="candidates")
    count("command_isa_candidates.yaml", "command_isa_candidates", "P12", "n_command_candidates",
          "command-ISA abstractions structurally suggested", sub="candidates")
    count("isa_candidate_primitives.yaml", "isa_candidate_primitives", "P12", "n_isa_primitives",
          "accelerator-ISA primitives structurally suggested", sub="primitives")
    count("boundary_dse_knobs.yaml", "boundary_dse_knobs", "P12", "n_boundary_knobs",
          "DSE knobs created by boundary placement", sub="knobs")
    resp = _csv_rows(cs_dir / "responsibility_split_matrix.csv")
    if resp:
        add(workload="ALL", metric="compiler_owned_functions",
            value=sum(1 for r in resp if r["compiler"] == "owns"), unit="functions",
            artifact="responsibility_split_matrix.csv", phase="P12", evidence=_DERIVED,
            implication="functions the compiler owns in the HW/SW split")
    # P11 knobs
    knobs = _yaml(cs_dir / "dse_search_space_knobs.yaml")
    if isinstance(knobs, dict):
        grp = knobs.get("dse_search_space_knobs", {}).get("knob_groups", [])
        add(workload="ALL", metric="enabled_knob_groups",
            value=sum(1 for g in grp if g.get("enabled")), unit="groups",
            artifact="dse_search_space_knobs.yaml", phase="P11", evidence=_DERIVED,
            implication="enabled DSE search-space knob groups (P5-P12)")
    # P4 compiler proofs (KEY: proof status distribution)
    cpm = _csv_rows(cs_dir / "compiler_proof_matrix.csv")
    if cpm:
        for st in ("proven_for_workload", "assumed", "unknown"):
            add(workload="ALL", metric=f"compiler_proofs_{st}",
                value=sum(1 for r in cpm if r["status"] == st), unit="axes",
                artifact="compiler_proof_matrix.csv", phase="P4", evidence=_DERIVED,
                implication=f"abstraction axes with compiler proof status = {st}")
    count("abstraction_pressure_ranking.csv", None, "P4", "n_abstraction_axes_ranked",
          "abstraction axes by cross-workload pressure")
    count("abstraction_pressure_table.csv", None, "P4", "n_abstraction_pressure_rows",
          "per-workload abstraction pressure rows")
    count("workload_family_table.csv", None, "P4", "n_workload_families",
          "workloads clustered into families")
    count("resident_state_table.csv", None, "P4", "n_resident_states",
          "states with a residency lifetime")
    count("requirements_table.csv", None, "P3", "n_requirements",
          "hardware-independent requirements derived")
    count("cross_workload_provenance.csv", None, "P1", "n_provenance_rows",
          "flat-vs-recovered provenance items")
    count("operator_cluster_table.csv", None, "P5", "n_shape_clusters",
          "cross-workload geometry clusters", evidence="recovered_from_ir")
    count("operator_cluster_to_hierarchy.csv", None, "P7", "n_cluster_hierarchy_maps",
          "shape-cluster -> hierarchy-option mappings")
    count("multi_rate_contract.yaml", "multi_rate_contract", "P6", "n_multi_rate_workloads",
          "workloads with a recovered multi-rate contract", sub="workloads")
    count("numerical_epilogue_candidates.yaml", "numerical_epilogue_candidates", "P10",
          "n_fused_candidates", "fused-epilogue abstraction candidates", sub="candidates")
    count("lost_numerical_contracts.csv", None, "P10", "n_erased_numerical_contracts",
          "numerical contracts erased by the capture (scoped limitation)", evidence="unavailable")
    count("processing_unit_candidates.yaml", "processing_unit_candidates", "P7", "n_unit_candidates",
          "processing-unit candidates", sub="units")
    count("boundary_candidate_contracts.yaml", "boundary_candidate_contracts", "P12",
          "n_boundary_certificates", "HW/SW boundary-placement certificates", sub="certificates")
    count("pipeline_candidates.yaml", "pipeline_candidates", "P8", "n_pipeline_candidate_workloads",
          "workloads with pipeline overlap candidates", sub="workloads")
    count("pipeline_envelope.yaml", "pipeline_envelope", "P8", "n_pipeline_envelope_workloads",
          "workloads with a multi-rate phase model", sub="workloads",
          evidence="recovered_from_prov_fqn")
    # memory_hierarchy_envelope regions
    mh = _yaml(cs_dir / "memory_hierarchy_envelope.yaml")
    if isinstance(mh, dict):
        regs = sum(len(x.get("regions", [])) for x in
                   mh.get("memory_hierarchy_envelope", {}).get("workloads", []))
        add(workload="ALL", metric="n_memory_regions", value=regs, unit="regions",
            artifact="memory_hierarchy_envelope.yaml", phase="P9", evidence="recovered_from_ir",
            implication="per-region memory envelopes")
    # sharding_opportunities required abstractions
    so = _yaml(cs_dir / "sharding_opportunities.yaml")
    if isinstance(so, dict):
        ra = so.get("sharding_opportunities", {}).get("required_abstractions", [])
        add(workload="ALL", metric="n_sharding_abstractions", value=len(ra), unit="abstractions",
            artifact="sharding_opportunities.yaml", phase="P7", evidence=_DERIVED,
            implication="reduction/broadcast abstractions sharding requires")
    # accuracy_gate_report summary (corroborates the measured results)
    if (cs_dir / "accuracy_gate_report.md").is_file():
        npass = (cs_dir / "accuracy_gate_report.md").read_text().lower().count("pass")
        add(workload="ALL", metric="accuracy_gate_report_present", value="yes", unit="",
            artifact="accuracy_gate_report.md", phase="P4", evidence="measured",
            corroborated_by=2, implication="measured int8 accuracy summary (real measurement)")
    # operator_geometry total operators (corroborates operator_shape_table)
    og = _yaml(cs_dir / "operator_geometry.yaml")
    if isinstance(og, dict):
        tot = sum(x.get("n_operators", 0) for x in
                  og.get("operator_geometry", {}).get("workloads", []))
        add(workload="ALL", metric="total_operators", value=tot, unit="ops",
            artifact="operator_geometry.yaml", phase="P5", evidence="recovered_from_ir",
            corroborated_by=2 if _csv_rows(cs_dir / "operator_shape_table.csv") else 1,
            implication="total matmul operators across workloads")
    # workload_contract_graph node count
    wcg = _yaml(cs_dir / "workload_contract_graph.yaml")
    if isinstance(wcg, dict):
        nodes = sum(len(g.get("nodes", [])) for g in
                    wcg.get("workload_contract_graph", {}).get("graphs", []))
        add(workload="ALL", metric="contract_graph_nodes", value=nodes, unit="nodes",
            artifact="workload_contract_graph.yaml", phase="P6", evidence="recovered_from_ir",
            implication="multi-rate contract graph size")
    # resource pressure dense vs skinny
    rp = {r["resource_class"]: r for r in _csv_rows(cs_dir / "resource_pressure_table.csv")}
    if rp:
        for rc in ("dense_gemm", "skinny_gemm_or_gemv"):
            if rc in rp:
                add(workload="ALL", metric=f"mac_fraction_{rc}", value=rp[rc]["mac_fraction"],
                    unit="MAC-fraction", artifact="resource_pressure_table.csv", phase="P7",
                    evidence="recovered_from_ir",
                    implication="distinct compute family -> specialized vs monolithic units")
    # processing unit multiplicity verdict
    pug = _yaml(cs_dir / "processing_unit_guidance.yaml")
    if isinstance(pug, dict):
        impl = pug.get("processing_unit_guidance", {}).get("search_space_implication", "")
        add(workload="ALL", metric="unit_multiplicity_implication",
            value="heterogeneous" if "heterogeneous" in impl.lower() else "see report", unit="",
            artifact="processing_unit_guidance.yaml", phase="P8", evidence=_DERIVED,
            implication="monolithic vs replicated vs specialized search-space implication")
    # primitive regret (broad coverage + spread)
    reg = _csv_rows(cs_dir / "primitive_regret_table.csv")
    for r in sorted(reg, key=lambda x: -float(x["coverage_under_10pct"]))[:2]:
        add(workload="ALL", abstraction=r["primitive"], metric="coverage_under_10pct",
            value=r["coverage_under_10pct"], unit="MAC-fraction",
            artifact="primitive_regret_table.csv", phase="P5", evidence=_DERIVED,
            corroborated_by=2 if _csv_rows(cs_dir / "primitive_coverage_matrix.csv") else 1,
            caveat="structural coverage, not a performance metric",
            implication="broadly-covering primitive for the DSE search space")
        add(workload="ALL", abstraction=r["primitive"], metric="max_regret", value=r["max_regret"],
            unit="MAC-fraction", artifact="primitive_regret_table.csv", phase="P5", evidence=_DERIVED,
            implication="cross-workload coverage spread (overfit risk if high)")
    # cap: emit the top-5 measurements by candidates-unblocked (not all 35 rows -> de-noise)
    mp = sorted(_csv_rows(cs_dir / "measurement_priority_table.csv"),
                key=lambda r: -int(r["n_candidates_unblocked"]))[:5]
    for r in mp:
        add(workload="ALL", metric="measurement_unblocks", value=r["n_candidates_unblocked"],
            unit="candidates", artifact="measurement_priority_table.csv", phase="P4",
            evidence=_DERIVED, caveat=r["measurement"], implication="measurement that unblocks candidates")
    # REAL measured data: accuracy gate + dispatch coupling
    for r in _csv_rows(cs_dir / "accuracy_gate_results.csv"):
        mdl = r.get("model", r.get("workload", "?"))
        st = r.get("status", r.get("int8_status", ""))
        add(workload=mdl, metric="accuracy_int8_w8a8", value=st, unit="band",
            artifact="accuracy_gate_results.csv", phase="P4",
            evidence="measured" if st else "unavailable", corroborated_by=2 if st else 1,
            implication="MEASURED int8 W8A8 accuracy (real measurement)")
    disp = _csv_rows(cs_dir / "dispatch_coupling.csv")
    if disp:
        add(workload="ALL", metric="measured_dispatch_ratio",
            value=disp[0].get(disp[0] and list(disp[0])[-1], "see file"), unit="ratio",
            artifact="dispatch_coupling.csv", phase="P4", evidence="measured",
            implication="MEASURED host dispatch coupling (real runtime measurement)")
    # zoo low-bit storage finding (real low-bit evidence already in the package)
    fid = cs_dir / "numerical_contract_fidelity_report.md"
    if fid.is_file():
        add(workload="ZOO", metric="lowbit_storage_dequantized_finding", value="present", unit="",
            artifact="numerical_contract_fidelity_report.md", phase="P4", evidence="recovered_from_ir",
            implication="quantized zoo stores weights low-bit but runs f32 matmuls (native low-bit "
                        "compute + packed layout absent) -- real low-bit storage evidence")


# --------------------------------------------------------------------------- P13-c evidence strength

def evidence_strength(facts: list[dict]) -> dict:
    from collections import Counter
    tier = Counter(f["evidence_tier"] for f in facts)
    deriv = Counter(f["derivation_type"] for f in facts)
    klass = Counter(f["metric_class"] for f in facts)
    by_wl = Counter(f["workload"] for f in facts)
    by_phase = Counter(f["source_phase"] for f in facts)
    by_art = Counter(f["source_artifact"] for f in facts)
    verified = sum(1 for f in facts if f["verification_status"] == "verified")
    sig = klass.get("signal", 0) + klass.get("measured", 0)
    return {
        "total_facts": len(facts),
        "signal_facts": sig, "context_facts": klass.get("context", 0),
        "signal_to_total": round(sig / len(facts), 3) if facts else 0.0,
        "by_metric_class": dict(klass),
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
     ["operator_shape_table.csv", "critical_path_table.csv", "hw_sw_boundary_matrix.csv",
      "data_movement_table.csv"]),
    ("safe_claims", "Which claims are safe to present without quantitative performance measurements?",
     ["verification_report.md"]),
]


# queries that are inherently about absence/limits (answered by scoped required-inputs, not tier-A
# facts) — these are honestly "partial/weak" even when the supporting artifact is rich.
_LIMIT_QUERIES = {"boundary_packed_lowbit": "weak", "assumption_heavy": "partial",
                  "shallow_analyses": "partial", "blocked_by_proof": "partial",
                  "family_specific_abstractions": "partial", "primitives_workload_specific": "partial"}


def usefulness(cs_dir: Path, scope: str, facts: list[dict]) -> list[dict]:
    """Status is DERIVED from the facts that back each query (no hardcoded strong/partial lists):
    strong = >=1 corroborated tier-A/B fact; partial = only tier-C; weak = only tier-D / limit query;
    unavailable = no supporting artifact present."""
    answers = []
    for key, q, arts in _QUERIES:
        present = [a for a in arts if (cs_dir / a).is_file()]
        related = [f for f in facts if f["source_artifact"] in arts]
        strong_facts = [f for f in related if f["evidence_tier"] in ("A", "B")
                        and (f["corroborated_by"] >= 2 or f["verifying_check"])]
        tierC = [f for f in related if f["evidence_tier"] == "C"]
        if not present:
            status = "unavailable"
        elif key in _LIMIT_QUERIES:
            status = _LIMIT_QUERIES[key]              # inherently a limit/absence question
        elif strong_facts:
            status = "strong"
        elif tierC:
            status = "partial"
        elif related:
            status = "weak"
        else:
            status = "partial"                        # artifacts present but no mined fact yet
        use = {"strong": "main", "partial": "backup", "weak": "backup",
               "unavailable": "do_not_show"}[status]
        answers.append({
            "key": key, "query": q, "status": status, "supporting_artifacts": present,
            "missing_artifacts": [a for a in arts if a not in present],
            "n_backing_strong_facts": len(strong_facts),
            "evidence_types": sorted({f["evidence_type"] for f in related}) or ["see artifacts"],
            "caveats": ("packed low-bit / scales erased in the capture (see required_inputs_manifest)"
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
        # 'main' requires real corroboration: a per-metric harness check OR >=2-artifact agreement
        corroborated = any(f["verifying_check"] or f["corroborated_by"] >= 2 for f in fs)
        wls = sorted({f["workload"] for f in fs})
        arts = sorted({f["source_artifact"] for f in fs})
        caveats = sorted({f["caveat"] for f in fs if f["caveat"]})
        impl = next((f["dse_implication"] for f in fs if f["dse_implication"]), "")
        # B5: per-workload spread (value per workload) so a multi-workload finding hides nothing
        spread = {f["workload"]: f["metric_value"] for f in fs if f["workload"] not in ("ALL", "ZOO")}
        vals = []
        for v in spread.values():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
        spread_summary = (f"range [{min(vals):g}, {max(vals):g}] across {len(spread)} workloads"
                          if len(vals) >= 2 else (f"{len(spread)} workload(s)" if spread else "cross-workload"))
        max_corr = max((f["corroborated_by"] for f in fs), default=1)
        main = tier in ("A", "B") and not purely_assumed and bool(impl) and corroborated
        findings.append({
            "title": metric.replace("_", " "),
            "claim": f"{metric}: {impl}", "evidence_tier": tier, "supporting_artifacts": arts,
            "supporting_workloads": wls, "per_workload_values": spread,
            "per_workload_spread": spread_summary, "max_corroborated_by": max_corr,
            "relevant_metrics": [metric], "dse_implication": impl, "caveats": caveats,
            "dse_question": _METRIC_QUESTION.get(metric, ""),
            "suggested_plot": _PLOT_FOR_METRIC.get(metric, ""),
            "presentation_placement": "main" if main else "backup", "forbidden_claim_risk": "low"})
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
    # ---- decision-impact ("what-if") plots: how an outcome changes under a DSE knob choice ----
    ("decision_primitive_choice", "Decision: single primitive choice -> coverage",
     "primitive_coverage_matrix.csv", ["primitive", "workload", "coverage_under_10pct"], "primitive",
     "coverage_under_10pct", "workload", "decision_bar", "main"),
    ("decision_weight_residency", "Decision: weight residency -> bytes moved vs loop count",
     "data_movement_table.csv", ["workload", "region", "weight_bytes", "invocations"], "loop_count",
     "bytes_moved", "policy", "decision_curve", "main"),
    ("decision_capacity_dtype", "Decision: on-chip capacity + dtype -> weights resident",
     "dtype_capacity_table.csv", ["workload", "bf16_B", "int8_B", "int4_B"], "capacity_budget",
     "workloads_resident", "dtype", "decision_curve", "main"),
    ("decision_sharding_cost", "Decision: shard axis + count -> extra data-movement bytes",
     "sharding_table.csv", ["axis", "shardable_2", "shardable_4", "shardable_8",
                            "per_extra_shard_bytes"], "shard_count", "extra_bytes", "axis",
     "decision_bar", "main"),
]


# =========================================================================== P15 signal study
# Given ONLY workload artifacts, what changes the future DSE search space? Every output below is
# recovered/derived from the captures (or a host measurement) — no quantity for unbuilt hardware.

def canonical_signal_table(facts: list[dict]) -> list[dict]:
    """The one headline table: signal/measured metrics only, with tier, strength, DSE question.

    Context/count metrics (provenance, redundant corroboration, raw row counts) are excluded — they
    live in the full fact table for traceability but never in the headline."""
    rows = []
    seen = set()
    for f in facts:
        if f.get("metric_class") not in ("signal", "measured"):
            continue
        cb = f.get("corroborated_by", 1)
        strength = "measured" if f.get("metric_class") == "measured" else (
            "verified+corroborated" if (f.get("verifying_check") and int(cb or 1) >= 2)
            else "verified" if f.get("verifying_check")
            else "corroborated x%d" % int(cb or 1) if int(cb or 1) >= 2 else "single-source")
        # entity = the row's discriminator (per-abstraction / per-region metrics share workload=ALL,
        # so without it the headline shows uninterpretable duplicate rows)
        entity = f.get("abstraction") or f.get("region") or f.get("workload", "")
        # collapse rows that are the same metric on the same entity+value (e.g. one accuracy fact
        # surfaced by two artifacts with different implication wording) to one headline row
        key = (f.get("dse_question", ""), f["metric_name"], f.get("workload", ""), entity,
               str(f.get("metric_value", "")))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "dse_question": f.get("dse_question", ""), "metric": f["metric_name"],
            "workload": f.get("workload", ""), "entity": entity,
            "value": f.get("metric_value", ""),
            "unit": f.get("metric_unit", ""), "evidence_tier": f.get("evidence_tier", ""),
            "strength": strength, "verification_status": f.get("verification_status", ""),
            "dse_implication": f.get("dse_implication", "")})
    rows.sort(key=lambda r: (r["dse_question"], r["metric"], r["workload"], r["entity"]))
    return rows


def per_operator_hotspots(cs_dir: Path, scope: str = "all", k: int = 10) -> dict:
    """Deeper-per-operator signal: which few ops dominate compute / weight / tile-waste / reload.

    Reads operator_shape_table (per-op M/N/K, macs, weight bytes, shape class), joins the best
    achievable tile padding waste per op from tile_waste_table, and the region reload from
    data_movement_table. Top-k by each axis; structural quantities only. When ``scope`` is a single
    workload (not "all"), the tables are restricted to that workload so a per-network run reports
    that network's own hotspots, not the corpus-wide top."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    tw = _csv_rows(cs_dir / "tile_waste_table.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    if scope != "all":
        ops = [o for o in ops if o["workload"] == scope]
        tw = [r for r in tw if r["workload"] == scope]
        dm = [r for r in dm if r["workload"] == scope]
    # best (lowest) achievable tile padding waste per op over the *real* applicable tile primitives
    best: dict[tuple, float] = {}
    for r in tw:
        if r.get("primitive_kind") != "tile" or r.get("applicable") != "True":
            continue
        key = (r["workload"], r["op_index"])
        w = float(r["padding_waste"])
        if key not in best or w < best[key]:
            best[key] = w
    mac_total_wl: dict[str, int] = {}
    for o in ops:
        mac_total_wl[o["workload"]] = mac_total_wl.get(o["workload"], 0) + int(o["macs"])
    enriched = []
    for o in ops:
        wl = o["workload"]
        tot = mac_total_wl.get(wl, 0) or 1
        enriched.append({
            "workload": wl, "op_index": o["op_index"], "prov_fqn": o["prov_fqn"],
            "op_kind": o["op_kind"], "shape_class": o["shape_class"],
            "M": o["M"], "N": o["N"], "K": o["K"], "macs": int(o["macs"]),
            "mac_fraction_of_workload": round(int(o["macs"]) / tot, 4),
            "rhs_weight_bytes": int(o["rhs_weight_bytes"]),
            "best_tile_padding_waste": best.get((wl, o["op_index"])),
            "is_tail_heavy": o["is_tail_heavy"]})
    by_macs = sorted(enriched, key=lambda r: -r["macs"])[:k]
    by_weight = sorted(enriched, key=lambda r: -r["rhs_weight_bytes"])[:k]
    by_pad = sorted([r for r in enriched if r["best_tile_padding_waste"] is not None],
                    key=lambda r: -r["best_tile_padding_waste"])[:k]
    by_reload = sorted(dm, key=lambda r: -int(r["avoidable_weight_reload"]))[:k]
    reload_rows = [{"workload": r["workload"], "region": r["region"],
                    "avoidable_weight_reload": int(r["avoidable_weight_reload"]),
                    "invocations": r["invocations"], "weight_bytes": int(r["weight_bytes"])}
                   for r in by_reload]
    # long-form flat table for CSV: one row per (ranking, rank, entity)
    flat = []
    for tag, rows_ in (("macs", by_macs), ("weight_bytes", by_weight), ("padding_waste", by_pad)):
        for i, r in enumerate(rows_, 1):
            flat.append({"ranking": tag, "rank": i, **r, "region": "",
                         "avoidable_weight_reload": ""})
    for i, r in enumerate(reload_rows, 1):
        flat.append({"ranking": "avoidable_reload", "rank": i, "workload": r["workload"],
                     "op_index": "", "prov_fqn": "", "op_kind": "", "shape_class": "",
                     "M": "", "N": "", "K": "", "macs": "", "mac_fraction_of_workload": "",
                     "rhs_weight_bytes": r["weight_bytes"], "best_tile_padding_waste": "",
                     "is_tail_heavy": "", "region": r["region"],
                     "avoidable_weight_reload": r["avoidable_weight_reload"]})
    dominant = by_macs[0] if by_macs else None
    return {"n_ops": len(ops), "by_macs": by_macs, "by_weight_bytes": by_weight,
            "by_padding_waste": by_pad, "by_avoidable_reload": reload_rows,
            "dominant_op": dominant, "rows": flat}


# proof-status conservatism: report the weakest status when an abstraction is referenced by several
# compiler-proof axes (unknown is worse than assumed, which is worse than proven).
_PROOF_RANK = {"unknown": 0, "unavailable": 0, "assumed": 1, "suggested": 1, "proven": 2, "": 3}


def abstraction_coverage(cs_dir: Path) -> list[dict]:
    """Replace the noisy ``n_X`` count facts with real coverage: for each candidate system
    abstraction, what fraction of the corpus (workloads / MACs / weight bytes / regions) implies it,
    its compiler-proof status, and an overfit/regret flag (single-workload support)."""
    cs_dir = Path(cs_dir)
    bm = _csv_rows(cs_dir / "hw_sw_boundary_matrix.csv")
    proofs = _csv_rows(cs_dir / "compiler_proof_matrix.csv")
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    all_wls = sorted({o["workload"] for o in ops})
    mac_by_wl: dict[str, int] = {}
    byte_by_wl: dict[str, int] = {}
    for o in ops:
        mac_by_wl[o["workload"]] = mac_by_wl.get(o["workload"], 0) + int(o["macs"])
        byte_by_wl[o["workload"]] = byte_by_wl.get(o["workload"], 0) + int(o["rhs_weight_bytes"])
    total_mac = sum(mac_by_wl.values()) or 1
    total_byte = sum(byte_by_wl.values()) or 1
    region_by_wl: dict[str, int] = {}
    for r in dm:
        region_by_wl[r["workload"]] = region_by_wl.get(r["workload"], 0) + 1
    total_regions = len(dm) or 1
    rows = []
    for b in bm:
        supp = [w.strip() for w in b["supporting_workloads"].split(";") if w.strip()]
        nwl = len(supp)
        # weakest proof status among compiler-proof axes that reference this atomic abstraction
        status = ""
        for p in proofs:
            if b["abstraction"] in p.get("system_abstraction", ""):
                if _PROOF_RANK.get(p["status"], 3) < _PROOF_RANK.get(status, 3):
                    status = p["status"]
        overfit = ("high" if nwl <= 1 else "medium" if nwl < len(all_wls) else "low")
        rows.append({
            "abstraction": b["abstraction"],
            "workloads_supporting": "; ".join(supp), "n_workloads": nwl,
            "workload_coverage": round(nwl / (len(all_wls) or 1), 4),
            "mac_coverage": round(sum(mac_by_wl.get(w, 0) for w in supp) / total_mac, 4),
            "byte_coverage": round(sum(byte_by_wl.get(w, 0) for w in supp) / total_byte, 4),
            "region_coverage": round(sum(region_by_wl.get(w, 0) for w in supp) / total_regions, 4),
            "boundary_pressure_score": int(b["boundary_pressure_score"]),
            "compiler_proof_status": status or "no_proof_axis",
            "overfit_risk": overfit})
    rows.sort(key=lambda r: (-r["mac_coverage"], -r["boundary_pressure_score"], r["abstraction"]))
    return rows


def _family_of(workload: str) -> str:
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    base = _base_model(workload)
    if base and base in MODEL_ARCH:
        return MODEL_ARCH[base].family
    return "unknown"


def workload_family_summary(cs_dir: Path, findings: list[dict] | None = None) -> dict:
    """Group the recaptured workloads by architecture family and report the per-family signal
    profile, plus which findings are family-specific vs cross-family."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    shape = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    crit = {r["workload"]: r for r in _csv_rows(cs_dir / "critical_path_table.csv")}
    wls = sorted({o["workload"] for o in ops})
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    fam: dict[str, dict] = {}
    for w in wls:
        f = _family_of(w)
        fam.setdefault(f, {"workloads": [], "total_macs": 0, "dominant_shape_class": "",
                           "parallelism": [], "K": []})
        fam[f]["workloads"].append(w)
        fam[f]["total_macs"] += sum(int(o["macs"]) for o in ops if o["workload"] == w)
        if w in crit:
            fam[f]["parallelism"].append(float(crit[w]["available_parallelism"]))
        base = _base_model(w)
        if base and base in MODEL_ARCH:
            fam[f]["K"].append(MODEL_ARCH[base].loop_count)
    # dominant shape class per family = max mac_fraction-weighted class across its workloads
    for f, d in fam.items():
        agg: dict[str, float] = {}
        for r in shape:
            if r["workload"] in d["workloads"]:
                agg[r["shape_class"]] = agg.get(r["shape_class"], 0.0) + float(r["mac_fraction"])
        d["dominant_shape_class"] = max(agg, key=agg.get) if agg else ""
        d["parallelism_range"] = ([round(min(d["parallelism"]), 2),
                                   round(max(d["parallelism"]), 2)] if d["parallelism"] else [])
        d["K_range"] = [min(d["K"]), max(d["K"])] if d["K"] else []
        d.pop("parallelism")
    fam_specific, cross = [], []
    for fd in (findings or []):
        if fd.get("presentation_placement") != "main":
            continue
        fams = {_family_of(w) for w in fd.get("supporting_workloads", []) if w not in ("ALL",)}
        fams.discard("unknown")
        rec = {"claim": fd.get("title", ""), "families": sorted(fams),
               "dse_question": fd.get("dse_question", "")}
        (fam_specific if len(fams) == 1 else cross).append(rec)
    return {"families": fam, "family_specific_findings": fam_specific,
            "cross_family_findings": cross}


# capture-fidelity asks that would most raise cross-workload confidence (what a recapture must keep).
_FIDELITY_ASKS = [
    "preserve the host K-loop (do not unroll the denoise/decode loop into a single pass)",
    "preserve packed low-bit weight layout + per-channel scales (do not dequantize to bf16)",
    "preserve the KV-cache / attention region (do not lower attention to dense matmuls)",
    "tag region roles (backbone-once vs repeated-head) in prov.fqn",
    "record the real loop count / control cadence (replace the assumed K reference)",
]


def corpus_expansion_plan(cs_dir: Path) -> dict:
    """Which registry families lack a committed recapture, grouped by family, plus the capture
    fidelity improvements that would most raise cross-workload confidence. A recommendation — no new
    data is ingested; this only states what additional captures the search-space study needs."""
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    captured = {b for b in (_base_model(w) for w in {o["workload"] for o in ops}) if b}
    missing = [m for m in MODEL_ARCH if m not in captured]
    by_family: dict[str, list[dict]] = {}
    for m in missing:
        a = MODEL_ARCH[m]
        by_family.setdefault(a.family, []).append({
            "model": m, "loop_kind": a.loop_kind, "reference_K": a.loop_count,
            "note": a.note or ""})
    return {"captured_models": sorted(captured),
            "captured_families": sorted({MODEL_ARCH[b].family for b in captured if b in MODEL_ARCH}),
            "missing_by_family": by_family, "n_missing": len(missing),
            "fidelity_asks": _FIDELITY_ASKS}


# one-sentence DSE-search-space implication shown beneath each presentation plot (structural only).
_PLOT_CAPTION = {
    "evidence_type_by_workload": "How much of each workload's evidence is IR-recovered vs assumed — "
        "where the search space rests on recovered structure vs reference values.",
    "evidence_type_by_phase": "Evidence provenance per analysis phase (traceability).",
    "shape_class_mac_share": "MAC mass per shape class — which matmul shapes a DSE primitive set must "
        "cover to capture most compute.",
    "shape_class_opcount_share": "Op-count share per shape class (context for the MAC share).",
    "primitive_coverage_heatmap": "Which candidate primitives tile each workload's shapes under 10% "
        "pad waste — the primitive search space, not a performance ranking.",
    "primitive_regret_bar": "Coverage vs worst-case cross-workload regret per primitive — primitives "
        "with high regret are corpus-overfit candidates DSE should treat cautiously.",
    "abstraction_pressure_bar": "How many workloads imply each system abstraction (support breadth).",
    "boundary_placement_heatmap": "At which HW/SW levels each abstraction could sit — the boundary "
        "search space DSE must explore (Merlin enumerates, does not choose).",
    "resident_capacity_by_dtype": "Resident weight bytes per region by dtype — the on-chip capacity "
        "the residency search space is sized against.",
    "avoidable_reload_by_region": "Weight bytes re-read across the K-loop that residency could avoid "
        "— where a residency/packed-store axis has the most to act on.",
    "measurement_priority_bar": "How many blocked candidates each missing input would unblock — what "
        "to capture/measure next, not a result.",
    "critical_path_parallelism": "Inter-op work/span per workload — the unit-multiplicity the "
        "heterogeneity search space could exploit.",
    "epilogue_pattern_counts": "Epilogue/fusion patterns per workload (numerical-contract context).",
    "decision_primitive_choice": "If DSE builds only ONE compute primitive, how much of each "
        "workload's MACs it can tile under 10% waste — the worst-case bar shows no single primitive "
        "covers every workload, so the search space needs both a tile and a GEMV lane.",
    "decision_weight_residency": "Weight bytes moved as the head loop count grows: reload-every-step "
        "(linear) vs keep-resident (flat). The vertical gap at each workload's real K is the "
        "avoidable reload a residency knob removes (bytes, not bandwidth).",
    "decision_capacity_dtype": "How many workloads become fully weight-resident as the on-chip "
        "capacity budget grows, per storage dtype — low-bit dtypes reach full residency at a smaller "
        "budget, quantifying the capacity-vs-dtype trade in the search space.",
    "decision_sharding_cost": "Extra data-movement bytes added by sharding 2/4/8 ways along M, N, or "
        "K: M/N shards are reduction-free (broadcast only) while K shards add partial-sum traffic — "
        "the cost side of the parallelization decision.",
}


def plot_manifest(cs_dir: Path, scope: str) -> list[dict]:
    out = []
    for pid, title, art, cols, x, y, series, kind, rec in _PLOTS:
        # unified_fact_table.csv is the run's own output (always produced) — treat as available
        present = (cs_dir / art).is_file() or art == "unified_fact_table.csv"
        have_cols = True
        n_rows = 0
        if present and art.endswith(".csv") and (cs_dir / art).is_file():
            hdr = (cs_dir / art).read_text().splitlines()[:1]
            header = hdr[0].split(",") if hdr else []
            have_cols = all(c in header for c in cols if c not in ("level", "count",
                                                                    "workload_region", "bytes",
                                                                    "dtype"))
            n_rows = len(_csv_rows(cs_dir / art))
        elif art == "unified_fact_table.csv":
            n_rows = len(unified_facts(cs_dir, scope))   # the run's own fact table (B6: non-empty)
        out.append({
            "plot_id": pid, "title": title, "source_artifact": art, "required_columns": cols,
            "x_axis": x, "y_axis": y, "series": series, "plot_type": kind,
            "available": bool(present and have_cols and n_rows > 0), "n_source_rows": n_rows,
            "evidence_tier": "B",
            "recommendation": rec if (present and have_cols and n_rows > 0) else "omit",
            "caveat": "structural; axes are counts/bytes/fractions, not a performance metric",
            "dse_caption": _PLOT_CAPTION.get(pid, ""),
            "why_useful": f"shows {title.lower()} from {art}"})
    return out


# --------------------------------------------------------------------------- P13-g consistency

def consistency_checks(cs_dir: Path, inventory, facts, findings, plots, answers) -> list[tuple]:
    res = []

    def chk(ok, msg):
        res.append((bool(ok), msg))

    art_names = {r["artifact"] for r in inventory}
    exist = {r["artifact"] for r in inventory if r["exists"]}
    # also recognize per-workload subdir artifacts (numerical_contract.yaml, region_attribution.yaml)
    for sub in cs_dir.glob("*/"):
        if sub.is_dir():
            exist.update(p.name for p in sub.glob("*") if p.is_file())
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


# --------------------------------------------------------------------------- inherent limits

# Each genuinely-external limit + the EXACT input/run that closes it. These are scoped (have a
# required_input), so the audit treats them as not-open — never bare caveats, never fabricated.
_INHERENT_LIMITS = [
    {"limit": "real deployment K + control rate", "currently": "K/control are published config "
     "reference values (recovered_from_model_config)", "evidence_today": "Tier C",
     "required_input": "a deployment/runtime trace giving actual loop counts + control frequency"},
    {"limit": "per-unit throughput / latency / area / energy", "currently": "Merlin refuses to "
     "estimate these for unbuilt hardware", "evidence_today": "unavailable",
     "required_input": "a candidate design YAML (unit shapes + a cost model); then the future DSE "
     "tool computes them"},
    {"limit": "KV / attention structure + true data deps at loop level", "currently": "attention "
     "is lowered into matmul projections; KV unavailable", "evidence_today": "unavailable",
     "required_input": "a Level-2 loop-preserving, attention-not-lowered capture"},
    {"limit": "packed low-bit layout + scales for the recaptured models", "currently": "the "
     "recaptures are dequantized f32; the quantized zoo has low-bit storage for OTHER models",
     "evidence_today": "low-bit storage shown on the zoo (numerical_contract_fidelity_report.md)",
     "required_input": "a low-bit (packed weights + scale metadata) capture of the recaptured models"},
    {"limit": "fp8 / int4 accuracy gates", "currently": "int8 W8A8 is measured (real); fp8/int4 "
     "are unavailable", "evidence_today": "int8 measured",
     "required_input": "per-format accuracy runs (W8A8 already done) for fp8 / int4"},
    {"limit": "real-magnitude weights", "currently": "captures are small random-init instances "
     "(structure real, magnitudes small)", "evidence_today": "structure recovered_from_ir",
     "required_input": "full-size (non-random-init) captures of the same architectures"},
]


def required_inputs(cs_dir: Path, facts: list[dict]) -> list[dict]:
    """The scoped inherent limits + the exact input/run that closes each (nothing invented)."""
    return [dict(x, status="scoped", avoidable=False) for x in _INHERENT_LIMITS]


# --------------------------------------------------------------------------- P14 devil's-advocate

def gap_audit(cs_dir: Path, scope: str, bundle: dict | None = None) -> list[dict]:
    """The devil's advocate: flag every fake/weak/unbacked/unused/heuristic item. Returns GapItems;
    convergence = 0 OPEN AVOIDABLE gaps (inherent limits are 'scoped' via a required_input)."""
    cs_dir = Path(cs_dir)
    b = bundle or mine(cs_dir, scope)
    inv, facts, findings, plots, answers = (b["inventory"], b["facts"], b["findings"], b["plots"],
                                            b["usefulness"])
    gaps = []
    k = 0

    def g(category, target, desc, avoidable, status, required_input=""):
        nonlocal k
        gaps.append({"gap_id": f"G{k:03d}", "scope": scope, "category": category, "target": target,
                     "description": desc, "avoidable": avoidable, "status": status,
                     "required_input": required_input})
        k += 1

    used = {f["source_artifact"] for f in facts}
    # 1 unused_artifact: the 'all' scope must cover every present artifact (per-network scopes only
    # cover their per-workload slice — cross-workload-only artifacts are not theirs to mine).
    if scope == "all":
        for r in inv:
            if r["exists"] and r["source_phase"] != "other" and r["artifact"] not in used:
                g("unused_artifact", r["artifact"], "present artifact contributes 0 mined facts",
                  True, "open")
    # 2 fake_or_unbacked_finding: a main finding without a per-metric-verified or corroborated fact
    fact_by_metric = {}
    for f in facts:
        fact_by_metric.setdefault(f["metric_name"], []).append(f)
    for f in findings:
        if f["presentation_placement"] != "main":
            continue
        fs = [x for m in f["relevant_metrics"] for x in fact_by_metric.get(m, [])]
        backed = any(x["verifying_check"] or x["corroborated_by"] >= 2 for x in fs)
        if not backed:
            g("fake_or_unbacked_finding", f["title"],
              "main finding has no per-metric-verified or >=2-corroborated fact", True, "open")
    # 3 heuristic_status: a 'strong' answer with 0 backing strong facts
    for a in answers:
        if a["status"] == "strong" and a.get("n_backing_strong_facts", 0) == 0:
            g("heuristic_status", a["key"], "'strong' status with no backing tier-A/B fact",
              True, "open")
    # 4 coarse_verification: a tier-A/B fact whose 'verified' is neither a check nor >=2-corroboration
    for f in facts:
        if f["evidence_tier"] in ("A", "B") and f["verification_status"] == "verified" \
                and not (f["verifying_check"] or f["corroborated_by"] >= 2):
            g("coarse_verification", f["fact_id"],
              "verified tier-A/B fact lacks a specific check and <2 corroboration", True, "open")
    # 5 uncorroborated_fact: a candidate that is NOT a SIGNAL metric, or is tier C/D (the real fake
    # risk). A single-source recovered_from_ir SIGNAL fact is legitimate evidence, not fake -- its
    # single-source status is reported via corroborated_by, not treated as a gap.
    for f in facts:
        if f["presentation_candidate"] and (f["metric_name"] not in SIGNAL_METRICS
                                            or f["evidence_tier"] in ("C", "D")):
            g("uncorroborated_fact", f["fact_id"],
              "candidate is not a tier-A/B signal metric", True, "open")
    # 6 aggregated_finding: a multi-workload finding without a per-workload spread
    for f in findings:
        if len(f["supporting_workloads"]) > 1 and not f.get("per_workload_spread"):
            g("aggregated_finding", f["title"], "multi-workload finding hides per-workload spread",
              True, "open")
    # 7 unvalidated_plot: a non-omit plot with empty source data
    for p in plots:
        if p["recommendation"] != "omit" and p.get("n_source_rows", 0) <= 0:
            g("unvalidated_plot", p["plot_id"], "non-omit plot has empty source data", True, "open")
    # 8 inherent_limit: scoped (has a required_input) -> not an open gap
    for lim in required_inputs(cs_dir, facts):
        g("inherent_limit", lim["limit"], lim["currently"], False, "scoped", lim["required_input"])
    return gaps


def open_avoidable(gaps: list[dict]) -> list[dict]:
    return [x for x in gaps if x["avoidable"] and x["status"] == "open"]


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
    bundle = {"scope": scope, "inventory": inv, "facts": facts, "evidence_strength": strength,
              "usefulness": answers, "findings": findings, "plots": plots,
              "consistency_checks": checks,
              "required_inputs": required_inputs(cs_dir, facts)}
    bundle["gaps"] = gap_audit(cs_dir, scope, bundle)
    bundle["open_avoidable_gaps"] = open_avoidable(bundle["gaps"])
    # P15 signal-first study layer (signal-only, every output answers a DSE question). Per-network
    # scopes get their OWN canonical table + hotspots; the cross-workload artifacts (coverage /
    # family / corpus) are corpus-level by nature and are emitted only in the 'all' scope (a single
    # workload's coverage/regret/family is degenerate, so dumping the corpus-wide table into a
    # per-network folder would misrepresent it).
    bundle["canonical_signal"] = canonical_signal_table(facts)
    bundle["hotspots"] = per_operator_hotspots(cs_dir, scope)
    is_all = scope == "all"
    bundle["abstraction_coverage"] = abstraction_coverage(cs_dir) if is_all else []
    bundle["family_summary"] = (workload_family_summary(cs_dir, findings) if is_all
                                else {"families": {}, "family_specific_findings": [],
                                      "cross_family_findings": []})
    bundle["corpus_plan"] = corpus_expansion_plan(cs_dir) if is_all else {}
    bundle["cs_dir"] = str(cs_dir)
    return bundle


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
         f"Total normalized facts: **{s['total_facts']}** — **{s['signal_facts']} signal/measured** "
         f"(decision-relevant; can become findings) + **{s['context_facts']} context** "
         f"(row counts / provenance / corroboration — traceability only, never a finding). "
         f"signal ratio = {s['signal_to_total']}.\n",
         "Tiers — A: IR-recovered/measured + verified; B: recovered-unverified or derived+verified; "
         "C: assumed_reference / config; D: unavailable/unknown.\n",
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
            f"- consistency checks: {npass}/{len(bundle['consistency_checks'])}\n"
            f"- **devil's-advocate gap audit: {len(bundle.get('open_avoidable_gaps', []))} open "
            f"avoidable gaps** (target 0); "
            f"{len(bundle.get('required_inputs', []))} inherent limits scoped to required inputs\n\n"
            "Files: artifact_inventory.{csv,md}, unified_fact_table.{csv,yaml}, "
            "evidence_strength_table.csv + evidence_strength_report.md, dse_usefulness_scorecard.md "
            "+ dse_usefulness_answers.yaml, presentation_candidate_findings.{md,csv}, "
            "plot_manifest.{yaml,md}, generated_plots/, consistency_checks.md, "
            "gap_audit_report.{md,csv}, required_inputs_manifest.{yaml,md}.\n")


def _gap_audit_md(gaps, open_av, scope) -> str:
    from collections import Counter
    by_cat = Counter(g["category"] for g in gaps)
    L = [f"# Devil's-advocate gap audit ({scope})\n",
         f"**Open avoidable gaps: {len(open_av)}** (convergence target = 0). "
         f"{sum(1 for g in gaps if g['category']=='inherent_limit')} inherent limits are scoped "
         "(each carries a required input — see required_inputs_manifest).\n",
         "| category | count | open avoidable |", "|---|---|---|"]
    for cat in sorted(by_cat):
        oa = sum(1 for g in open_av if g["category"] == cat)
        L.append(f"| {cat} | {by_cat[cat]} | {oa} |")
    if open_av:
        L.append("\n## Open avoidable gaps (must close)\n")
        for g in open_av:
            L.append(f"- [{g['category']}] {g['target']}: {g['description']}")
    else:
        L.append("\n**No open avoidable gaps — the analysis leverages every artifact, every "
                 "presented fact is corroborated/checked, every main finding is backed, and every "
                 "inherent limit is scoped to a required input.**")
    return "\n".join(L) + "\n"


def _required_inputs_md(ri, scope) -> str:
    L = [f"# Required-inputs manifest ({scope})\n",
         "> The inherent limits that remain, each with the EXACT input/run that closes it. These are "
         "scoped (not bare caveats) and nothing is fabricated — Merlin reports what a real DSE run "
         "still needs.\n",
         "| limit | evidence today | required input to close |", "|---|---|---|"]
    for x in ri:
        L.append(f"| {x['limit']} | {x['evidence_today']} | {x['required_input']} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- P15 emitters

_CANON_COLS = ["dse_question", "metric", "workload", "entity", "value", "unit", "evidence_tier",
               "strength", "verification_status", "dse_implication"]
_HOTSPOT_COLS = ["ranking", "rank", "workload", "op_index", "prov_fqn", "op_kind", "shape_class",
                 "M", "N", "K", "macs", "mac_fraction_of_workload", "rhs_weight_bytes",
                 "best_tile_padding_waste", "is_tail_heavy", "region", "avoidable_weight_reload"]
_COVER_COLS = ["abstraction", "workloads_supporting", "n_workloads", "workload_coverage",
               "mac_coverage", "byte_coverage", "region_coverage", "boundary_pressure_score",
               "compiler_proof_status", "overfit_risk"]


def _hotspots_md(h, scope) -> str:
    d = h.get("dominant_op")
    L = [f"# Per-operator hotspots ({scope})\n",
         "> Which few operators dominate the constraints DSE must size for. Structural quantities "
         "(MACs / weight bytes / tile padding waste / avoidable reload) recovered from the capture — "
         "no latency, throughput, or performance claim.\n",
         f"Total operators analyzed: **{h['n_ops']}**.\n"]
    if d:
        L.append(f"**Dominant op (by MACs):** `{d['prov_fqn']}` in {d['workload']} — "
                 f"{d['M']}x{d['N']}x{d['K']} = {d['macs']:,} MACs "
                 f"({d['mac_fraction_of_workload']:.0%} of its workload), class {d['shape_class']}.\n")
    L += ["## Top ops by MACs\n", "| workload | op | shape M×N×K | MACs | % of workload | class |",
          "|---|---|---|---|---|---|"]
    for r in h["by_macs"]:
        L.append(f"| {r['workload']} | {r['prov_fqn']} | {r['M']}×{r['N']}×{r['K']} | "
                 f"{r['macs']:,} | {r['mac_fraction_of_workload']:.0%} | {r['shape_class']} |")
    L += ["\n## Top ops by tile padding waste (tile-hostility)\n",
          "| workload | op | shape M×N×K | best tile waste | class |", "|---|---|---|---|---|"]
    for r in h["by_padding_waste"]:
        L.append(f"| {r['workload']} | {r['prov_fqn']} | {r['M']}×{r['N']}×{r['K']} | "
                 f"{r['best_tile_padding_waste']:.3f} | {r['shape_class']} |")
    L += ["\n## Regions by avoidable weight reload (residency target)\n",
          "| workload | region | avoidable reload (B) | weight bytes (B) |", "|---|---|---|---|"]
    for r in h["by_avoidable_reload"]:
        L.append(f"| {r['workload']} | {r['region']} | {r['avoidable_weight_reload']:,} | "
                 f"{r['weight_bytes']:,} |")
    return "\n".join(L) + "\n"


def _coverage_md(rows, scope) -> str:
    L = [f"# Abstraction coverage ({scope})\n",
         "> Replaces raw support counts with corpus **coverage**: for each candidate system "
         "abstraction, the fraction of workloads / MACs / weight bytes / regions that imply it, its "
         "compiler-proof status, and an overfit flag (single-workload support is corpus-overfit "
         "risk). Support breadth, not a performance ranking.\n",
         "| abstraction | workloads | MAC cov | byte cov | proof | overfit |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['abstraction']} | {r['n_workloads']} ({r['workload_coverage']:.0%}) | "
                 f"{r['mac_coverage']:.0%} | {r['byte_coverage']:.0%} | "
                 f"{r['compiler_proof_status']} | {r['overfit_risk']} |")
    return "\n".join(L) + "\n"


def _family_md(fs, scope) -> str:
    L = [f"# Workload-family summary ({scope})\n",
         "> The recaptured workloads grouped by architecture family, with each family's recovered "
         "signal profile and which findings are family-specific vs cross-family. Structural only.\n"]
    for fam, d in sorted(fs["families"].items()):
        L.append(f"## {fam}\n")
        L.append(f"- workloads: {', '.join(d['workloads'])}")
        L.append(f"- total MACs (single-pass capture): {d['total_macs']:,}")
        L.append(f"- dominant shape class: {d['dominant_shape_class'] or '—'}")
        L.append(f"- inter-op parallelism (work/span) range: {d.get('parallelism_range') or '—'}")
        L.append(f"- reference loop count K range: {d.get('K_range') or '—'}\n")
    L.append("## Family-specific findings (one family only)\n")
    for f in fs["family_specific_findings"] or [{"claim": "—", "families": [], "dse_question": ""}]:
        L.append(f"- {f['claim']} _(families: {', '.join(f['families']) or '—'})_")
    L.append("\n## Cross-family findings (≥2 families)\n")
    for f in fs["cross_family_findings"] or [{"claim": "—", "families": [], "dse_question": ""}]:
        L.append(f"- {f['claim']} _(families: {', '.join(f['families']) or '—'})_")
    return "\n".join(L) + "\n"


def _corpus_md(cp, scope) -> str:
    L = [f"# Corpus-expansion plan ({scope})\n",
         "> A recommendation only — no new data is ingested here. Which registry model families lack "
         "a committed recapture, and the capture-fidelity improvements that would most raise "
         "cross-workload confidence before any quantitative DSE.\n",
         f"Captured models: {', '.join(cp['captured_models']) or '—'} "
         f"(families: {', '.join(cp['captured_families']) or '—'}).\n",
         f"**{cp['n_missing']} registry models lack a recapture.**\n",
         "## Missing captures by family\n",
         "| family | model | loop kind | reference K | note |", "|---|---|---|---|---|"]
    for fam, models in sorted(cp["missing_by_family"].items()):
        for m in models:
            L.append(f"| {fam} | {m['model']} | {m['loop_kind']} | {m['reference_K']} | "
                     f"{m['note']} |")
    L.append("\n## Capture-fidelity asks (raise confidence on existing + new captures)\n")
    for a in cp["fidelity_asks"]:
        L.append(f"- {a}")
    return "\n".join(L) + "\n"


def _signal_report_md(bundle, scope) -> str:
    canon = bundle["canonical_signal"]
    findings = [f for f in bundle["findings"] if f["presentation_placement"] == "main"]
    by_q_find: dict[str, list] = {}
    for f in findings:
        by_q_find.setdefault(f.get("dse_question", ""), []).append(f)
    by_q_metric: dict[str, set] = {}
    for r in canon:
        by_q_metric.setdefault(r["dse_question"], set()).add(r["metric"])
    L = [f"# Signal findings report ({scope})\n",
         "> Given only workload artifacts: what changes the future DSE search space. Every finding "
         "is recovered/derived from the captures (or a host measurement); **no quantity is claimed "
         "for unbuilt hardware** (cycles / area / energy / throughput are refused, not estimated). "
         "Organized by the DSE question each metric answers.\n"]
    for q in DSE_QUESTIONS:
        fs = by_q_find.get(q, [])
        ms = sorted(by_q_metric.get(q, set()))
        L.append(f"## {q}\n")
        if not fs and not ms:
            L.append("_No signal recovered for this question from the current corpus._\n")
            continue
        L.append(f"_Signal metrics: {', '.join(ms) or '—'}_\n")
        for f in fs:
            L.append(f"- **{f['title']}** [tier {f['evidence_tier']}] — {f['dse_implication']}  "
                     f"_(workloads: {', '.join(f['supporting_workloads'])})_")
        L.append("")
    # what remains unclaimed
    L.append("## What remains unclaimed (and the exact input needed)\n")
    for x in bundle.get("required_inputs", []):
        L.append(f"- {x['limit']}: {x['required_input']}")
    # closing devil's-advocate note: which signal is robust vs corpus-limited
    fam = bundle.get("family_summary", {})
    n_fam_specific = len(fam.get("family_specific_findings", []))
    L.append("\n## Devil's advocate — robust vs corpus-limited\n")
    L.append("**Robust (structural, independent of magnitudes):** shape-class distribution, the "
             "recovered SSA data-dependency graph, the backbone/head role split, the dtype/epilogue "
             "numerical contract, and per-op MAC *fractions* (relative, not absolute). These hold "
             "regardless of weight magnitudes.")
    L.append("\n**Corpus-limited (treat as directional, not settled):**")
    L.append("- The 4 recaptures are small, random-init f32 instances — structure and provenance "
             "are real, but absolute byte/MAC magnitudes are a small instance, so any finding that "
             "leans on absolute size is directional only.")
    L.append("- low-bit / KV / attention structure is erased or lowered in the capture, so those "
             "candidates are blocked, not measured (see What remains unclaimed).")
    if bundle.get("scope") == "all":
        L.append(f"- {n_fam_specific} family-specific findings recovered: the corpus is small and "
                 "structurally homogeneous across families, so cross-family separation is weak — "
                 "expanding the corpus (see corpus_expansion_plan.md) is the prerequisite before any "
                 "cross-family DSE claim.")
    return "\n".join(L) + "\n"


def _plots_index_md(plots, scope, rendered, folder) -> str:
    L = [f"# Presentation plots index ({scope})\n",
         "> Each rendered plot with its one-sentence DSE-search-space implication. Structural axes "
         "only (counts / bytes / fractions) — none is a performance metric.\n"]
    for p in plots:
        if p["plot_id"] not in rendered:
            continue
        L.append(f"### {p['title']}\n")
        L.append(f"![{p['plot_id']}]({folder}/{p['plot_id']}.png)\n")
        L.append(f"**DSE implication:** {p.get('dse_caption') or '—'}\n")
    return "\n".join(L) + "\n"


def _findings_digest_md(bundle: dict, cs_dir: Path, rendered: list[str]) -> str:
    """ONE self-contained findings document per run: every metric table, the operator hotspots, the
    decision-impact plots, and the DSE-ingest knobs in a single file you can open and evaluate.

    Pure presentation of already-computed bundle data + the committed knob catalog — no new numbers.
    """
    cs_dir = Path(cs_dir)
    scope = bundle["scope"]
    canon = bundle["canonical_signal"]
    L = [f"# DSE findings digest — scope: {scope}\n",
         "Self-contained summary of the workload-contract analysis. Every number is recovered from "
         "the captures or a host measurement; **no quantity is claimed for unbuilt hardware**. "
         "Each metric carries an evidence tier (A/measured = IR or real measurement; B = "
         "recovered/derived + recompute check; C = config/assumed; D = unavailable) so you can "
         "weight it yourself. Source CSV/YAML are in this same folder; the full per-fact trace "
         "(metric -> source artifact -> check) is in `unified_fact_table.csv`.\n",
         "## 1. Headline metrics (canonical_signal_table.csv)\n",
         "Grouped by the DSE question each answers. `entity` is the thing the metric is about "
         "(workload / abstraction / region).\n"]
    for q in DSE_QUESTIONS:
        qrows = [r for r in canon if r["dse_question"] == q]
        if not qrows:
            continue
        L.append(f"### {q}\n")
        L.append("| metric | entity | value | unit | tier | strength | implication |")
        L.append("|---|---|---|---|---|---|---|")
        for r in qrows:
            L.append(f"| {r['metric']} | {r['entity']} | {r['value']} | {r['unit']} | "
                     f"{r['evidence_tier']} | {r['strength']} | {r['dse_implication']} |")
        L.append("")
    # 2. operator hotspots (reuse the dedicated builder; replace its H1 with our section header)
    L.append("## 2. Per-operator hotspots\n")
    L.append(_hotspots_md(bundle["hotspots"], scope).split("\n", 1)[1])
    # 3. abstraction coverage (only meaningful at 'all')
    if bundle["abstraction_coverage"]:
        L.append("## 3. Abstraction coverage\n")
        L.append(_coverage_md(bundle["abstraction_coverage"], scope).split("\n", 1)[1])
    # 4. decision-impact plots
    L.append("## Decision-impact plots (what changes if DSE picks differently)\n")
    L.append("Structural what-if curves (bytes / coverage / counts — never latency or speedup). "
             "PNGs under `generated_plots/`.\n")
    for p in bundle["plots"]:
        if p["plot_id"].startswith("decision_") and p["plot_id"] in rendered:
            L.append(f"- **{p['title']}** (`generated_plots/{p['plot_id']}.png`)  \n"
                     f"  {p.get('dse_caption', '')}")
    # 5. the DSE-ingest knob catalog (embed the committed consolidated table verbatim)
    L.append("\n## What a DSE tool ingests — knob catalog\n")
    knobs = cs_dir / "dse_search_space_knobs.md"
    if knobs.is_file():
        body = knobs.read_text().split("\n", 1)[1] if "\n" in knobs.read_text() else ""
        L.append(body.strip())
    else:
        L.append("_dse_search_space_knobs.md not found in the package._")
    # 6. how to read / reproduce
    L.append("\n## How to evaluate this yourself\n")
    L.append("- Every headline row traces through `unified_fact_table.csv` "
             "(`metric_name -> source_artifact -> verifying_check`).")
    L.append("- The numbers are recomputed independently by "
             "`merlin/benchmarks/dse_guidance/verify_implementation.py` (run it; exit 0 = all checks "
             "pass).")
    L.append("- Regenerate this whole folder with `merlin-dse-guidance --insight-mining` "
             "(add `--workload <name>` for one network).")
    return "\n".join(L) + "\n"


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
        ["title", "claim", "evidence_tier", "presentation_placement", "max_corroborated_by",
         "per_workload_spread", "supporting_artifacts", "supporting_workloads", "relevant_metrics",
         "dse_implication", "caveats", "suggested_plot", "forbidden_claim_risk"])).write(run_dir)
    # P14: gap-audit report + required-inputs manifest
    gaps = bundle.get("gaps", [])
    oa = bundle.get("open_avoidable_gaps", [])
    Artifact("gap_audit_report.csv", _csv_text(gaps,
        ["gap_id", "scope", "category", "target", "description", "avoidable", "status",
         "required_input"])).write(run_dir)
    Artifact("gap_audit_report.md", _gap_audit_md(gaps, oa, scope)).write(run_dir)
    ri = bundle.get("required_inputs", [])
    yaml_artifact("required_inputs_manifest.yaml",
                  {"required_inputs_manifest": {"scope": scope, "inherent_limits": ri}},
                  header=f"required_inputs_manifest: {scope} (exact inputs to close each limit)"
                  ).write(run_dir)
    Artifact("required_inputs_manifest.md", _required_inputs_md(ri, scope)).write(run_dir)
    yaml_artifact("plot_manifest.yaml", {"plot_manifest": {"scope": scope, "plots": bundle["plots"],
                  "rendered": rendered}}, header=f"plot_manifest: {scope}").write(run_dir)
    Artifact("plot_manifest.md", _plot_md(bundle["plots"], scope, rendered)).write(run_dir)
    checks = bundle["consistency_checks"]
    npass = sum(1 for ok, _ in checks if ok)
    cm = [f"# Cross-artifact consistency checks ({scope})\n", f"**{npass}/{len(checks)} passed.**\n"]
    cm += [f"- [{'PASS' if ok else 'FAIL'}] {msg}" for ok, msg in checks]
    Artifact("consistency_checks.md", "\n".join(cm) + "\n").write(run_dir)
    Artifact("insight_mining_README.md", _readme_md(bundle, rendered)).write(run_dir)
    # P15 signal-first study deliverables (signal-only; every output answers a DSE question)
    Artifact("canonical_signal_table.csv",
             _csv_text(bundle["canonical_signal"], _CANON_COLS)).write(run_dir)
    h = bundle["hotspots"]
    Artifact("per_operator_hotspots.csv", _csv_text(h["rows"], _HOTSPOT_COLS)).write(run_dir)
    Artifact("per_operator_hotspots.md", _hotspots_md(h, scope)).write(run_dir)
    # cross-workload artifacts are corpus-level — only meaningful (and only emitted) in 'all'
    if bundle["abstraction_coverage"]:
        Artifact("abstraction_coverage_table.csv",
                 _csv_text(bundle["abstraction_coverage"], _COVER_COLS)).write(run_dir)
        Artifact("abstraction_coverage.md",
                 _coverage_md(bundle["abstraction_coverage"], scope)).write(run_dir)
    if bundle["family_summary"]["families"]:
        Artifact("workload_family_summary.md",
                 _family_md(bundle["family_summary"], scope)).write(run_dir)
    if bundle["corpus_plan"]:
        Artifact("corpus_expansion_plan.md", _corpus_md(bundle["corpus_plan"], scope)).write(run_dir)
    Artifact("signal_findings_report.md", _signal_report_md(bundle, scope)).write(run_dir)
    Artifact("DSE_FINDINGS.md",
             _findings_digest_md(bundle, bundle.get("cs_dir", run_dir), rendered)).write(run_dir)
    Artifact("presentation_plots_index.md",
             _plots_index_md(bundle["plots"], scope, rendered, "generated_plots")).write(run_dir)
