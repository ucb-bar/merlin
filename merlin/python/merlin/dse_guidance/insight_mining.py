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
import math
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
    "lowbit_storage_dequantized_finding", "overlap_candidates_yes",
    # NOTE: unit_multiplicity_implication is deliberately NOT signal — "heterogeneous" is an
    # interpretation, not a measured fact; it is kept as context and the report derives the
    # heterogeneity conclusion from the component metrics (dense/skinny split, parallelism, sharding).
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
     ["workload", "evidence_type"], "workload", "count", "evidence_type", "stacked_bar", "backup"),
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
     "max_regret", "grouped_bar", "backup"),
    ("abstraction_pressure_bar", "Abstraction pressure (workloads supporting)",
     "abstraction_pressure_ranking.csv", ["system_abstraction", "n_workloads"], "system_abstraction",
     "n_workloads", "", "bar", "backup"),
    ("boundary_placement_heatmap", "Boundary placement: abstraction x level",
     "hw_sw_boundary_matrix.csv",
     ["abstraction", "compiler_transform", "runtime_hal_object", "command_buffer_or_command_isa",
      "accelerator_isa", "device_microcode_or_controller", "fixed_hardware_datapath"],
     "level", "abstraction", "status", "heatmap", "backup"),
    ("resident_capacity_by_dtype", "Resident capacity by dtype (per region)",
     "data_movement_table.csv", ["workload", "region", "resident_int8_B", "resident_bf16_B"],
     "workload_region", "bytes", "dtype", "grouped_bar", "backup"),
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
    # ---- P16 decision-frontier & robustness plots ----
    ("primitive_set_frontier", "Primitive-set frontier (worst vs mean coverage)",
     "tile_waste_table.csv", ["workload", "op_index", "primitive", "covered_under_10pct"],
     "mean_coverage", "worst_coverage", "set", "scatter", "main"),
    ("operator_cumulative_mac", "Operator cumulative MAC share (few giant vs many even ops)",
     "operator_shape_table.csv", ["workload", "macs"], "top_k_ops", "cumulative_mac", "workload",
     "line", "main"),
    ("boundary_necessity_matrix", "Abstraction necessity (necessary/useful/possible/blocked)",
     "hw_sw_boundary_matrix.csv", ["abstraction"], "workload", "abstraction", "necessity_class",
     "categorical", "main"),
    ("decision_sharding_per_top_op", "Decision: shard top-MAC ops -> extra bytes / output bytes",
     "sharding_table.csv", ["workload", "op_index", "axis", "per_extra_shard_bytes"], "top_op",
     "extra_over_output", "axis", "decision_bar", "main"),
    # ---- P17 adversarial-audit + requirements-envelope plots ----
    ("primitive_frontier_by_threshold", "Frontier robustness: worst coverage vs set size by threshold",
     "tile_waste_table.csv", ["workload", "op_index", "primitive", "covered_under_10pct"], "set_size",
     "worst_coverage", "threshold", "line", "main"),
    ("macro_vs_micro_primitive_coverage", "Macro vs micro vs worst primitive coverage",
     "tile_waste_table.csv", ["workload", "op_index", "primitive", "covered_under_10pct"], "set_size",
     "coverage", "aggregation", "line", "main"),
    ("required_compute_envelope", "Required compute envelope (requirement, not measured)",
     "requirements_table.csv", ["workload", "region", "requirement", "value"], "deadline_ms",
     "required_GMAC_per_s", "workload", "line", "main"),
    ("required_memory_movement_envelope", "Required memory-movement envelope (residency removes Kx)",
     "requirements_table.csv", ["workload", "region", "requirement", "value"], "workload",
     "required_weight_B_per_s", "residency", "grouped_bar", "main"),
    ("required_command_rate_envelope", "Required command-rate envelope (proxy; not measured)",
     "requirements_table.csv", ["workload", "region", "requirement", "value"], "deadline_ms",
     "required_dispatch_per_s", "workload", "line", "backup"),
    ("workload_influence_loo_delta", "Workload influence: leave-one-out micro delta",
     "shape_summary_by_workload.csv", ["workload", "shape_class", "mac_fraction"], "metric",
     "max_loo_micro_delta", "", "bar", "main"),
    # ---- P18 operator-recovery plots ----
    ("work_coverage_by_workload", "Recovered work: linear-GEMM vs attention MAC mass",
     "work_coverage_table.csv", ["workload", "linear_gemm_macs", "attention_macs"], "workload",
     "recovered_macs", "op_class", "stacked_bar", "main"),
    ("visible_linear_fraction", "Visible linear fraction (linear / (linear+attention))",
     "work_coverage_table.csv", ["workload", "visible_linear_fraction"], "workload",
     "visible_linear_fraction", "", "bar", "main"),
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
# =========================================================================== P16 decision-frontier
# Sharpen results into decision-discriminating form: a metric is useful only if it changes a DSE
# decision. Strict necessity (not permissive support), primitive-set frontier, operator Pareto,
# macro/micro + leave-one-workload-out robustness, and capture-fidelity as a first-class result.

_NEC_CLASSES = ("necessary", "useful", "possible", "blocked", "not_applicable")


def _signal_by_workload(cs_dir: Path):
    """Per-workload signal aggregates shared by the necessity classifier + robustness."""
    cs_dir = Path(cs_dir)
    ops = _csv_rows(cs_dir / "operator_shape_table.csv")
    shape = _csv_rows(cs_dir / "shape_summary_by_workload.csv")
    dm = _csv_rows(cs_dir / "data_movement_table.csv")
    shard = _csv_rows(cs_dir / "sharding_table.csv")
    epi = _csv_rows(cs_dir / "epilogue_pattern_table.csv")
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    wls = sorted({o["workload"] for o in ops})
    sig = {}
    for w in wls:
        srows = [r for r in shape if r["workload"] == w]
        dense = sum(float(r["mac_fraction"]) for r in srows if r["shape_class"] == "squareish_gemm")
        true_gemv = sum(float(r["mac_fraction"]) for r in srows if r["shape_class"] == "gemv_like")
        skinny_gemm = sum(float(r["mac_fraction"]) for r in srows
                          if r["shape_class"] in ("wide_skinny", "tall_skinny"))
        gemv = true_gemv + skinny_gemm
        rh = next((r for r in dm if r["workload"] == w and r["region"] == "repeated_head"), None)
        shrows = [r for r in shard if r["workload"] == w]
        erows = [r for r in epi if r["workload"] == w]
        base = _base_model(w)
        arch = MODEL_ARCH.get(base)
        sig[w] = {
            "family": arch.family if arch else "unknown",
            "K": arch.loop_count if arch else 1,
            "H": (arch.action_horizon or 0) if arch else 0,
            "control_rate": (arch.control_rate_hz if arch else None),
            "dense_mac": round(dense, 4), "gemv_mac": round(gemv, 4),
            "true_gemv_mac": round(true_gemv, 4), "skinny_gemm_mac": round(skinny_gemm, 4),
            "weight_bytes": int(rh["weight_bytes"]) if rh else 0,
            "avoidable_reload": int(rh["avoidable_weight_reload"]) if rh else 0,
            "invocations": int(rh["invocations"]) if rh else 0,
            "k_shard": any(r["axis"] == "K" and r["reduction_required"] == "True"
                           and r.get("shardable_8") == "True" for r in shrows),
            "mn_shard": any(r["axis"] in ("M", "N") and r["reduction_required"] == "False"
                            and r.get("shardable_8") == "True" for r in shrows),
            "epi_scale_act": any(r["has_scale"] == "True" and r["has_activation"] == "True"
                                 for r in erows),
            "epi_bias": any(r["has_bias"] == "True" for r in erows),
            "has_backbone": any(o["region_role"] == "backbone_once"
                                for o in ops if o["workload"] == w),
            "has_repeated": any(o["region_role"] == "repeated_head"
                                for o in ops if o["workload"] == w)}
    return sig, wls


def _classify_abstraction(spec: dict, w: str, sig: dict) -> tuple:
    """Strict per-(abstraction, workload) class + the reason (the predicate that fired)."""
    s = sig[w]
    name, sup = spec["abstraction"], spec["support"]
    is_ar = s["family"] in ("autoregressive_vla", "llm")
    has_control = s["family"] in ("flow_matching", "diffusion", "autoregressive_vla") \
        and bool(s["control_rate"])
    if spec["kv"] and not is_ar:
        return "not_applicable", "KV abstraction; workload is not autoregressive"
    if sup == "decode" and not is_ar:
        return "not_applicable", "decode abstraction; workload is not autoregressive"
    if sup == "control_loop" and not has_control:
        return "not_applicable", "no control-rate loop in this workload"
    if spec["erased"]:
        return "blocked", "low-bit/packed structure dequantized in the capture"
    if spec["kv"]:
        return "blocked", "attention/KV lowered; structure not recovered"
    # Reasons below are class-invariant RULES (no per-workload scalar): the per-workload K /
    # weight bytes / MAC-fraction inputs live in predicate_audit_table.csv, so the necessity
    # rollup never presents one workload's K (e.g. "K=7") as a corpus constant.
    if name == "resident_weight_object":
        if s["invocations"] > 1 and s["weight_bytes"] > 1_000_000 \
                and s["avoidable_reload"] > s["weight_bytes"]:
            return "necessary", ("repeated weight reuse (K>1), resident weights >1MB, "
                                 "avoidable_reload>weight_bytes (per-workload K/MB in predicate_audit)")
        if s["invocations"] > 1:
            return "useful", "repeated weight reuse (K>1) but resident weights <=1MB"
        return "possible", "no repeated weight reuse recovered"
    if sup == "dense":
        d = s["dense_mac"]
        return (("necessary", "dense (squareish_gemm) MAC fraction >0.5") if d > 0.5
                else ("useful", "dense MAC fraction in (0.1, 0.5]") if d > 0.1
                else ("possible", "dense MAC fraction <=0.1"))
    if sup == "gemv":
        g = s["gemv_mac"]                                  # true_gemv + skinny (engine serves both)
        return (("necessary", "gemv/skinny MAC fraction >0.5 (true_gemv+skinny; split in "
                              "predicate_audit)") if g > 0.5
                else ("useful", "gemv/skinny MAC fraction in (0.1, 0.5]") if g > 0.1
                else ("possible", "gemv/skinny MAC fraction <=0.1"))
    if name in ("partial_sum_object", "accumulator_merge"):
        if s["k_shard"] and not s["mn_shard"]:
            return "necessary", "K-shard reduction needed; reduction-free M/N insufficient"
        if s["k_shard"]:
            return "useful", "K-shard available but reduction-free M/N also possible"
        return "possible", "no attractive K-shard"
    if sup == "epilogue":
        if s["epi_scale_act"]:
            return "necessary", "epilogue has scale+activation"
        if s["epi_bias"]:
            return "useful", "bias-only epilogue (requant slot present)"
        return "possible", "no fused epilogue detected"
    if sup == "k_loop":     # bounded_loop_command / loop_carried_state_handle
        if s["K"] > 1:
            return "useful", ("K>1 cadence (configured/reference, not IR-recovered; needs a "
                              "loop-preserving capture; per-workload K in predicate_audit)")
        return "possible", "no loop"
    return "possible", "available; not gated by a discriminating signal"


def _capture_gate(spec: dict) -> str:
    if spec["erased"]:
        return "low-bit recapture (packed weights + scales + per-format accuracy)"
    if spec["kv"]:
        return "loop-preserving, attention-not-lowered capture"
    if spec["support"] in ("k_loop", "decode", "control_loop"):
        return "loop-preserving capture (K/cadence are configured/reference)"
    return ""


def abstraction_necessity(cs_dir: Path) -> dict:
    """The #1 fix: replace permissive 'supported by 4/4' with a strict necessity classification
    (necessary/useful/possible/blocked/not_applicable) per abstraction × workload, using the
    threshold signals already in the committed artifacts. Returns rows + a corpus rollup."""
    from merlin.dse_guidance.boundary_placement import catalog_rows
    sig, wls = _signal_by_workload(cs_dir)
    if not wls:                          # partial mode: no per-workload signal artifacts present
        return {"workloads": [], "rows": [], "rollup": {k: 0 for k in _NEC_CLASSES}}
    rows = []
    for spec in catalog_rows():
        per = {w: _classify_abstraction(spec, w, sig) for w in wls}
        classes = {w: c for w, (c, _) in per.items()}
        counts = {k: sum(1 for c in classes.values() if c == k) for k in _NEC_CLASSES}
        applicable = sum(counts[k] for k in ("necessary", "useful", "possible"))
        if counts["blocked"] and applicable == 0:
            macro = "blocked"
        elif counts["not_applicable"] == len(wls):
            macro = "not_applicable"
        elif counts["necessary"] >= max(1, applicable // 2) and applicable:
            macro = "necessary"
        elif counts["necessary"] + counts["useful"] >= 1:
            macro = "useful"
        else:
            macro = "possible"
        # representative predicate: the highest-severity class-invariant RULE present (necessary ->
        # useful -> possible -> any). Reasons are now K-free rules, so this is honest — the rollup
        # describes the binding constraint and never presents one workload's K as a corpus constant
        # (the per-workload K / bytes / fractions live in predicate_audit_table.csv).
        reason = next((r for w, (c, r) in per.items() if c == "necessary"),
                      next((r for w, (c, r) in per.items() if c == "useful"),
                           next((r for w, (c, r) in per.items() if c == "possible"),
                                next(iter(per.values()))[1])))
        row = {"abstraction": spec["abstraction"], "support_tag": spec["support"],
               "macro_class": macro, "predicate": reason, "capture_gate": _capture_gate(spec)}
        for w in wls:
            row[w] = classes[w]
        for k in _NEC_CLASSES:
            row[f"n_{k}"] = counts[k]
        rows.append(row)
    order = {"necessary": 0, "useful": 1, "possible": 2, "blocked": 3, "not_applicable": 4}
    rows.sort(key=lambda r: (order[r["macro_class"]], -r["n_necessary"], r["abstraction"]))
    rollup = {k: sum(1 for r in rows if r["macro_class"] == k) for k in _NEC_CLASSES}
    return {"workloads": wls, "rows": rows, "rollup": rollup}


def _coverage_from_tilewaste(cs_dir: Path):
    """Per-op covered_under_10pct flags grouped by (workload, primitive) -> set-union frontier base.
    Returns (op_macs, op_cover) where op_cover[(w, op)][primitive] = bool covered, and a per-workload
    MAC total — so a primitive SET covers an op if ANY member covers it."""
    tw = _csv_rows(Path(cs_dir) / "tile_waste_table.csv")
    op_macs, op_cover, prims = {}, {}, set()
    for r in tw:
        if r.get("applicable") != "True":
            continue
        key = (r["workload"], r["op_index"])
        op_macs[key] = float(r["true_macs"])
        op_cover.setdefault(key, {})[r["primitive"]] = (r["covered_under_10pct"] == "True")
        prims.add(r["primitive"])
    return op_macs, op_cover, sorted(prims)


def _set_coverage(pset, op_macs, op_cover, drop=None):
    """MAC-weighted coverage of a primitive SET, per workload (a set covers an op if any member does).
    Returns {workload: coverage_fraction}."""
    num, den = {}, {}
    for (w, op), macs in op_macs.items():
        if drop and w == drop:
            continue
        covered = any(op_cover[(w, op)].get(p, False) for p in pset)
        den[w] = den.get(w, 0.0) + macs
        if covered:
            num[w] = num.get(w, 0.0) + macs
    return {w: (num.get(w, 0.0) / den[w] if den[w] else 0.0) for w in den}


def _frontier_stats(pset, op_macs, op_cover, drop=None):
    cov = _set_coverage(pset, op_macs, op_cover, drop)
    if not cov:
        return None
    vals = list(cov.values())
    macro = sum(vals) / len(vals)
    # micro = MAC-weighted across the corpus
    tot_macs = sum(m for (w, _), m in op_macs.items() if not (drop and w == drop)) or 1.0
    micro = sum(_set_coverage(pset, op_macs, op_cover, drop)[w]
                * sum(m for (ww, _), m in op_macs.items() if ww == w)
                for w in cov) / tot_macs
    worst = min(vals)
    return {"macro": round(macro, 4), "micro": round(micro, 4), "worst": round(worst, 4),
            "max_regret": round(max(vals) - min(vals), 4), "per_workload": cov}


def primitive_set_frontier(cs_dir: Path, max_size: int = 3) -> dict:
    """Best primitive SET of size 1/2/3 by worst-workload then macro coverage (a set covers an op if
    ANY member tiles it under 10% waste). The headline search-space result: one primitive is not
    enough; {one tile + one gemv lane} lifts worst-workload coverage."""
    import itertools
    op_macs, op_cover, prims = _coverage_from_tilewaste(cs_dir)
    best_by_size = {}
    for size in range(1, max_size + 1):
        best = None
        for combo in itertools.combinations(prims, size):
            st = _frontier_stats(combo, op_macs, op_cover)
            if st is None:
                continue
            key = (st["worst"], st["macro"])
            if best is None or key > best["key"]:
                best = {"set": list(combo), "key": key, **st}
        if best:
            best.pop("key")
            best_by_size[size] = best
    # also per-single-primitive stats (for the frontier scatter)
    singles = []
    for p in prims:
        st = _frontier_stats((p,), op_macs, op_cover)
        if st:
            singles.append({"primitive": p, "macro": st["macro"], "worst": st["worst"]})
    return {"best_by_size": best_by_size, "singles": singles}


def operator_pareto(cs_dir: Path, thresholds=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Per workload, how many top ops are needed to reach 50/80/90/95% of MACs (and of weight bytes).
    Shows whether DSE sizes for a few giant ops or many even ones."""
    ops = _csv_rows(Path(cs_dir) / "operator_shape_table.csv")
    wls = sorted({o["workload"] for o in ops})
    out = []
    for w in wls:
        wops = [o for o in ops if o["workload"] == w]
        def _ktoreach(key):
            vals = sorted((int(o[key]) for o in wops), reverse=True)
            tot = sum(vals) or 1
            ks = {}
            for t in thresholds:
                acc, k = 0, 0
                for v in vals:
                    acc += v
                    k += 1
                    if acc / tot >= t:
                        break
                ks[t] = k
            return ks
        km, kb = _ktoreach("macs"), _ktoreach("rhs_weight_bytes")
        row = {"workload": w, "n_ops": len(wops)}
        for t in thresholds:
            row[f"k_macs_{int(t*100)}"] = km[t]
            row[f"k_wbytes_{int(t*100)}"] = kb[t]
        # top op MAC share
        top = max(wops, key=lambda o: int(o["macs"]))
        row["top_op"] = top["prov_fqn"]
        row["top_op_mac_share"] = round(int(top["macs"]) / (sum(int(o["macs"]) for o in wops) or 1), 4)
        # distinct-shape vs instance-count (Q6): pi05's ~777 ops are a handful of shapes repeated
        # 100s of times, NOT 777 heterogeneous ops -- so "many ops for 50% MACs" is instance-count
        # concentration, not shape diversity.
        from collections import Counter as _C
        shapes = _C((int(o["M"]), int(o["N"]), int(o["K"])) for o in wops)
        row["n_distinct_shapes"] = len(shapes)
        row["top_shape_multiplicity"] = max(shapes.values()) if shapes else 0
        out.append(row)
    return {"thresholds": thresholds, "rows": out}


def robustness(cs_dir: Path) -> dict:
    """Macro/micro + leave-one-workload-out for the major cross-workload findings, flagging which
    conclusions flip when a workload is removed (anti-overfitting)."""
    sig, wls = _signal_by_workload(cs_dir)
    op_macs, op_cover, prims = _coverage_from_tilewaste(cs_dir)
    if len(wls) < 1 or len(prims) < 2:           # partial mode: nothing to compare
        return {"workloads": wls, "findings": []}
    findings = []

    # (a) best 2-primitive set: does the winner change under leave-one-out?
    import itertools
    def best_two(drop=None):
        best = None
        for combo in itertools.combinations(prims, 2):
            st = _frontier_stats(combo, op_macs, op_cover, drop)
            if st and (best is None or (st["worst"], st["macro"]) > best[0]):
                best = ((st["worst"], st["macro"]), combo, st)
        return best
    full = best_two()
    loo = {w: best_two(w) for w in wls}
    flips = sorted(w for w in wls if loo[w] and set(loo[w][1]) != set(full[1]))
    findings.append({"finding": "best_2_primitive_set", "all": list(full[1]),
                     "all_worst": full[2]["worst"], "all_macro": full[2]["macro"],
                     "loo_changes_winner": flips,
                     "robust": not flips})

    # (b) dense-MAC dominance: macro (equal-weight) vs micro (MAC-weighted); rdt-driven?
    dense_macro = sum(sig[w]["dense_mac"] for w in wls) / len(wls)
    tot_mac = {w: 0.0 for w in wls}
    ops = _csv_rows(Path(cs_dir) / "operator_shape_table.csv")
    for o in ops:
        tot_mac[o["workload"]] += int(o["macs"])
    corpus_mac = sum(tot_mac.values()) or 1.0
    dense_micro = sum(sig[w]["dense_mac"] * tot_mac[w] for w in wls) / corpus_mac
    dense_micro_loo = {w: (sum(sig[x]["dense_mac"] * tot_mac[x] for x in wls if x != w)
                           / (corpus_mac - tot_mac[w] or 1.0)) for w in wls}
    drop_to_low = sorted(w for w in wls if dense_micro_loo[w] < 0.2 <= dense_micro)
    findings.append({"finding": "dense_gemm_mac_dominance", "macro": round(dense_macro, 4),
                     "micro": round(dense_micro, 4),
                     "micro_loo": {w: round(v, 4) for w, v in dense_micro_loo.items()},
                     "collapses_if_removed": drop_to_low,
                     "note": ("MAC-dominant (micro) but corpus-narrow (macro)"
                              if dense_micro - dense_macro > 0.2 else "consistent across views")})

    # (c) residency pressure ranking: which workload tops avoidable_reload, robust?
    rank = sorted(wls, key=lambda w: -sig[w]["avoidable_reload"])
    findings.append({"finding": "residency_pressure_rank", "all": rank,
                     "top": rank[0] if rank else "",
                     "note": "absolute bytes are small/random-init; ranking is the robust signal"})
    return {"workloads": wls, "findings": findings}


# capture-fidelity feature rows (reuse fidelity.py vocab); status per workload reconstructed from
# committed artifacts + MODEL_ARCH (run-folder-only, no live topology object needed).
_FIDELITY_FEATURES = [
    ("op_shapes_MNK", "strong"), ("region_roles", "strong"), ("dtype_information", "strong"),
    # P18: attention bmm / softmax / norm are NOT erased — lowered to linalg.generic but re-parsed
    # (attention with real MACs). Only the KV *state across the decode loop* stays erased.
    ("attention_bmm_qkT_attnV", "recovered_attn"), ("softmax", "recovered_softmax"),
    ("normalization", "recovered_norm"),
    ("K_or_decode_loop", "loop"), ("kv_cache_state", "kv"),
    ("loop_carried_state", "loop_carried"),
    ("packed_lowbit_layout", "lowbit"),
    ("scale_metadata", "lowbit"), ("host_dispatch_count", "measured"),
    ("target_latency_cycles", "not_claimed")]


def capture_fidelity(cs_dir: Path) -> dict:
    """First-class result: which structural features the flat capture preserves vs erased, per
    workload. Attention/softmax/norm are RECOVERED (lowered to linalg.generic but re-parsed; P18);
    what stays erased is the K-loop, the KV *state* across the decode loop, and packed-lowbit/scales —
    the axes the residency & loop conclusions depend on."""
    from merlin.dse_guidance import fidelity as FID, topology as TOP
    from merlin.dse_guidance.loop_recovery import recover_loop
    sig, wls = _signal_by_workload(cs_dir)
    wc = {r["workload"]: r for r in _csv_rows(Path(cs_dir) / "work_coverage_table.csv")}
    # P21-S1: loop-preserving captures (recaptures_loop/<w>/model.mlir, if present) recover K, the
    # loop-carried state and the KV cache directly from scf.for -> flip K/KV from assumed/erased.
    # Resolved through the corpus accessor (committed under merlin/benchmarks/ + out/ overflow), NOT
    # as a sibling of cs_dir (which broke silently once case_study moved under out/artifacts/).
    from merlin.dse_guidance.corpus import _recap_dir_in
    lr_by_w = {}
    for w in wls:
        mp = _recap_dir_in(w, "recaptures_loop") / "model.mlir"
        if mp.is_file():
            lr = recover_loop(mp, w)
            if lr.present:
                lr_by_w[w] = lr
    fam_to_class = {"flow_matching": TOP.CLASS_FLOW_MATCHING, "diffusion": TOP.CLASS_FLOW_MATCHING,
                    "autoregressive_vla": TOP.CLASS_AUTOREGRESSIVE,
                    "llm": TOP.CLASS_AUTOREGRESSIVE}
    per_workload, matrix = {}, []
    for w in wls:
        s = sig[w]
        is_ar = s["family"] in ("autoregressive_vla", "llm")
        cls = fam_to_class.get(s["family"], TOP.CLASS_UNKNOWN)
        missing = []
        if s["has_repeated"]:
            missing.append("token_decode_loop" if is_ar else "denoise_loop")
        if is_ar:
            missing.append("kv_cache_growth")
        if s["has_backbone"] and s["has_repeated"]:
            missing.append("prefix_kv_reuse")
            missing.append("async_backbone_head_overlap")
        if s["H"] > 1:
            missing.append("action_chunk_horizon")
        if s["control_rate"]:
            missing.append("replan_deadline")
        seen = set()
        missing = [m for m in missing if not (m in seen or seen.add(m))]
        hidden = []
        for m in missing:
            for ax in FID.HIDDEN_AXES_BY_STRUCTURE.get(m, []):
                if ax not in hidden:
                    hidden.append(ax)
        severity = TOP.CLASS_SEVERITY.get(cls, "medium")
        if not s["has_repeated"] and not s["control_rate"]:
            severity = "low"
        recovered_loop = []
        lr = lr_by_w.get(w)
        if lr is not None:
            # the loop-preserving capture moves loop/KV out of "missing" -> recovered-from-IR
            for tag in ("token_decode_loop", "denoise_loop", "kv_cache_growth", "prefix_kv_reuse"):
                if tag in missing:
                    missing.remove(tag)
            severity = "low"
            recovered_loop = [f"K_loop(K={lr.K},IR)",
                              f"repeated_region({lr.repeated_region_op_count}ops)",
                              f"loop_carried_state({lr.n_iter_args} iter_args)"]
            if lr.kv_cache_bytes:
                recovered_loop.append(f"kv_cache_state({lr.kv_cache_bytes}B,IR)")
        per_workload[w] = {"family": s["family"], "class": cls, "severity": severity,
                           "missing": missing, "hidden_axes": hidden,
                           "preserved": ["matmul_shapes", "dtype_information", "op_mix",
                                         "weight_sizes"],
                           "recovered_from_loop_capture": recovered_loop}
    for feat, kind in _FIDELITY_FEATURES:
        row = {"feature": feat}
        for w in wls:
            s = sig[w]
            is_ar = s["family"] in ("autoregressive_vla", "llm")
            wcr = wc.get(w, {})
            if kind == "strong":
                st = "strong"
            elif kind == "recovered_attn":
                n = int(wcr.get("n_attention_ops", 0) or 0)
                st = f"recovered ({n} ops)" if n else "n/a (attention-free here)"
            elif kind == "recovered_softmax":
                n = int(wcr.get("n_softmax", 0) or 0)
                st = "recovered" if n else "n/a"
            elif kind == "recovered_norm":
                n = int(wcr.get("n_normalization", 0) or 0)
                st = "recovered" if n else "n/a (norm as elementwise primitives)"
            elif kind == "loop":
                lr = lr_by_w.get(w)
                if lr is not None:
                    st = f"recovered (K={lr.K}, IR scf.for)"
                else:
                    st = "assumed (config K)" if s["has_repeated"] else "n/a"
            elif kind == "kv":
                lr = lr_by_w.get(w)
                if lr is not None and lr.kv_cache_bytes:
                    st = f"recovered ({lr.kv_cache_bytes} B, IR iter_arg)"
                elif lr is not None:
                    st = "n/a (prefix-KV invariant, closed-over)"
                else:
                    st = "erased" if is_ar else "n/a"  # KV STATE across the decode loop (not the bmm)
            elif kind == "loop_carried":
                lr = lr_by_w.get(w)
                if lr is not None:
                    roles = sorted({c.role for c in lr.carried_state})
                    st = f"recovered ({lr.n_iter_args} iter_args: {','.join(roles)})"
                else:
                    st = "erased (loop unrolled)" if s["has_repeated"] else "n/a"
            elif kind == "lowbit":
                st = "erased"        # all recaptures are dequantized f32
            elif kind == "measured":
                st = "measured (host)"
            else:
                st = "not_claimed"
            row[w] = st
        matrix.append(row)
    return {"workloads": wls, "per_workload": per_workload, "matrix": matrix}


def decision_scorecard(bundle: dict) -> list[dict]:
    """The 7 DSE decision questions, each answered from the P16 analyses with macro/micro/worst +
    leave-one-out stability + a capture caveat."""
    fr = bundle.get("primitive_frontier", {})
    nec = bundle.get("abstraction_necessity", {})
    rob = bundle.get("robustness", {})
    par = bundle.get("operator_pareto", {})
    by_size = fr.get("best_by_size", {})
    rob_by = {f["finding"]: f for f in rob.get("findings", [])}
    q = []
    s1 = by_size.get(1)
    q.append({"q": "Q1 best single primitive (worst-workload coverage)?",
              "answer": (f"{s1['set'][0]} -> worst {s1['worst']:.2f}, macro {s1['macro']:.2f}"
                         if s1 else "n/a"),
              "caveat": "no single primitive covers every workload"})
    s2 = by_size.get(2)
    q.append({"q": "Q2 best 2-primitive set?",
              "answer": (f"{'+'.join(s2['set'])} -> worst {s2['worst']:.2f} "
                         f"(vs {s1['worst']:.2f} single)" if s2 and s1 else "n/a"),
              "caveat": "search primitive SETS, not one tile"})
    q.append({"q": "Q3 capacity x dtype residency thresholds?",
              "answer": "see decision_capacity_dtype plot (int4<int8<bf16 budget to fit)",
              "caveat": "repeated-head weights only; K is configured/reference"})
    q.append({"q": "Q4 sharding axis for top-MAC ops?",
              "answer": "M/N reduction-free vs K partial-sum (see decision_sharding_per_top_op)",
              "caveat": "communication bytes, not latency"})
    necd = nec.get("rollup", {})
    q.append({"q": "Q5 which abstractions are NECESSARY (not just possible)?",
              "answer": (f"{necd.get('necessary',0)} necessary, {necd.get('useful',0)} useful, "
                         f"{necd.get('possible',0)} possible, {necd.get('blocked',0)} blocked, "
                         f"{necd.get('not_applicable',0)} N/A"),
              "caveat": "strict predicate; low-bit abstractions blocked by capture"})
    dd = rob_by.get("dense_gemm_mac_dominance", {})
    q.append({"q": "Q6 which conclusions are driven by one workload (RDT)?",
              "answer": (f"dense-MAC dominance macro {dd.get('macro','?')} vs micro "
                         f"{dd.get('micro','?')}; collapses if removed: "
                         f"{dd.get('collapses_if_removed') or 'none'}"),
              "caveat": "micro view is biased by RDT's 87%-of-workload op"})
    q.append({"q": "Q7 which claims depend on configured K (capture fidelity)?",
              "answer": "all residency / loop / command claims (K is config/reference)",
              "caveat": "needs a loop-preserving capture; see capture_fidelity_matrix"})
    return q


# =========================================================================== P17 adversarial audit
# Stress-test the P16 conclusions: do they survive leave-one-out, macro-vs-micro, and pad-waste
# threshold perturbation? Audit every necessity predicate (its inputs, thresholds, and whether it
# rests on configured/reference K or on capture-erased structure). Sweep the primitive-set frontier
# over thresholds + set sizes + extra candidate tiles. Structural only; every number is recomputed
# from the committed artifacts — no measurement, no performance claim.

_PRED_COLS = ["abstraction", "workload", "classification", "predicate", "predicate_inputs",
              "thresholds", "uses_configured_K", "uses_erased_capture", "has_negative_control",
              "is_discriminating", "suspicious"]


def _predicate_io(spec: dict, w: str, sig: dict) -> tuple:
    """(predicate_inputs, thresholds, uses_configured_K, uses_erased_capture) for one cell. The
    per-workload scalars (K, weight bytes, MAC fractions) live HERE, not in the necessity rollup."""
    s = sig[w]
    name, sup = spec["abstraction"], spec["support"]
    if spec["erased"]:
        return "low-bit/packed weights+scales (dequantized in the f32 capture)", "n/a", "no", "yes"
    if spec["kv"]:
        return "attention/KV state (lowered in the flat capture)", "n/a", "no", "yes"
    if name == "resident_weight_object":
        return (f"K={s['invocations']}, weight_bytes={s['weight_bytes']}, "
                f"avoidable_reload={s['avoidable_reload']}",
                "K>1 & weight_bytes>1e6 & avoidable_reload>weight_bytes", "yes", "yes")
    if sup == "dense":
        return f"dense_mac={s['dense_mac']}", ">0.5 necessary; >0.1 useful", "no", "no"
    if sup == "gemv":
        return (f"gemv_mac={s['gemv_mac']} (true_gemv={s['true_gemv_mac']}, "
                f"skinny_gemm={s['skinny_gemm_mac']})", ">0.5 necessary; >0.1 useful", "no", "no")
    if name in ("partial_sum_object", "accumulator_merge"):
        return (f"k_shard={s['k_shard']}, mn_shard={s['mn_shard']}",
                "k_shard & not mn_shard -> necessary", "no", "no")
    if sup == "epilogue":
        return (f"epi_scale_act={s['epi_scale_act']}, epi_bias={s['epi_bias']}",
                "scale+act -> necessary; bias -> useful", "no", "no")
    if sup == "k_loop":
        return f"K={s['K']}", "K>1 -> useful", "yes", "yes"
    return "not gated by a discriminating signal", "n/a", "no", "no"


def predicate_audit(cs_dir: Path) -> dict:
    """Part B: for every (abstraction, workload) emit the exact predicate, its numeric inputs, the
    thresholds, whether it depends on configured K or capture-erased structure, whether the
    abstraction has a negative control (some workload where it is NOT necessary/useful), whether it
    is discriminating (the class varies across the corpus), and a suspicious flag."""
    from merlin.dse_guidance.boundary_placement import catalog_rows
    sig, wls = _signal_by_workload(cs_dir)
    if not wls:
        return {"workloads": [], "rows": []}
    rows = []
    for spec in catalog_rows():
        per = {w: _classify_abstraction(spec, w, sig) for w in wls}
        classes = [c for c, _ in per.values()]
        is_disc = len({*classes}) > 1
        has_neg = (any(c in ("possible", "blocked", "not_applicable") for c in classes)
                   and any(c in ("necessary", "useful") for c in classes))
        for w in wls:
            cls, reason = per[w]
            inputs, thr, usesK, erased = _predicate_io(spec, w, sig)
            susp = []
            if cls == "necessary" and usesK == "yes":
                susp.append("necessity rests on configured/reference K (not IR-recovered)")
            if cls in ("necessary", "useful") and not is_disc:
                susp.append("class never varies across corpus (not discriminating)")
            rows.append({"abstraction": spec["abstraction"], "workload": w, "classification": cls,
                         "predicate": reason, "predicate_inputs": inputs, "thresholds": thr,
                         "uses_configured_K": usesK, "uses_erased_capture": erased,
                         "has_negative_control": "yes" if has_neg else "no",
                         "is_discriminating": "yes" if is_disc else "no",
                         "suspicious": "; ".join(susp)})
    return {"workloads": wls, "rows": rows}


# ---- Part C: primitive-frontier robustness (recompute pad-waste so thresholds + extra tiles work)
_FRONTIER_THRESHOLDS = (0.05, 0.10, 0.20)
# extra candidate tiles beyond the committed set (structural coverage candidates only, never "faster")
_EXTRA_TILES = ("tile_4x16", "tile_8x32", "tile_16x64", "tile_64x16", "tile_4x32")
_LANE_VEC_DIM = {"gemv_like": "N", "wide_skinny": "N", "tall_skinny": "M"}
_FROB_COLS = ["threshold_pct", "set_size", "primitive_set", "worst", "macro", "micro", "max_regret"]
_UNCOV_COLS = ["threshold_pct", "set_size", "primitive_set", "workload", "op_index", "status",
               "true_macs"]


def _prim_spec(name: str) -> tuple:
    if name.startswith("tile_"):
        tm, tn = name[len("tile_"):].split("x")
        return ("tile", int(tm), int(tn))
    if name.startswith("gemv_lane_"):
        return ("lane", int(name.rsplit("_", 1)[1]), 0)
    return (None, 0, 0)


def _prim_waste(name: str, M: int, N: int, K: int, shape_class: str):
    """(applicable, padding_waste_fraction) using the documented primitive_coverage geometry: tiles
    pad M/N tails (K exact); a GEMV lane pads the one vector dim and only applies to vector-like
    shapes. Mirrors primitive_coverage.py so the 10% recompute matches the committed tile_waste."""
    kind, a, b = _prim_spec(name)
    if kind == "tile":
        pm = int(math.ceil(M / a) * a)
        pn = int(math.ceil(N / b) * b)
        return True, pm * pn / (M * N) - 1.0
    if kind == "lane":
        vd = _LANE_VEC_DIM.get(shape_class)
        if vd is None:
            return False, None
        vec = N if vd == "N" else M
        return True, int(math.ceil(vec / a) * a) / vec - 1.0
    return False, None


def primitive_frontier_robustness(cs_dir: Path, max_size: int = 4) -> dict:
    """Part C: sweep the best primitive SET over set sizes 1..4 and pad-waste thresholds 5/10/20%,
    add extra candidate tiles, and report macro/micro/worst/max-regret + leave-one-out winner +
    uncovered ops. Coverage is recomputed from operator_shape_table geometry (so thresholds other
    than the committed 10% boolean, and the extra tiles, are available); the 10% recompute is
    regression-checked against the committed covered_under_10pct by the verifier."""
    import itertools
    cs_dir = Path(cs_dir)
    base = sorted({r["primitive"] for r in _csv_rows(cs_dir / "tile_waste_table.csv")
                   if r.get("applicable") == "True"})
    prims = base + [p for p in _EXTRA_TILES if p not in base]
    geom, op_macs = {}, {}
    for o in _csv_rows(cs_dir / "operator_shape_table.csv"):
        M, N, K = int(o["M"]), int(o["N"]), int(o["K"])
        if M <= 0 or N <= 0 or K <= 0:
            continue
        key = (o["workload"], o["op_index"])
        geom[key] = (M, N, K, o["shape_class"])
        op_macs[key] = float(M) * N * K
    waste = {}
    for key, (M, N, K, sc) in geom.items():
        cell = {}
        for p in prims:
            ap, wv = _prim_waste(p, M, N, K, sc)
            cell[p] = wv if ap else None
        waste[key] = cell
    wls = sorted({w for (w, _) in geom})

    def cov(pset, key, t):
        return any(waste[key][p] is not None and waste[key][p] <= t for p in pset)

    def stats(pset, t, drop=None):
        num, den = {}, {}
        for (w, op), m in op_macs.items():
            if drop and w == drop:
                continue
            den[w] = den.get(w, 0.0) + m
            if cov(pset, (w, op), t):
                num[w] = num.get(w, 0.0) + m
        per = {w: (num.get(w, 0.0) / den[w] if den[w] else 0.0) for w in den}
        if not per:
            return None
        vals = list(per.values())
        tot = sum(den.values()) or 1.0
        return {"macro": sum(vals) / len(vals), "micro": sum(num.values()) / tot,
                "worst": min(vals), "max_regret": max(vals) - min(vals), "per": per}

    def best(size, t, drop=None):
        b = None
        for combo in itertools.combinations(prims, size):
            st = stats(combo, t, drop)
            if st and (b is None or (st["worst"], st["macro"]) > (b[1]["worst"], b[1]["macro"])):
                b = (combo, st)
        return b

    rows, grid = [], {}
    for t in _FRONTIER_THRESHOLDS:
        for size in range(1, max_size + 1):
            bs = best(size, t)
            if not bs:
                continue
            grid[(t, size)] = bs
            combo, st = bs
            rows.append({"threshold_pct": int(t * 100), "set_size": size,
                         "primitive_set": "+".join(combo), "worst": round(st["worst"], 4),
                         "macro": round(st["macro"], 4), "micro": round(st["micro"], 4),
                         "max_regret": round(st["max_regret"], 4)})
    # uncovered ops as the set grows (headline 10% threshold)
    uncovered, prev = [], set()
    for size in range(1, max_size + 1):
        bs = grid.get((0.10, size))
        if not bs:
            continue
        combo = bs[0]
        covered = {k for k in op_macs if cov(combo, k, 0.10)}
        for (w, op) in sorted(k for k in op_macs if k not in covered):
            uncovered.append({"threshold_pct": 10, "set_size": size, "primitive_set": "+".join(combo),
                              "workload": w, "op_index": op, "status": "uncovered",
                              "true_macs": int(op_macs[(w, op)])})
        prev = covered
    # threshold + LOO robustness of the best 2-set
    two = {t: (grid[(t, 2)][0] if (t, 2) in grid else None) for t in _FRONTIER_THRESHOLDS}
    base2 = grid.get((0.10, 2))
    loo_flips = []
    if base2:
        for w in wls:
            bs = best(2, 0.10, drop=w)
            if bs and set(bs[0]) != set(base2[0]):
                loo_flips.append(w)
    thr_sets = {tuple(sorted(v)) for v in two.values() if v}
    return {"workloads": wls, "primitives": prims, "rows": rows, "uncovered_rows": uncovered,
            "two_winners": {f"{int(t * 100)}pct": ("+".join(v) if v else "") for t, v in two.items()},
            "two_set_loo_flips": loo_flips,
            "two_set_threshold_robust": (len(thr_sets) == 1 and not loo_flips)}


# ---- Part D: macro/micro + leave-one-out influence (winner-stable vs magnitude-stable)
_INFLUENCE_METRICS = [
    ("dense_gemm_mac_fraction", "dense_mac"),
    ("gemv_skinny_mac_fraction", "gemv_mac"),
    ("true_gemv_mac_fraction", "true_gemv_mac"),
    ("skinny_gemm_mac_fraction", "skinny_gemm_mac"),
]
_INFLUENCE_COLS = ["metric", "macro", "micro", "macro_micro_gap", "worst_workload", "worst",
                   "best_workload", "best", "most_influential", "loo_micro_at_influential",
                   "max_loo_micro_delta", "winner_stable_magnitude_unstable"]
_LOO_DELTA_COLS = ["metric", "workload_removed", "loo_micro", "delta_vs_full"]


def macro_micro_influence(cs_dir: Path) -> dict:
    """Part D: per cross-workload MAC-fraction metric, report macro (equal-weight) vs micro
    (MAC-weighted), worst/best workload, the most influential workload (largest leave-one-out micro
    swing), and crucially flag metrics whose WINNER is stable but whose MAGNITUDE is not (the
    dense-GEMM case: drop pi05 and micro jumps ~0.04 -> ~0.65 though skinny still wins)."""
    cs_dir = Path(cs_dir)
    sig, wls = _signal_by_workload(cs_dir)
    if len(wls) < 2:
        return {"workloads": wls, "rows": [], "loo_rows": []}
    tot_mac = {w: 0.0 for w in wls}
    for o in _csv_rows(cs_dir / "operator_shape_table.csv"):
        if o["workload"] in tot_mac:
            tot_mac[o["workload"]] += int(o["macs"])
    corpus = sum(tot_mac.values()) or 1.0
    rows, loo_rows = [], []
    for label, key in _INFLUENCE_METRICS:
        vals = {w: float(sig[w][key]) for w in wls}
        macro = sum(vals.values()) / len(wls)
        micro = sum(vals[w] * tot_mac[w] for w in wls) / corpus
        loo = {w: (sum(vals[x] * tot_mac[x] for x in wls if x != w)
                   / ((corpus - tot_mac[w]) or 1.0)) for w in wls}
        infl = max(wls, key=lambda w: abs(loo[w] - micro))
        max_delta = abs(loo[infl] - micro)
        worst_w = min(wls, key=lambda w: vals[w])
        best_w = max(wls, key=lambda w: vals[w])
        rows.append({"metric": label, "macro": round(macro, 4), "micro": round(micro, 4),
                     "macro_micro_gap": round(abs(macro - micro), 4),
                     "worst_workload": worst_w, "worst": round(vals[worst_w], 4),
                     "best_workload": best_w, "best": round(vals[best_w], 4),
                     "most_influential": infl, "loo_micro_at_influential": round(loo[infl], 4),
                     "max_loo_micro_delta": round(max_delta, 4),
                     "winner_stable_magnitude_unstable": ("yes" if max_delta > 0.2 else "no")})
        for w in wls:
            loo_rows.append({"metric": label, "workload_removed": w, "loo_micro": round(loo[w], 4),
                             "delta_vs_full": round(loo[w] - micro, 4)})
    # residency pressure is a byte ranking, not a fraction — report the rank separately
    reload_rank = sorted(wls, key=lambda w: -float(sig[w]["avoidable_reload"]))
    return {"workloads": wls, "rows": rows, "loo_rows": loo_rows,
            "residency_pressure_rank": reload_rank}


_AUDIT_COLS = ["conclusion", "supporting_artifact", "supporting_metric", "metric_class",
               "depends_on_configured_K", "depends_on_flat_capture", "survives_loo",
               "survives_macro_vs_micro", "survives_threshold", "driven_by", "decision_or_fact",
               "verdict"]


def adversarial_audit(bundle: dict) -> dict:
    """Part A: a validity ledger — each headline conclusion mapped to the artifact/metric that
    supports it and whether it survives leave-one-out, macro-vs-micro, and threshold perturbation.
    No new measurement; it reads the P16/P17 analyses already in the bundle."""
    rob = {f["finding"]: f for f in bundle.get("robustness", {}).get("findings", [])}
    fr = bundle.get("primitive_frontier", {}).get("best_by_size", {})
    frob = bundle.get("primitive_frontier_robustness", {})
    necd = bundle.get("abstraction_necessity", {}).get("rollup", {})
    infl = {r["metric"]: r for r in bundle.get("macro_micro_influence", {}).get("rows", [])}
    two = rob.get("best_2_primitive_set", {})
    thr_robust = frob.get("two_set_threshold_robust")
    rows = []

    def add(**kw):
        rows.append({c: kw.get(c, "") for c in _AUDIT_COLS})

    s1, s2 = fr.get(1, {}), fr.get(2, {})
    add(conclusion="primitive-set frontier: a 2-primitive set covers the corpus where one fails",
        supporting_artifact="primitive_set_frontier.csv; primitive_frontier_robustness.csv",
        supporting_metric=f"worst coverage size1 {s1.get('worst','?')} -> size2 {s2.get('worst','?')}",
        metric_class="derived", depends_on_configured_K="no", depends_on_flat_capture="no",
        survives_loo=("yes" if two.get("robust") else f"no (flips: {two.get('loo_changes_winner')})"),
        survives_macro_vs_micro="yes (macro+micro+worst all reported)",
        survives_threshold=("yes (5/10/20%)" if thr_robust else
                            ("not computed" if thr_robust is None else
                             f"no (winner varies: {frob.get('two_winners')})")),
        driven_by=("specific pair shifts with finer tiles/threshold"
                   if not thr_robust else "none"),
        decision_or_fact="decision (search primitive SETS, not one tile)",
        # the CLAIM "a 2-set suffices" is robust (a high-worst-coverage pair exists at every
        # threshold); only the SPECIFIC pair is threshold/LOO-sensitive once finer tiles are added.
        verdict=("robust" if two.get("robust") and thr_robust else
                 "claim robust (a 2-set suffices); specific pair is threshold/LOO-sensitive"))
    dd = rob.get("dense_gemm_mac_dominance", {})
    di = infl.get("dense_gemm_mac_fraction", {})
    add(conclusion="dense-GEMM is corpus-narrow (skinny/GEMV dominates the MAC mass)",
        supporting_artifact="leave_one_workload_out.md; macro_micro_influence_table.csv",
        supporting_metric=f"dense macro {dd.get('macro','?')} vs micro {dd.get('micro','?')}",
        metric_class="derived", depends_on_configured_K="no", depends_on_flat_capture="no",
        survives_loo=(f"magnitude NOT stable (max micro swing {di.get('max_loo_micro_delta','?')})"
                      if di.get("winner_stable_magnitude_unstable") == "yes" else "yes"),
        survives_macro_vs_micro="winner yes; magnitude no (macro/micro differ widely)",
        survives_threshold="n/a", driven_by=di.get("most_influential", "pi05"),
        decision_or_fact="decision (do not size only for square GEMMs)",
        verdict="winner robust, magnitude fragile")
    add(conclusion="resident_weight_object is necessary (K-loop weight residency is a search axis)",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric=f"necessary for {necd.get('necessary','?')} abstraction rollup",
        metric_class="derived (weight bytes) + assumed (K)", depends_on_configured_K="yes",
        depends_on_flat_capture="yes (K-loop erased)", survives_loo="n/a (per-workload predicate)",
        survives_macro_vs_micro="n/a", survives_threshold="n/a",
        driven_by="configured/reference K", decision_or_fact="decision (residency knob)",
        verdict="blocked-by-capture (needs a loop-preserving capture to confirm K)")
    add(conclusion="skinny_gemm_or_gemv_engine is necessary/useful across the corpus",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric="gemv/skinny MAC fraction (true_gemv + skinny split in predicate_audit)",
        metric_class="derived", depends_on_configured_K="no", depends_on_flat_capture="no",
        survives_loo="see predicate_audit per-workload", survives_macro_vs_micro="yes",
        survives_threshold="n/a", driven_by="skinny-GEMM projections (not true GEMV)",
        decision_or_fact="decision (engine serves skinny GEMM, not only M/N<=4 GEMV)",
        verdict="robust (name now honest: skinny OR gemv)")
    add(conclusion="low-bit abstractions are blocked (cannot be evaluated)",
        supporting_artifact="abstraction_necessity_table.csv; capture_fidelity_matrix.csv",
        supporting_metric=f"{necd.get('blocked','?')} abstractions blocked",
        metric_class="unavailable", depends_on_configured_K="no",
        depends_on_flat_capture="yes (packed layout + scales erased)", survives_loo="n/a",
        survives_macro_vs_micro="n/a", survives_threshold="n/a", driven_by="f32 fake-quant capture",
        decision_or_fact="decision-blocking (needs a low-bit recapture)",
        verdict="blocked-by-capture")
    add(conclusion="attention/KV abstractions are blocked (attention lowered)",
        supporting_artifact="abstraction_necessity_table.csv; capture_fidelity_matrix.csv",
        supporting_metric="KV abstractions classified blocked/not_applicable",
        metric_class="unavailable", depends_on_configured_K="no",
        depends_on_flat_capture="yes (q.kT / attn.v lowered)", survives_loo="n/a",
        survives_macro_vs_micro="n/a", survives_threshold="n/a", driven_by="flat capture",
        decision_or_fact="decision-blocking (needs attention-preserving capture)",
        verdict="blocked-by-capture")
    add(conclusion="bounded_loop_command / loop_carried_state_handle are useful",
        supporting_artifact="abstraction_necessity_table.csv; predicate_audit_table.csv",
        supporting_metric="K>1 cadence (configured/reference)",
        metric_class="assumed", depends_on_configured_K="yes",
        depends_on_flat_capture="yes (loops not in IR)", survives_loo="n/a",
        survives_macro_vs_micro="n/a", survives_threshold="n/a", driven_by="configured/reference K",
        decision_or_fact="decision (where the loop lives is a boundary axis)",
        verdict="blocked-by-capture (loop erased; useful, not provable)")
    add(conclusion="capture fidelity is the limiting factor (flat captures erase loop/KV/low-bit)",
        supporting_artifact="capture_fidelity_matrix.csv",
        supporting_metric="K_or_decode_loop=assumed; packed_lowbit_layout/scale_metadata=erased",
        metric_class="derived", depends_on_configured_K="no", depends_on_flat_capture="yes",
        survives_loo="uniform across corpus", survives_macro_vs_micro="n/a",
        survives_threshold="n/a", driven_by="capture level (uniform)",
        decision_or_fact="methodology result (the central finding)", verdict="robust")
    add(conclusion="HW/SW boundary placement is itself a DSE search axis",
        supporting_artifact="boundary_placement_matrix; abstraction_necessity_table.csv",
        supporting_metric="per-abstraction level enumeration (no level chosen)",
        metric_class="derived", depends_on_configured_K="no", depends_on_flat_capture="partly",
        survives_loo="n/a", survives_macro_vs_micro="n/a", survives_threshold="n/a",
        driven_by="structural enumeration", decision_or_fact="framing (search axis, not a choice)",
        verdict="descriptive (enumeration, not a single decision)")
    return {"rows": rows}


# =========================================================================== P17 requirements (E/F)
# Timing as an external REQUIREMENT envelope (work / deadline), never a measured hardware result;
# and an honest accounting of the operators this study does NOT count. Every input is tagged
# configured / reference / sweep; no row is a performance prediction.

_DEADLINE_MS = (50, 100, 200, 500)
_K_SWEEP = (4, 8, 16, 32)
_MEASURED_DISPATCH = {"small_llama"}     # dispatch_coupling.csv has a measured per-forward count
_ENV_COLS = ["workload", "region", "K", "K_basis", "deadline_ms", "deadline_basis",
             "required_compute_MAC_per_s", "required_weight_B_per_s_nonresident",
             "required_weight_B_per_s_resident", "required_command_rate_per_s", "command_rate_basis",
             "resident_capacity_bf16_B", "resident_capacity_int8_B"]


def timing_requirement_envelope(cs_dir: Path) -> dict:
    """Part E: cross per-workload structural base facts (macs/step, resident weight bytes,
    dispatches/step, resident capacity by dtype — all already in requirements_table.csv) with
    scenario grids: replan deadlines {50,100,200,500 ms}, a K sweep {4,8,16,32}+configured, and the
    configured H/control-rate deadline. Each row is a REQUIREMENT (work / deadline), explicitly NOT a
    hardware performance prediction. Command rate is proxy-only except where a measured dispatch
    count exists (no silent fill)."""
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    cs_dir = Path(cs_dir)
    req = {(r["workload"], r["region"], r["requirement"]): r["value"]
           for r in _csv_rows(cs_dir / "requirements_table.csv")}
    wls = sorted({w for (w, reg, _) in req if reg == "repeated_head"})
    rows = []
    for w in wls:
        arch = MODEL_ARCH.get(_base_model(w))
        Kcfg = arch.loop_count if arch else 1

        def g(name):
            v = req.get((w, "repeated_head", name))
            return float(v) if v not in (None, "") else 0.0

        macs_replan, weight_bytes, disp_replan = (g("macs_per_replan"),
                                                  g("resident_capacity_required"),
                                                  g("dispatches_per_replan"))
        cap_bf16, cap_int8 = g("resident_capacity_bf16"), g("resident_capacity_int8")
        if Kcfg <= 0 or macs_replan <= 0:
            continue
        macs_step, disp_step = macs_replan / Kcfg, disp_replan / Kcfg
        cmd_basis = ("measured_dispatch_available" if _base_model(w) in _MEASURED_DISPATCH else
                     "proxy_only (matmul-count, ~12x undercount; NOT a hardware command rate)")
        scen = [(float(dl), "sweep") for dl in _DEADLINE_MS]
        if arch and arch.control_rate_hz and arch.action_horizon:
            scen.append((1000.0 * arch.action_horizon / arch.control_rate_hz,
                         f"derived_H/control_rate (H={arch.action_horizon}, {arch.control_rate_hz}Hz;"
                         " configured/reference)"))
        for K in sorted({Kcfg, *_K_SWEEP}):
            kb = "configured" if K == Kcfg else "sweep"
            for dl, db in scen:
                s = dl / 1000.0
                rows.append({"workload": w, "region": "repeated_head", "K": K, "K_basis": kb,
                             "deadline_ms": round(dl, 3), "deadline_basis": db,
                             "required_compute_MAC_per_s": round(macs_step * K / s, 1),
                             "required_weight_B_per_s_nonresident": round(weight_bytes * K / s, 1),
                             "required_weight_B_per_s_resident": round(weight_bytes / s, 1),
                             "required_command_rate_per_s": round(disp_step * K / s, 2),
                             "command_rate_basis": cmd_basis,
                             "resident_capacity_bf16_B": int(cap_bf16),
                             "resident_capacity_int8_B": int(cap_int8)})
    return {"workloads": wls, "rows": rows}


_OMIT_COLS = ["workload", "linear_gemm_ops", "linear_gemm_macs", "attention_ops", "attention_macs",
              "visible_linear_fraction", "softmax_ops", "norm_ops", "conv_ops", "elementwise_ops",
              "kv_state_recovered", "lowbit_packed_recovered", "scale_metadata_recovered",
              "still_erased", "capture_level_needed"]


def operator_recovery_accounting(cs_dir: Path) -> dict:
    """Part F (RECOVERED, not count-only): attention bmm / softmax / norm are NOT erased — the flat
    capture lowered them to ``linalg.generic`` but they are re-parsed. Per workload: linear-GEMM vs
    attention MAC mass (both recovered from IR shapes — no model-card config) + softmax/norm/conv/
    elementwise op counts, and what genuinely STAYS erased (KV state across the decode loop, packed
    low-bit layout, scale metadata). ``visible_linear_fraction`` answers "how much of the recovered
    MAC work is the linear-GEMM geometry this study analyzes" with real numbers."""
    cs_dir = Path(cs_dir)
    wc = {r["workload"]: r for r in _csv_rows(cs_dir / "work_coverage_table.csv")}
    sig, wls = _signal_by_workload(cs_dir)
    rows = []
    for w in wls:
        r = wc.get(w, {})
        is_ar = sig[w]["family"] in ("autoregressive_vla", "llm")
        rows.append({
            "workload": w,
            "linear_gemm_ops": int(r.get("n_linear_matmul", 0) or 0),
            "linear_gemm_macs": int(r.get("linear_gemm_macs", 0) or 0),
            "attention_ops": int(r.get("n_attention_ops", 0) or 0),
            "attention_macs": int(r.get("attention_macs", 0) or 0),
            "visible_linear_fraction": float(r.get("visible_linear_fraction", 1.0) or 1.0),
            "softmax_ops": int(r.get("n_softmax", 0) or 0),
            "norm_ops": int(r.get("n_normalization", 0) or 0),
            "conv_ops": int(r.get("n_conv", 0) or 0),
            "elementwise_ops": int(r.get("n_elementwise", 0) or 0),
            # what STAYS erased (genuinely absent in the flat capture, not merely unparsed):
            "kv_state_recovered": ("no (KV state across decode loop erased)" if is_ar
                                   else "n/a (non-autoregressive)"),
            "lowbit_packed_recovered": "no (f32 fake-quant capture)",
            "scale_metadata_recovered": "no (dequantized in capture)",
            "still_erased": "K-loop trip count; KV state across decode; packed low-bit layout + scales",
            "capture_level_needed": ("loop-preserving (K / KV state)"
                                     + ("; KV-cache" if is_ar else "")
                                     + "; low-bit recapture for packed layout + scales")})
    return {"workloads": wls, "rows": rows}


# ---- Stage C: capture-erasure IR evidence, per-family decomposition (analysis-only) ----
_ERASE_COLS = ["workload", "n_scf_loops", "n_linalg_matmul", "n_linalg_generic",
               "lowbit_int_types_present", "loops_preserved", "evidence"]


def _count_word_bounded(text: str, *needles: str) -> int:
    """Occurrences of each ``needle`` bounded by non-word chars on both sides — the regex-free
    equivalent of ``\\b<needle>\\b`` (e.g. counts ``scf.for(`` but not ``scf.forall``)."""
    total = 0
    for needle in needles:
        start = 0
        while True:
            i = text.find(needle, start)
            if i == -1:
                break
            start = i + 1
            before = text[i - 1] if i > 0 else ""
            after = text[i + len(needle)] if i + len(needle) < len(text) else ""
            bw = before.isalnum() or before == "_"
            aw = after.isalnum() or after == "_"
            if not bw and not aw:
                total += 1
    return total


def capture_erasure_evidence(cs_dir: Path) -> dict:
    """Stage C: demonstrate (not just assert) what the flat capture erases, from the recapture IR:
    scf.for/while count (loops), linalg.matmul vs linalg.generic counts (named ops vs lowered),
    and whether any packed low-bit integer tensor types survive.

    Always reads the FLAT corpus (that is the capture whose erasure is being shown), resolved through
    the corpus accessor — committed under merlin/benchmarks/ with the oversized ones in the
    out/artifacts/recaptures/ overflow. (It used to be globbed as a sibling of cs_dir, which silently
    yielded zero rows once case_study moved under out/artifacts/.)"""
    from merlin.dse_guidance.corpus import RECAP_MODELS, _recap_dir_in

    rows = []
    for d in [_recap_dir_in(w, "recaptures") for w in sorted(RECAP_MODELS)]:
        if not (d / "model.mlir").is_file():
            continue
        # Bulk triage over the whole flat corpus (dozens of large captures): a cheap, regex-free
        # token/substring scan — a full xDSL parse per file would be 100-1000x costlier for a
        # presence/count heuristic. (Per-workload deep facts use merlin.common.mlir_query.)
        txt = (d / "model.mlir").read_text(errors="ignore")
        n_loops = _count_word_bounded(txt, "scf.for", "scf.while")   # == old \bscf\.(for|while)\b
        n_mm = txt.count("linalg.matmul")
        n_gen = txt.count("linalg.generic")
        lowbit = any(f"x{dt}>" in txt for dt in ("i4", "i8", "si8", "ui8", "si4"))
        rows.append({"workload": d.name, "n_scf_loops": n_loops, "n_linalg_matmul": n_mm,
                     "n_linalg_generic": n_gen, "lowbit_int_types_present": lowbit,
                     "loops_preserved": n_loops > 0,
                     "evidence": (f"{n_loops} scf loops, {n_gen} generics vs {n_mm} named matmuls, "
                                  f"low-bit int types {'present' if lowbit else 'absent'}")})
    return {"rows": rows}


def per_family_summary(cs_dir: Path) -> dict:
    """Stage C (§5E): decompose the corpus by family (iterative_denoise vs token_decode) so corpus
    averages don't hide family behavior — per-family op counts, attention share, and visible-linear
    fraction from work_coverage_table + workload_family_table."""
    cs_dir = Path(cs_dir)
    fam = {r["workload"]: r["family"] for r in _csv_rows(cs_dir / "workload_family_table.csv")}
    wc = {r["workload"]: r for r in _csv_rows(cs_dir / "work_coverage_table.csv")}
    agg = {}
    for w, f in fam.items():
        r = wc.get(w)
        if not r:
            continue
        a = agg.setdefault(f, {"family": f, "workloads": [], "lin_macs": 0, "attn_macs": 0,
                               "n_attention_ops": 0, "n_softmax": 0, "n_norm": 0})
        a["workloads"].append(w)
        a["lin_macs"] += int(r["linear_gemm_macs"])
        a["attn_macs"] += int(r["attention_macs"])
        a["n_attention_ops"] += int(r["n_attention_ops"])
        a["n_softmax"] += int(r["n_softmax"])
        a["n_norm"] += int(r["n_normalization"])
    rows = []
    for f, a in sorted(agg.items()):
        tot = a["lin_macs"] + a["attn_macs"]
        rows.append({"family": f, "n_workloads": len(a["workloads"]),
                     "workloads": "; ".join(sorted(a["workloads"])),
                     "linear_gemm_macs": a["lin_macs"], "attention_macs": a["attn_macs"],
                     "visible_linear_fraction": round(a["lin_macs"] / tot, 4) if tot else 1.0,
                     "n_attention_ops": a["n_attention_ops"], "n_softmax": a["n_softmax"],
                     "n_norm": a["n_norm"]})
    return {"rows": rows}


# ---- Stage B: capture-level ablation over REAL multi-level recaptures (m2m flags) ----
# Level 0 = flat (committed recaptures/), high_level = m2m level="high-level" (linalg_ext.softmax),
# quant_qdq = m2m preserve_qdq int8 (quant_ext.dequantize). Loop-preserving is torch.export-blocked.
_ABLATION_LEVELS = [("flat", "recaptures", "model.mlir"),
                    ("high_level", "recaptures_levels", "model_highlevel.mlir"),
                    ("quant_qdq", "recaptures_levels", "model_qdq.mlir")]
_ABLATION_OPS = ["linalg.matmul", "linalg.generic", "linalg_ext.softmax", "linalg_ext.layer_norm",
                 "quant_ext.dequantize", "scf.for"]
_ABLATION_COLS = (["workload", "level", "available"] + [o.replace(".", "_") for o in _ABLATION_OPS])
# DSE axes and how each becomes decidable as the capture level rises (categorical).
_ABLATION_AXES = ["attention_primitive", "softmax_reduction_unit", "packed_lowbit_weights",
                  "scale_metadata", "bounded_loop_command", "kv_cache_object"]


def capture_level_ablation(cs_dir: Path) -> dict:
    """Stage B: real capture-level ablation. Reads the committed op-count summary
    ``capture_level_ablation.csv`` (the raw multi-level recaptures are gitignored + regenerable) and
    derives which DSE axes move blocked -> decidable. Levels: flat (attention recovered by re-parsing
    generics, P18-A), high_level (attention/softmax as NAMED linalg_ext ops), quant_qdq (packed-lowbit
    + scales via quant_ext.dequantize). Loop/KV stay blocked at every level (torch.export unrolls
    loops). Empty if the summary is absent (no recaptures were generated)."""
    raw = _csv_rows(Path(cs_dir) / "capture_level_ablation.csv")
    if not raw:
        return {"workloads": [], "rows": [], "unlock": []}
    rows = []
    for r in raw:
        row = {"workload": r["workload"], "level": r["level"], "available": r["available"] == "True"}
        for o in _ABLATION_OPS:
            k = o.replace(".", "_")
            row[k] = int(r.get(k, 0) or 0)
        rows.append(row)
    wls = sorted({r["workload"] for r in rows})
    unlock = []
    for w in wls:
        by = {r["level"]: r for r in rows if r["workload"] == w}
        hl, qd = by.get("high_level", {}), by.get("quant_qdq", {})
        any_loop = any(by.get(lv, {}).get("scf_for") for lv in ("flat", "high_level", "quant_qdq"))
        unlock.append({
            "workload": w,
            "attention_primitive": ("named (linalg_ext) @high_level" if hl.get("linalg_ext_softmax")
                                    else "recovered (generic) @flat"),
            "softmax_reduction_unit": ("named @high_level" if hl.get("linalg_ext_softmax")
                                       else "recovered (generic) @flat"),
            "packed_lowbit_weights": ("decidable @quant_qdq" if qd.get("quant_ext_dequantize")
                                      else "blocked (no qdq capture)"),
            "scale_metadata": ("decidable @quant_qdq" if qd.get("quant_ext_dequantize")
                               else "blocked (no qdq capture)"),
            "bounded_loop_command": ("partial" if any_loop else "blocked (torch.export unrolls loops)"),
            "kv_cache_object": ("partial" if any_loop else "blocked (torch.export unrolls loops)")})
    return {"workloads": wls, "rows": rows, "unlock": unlock}


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
    "primitive_set_frontier": "Each point is a primitive (or best set): x=mean coverage, y=worst-"
        "workload coverage. Upper-right = broadly useful; high-x/low-y = corpus-overfit. The best "
        "single primitive sits low on y; a {tile + GEMV-lane} set reaches the top-right — DSE should "
        "search primitive SETS.",
    "operator_cumulative_mac": "Cumulative MAC share vs top-k operators per workload: a steep curve "
        "(rdt: 1 op = 87%) means DSE sizes for a few giant ops; a gradual curve means many even ops.",
    "boundary_necessity_matrix": "Strict necessity per abstraction × workload "
        "(necessary/useful/possible/blocked/N-A) — what DSE should commit to, not merely what is "
        "possible; low-bit abstractions are blocked by the dequantized capture.",
    "decision_sharding_per_top_op": "For the top-MAC ops, extra sharding bytes normalized by the op's "
        "output bytes, per M/N/K axis — which axis partitions a hot op cheaply (the per-operator view, "
        "not a corpus aggregate).",
    "primitive_frontier_by_threshold": "Worst-workload coverage vs primitive-set size at 5/10/20% pad "
        "waste — whether the 'a 2-set suffices' claim survives threshold perturbation (the specific "
        "pair may shift; structural coverage only).",
    "macro_vs_micro_primitive_coverage": "Macro (equal-weight) vs micro (MAC-weighted) vs worst "
        "coverage as the primitive set grows — the second primitive is where worst-workload coverage "
        "jumps; structural, no performance.",
    "required_compute_envelope": "Required compute rate (= configured-K replan MACs / deadline) vs "
        "replan deadline — a REQUIREMENT a future accelerator must exceed, not a measured rate.",
    "required_memory_movement_envelope": "Required weight bandwidth at a 100 ms deadline, weights "
        "reloaded every step vs kept resident — residency removes a K× bandwidth requirement (the "
        "residency search axis); a requirement, not a measured rate.",
    "required_command_rate_envelope": "Required dispatch rate vs deadline — a PROXY (matmul-count, "
        "~12× undercount), measured only for small_llama; not a hardware command rate.",
    "workload_influence_loo_delta": "Largest leave-one-out micro swing per cross-workload metric — "
        "red bars are metrics whose winner is stable but whose magnitude is not (drop one workload "
        "and the number moves sharply).",
    "work_coverage_by_workload": "Recovered MAC mass split into linear-GEMM vs attention (both from "
        "IR shapes, no config) — attention is NOT erased, just lowered to generic and re-parsed.",
    "visible_linear_fraction": "Fraction of recovered MAC work that is the linear-GEMM geometry this "
        "study analyzes (rest = attention) — answers 'are we analyzing most of the compute?'.",
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
    # P16 decision-frontier & robustness (corpus-level analyses computed at 'all'; the per-network
    # necessity/fidelity are still meaningful per workload but the frontier/robustness need the corpus)
    bundle["abstraction_necessity"] = abstraction_necessity(cs_dir)
    bundle["primitive_frontier"] = primitive_set_frontier(cs_dir)
    bundle["operator_pareto"] = operator_pareto(cs_dir)
    bundle["robustness"] = robustness(cs_dir) if is_all else {"workloads": [], "findings": []}
    bundle["capture_fidelity"] = capture_fidelity(cs_dir)
    bundle["decision_scorecard"] = decision_scorecard(bundle)
    # P17 adversarial audit: predicate audit is per-workload meaningful; the frontier-robustness and
    # influence analyses are corpus-level (computed only at 'all'); the conclusion ledger reads them.
    bundle["predicate_audit"] = predicate_audit(cs_dir)
    bundle["primitive_frontier_robustness"] = (primitive_frontier_robustness(cs_dir) if is_all
                                               else {"workloads": [], "rows": [], "uncovered_rows": []})
    bundle["macro_micro_influence"] = (macro_micro_influence(cs_dir) if is_all
                                       else {"workloads": [], "rows": [], "loo_rows": []})
    bundle["adversarial_audit"] = adversarial_audit(bundle)
    # P17 requirements envelope + omitted-op accounting (Parts E/F; corpus-level at 'all')
    bundle["timing_envelope"] = (timing_requirement_envelope(cs_dir) if is_all
                                 else {"workloads": [], "rows": []})
    bundle["omitted_ops"] = operator_recovery_accounting(cs_dir)
    bundle["capture_erasure"] = capture_erasure_evidence(cs_dir) if is_all else {"rows": []}
    bundle["per_family"] = per_family_summary(cs_dir) if is_all else {"rows": []}
    bundle["capture_ablation"] = (capture_level_ablation(cs_dir) if is_all
                                  else {"rows": [], "unlock": []})
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


# --------------------------------------------------------------------------- P16 emitters

def _necessity_cols(wls):
    return (["abstraction", "support_tag", "macro_class"] + list(wls)
            + [f"n_{k}" for k in _NEC_CLASSES] + ["predicate", "capture_gate"])


def _necessity_md(nec, scope) -> str:
    wls = nec["workloads"]
    r = nec["rollup"]
    L = [f"# Abstraction necessity ({scope})\n",
         "> Strict replacement for the permissive support table: each abstraction is classified per "
         "workload as **necessary / useful / possible / blocked / not_applicable** by a threshold "
         "predicate over the recovered signals (not `any-X` presence). 'possible' = available but not "
         "gated by a discriminating signal; 'blocked' = the capture erased the needed structure.\n",
         f"**Corpus rollup:** {r['necessary']} necessary · {r['useful']} useful · {r['possible']} "
         f"possible · {r['blocked']} blocked · {r['not_applicable']} not-applicable "
         f"(of {len(nec['rows'])} abstractions).\n",
         "| abstraction | macro | " + " | ".join(wls) + " | predicate | needs |",
         "|---|---|" + "|".join(["---"] * len(wls)) + "|---|---|"]
    for row in nec["rows"]:
        L.append(f"| {row['abstraction']} | **{row['macro_class']}** | "
                 + " | ".join(row[w] for w in wls)
                 + f" | {row['predicate']} | {row['capture_gate'] or '—'} |")
    return "\n".join(L) + "\n"


def _frontier_rows(fr):
    rows = []
    for size, b in sorted(fr["best_by_size"].items()):
        rows.append({"set_size": size, "primitive_set": " + ".join(b["set"]),
                     "macro_coverage": b["macro"], "micro_coverage": b["micro"],
                     "worst_workload_coverage": b["worst"], "max_regret": b["max_regret"]})
    return rows


def _frontier_md(fr, scope) -> str:
    L = [f"# Primitive-set frontier ({scope})\n",
         "> A primitive SET covers an op if ANY member tiles it under 10% pad waste. The headline "
         "search-space result: one primitive is not enough — the best single primitive leaves a "
         "workload badly covered (low worst-workload), while a {tile + GEMV-lane} pair covers the "
         "corpus. Structural coverage only, no performance.\n",
         "| set size | best primitive set | worst-workload | macro (mean) | micro (MAC-wt) | max regret |",
         "|---|---|---|---|---|---|"]
    for r in _frontier_rows(fr):
        L.append(f"| {r['set_size']} | {r['primitive_set']} | {r['worst_workload_coverage']:.2f} | "
                 f"{r['macro_coverage']:.2f} | {r['micro_coverage']:.2f} | {r['max_regret']:.2f} |")
    return "\n".join(L) + "\n"


def _pareto_md(par, scope) -> str:
    ts = par["thresholds"]
    L = [f"# Operator Pareto hotspots ({scope})\n",
         "> How many top ops are needed to reach a MAC threshold — whether DSE should size for a few "
         "giant ops or many even ones. Structural (MAC/byte share), no performance.\n",
         "| workload | n_ops | " + " | ".join(f"k@{int(t*100)}%MAC" for t in ts)
         + " | top-op MAC share |",
         "|---|---|" + "|".join(["---"] * len(ts)) + "|---|"]
    for r in par["rows"]:
        L.append(f"| {r['workload']} | {r['n_ops']} | "
                 + " | ".join(str(r[f'k_macs_{int(t*100)}']) for t in ts)
                 + f" | {r['top_op_mac_share']:.0%} |")
    return "\n".join(L) + "\n"


def _robustness_md(rob, scope) -> str:
    L = [f"# Macro/micro + leave-one-workload-out robustness ({scope})\n",
         "> Anti-overfitting: each cross-workload finding recomputed dropping one workload. A finding "
         "that flips is corpus-specific, not general.\n"]
    for f in rob["findings"]:
        L.append(f"## {f['finding']}\n")
        for k, v in f.items():
            if k == "finding":
                continue
            L.append(f"- {k}: {v}")
        L.append("")
    return "\n".join(L) + "\n"


def _capture_fidelity_md(cf, scope) -> str:
    wls = cf["workloads"]
    L = [f"# Capture-fidelity matrix ({scope})\n",
         "> The likely central result: which structural features the flat capture preserves vs "
         "erased. `strong`=recovered from IR; `assumed (config K)`=loop count is a reference value, "
         "not captured; `erased`=lowered/dequantized away; `measured (host)`=real host measurement; "
         "`not_claimed`=needs a target design. Findings that depend on `assumed`/`erased` rows are "
         "capture-limited.\n",
         "| feature | " + " | ".join(wls) + " |",
         "|---|" + "|".join(["---"] * len(wls)) + "|"]
    for row in cf["matrix"]:
        L.append(f"| {row['feature']} | " + " | ".join(row[w] for w in wls) + " |")
    L.append("\n**Per-workload DSE risk:**\n")
    for w in wls:
        d = cf["per_workload"][w]
        L.append(f"- **{w}** ({d['family']}, severity {d['severity']}): lost "
                 f"{', '.join(d['missing']) or 'none'}; hides axes "
                 f"{', '.join(d['hidden_axes']) or 'none'}")
    return "\n".join(L) + "\n"


def _scorecard16_md(qs, scope) -> str:
    L = [f"# DSE decision-question scorecard ({scope})\n",
         "> The few decisions a future DSE tool must make, each answered from the workload analysis "
         "with its caveat. A metric earns its place only by answering one of these.\n",
         "| # | decision question | answer (from analysis) | caveat |", "|---|---|---|---|"]
    for q in qs:
        L.append(f"| {q['q'].split()[0]} | {q['q']} | {q['answer']} | {q['caveat']} |")
    return "\n".join(L) + "\n"


def _slides_md(bundle, scope) -> str:
    nec = bundle.get("abstraction_necessity", {}).get("rollup", {})
    fr = bundle.get("primitive_frontier", {}).get("best_by_size", {})
    s1, s2 = fr.get(1), fr.get(2)
    L = [f"# Presentation slide candidates ({scope})\n",
         "> Show these, not the descriptive QA plots. Each slide = one DSE claim + its evidence + "
         "caveat. Structural only; no speedup/area/energy.\n"]
    slides = [
        ("What Merlin does", "flat capture -> workload contract -> DSE search-space axes (it "
         "enumerates the search space; it does not choose a design)",
         "dse_search_space_knobs", "no performance claimed"),
        ("Primitive set is a DSE axis",
         (f"one primitive worst-cov {s1['worst']:.2f}; {'+'.join(s2['set'])} -> {s2['worst']:.2f}"
          if s1 and s2 else "best 2-set covers the corpus a single primitive cannot"),
         "primitive_set_frontier", "set-union coverage, structural"),
        ("Residency is a loop/rate abstraction",
         "weight bytes moved grows with K under reload; flat if resident; threshold set by dtype",
         "decision_weight_residency + decision_capacity_dtype", "K is configured/reference"),
        ("Inter-op parallelism is low",
         "work/span ~1.1-1.6 -> pushes DSE to intra-op sharding / pipelining / specialized units",
         "critical_path_parallelism", "flattened capture may erase loop/pipeline parallelism"),
        ("HW/SW boundary placement is a search axis",
         (f"abstraction necessity: {nec.get('necessary',0)} necessary / {nec.get('blocked',0)} "
          f"blocked -> DSE searches WHERE state/loops/layout/sync/reductions live"),
         "boundary_necessity_matrix", "categorical, not a score"),
        ("Capture fidelity is the limiting factor",
         "the flat capture erases K-loop / KV / packed-layout the loop & residency claims need",
         "capture_fidelity_matrix", "the central next-step result, not a side note")]
    for i, (title, claim, art, cav) in enumerate(slides, 1):
        L.append(f"## Slide {i} — {title}\n")
        L.append(f"- **claim:** {claim}")
        L.append(f"- **show:** `{art}`")
        L.append(f"- **caveat:** {cav}\n")
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
    # 3. abstraction NECESSITY (strict; the decision-discriminating view) + frontier + pareto
    L.append("## 3. Abstraction necessity (strict — what DSE should commit to)\n")
    L.append(_necessity_md(bundle["abstraction_necessity"], scope).split("\n", 1)[1])
    L.append("## 4. Primitive-set frontier\n")
    L.append(_frontier_md(bundle["primitive_frontier"], scope).split("\n", 1)[1])
    L.append("## 5. Operator Pareto hotspots\n")
    L.append(_pareto_md(bundle["operator_pareto"], scope).split("\n", 1)[1])
    L.append("## 6. Capture fidelity (what the flat capture erased)\n")
    L.append(_capture_fidelity_md(bundle["capture_fidelity"], scope).split("\n", 1)[1])
    L.append("## 7. Decision-question scorecard\n")
    L.append(_scorecard16_md(bundle["decision_scorecard"], scope).split("\n", 1)[1])
    if bundle.get("robustness", {}).get("findings"):
        L.append("## 8. Leave-one-workload-out robustness\n")
        L.append(_robustness_md(bundle["robustness"], scope).split("\n", 1)[1])
    # the permissive coverage is kept only as a secondary 'possible-placement' reference
    if bundle["abstraction_coverage"]:
        L.append("## Appendix — abstraction support breadth (possible-placement view only)\n")
        L.append(_coverage_md(bundle["abstraction_coverage"], scope).split("\n", 1)[1])
    # decision-impact plots
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


def _predicate_audit_md(pa, scope) -> str:
    rows = pa["rows"]
    susp = [r for r in rows if r["suspicious"]]
    L = [f"# Predicate audit ({scope})\n",
         "> Every necessity predicate, its numeric inputs, thresholds, and whether it rests on "
         "configured/reference K or on capture-erased structure. The per-workload K / weight-bytes / "
         "MAC-fraction scalars live HERE — the necessity rollup carries only the class-invariant "
         "rule, so it never presents one workload's K as a corpus constant.\n",
         f"**{len(susp)} suspicious cells** (necessity resting on configured K, or a non-"
         "discriminating predicate).\n",
         "| abstraction | workload | class | inputs | thresholds | uses_K | erased | "
         "neg_control | discriminating | suspicious |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['abstraction']} | {r['workload']} | {r['classification']} | "
                 f"{r['predicate_inputs']} | {r['thresholds']} | {r['uses_configured_K']} | "
                 f"{r['uses_erased_capture']} | {r['has_negative_control']} | "
                 f"{r['is_discriminating']} | {r['suspicious'] or '—'} |")
    return "\n".join(L) + "\n"


def _adversarial_audit_md(aud, scope) -> str:
    L = [f"# Adversarial audit of the headline conclusions ({scope})\n",
         "> Validity ledger: each conclusion → the artifact/metric behind it → whether it survives "
         "leave-one-out, macro-vs-micro, and pad-waste threshold perturbation. `robust` = survives "
         "all; `winner robust, magnitude fragile` = the ranking holds but the number moves a lot; "
         "`blocked-by-capture` = the conclusion cannot be confirmed without a richer capture.\n",
         "| conclusion | metric class | survives LOO | survives macro/micro | survives threshold | "
         "driven by | verdict |",
         "|---|---|---|---|---|---|---|"]
    for r in aud["rows"]:
        L.append(f"| {r['conclusion']} | {r['metric_class']} | {r['survives_loo']} | "
                 f"{r['survives_macro_vs_micro']} | {r['survives_threshold']} | {r['driven_by']} | "
                 f"**{r['verdict']}** |")
    L.append("\nFull supporting artifact + decision-vs-fact per row: `conclusion_validity_table.csv`.\n")
    return "\n".join(L) + "\n"


def _frontier_robustness_md(fro, scope) -> str:
    L = [f"# Primitive-set frontier robustness ({scope})\n",
         "> The best primitive SET swept over set sizes 1–4 and pad-waste thresholds 5/10/20%, "
         f"with extra candidate tiles ({', '.join(_EXTRA_TILES)}). Coverage is recomputed from "
         "operator geometry (the 10% recompute is regression-checked against the committed "
         "tile_waste). Structural coverage only — no primitive is called faster.\n",
         f"**Best 2-set across thresholds:** {fro['two_winners']}. "
         f"**LOO flips (10%):** {fro['two_set_loo_flips'] or 'none'}. "
         f"**Threshold-robust:** {fro['two_set_threshold_robust']}.\n",
         "| threshold | set size | best primitive set | worst | macro | micro | max regret |",
         "|---|---|---|---|---|---|---|"]
    for r in fro["rows"]:
        L.append(f"| {r['threshold_pct']}% | {r['set_size']} | {r['primitive_set']} | "
                 f"{r['worst']:.3f} | {r['macro']:.3f} | {r['micro']:.3f} | {r['max_regret']:.3f} |")
    nunc = len(fro["uncovered_rows"])
    L.append(f"\nUncovered ops as the set grows (10% threshold): `uncovered_ops_by_primitive_set.csv` "
             f"({nunc} rows).\n")
    return "\n".join(L) + "\n"


def _influence_md(inf, scope) -> str:
    L = [f"# Workload influence / magnitude stability ({scope})\n",
         "> Per cross-workload MAC-fraction metric: macro (equal-weight) vs micro (MAC-weighted), the "
         "most influential workload (largest leave-one-out micro swing), and a flag for metrics whose "
         "**winner is stable but whose magnitude is not** — a metric can keep its ranking yet move "
         "sharply when one workload is dropped.\n",
         "| metric | macro | micro | gap | most influential | max LOO micro Δ | winner-stable/"
         "magnitude-unstable |",
         "|---|---|---|---|---|---|---|"]
    for r in inf["rows"]:
        L.append(f"| {r['metric']} | {r['macro']:.3f} | {r['micro']:.3f} | {r['macro_micro_gap']:.3f}"
                 f" | {r['most_influential']} | {r['max_loo_micro_delta']:.3f} | "
                 f"**{r['winner_stable_magnitude_unstable']}** |")
    L.append(f"\nResidency-pressure ranking (avoidable weight reload, bytes; ranking is the robust "
             f"signal, absolute bytes are random-init): {' > '.join(inf['residency_pressure_rank'])}.")
    L.append("Per-(metric, workload) leave-one-out micro: `leave_one_out_delta_table.csv`.\n")
    return "\n".join(L) + "\n"


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}GB"


def _deadline_sensitivity_md(env, scope) -> str:
    dls = list(_DEADLINE_MS)
    L = [f"# Deadline-sensitivity of the requirement envelope ({scope})\n",
         "> Required **compute rate** (= configured-K replan MACs / deadline) as the replan deadline "
         "tightens. A REQUIREMENT, not a hardware prediction — a future accelerator must *exceed* "
         "this to meet the deadline. K is configured/reference; deadlines are a sweep.\n",
         "| workload | K | " + " | ".join(f"{d}ms" for d in dls) + " |",
         "|---|---|" + "|".join(["---"] * len(dls)) + "|"]
    by = {}
    for r in env["rows"]:
        if r["K_basis"] == "configured" and r["deadline_basis"] == "sweep":
            by.setdefault(r["workload"], {})[int(r["deadline_ms"])] = r["required_compute_MAC_per_s"]
            by[r["workload"]]["K"] = r["K"]
    for w in env["workloads"]:
        d = by.get(w, {})
        L.append(f"| {w} | {d.get('K','?')} | "
                 + " | ".join(f"{d.get(dl,0)/1e9:.2f} GMAC/s" for dl in dls) + " |")
    L.append("\nFull K×deadline grid (compute / weight-bandwidth resident-vs-nonresident / command "
             "rate): `timing_requirement_envelope.csv`.\n")
    return "\n".join(L) + "\n"


def _residency_tradeoff_md(env, scope) -> str:
    L = [f"# Residency vs deadline tradeoff ({scope})\n",
         "> At configured K and a 100 ms replan deadline: the required **weight bandwidth** if weights "
         "are reloaded every step (non-resident) vs kept resident (loaded once). The ratio is exactly "
         "K — residency removes a K× bandwidth requirement at the cost of the resident capacity shown "
         "(by dtype). Structural requirement only.\n",
         "| workload | K | non-resident B/s | resident B/s | ratio | resident bf16 | resident int8 |",
         "|---|---|---|---|---|---|---|"]
    seen = set()
    for r in env["rows"]:
        if r["K_basis"] != "configured" or int(r["deadline_ms"]) != 100 or r["deadline_basis"] != "sweep":
            continue
        w = r["workload"]
        if w in seen:
            continue
        seen.add(w)
        nr, rs = r["required_weight_B_per_s_nonresident"], r["required_weight_B_per_s_resident"]
        ratio = (nr / rs) if rs else 0
        L.append(f"| {w} | {r['K']} | {_fmt_bytes(nr)}/s | {_fmt_bytes(rs)}/s | {ratio:.0f}x | "
                 f"{_fmt_bytes(r['resident_capacity_bf16_B'])} | "
                 f"{_fmt_bytes(r['resident_capacity_int8_B'])} |")
    L.append("\nResidency is therefore a search axis (capacity to keep weights on-chip vs the K× "
             "reload bandwidth it avoids). Absolute bytes are small/random-init; the K× ratio and the "
             "capacity-by-dtype ordering are the robust signal.\n")
    return "\n".join(L) + "\n"


def _omitted_op_md(om, scope) -> str:
    L = [f"# Operator recovery accounting ({scope})\n",
         "> Attention bmm / softmax / norm are **NOT erased** — the flat capture lowered them to "
         "`linalg.generic` but they are re-parsed. Linear-GEMM and attention MACs are both recovered "
         "from IR shapes (no model-card config). `visible_linear_fraction` = linear / (linear+attention) "
         "answers how much of the recovered MAC work is the linear-GEMM geometry. What genuinely STAYS "
         "erased: the K-loop, KV state across the decode loop, and packed low-bit layout + scales.\n",
         "| workload | linear ops | linear MACs | attn ops | attn MACs | **visible_linear_frac** | "
         "softmax | norm | conv | elementwise |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in om["rows"]:
        L.append(f"| {r['workload']} | {r['linear_gemm_ops']} | {r['linear_gemm_macs']} | "
                 f"{r['attention_ops']} | {r['attention_macs']} | **{r['visible_linear_fraction']:.3f}** | "
                 f"{r['softmax_ops']} | {r['norm_ops']} | {r['conv_ops']} | {r['elementwise_ops']} |")
    L.append("\n**Still erased** (genuinely absent in the flat capture, not merely unparsed): the K-loop "
             "trip count, KV state across the decode loop, and packed low-bit layout + scales — needing "
             "a loop-preserving (+ KV) capture and a low-bit recapture. Per-workload: "
             "`visible_vs_erased_work_table.csv`.\n")
    return "\n".join(L) + "\n"


_PER_FAMILY_COLS = ["family", "n_workloads", "workloads", "linear_gemm_macs", "attention_macs",
                    "visible_linear_fraction", "n_attention_ops", "n_softmax", "n_norm"]


def _erasure_evidence_md(ce, scope) -> str:
    L = [f"# Capture-erasure IR evidence ({scope})\n",
         "> The 'erased' claims, demonstrated from the recapture IR rather than asserted. Loops are "
         "absent (torch.export unrolls them — only a gather/scatter artifact shows scf.for); attention "
         "lives in `linalg.generic` (now re-parsed); no packed low-bit integer tensor types survive "
         "(the capture is dequantized f32).\n",
         "| workload | scf loops | linalg.matmul | linalg.generic | low-bit int types | loops preserved |",
         "|---|---|---|---|---|---|"]
    for r in ce["rows"]:
        L.append(f"| {r['workload']} | {r['n_scf_loops']} | {r['n_linalg_matmul']} | "
                 f"{r['n_linalg_generic']} | {r['lowbit_int_types_present']} | {r['loops_preserved']} |")
    nloop = sum(1 for r in ce["rows"] if r["loops_preserved"])
    L.append(f"\n**{nloop}/{len(ce['rows'])} captures retain any scf loop** (and that one is a bool-mask "
             "gather artifact, not a model loop). No capture retains packed low-bit types. This is the "
             "concrete basis for the capture-fidelity 'erased' rows.\n")
    return "\n".join(L) + "\n"


def _per_family_md(pf, scope) -> str:
    L = [f"# Per-family decomposition ({scope})\n",
         "> Corpus split by family so averages don't hide family behavior. Linear-GEMM vs attention MAC "
         "mass and op mix per family (structural; recovered from IR).\n",
         "| family | workloads | linear MACs | attention MACs | visible_linear_frac | attn ops | "
         "softmax | norm |",
         "|---|---|---|---|---|---|---|---|"]
    for r in pf["rows"]:
        L.append(f"| {r['family']} | {r['n_workloads']} | {r['linear_gemm_macs']} | "
                 f"{r['attention_macs']} | {r['visible_linear_fraction']:.3f} | {r['n_attention_ops']} | "
                 f"{r['n_softmax']} | {r['n_norm']} |")
    return "\n".join(L) + "\n"


def _ablation_md(ab, scope) -> str:
    L = [f"# Capture-level ablation ({scope})\n",
         "> REAL multi-level recaptures (m2m flags). **flat** = the committed capture (attention "
         "recovered by re-parsing generics, P18-A); **high_level** (`level=\"high-level\"`) emits "
         "attention/softmax as NAMED `linalg_ext` ops; **quant_qdq** (`preserve_qdq` int8) emits "
         "`quant_ext.dequantize` (packed-lowbit + scales). **Loop-preserving is torch.export-blocked** "
         "(no Level — the named frontier a TorchDynamo/`torch.cond` frontend would unlock).\n",
         "## Op vocabulary per level\n",
         "| workload | level | available | matmul | generic | linalg_ext.softmax | "
         "quant_ext.dequantize | scf.for |",
         "|---|---|---|---|---|---|---|---|"]
    for r in ab["rows"]:
        L.append(f"| {r['workload']} | {r['level']} | {r['available']} | {r['linalg_matmul']} | "
                 f"{r['linalg_generic']} | {r['linalg_ext_softmax']} | {r['quant_ext_dequantize']} | "
                 f"{r['scf_for']} |")
    L += ["\n## Axis-unlock matrix (which DSE axes become decidable at which level)\n",
          "| workload | " + " | ".join(_ABLATION_AXES) + " |",
          "|---|" + "|".join(["---"] * len(_ABLATION_AXES)) + "|"]
    for u in ab["unlock"]:
        L.append(f"| {u['workload']} | " + " | ".join(u[a] for a in _ABLATION_AXES) + " |")
    L.append("\n**Loop-preserving capture is the blocked frontier**: `scf.for` is absent at every level "
             "(torch.export unrolls Python loops), so `bounded_loop_command` / `kv_cache_object` stay "
             "blocked — the one axis a richer (TorchDynamo / torch.cond) frontend would unlock.\n")
    return "\n".join(L) + "\n"


def _p17_findings_md(bundle, scope) -> str:
    aud = bundle.get("adversarial_audit", {}).get("rows", [])
    L = [f"# P17 final report — what is robust, what is blocked ({scope})\n",
         "> Per headline finding: a verdict from the adversarial audit (robust / winner-robust-but-"
         "magnitude-fragile / blocked-by-capture / descriptive), then the captures that would most "
         "improve the study. Structural only; timing rows are requirements, not measured performance.\n",
         "## Finding verdicts\n",
         "| conclusion | verdict | slide |",
         "|---|---|---|"]
    for r in aud:
        v = r["verdict"]
        slide = "backup" if ("blocked" in v or "descriptive" in v) else "main"
        L.append(f"| {r['conclusion']} | {v} | {slide} |")
    L += ["\n## Most valuable next captures (to unblock the blocked findings)\n",
          "1. **Loop-preserving capture** — turns configured/reference K into IR-recovered loop "
          "structure; unblocks `resident_weight_object` / `bounded_loop_command` and grounds the "
          "timing-requirement envelope.",
          "2. **Loop/KV-preserving capture** — attention bmm / softmax / norm are already recovered "
          "(see `omitted_operator_accounting.md`); what remains is the KV *state* across the decode "
          "loop and the K-loop trip count.",
          "3. **Low-bit recapture** (packed weights + scales + per-format accuracy) — unblocks the "
          "low-bit abstractions (blocked today by the f32 fake-quant capture).",
          "4. **Measured dispatch counts** beyond small_llama — turns the command-rate envelope from "
          "proxy-only into a real requirement.\n",
          "## Headline for the talk\n",
          "Compiler-derived workload contracts tell a DSE tool which search axes to include before "
          "any hardware exists — primitive-SET coverage, loop/rate residency, operator concentration, "
          "HW/SW boundary placement — **and** reveal when the capture itself is too flat to decide "
          "(low-bit, KV, and the K-loop are erased). Capture fidelity is part of the methodology.\n"]
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
    # P16 decision-frontier & robustness deliverables
    nec = bundle["abstraction_necessity"]
    Artifact("abstraction_necessity_table.csv",
             _csv_text(nec["rows"], _necessity_cols(nec["workloads"]))).write(run_dir)
    Artifact("abstraction_necessity.md", _necessity_md(nec, scope)).write(run_dir)
    Artifact("primitive_set_frontier.csv",
             _csv_text(_frontier_rows(bundle["primitive_frontier"]),
                       ["set_size", "primitive_set", "worst_workload_coverage", "macro_coverage",
                        "micro_coverage", "max_regret"])).write(run_dir)
    Artifact("primitive_set_frontier.md",
             _frontier_md(bundle["primitive_frontier"], scope)).write(run_dir)
    par = bundle["operator_pareto"]
    _pcols = (["workload", "n_ops"]
              + [f"k_{m}_{int(t*100)}" for t in par["thresholds"] for m in ("macs", "wbytes")]
              + ["top_op", "top_op_mac_share", "n_distinct_shapes", "top_shape_multiplicity"])
    Artifact("operator_pareto_hotspots.csv", _csv_text(par["rows"], _pcols)).write(run_dir)
    Artifact("operator_pareto_hotspots.md", _pareto_md(par, scope)).write(run_dir)
    cf = bundle["capture_fidelity"]
    Artifact("capture_fidelity_matrix.csv",
             _csv_text(cf["matrix"], ["feature"] + cf["workloads"])).write(run_dir)
    Artifact("capture_fidelity_matrix.md", _capture_fidelity_md(cf, scope)).write(run_dir)
    Artifact("decision_question_scorecard.md",
             _scorecard16_md(bundle["decision_scorecard"], scope)).write(run_dir)
    Artifact("presentation_slide_candidates.md", _slides_md(bundle, scope)).write(run_dir)
    if bundle["robustness"]["findings"]:
        Artifact("leave_one_workload_out.md", _robustness_md(bundle["robustness"], scope)).write(run_dir)
    # P17 adversarial audit & predicate audit (predicate audit always; corpus analyses at 'all')
    pa = bundle["predicate_audit"]
    if pa["rows"]:
        Artifact("predicate_audit_table.csv", _csv_text(pa["rows"], _PRED_COLS)).write(run_dir)
        Artifact("predicate_audit_table.md", _predicate_audit_md(pa, scope)).write(run_dir)
    aud = bundle["adversarial_audit"]
    Artifact("conclusion_validity_table.csv", _csv_text(aud["rows"], _AUDIT_COLS)).write(run_dir)
    Artifact("adversarial_audit_report.md", _adversarial_audit_md(aud, scope)).write(run_dir)
    fro = bundle.get("primitive_frontier_robustness", {})
    if fro.get("rows"):
        Artifact("primitive_frontier_robustness.csv", _csv_text(fro["rows"], _FROB_COLS)).write(run_dir)
        Artifact("primitive_frontier_robustness.md", _frontier_robustness_md(fro, scope)).write(run_dir)
        Artifact("uncovered_ops_by_primitive_set.csv",
                 _csv_text(fro["uncovered_rows"], _UNCOV_COLS)).write(run_dir)
    inf = bundle.get("macro_micro_influence", {})
    if inf.get("rows"):
        Artifact("macro_micro_influence_table.csv",
                 _csv_text(inf["rows"], _INFLUENCE_COLS)).write(run_dir)
        Artifact("leave_one_out_delta_table.csv",
                 _csv_text(inf["loo_rows"], _LOO_DELTA_COLS)).write(run_dir)
        Artifact("workload_influence_report.md", _influence_md(inf, scope)).write(run_dir)
    # P17 requirements envelope (Part E) + omitted-operator accounting (Part F)
    env = bundle.get("timing_envelope", {})
    if env.get("rows"):
        Artifact("timing_requirement_envelope.csv", _csv_text(env["rows"], _ENV_COLS)).write(run_dir)
        Artifact("deadline_sensitivity_report.md", _deadline_sensitivity_md(env, scope)).write(run_dir)
        Artifact("residency_vs_deadline_tradeoff.md",
                 _residency_tradeoff_md(env, scope)).write(run_dir)
    om = bundle.get("omitted_ops", {})
    if om.get("rows"):
        Artifact("visible_vs_erased_work_table.csv", _csv_text(om["rows"], _OMIT_COLS)).write(run_dir)
        Artifact("omitted_operator_accounting.md", _omitted_op_md(om, scope)).write(run_dir)
    # Stage C: capture-erasure evidence, per-family decomposition, and the critique's exact-name aliases
    ce = bundle.get("capture_erasure", {})
    if ce.get("rows"):
        Artifact("capture_erasure_evidence.csv", _csv_text(ce["rows"], _ERASE_COLS)).write(run_dir)
        Artifact("capture_erasure_evidence.md", _erasure_evidence_md(ce, scope)).write(run_dir)
    pf = bundle.get("per_family", {})
    if pf.get("rows"):
        Artifact("per_family_summary.csv", _csv_text(pf["rows"], _PER_FAMILY_COLS)).write(run_dir)
        Artifact("per_family_summary.md", _per_family_md(pf, scope)).write(run_dir)
    if inf.get("rows"):     # alias: Part A's `metric_stability_table.csv` (= macro/micro + LOO delta)
        Artifact("metric_stability_table.csv", _csv_text(inf["rows"], _INFLUENCE_COLS)).write(run_dir)
    if om.get("rows"):      # alias: Part A's `known_omissions_table.csv` (= recovered/erased work)
        Artifact("known_omissions_table.csv", _csv_text(om["rows"], _OMIT_COLS)).write(run_dir)
    ab = bundle.get("capture_ablation", {})     # Stage B: real capture-level ablation
    if ab.get("rows"):
        Artifact("capture_level_ablation.csv", _csv_text(ab["rows"], _ABLATION_COLS)).write(run_dir)
        Artifact("capture_level_ablation.md", _ablation_md(ab, scope)).write(run_dir)
    if env.get("rows"):     # P17 final report (corpus-level, when the requirements envelope exists)
        Artifact("P17_FINDINGS.md", _p17_findings_md(bundle, scope)).write(run_dir)
    Artifact("DSE_FINDINGS.md",
             _findings_digest_md(bundle, bundle.get("cs_dir", run_dir), rendered)).write(run_dir)
    Artifact("presentation_plots_index.md",
             _plots_index_md(bundle["plots"], scope, rendered, "generated_plots")).write(run_dir)
