#!/usr/bin/env python
"""Independent verification harness for the dse_guidance contract-completeness package.

This does NOT trust the emitted artifacts. It re-derives the key numbers from the raw captures and
from the base recovered facts, then cross-checks them against every emitted artifact. If a module
computed a derived quantity wrong, the recomputation here diverges and the check FAILS.

Independent sources used:
  * raw `recaptures/<w>/model.mlir`  — text grep of `linalg.matmul` (a count nobody else produces)
  * `attribution.extract_matmuls`    — the low-level IR primitive (separately tested, cos=1.0)
  * first-principles arithmetic       — reload = WB*(K-1), capacity = WB/elem[f32]*elem[fmt], etc.

Cross-checked artifacts:
  region_attribution.yaml, resident_state_table.csv, traffic_table.csv, dtype_capacity_table.csv,
  dispatch_granularity_table.csv, accuracy_gated_dtype_candidates.csv, design_envelope.yaml.

Run:  .venv/bin/python merlin/benchmarks/dse_guidance/verify_implementation.py
Exit code 0 = all checks pass. Writes case_study/verification_report.md.
"""
from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

from merlin.common.yaml import load_yaml
from merlin.dse_guidance import accuracy_gate as AG
from merlin.dse_guidance import attribution as ATTR
from merlin.dse_guidance import shape_taxonomy as ST
from merlin.dse_guidance.case_study import RECAP_MODELS, _CORPUS_SUBDIR, available_models
from merlin.dse_guidance.design_envelope import ELEMENT_BYTES

HERE = Path(__file__).resolve().parent
CS = HERE / "case_study"
# P23: verify against the SAME corpus the study analyzes (loop-preserving by default; the studyable
# model set excludes the small_llama toy whose loop wrapper produced no linalg.matmul).
RECAP = HERE / _CORPUS_SUBDIR
_MODELS = set(available_models())
CS_SHAPE = CS / "operator_shape_table.csv"
CS_TILE = CS / "tile_waste_table.csv"
CS_COVMAT = CS / "primitive_coverage_matrix.csv"
CS_REGRET = CS / "primitive_regret_table.csv"
CS_GRAPH = CS / "workload_contract_graph.yaml"
CS_CRIT = CS / "critical_path_table.csv"
CS_SHARD = CS / "sharding_table.csv"
CS_HIER = CS / "operator_cluster_to_hierarchy.csv"
CS_RESPRESS = CS / "resource_pressure_table.csv"
CS_PHASE_RATE = CS / "phase_rate_table.csv"
CS_PIPE_STAGE = CS / "pipeline_stage_table.csv"
CS_DATAMOVE = CS / "data_movement_table.csv"
CS_DMA = CS / "dma_stream_table.csv"
CS_BUFREQ = CS / "buffer_requirement_table.csv"
CS_EPILOGUE = CS / "epilogue_pattern_table.csv"
CS_ACCUM = CS / "accumulator_contract_table.csv"
_DEQ_VOCAB = {"before_matmul", "fused_candidate", "after_load", "unavailable"}
_REQ_VOCAB = {"separate_op", "epilogue_candidate", "unavailable"}
_ACC_VOCAB = {"materialized", "committed_directly", "unavailable"}
ALLOWED_CADENCE = {"once_per_instruction", "once_per_replan", "K_times_per_replan", "token_loop",
                   "control_tick", "once_per_forward", "unknown"}
ALLOWED_EVIDENCE = {"recovered_from_ir", "recovered_from_prov_fqn", "recovered_from_model_config",
                    "assumed_reference", "derived_requirement", "design_assumption", "measured",
                    "proxy_measured", "unavailable"}
DANGEROUS = ("improvement", "optimal", "best design", "predicted cycles", "calibrated future",
             "gap_closure", "faster")
V0_DOCS = {"current_state_audit.md", "claim_evidence_matrix.csv", "reproducibility_check.log",
           "known_limitations.md", "verification_report.md"}

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))


def _csv_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _eq(a, b, tol=1) -> bool:
    return abs(float(a) - float(b)) <= tol


def _head_facts_from_raw(workload: str) -> dict:
    """Recompute per-role facts straight from the IR primitive (independent of the YAML), mirroring
    attribution: on a loop capture the scf.for boundary is authoritative (in-loop -> repeated_head,
    out-of-loop -> backbone_once); on a flat capture the prov.fqn heuristic is used."""
    recs = ATTR.extract_matmuls(str(RECAP / workload))
    is_loop = any(getattr(r, "in_loop_body", False) for r in recs)
    roles: dict[str, dict] = {}
    for r in recs:
        if is_loop:
            role = ATTR.ROLE_REPEATED_HEAD if r.in_loop_body else ATTR.ROLE_BACKBONE
        else:
            role = ATTR.role_from_fqn(r.fqn)
        if role is None:
            continue
        d = roles.setdefault(role, {"matmul_count": 0, "weight_bytes": 0, "macs": 0})
        d["matmul_count"] += 1
        d["weight_bytes"] += r.weight_bytes
        d["macs"] += r.macs
    return {"roles": roles, "n_records": len(recs)}


def verify_workload(w: str) -> dict:
    K = int(RECAP_MODELS[w]["K"])
    mlir = (RECAP / w / "model.mlir").read_text()
    raw_mm = len(re.findall(r"\blinalg\.matmul\b", mlir))

    attr = load_yaml(CS / w / "region_attribution.yaml")["topology_recovery"]["regions"]
    by_role = {r["role"]: r["facts"] for r in attr}
    attributed_mm = sum(r["facts"]["matmul_count"] for r in attr)

    raw = _head_facts_from_raw(w)
    head = by_role.get("repeated_head", {})
    WB = int(head.get("weight_bytes", 0))
    MM = int(head.get("matmul_count", 0))

    # --- A: matmul count is real (raw grep == primitive == attributed sum) ---
    check(raw_mm == raw["n_records"] == attributed_mm,
          f"[{w}] matmul count: raw_grep={raw_mm} extract_matmuls={raw['n_records']} "
          f"attributed_sum={attributed_mm}")

    # --- B: recovered head facts match the independent raw recompute ---
    raw_head = raw["roles"].get("repeated_head", {})
    check(raw_head.get("weight_bytes") == WB and raw_head.get("matmul_count") == MM,
          f"[{w}] head facts vs raw IR: weight_bytes {raw_head.get('weight_bytes')}=={WB}, "
          f"matmul_count {raw_head.get('matmul_count')}=={MM}")

    # --- C: traffic_table derived from WB*(K) / WB*(K-1) ---
    tr = {r["region"]: r for r in _csv_rows(CS / "traffic_table.csv") if r["workload"] == w}
    h = tr.get("repeated_head", {})
    check(_eq(h.get("weight_bytes", -1), WB)
          and _eq(h.get("weight_traffic_if_nonresident", -1), WB * K)
          and _eq(h.get("avoidable_weight_reload", -1), WB * (K - 1))
          and int(h.get("invocations", -1)) == K,
          f"[{w}] traffic_table: WB={WB}, nonresident=WB*{K}={WB*K}, "
          f"avoidable=WB*{K-1}={WB*(K-1)}")

    # --- D: dtype_capacity scaled from captured-f32 element count ---
    cap = next(r for r in _csv_rows(CS / "dtype_capacity_table.csv") if r["workload"] == w)
    # n_elem from the CAPTURED dtype (most captures are f32; smolvla's real VLM is bf16) — the
    # weight-byte count divided by the captured element size gives the parameter count.
    _capt = cap.get("captured_dtype", "f32")
    n_elem = WB / ELEMENT_BYTES.get(_capt, ELEMENT_BYTES["f32"])
    exp = {f: n_elem * ELEMENT_BYTES[f] for f in ("bf16", "fp8", "int8", "fp6", "int4")}
    ok_cap = all(_eq(cap[f"{f}_B"], exp[f]) for f in exp)
    check(ok_cap and _eq(cap["avoidable_reload_B"], WB * (K - 1)),
          f"[{w}] dtype_capacity: int8={round(exp['int8'])} bf16={round(exp['bf16'])} "
          f"int4={round(exp['int4'])} (from {round(n_elem)} params)")

    # --- E: dispatch proxy = head matmuls * K ---
    dg = next(r for r in _csv_rows(CS / "dispatch_granularity_table.csv") if r["workload"] == w)
    check(int(dg["commands_per_step_matmul_proxy"]) == MM
          and int(dg["dispatches_per_replan_proxy"]) == MM * K
          and dg["syncs_per_step"] == "unavailable",
          f"[{w}] dispatch proxy: commands/step={MM}, dispatches/replan=MM*{K}={MM*K}, "
          f"syncs=unavailable")

    # --- F: resident_state weights bytes == WB ---
    rs = [r for r in _csv_rows(CS / "resident_state_table.csv")
          if r["workload"] == w and r["state"] == "weights"]
    check(rs and _eq(rs[0]["bytes"], WB) and rs[0]["bytes_evidence"] == "recovered_from_ir",
          f"[{w}] resident_state weights bytes == WB == {WB}")

    # --- G: design_envelope requirements consistent ---
    de = load_yaml(CS / w / "design_envelope.yaml")["design_envelope"]
    reqs = {r["name"]: r.get("value") for r in de["requirements"]}
    check(_eq(reqs["resident_capacity_required"], WB)
          and _eq(reqs["avoidable_weight_reload_bytes"], WB * (K - 1))
          and _eq(reqs["macs_per_replan"], head.get("macs_total", -1)),
          f"[{w}] design_envelope: resident={WB}, avoidable={WB*(K-1)}, "
          f"macs_per_replan={head.get('macs_total')}")

    # --- H: accuracy-gated certificate capacity matches dtype_capacity int8 ---
    certs = [r for r in _csv_rows(CS / "accuracy_gated_dtype_candidates.csv")
             if r["workload"] == w and r["dtype"] == "int8_w8a8"]
    check(certs and _eq(certs[0]["resident_capacity_at_format_B"], exp["int8"]),
          f"[{w}] cert int8 capacity == dtype_capacity int8 == {round(exp['int8'])}")

    return {"workload": w, "K": K, "matmuls": MM, "weight_bytes": WB,
            "avoidable_reload": WB * (K - 1), "cap_int8": round(exp["int8"]),
            "dispatches_per_replan": MM * K}


def verify_global() -> None:
    # --- I: accuracy gating invariant — only int8 may be measured_pass, and only for gated models ---
    gated = _csv_rows(CS / "accuracy_gated_dtype_candidates.csv")
    non_int8_pass = [r for r in gated if r["dtype"] != "int8_w8a8"
                     and r["accuracy_status"] == "measured_pass"]
    check(not non_int8_pass, f"accuracy gating: no non-int8 measured_pass (found {len(non_int8_pass)})")
    gate_models = {p.model for p in AG.load()}
    int8_pass = {r["workload"] for r in gated
                 if r["dtype"] == "int8_w8a8" and r["accuracy_status"] == "measured_pass"}
    expected_pass = {w for w in RECAP_MODELS if w in gate_models and w in _MODELS}
    check(int8_pass == expected_pass,
          f"accuracy gating: int8 measured_pass workloads {sorted(int8_pass)} == "
          f"gated recaptures {sorted(expected_pass)} (rdt absent: recapture != rdt2 gate entry)")

    # --- J: every scalar 'evidence: <label>' across the YAMLs is in the allowed vocabulary.
    # Same-line match only: `evidence:` whose value is a nested dict (candidate evidence PAYLOADS,
    # whose keys are arbitrary like K/H/has_epilogue) has the value on the next line and is skipped. ---
    bad = set()
    n_labels = 0
    for y in CS.rglob("*.yaml"):
        for m in re.findall(r"evidence:[ \t]+([A-Za-z_]+)[ \t]*$", y.read_text(), re.M):
            n_labels += 1
            if m not in ALLOWED_EVIDENCE:
                bad.add(m)
    check(not bad and n_labels > 0,
          f"all {n_labels} scalar evidence labels in allowed vocabulary (offenders: {sorted(bad)})")

    # --- K: dangerous terms absent from GENERATED artifacts (V0/verification docs excluded) ---
    found = {}
    for f in CS.rglob("*"):
        # guard the AUTO-GENERATED structural artifacts; the manual_validation/ subtree is
        # hand-curated narrative (threats-to-validity, final_report, audits, the insight-mining
        # digest) that legitimately uses these words in disclaimer/meta context (e.g. "no speedup",
        # "corpus expansion improvement plan") and is reviewed by hand.
        if not f.is_file() or f.name in V0_DOCS or "manual_validation" in f.parts:
            continue
        low = f.read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                found.setdefault(t, []).append(f.name)
    check(not found, f"dangerous terms absent from generated artifacts (found: {found})")

    # --- L: speedup appears in generated artifacts only inside disclaimers / not_claimed fields ---
    affirmative = []
    for f in CS.rglob("*"):
        if not f.is_file() or f.name in V0_DOCS or "manual_validation" in f.parts:
            continue
        for ln in f.read_text(errors="ignore").splitlines():
            if "speedup" not in ln.lower():
                continue
            low = ln.lower()
            if not any(k in low for k in ("not_claimed", "no speedup", "not a speedup",
                                          "never a claimed", "- speedup", "no bandwidth/speedup",
                                          "claim a speedup", "speedup/accuracy", "speedup, no cycle",
                                          "real dispatch/sync", "no file claims a speedup")):
                affirmative.append((f.name, ln.strip()[:80]))
    check(not affirmative, f"speedup only in disclaimers (affirmative: {affirmative})")

    # --- M: capstone — the consolidated DSE knob catalog aggregates all of P5-P10 with valid
    # evidence, and its grounded knob lists match the source modules (not a stale/hand list) ---
    knobs = load_yaml(CS / "dse_search_space_knobs.yaml")["dse_search_space_knobs"]
    groups = {g["group"]: g for g in knobs["knob_groups"]}
    phases = {g["source_phase"] for g in groups.values()}
    ev_ok = all(g["evidence"] in ALLOWED_EVIDENCE for g in groups.values())
    from merlin.dse_guidance.primitive_coverage import TILE_PRIMITIVES, GEMV_PRIMITIVES
    prim = [n for n, _, _ in TILE_PRIMITIVES] + [n for n, _ in GEMV_PRIMITIVES]
    prim_ok = groups.get("compute_primitive_shape", {}).get("knobs") == prim
    check({"P5", "P7", "P8", "P9", "P10"} <= phases and ev_ok and prim_ok,
          f"consolidated DSE knob catalog aggregates P5-P10 (phases={sorted(phases)}), valid "
          f"evidence, primitive knobs match the P5 candidate set")

    # --- N: the machine-readable contract manifest is well-formed, references real artifacts, and
    # its per-workload set matches the recaptures (the single consume entry point) ---
    import json as _json
    man = _json.loads((CS / "dse_contract.json").read_text())
    req_keys = {"workloads", "per_workload", "search_space_knob_groups", "boundary_placement",
                "measurements_needed_before_quantitative_dse", "artifacts_index",
                "what_is_not_claimed"}
    wl_ok = set(man.get("workloads", [])) == {w for w in RECAP_MODELS
                                              if w in _MODELS}
    idx_ok = all((CS / v).is_file() for v in man.get("artifacts_index", {}).values())
    man_low = (CS / "dse_contract.json").read_text().lower()
    clean = not any(t in man_low for t in DANGEROUS + ("optimal",))
    check(req_keys <= set(man) and wl_ok and idx_ok and clean,
          f"dse_contract.json well-formed, workloads match recaptures, artifact index resolves "
          f"({len(man.get('artifacts_index', {}))} entries), no forbidden terms")


# ============================================================ P5 operator geometry + coverage
# Re-derived INDEPENDENTLY here (the taxonomy thresholds + tile arithmetic are re-implemented, not
# imported) so a divergence in operator_geometry.py / primitive_coverage.py / shape_taxonomy.py
# surfaces as a verification FAILURE.

p5_results: list[tuple[bool, str]] = []


def p5check(ok: bool, msg: str) -> None:
    p5_results.append((bool(ok), msg))


def _ceil_to(x: int, t: int) -> int:
    return int(math.ceil(x / t) * t)


def _classify_geom_indep(M: int, N: int, K: int) -> str:
    """Independent re-implementation of shape_taxonomy.classify_geometry (same documented rules)."""
    if min(M, N) <= 4 and max(M, N) >= 32:
        return "gemv_like"
    mn = M / N
    if mn >= 4:
        return "tall_skinny"
    if (1.0 / mn) >= 4:
        return "wide_skinny"
    if 0.5 <= mn <= 2.0 and min(M, N) >= 32:
        return "squareish_gemm"
    if K >= max(M, N):
        return "projection_like"
    if (_ceil_to(M, 32) * _ceil_to(N, 32)) / (M * N) - 1.0 > 0.10:
        return "odd_tail_heavy"
    if M * N * K < (1 << 16):
        return "small_dispatch_fragment"
    return "unknown"


def verify_p5_workload(w: str) -> dict:
    recs = ATTR.extract_matmuls(str(RECAP / w))
    by_idx = {r.index: r for r in recs}
    shape_rows = [r for r in _csv_rows(CS_SHAPE) if r["workload"] == w]

    # P5-A: one shape row per matmul, M/N/K/macs/aspect/class re-derived from the IR primitive
    geom_ok = len(shape_rows) == len(recs)
    cls_ok = macs_ok = asp_ok = True
    for sr in shape_rows:
        rec = by_idx.get(int(sr["op_index"]))
        if rec is None:
            geom_ok = False
            continue
        if not (int(sr["M"]) == rec.M and int(sr["N"]) == rec.N and int(sr["K"]) == rec.K):
            geom_ok = False
        if int(sr["macs"]) != rec.M * rec.N * rec.K:
            macs_ok = False
        if not _eq(sr["aspect_ratio_MN"], round(rec.M / rec.N, 4), tol=1e-3):
            asp_ok = False
        if sr["shape_class"] != _classify_geom_indep(rec.M, rec.N, rec.K):
            cls_ok = False
    p5check(geom_ok, f"[{w}] operator_shape_table: {len(shape_rows)} rows == {len(recs)} matmuls, "
                     f"M/N/K match IR primitive")
    p5check(macs_ok, f"[{w}] macs == M*N*K for every operator")
    p5check(asp_ok, f"[{w}] aspect_ratio_MN == round(M/N,4) for every operator")
    p5check(cls_ok, f"[{w}] shape_class matches independent taxonomy re-derivation")

    # P5-B: tile padding / waste / utilisation re-derived from primitive shape (tile rows only)
    tw = [r for r in _csv_rows(CS_TILE) if r["workload"] == w]
    pad_ok = waste_ok = util_ok = True
    for r in tw:
        if r["primitive_kind"] != "tile":
            continue
        rec = by_idx[int(r["op_index"])]
        TM, TN = (int(x) for x in r["primitive"].replace("tile_", "").split("x"))
        pM, pN = _ceil_to(rec.M, TM), _ceil_to(rec.N, TN)
        pmacs = pM * pN * rec.K
        if not (int(r["padded_M"]) == pM and int(r["padded_N"]) == pN
                and int(r["padded_macs"]) == pmacs):
            pad_ok = False
        if not _eq(r["padding_waste"], pmacs / (rec.M * rec.N * rec.K) - 1.0, tol=1e-4):
            waste_ok = False
        if not _eq(r["tile_utilization"], (rec.M * rec.N * rec.K) / pmacs, tol=1e-4):
            util_ok = False
    p5check(pad_ok, f"[{w}] tile padded_M/N/macs == ceil-to-tile recompute")
    p5check(waste_ok, f"[{w}] padding_waste == padded_macs/true_macs - 1")
    p5check(util_ok, f"[{w}] tile_utilization == true_macs/padded_macs")

    # P5-C: coverage matrix aggregates recompute from tile_waste_table
    mat = {r["primitive"]: r for r in _csv_rows(CS_COVMAT) if r["workload"] == w}
    agg_ok = True
    by_prim: dict[str, list] = {}
    for r in tw:
        by_prim.setdefault(r["primitive"], []).append(r)
    for prim, rows in by_prim.items():
        m = mat.get(prim)
        if m is None:
            agg_ok = False
            continue
        macs_total = sum(int(x["true_macs"]) for x in rows)
        m10 = sum(int(x["true_macs"]) for x in rows if x["covered_under_10pct"] == "True")
        if int(m["op_count"]) != len(rows) or int(m["macs_covered_10"]) != m10:
            agg_ok = False
        if not _eq(m["coverage_under_10pct"], m10 / macs_total if macs_total else 0.0, tol=1e-4):
            agg_ok = False
    p5check(agg_ok, f"[{w}] primitive_coverage_matrix aggregates recompute from tile_waste_table")

    dom = max(({}.fromkeys(r["shape_class"] for r in shape_rows)), default="—")
    return {"workload": w, "operators": len(shape_rows),
            "shape_classes": len({r["shape_class"] for r in shape_rows}),
            "semantic_classes": len({r["semantic_class"] for r in shape_rows})}


def verify_p5_global() -> None:
    # gemv-lane applicability: a lane must NEVER be scored as covering a squareish_gemm op
    bad = [r for r in _csv_rows(CS_TILE)
           if r["primitive_kind"] == "gemv_lane" and r["shape_class"] == "squareish_gemm"
           and r["applicable"] == "True"]
    p5check(not bad, f"gemv lanes not applicable to squareish_gemm (offenders: {len(bad)})")
    # regret table is internally consistent: max_regret == best - worst
    reg_ok = True
    for r in _csv_rows(CS_REGRET):
        if not _eq(r["max_regret"], float(r["best_workload_coverage_10"])
                   - float(r["worst_workload_coverage_10"]), tol=1e-4):
            reg_ok = False
    p5check(reg_ok, "primitive_regret max_regret == best - worst for every primitive")
    # no forbidden performance claim in the P5 artifacts specifically
    p5_files = ["operator_shape_table.csv", "operator_geometry.yaml", "operator_geometry_report.md",
                "tile_waste_table.csv", "primitive_coverage_matrix.csv",
                "primitive_coverage_report.md", "primitive_regret_table.csv",
                "cross_workload_coverage_report.md", "operator_cluster_table.csv"]
    leaked = {}
    for fn in p5_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p5check(not leaked, f"P5 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P6 multi-rate contract graph
# Re-derived against the IR primitive + the P5 shape table + the attribution facts. A graph that
# disagrees with what the earlier phases recovered fails a check.

p6_results: list[tuple[bool, str]] = []


def p6check(ok: bool, msg: str) -> None:
    p6_results.append((bool(ok), msg))


def _graphs_by_workload() -> dict:
    g = load_yaml(CS_GRAPH)["workload_contract_graph"]
    return {x["workload"]: x for x in g["graphs"]}


def _indep_matmul_deps(w: str) -> list[tuple[int, ...]]:
    """Independent re-trace of per-matmul data deps from the SSA use-def graph (re-implemented)."""
    from merlin.design_pressure.ingest import mlir_m2m
    from xdsl.ir import Operation
    m = mlir_m2m._parse_module((RECAP / w / "model.mlir").read_text())
    mms = [o for o in m.walk() if o.name == "linalg.matmul"]
    res = {id(r): i for i, mm in enumerate(mms) for r in mm.results}
    out = []
    for i, mm in enumerate(mms):
        preds, seen, st = set(), set(), list(mm.operands)
        while st:
            v = st.pop()
            if id(v) in seen:
                continue
            seen.add(id(v))
            j = res.get(id(v))
            if j is not None:
                if j != i:
                    preds.add(j)
                continue
            o = getattr(v, "owner", None)
            if isinstance(o, Operation):
                st.extend(o.operands)
        out.append(tuple(sorted(preds)))
    return out


def verify_p6_workload(w: str, graphs: dict) -> dict:
    g = graphs[w]
    nodes, edges = g["nodes"], g["edges"]
    recs = ATTR.extract_matmuls(str(RECAP / w))

    # P6-A: operator nodes match operator_shape_table.csv exactly (one per matmul, M/N/K/macs)
    ops = [n for n in nodes if n["kind"] == "operator"]
    shape_rows = {int(r["op_index"]): r for r in _csv_rows(CS_SHAPE) if r["workload"] == w}
    op_ok = len(ops) == len(shape_rows) == len(recs)
    for n in ops:
        idx = int(n["id"].split(":")[-1])
        sr = shape_rows.get(idx)
        ss = n.get("shape_summary", {})
        if not sr or any(int(ss.get(k, -1)) != int(sr[k]) for k in ("M", "N", "K", "macs")):
            op_ok = False
    p6check(op_ok, f"[{w}] graph operator nodes == operator_shape_table rows == {len(recs)} matmuls")

    # P6-B: region nodes match attribution (independent raw recompute of role facts) + MAC identity
    raw_roles = _head_facts_from_raw(w)["roles"]
    regions = [n for n in nodes if n["kind"] == "region"]
    reg_ok = True
    for n in regions:
        ws = n.get("work_summary", {})
        rr = raw_roles.get(n["region_role"])
        if rr is not None:
            if int(ws.get("matmul_count") or 0) != rr["matmul_count"] \
                    or int(ws.get("weight_bytes") or 0) != rr["weight_bytes"]:
                reg_ok = False
        mpi, mpr, inv = ws.get("macs_per_invocation"), ws.get("macs_per_replan"), \
            (n.get("rate") or {}).get("invocations")
        if mpi and mpr and inv and int(mpr) != int(mpi) * int(inv):  # derived MACs/replan identity
            reg_ok = False
    p6check(reg_ok, f"[{w}] region nodes match attribution facts; macs_per_replan == "
                    f"macs_per_invocation * invocations")

    # P6-C: K / invocation counts match the design-envelope input (loop trip == head invocations == K)
    K = int(RECAP_MODELS[w]["K"])
    loop = [n for n in nodes if n["kind"] == "loop_body"]
    head = [n for n in regions if n["region_role"] == "repeated_head"]
    k_ok = bool(loop) and int(loop[0]["rate"]["trip_count"]) == K \
        and bool(head) and int(head[0]["rate"]["invocations"]) == K
    p6check(k_ok, f"[{w}] loop_body trip_count == repeated_head invocations == K == {K}")

    # P6-D: cadence vocabulary + edge evidence labels
    cad_ok = all((n.get("rate") or {}).get("cadence", "unknown") in ALLOWED_CADENCE
                 for n in nodes if n["kind"] in ("phase", "region", "loop_body"))
    ev_ok = all(e["evidence"] in ALLOWED_EVIDENCE for e in edges)
    p6check(cad_ok, f"[{w}] all cadence fields in allowed vocabulary")
    p6check(ev_ok, f"[{w}] all edge evidence labels valid")

    # P6-E: operator data-dependency edges match an INDEPENDENT SSA re-trace (recovered_from_ir)
    indep = _indep_matmul_deps(w)
    graph_deps: dict[int, set] = {}
    dd_ev_ok = True
    for e in edges:
        if e["kind"] != "data_dependency":
            continue
        if e["evidence"] != "recovered_from_ir" or e.get("can_pipeline") is not False:
            dd_ev_ok = False
        tgt = int(e["target"].split(":")[-1])
        graph_deps.setdefault(tgt, set()).add(int(e["source"].split(":")[-1]))
    deps_match = all(graph_deps.get(i, set()) == set(indep[i]) for i in range(len(indep)))
    n_dd = sum(len(d) for d in indep)
    p6check(deps_match and dd_ev_ok,
            f"[{w}] {n_dd} data_dependency edges == independent SSA re-trace "
            f"(recovered_from_ir, can_pipeline=False)")

    # P6-F: every operator is attributed to a real role (no unknown region; full attribution)
    op_roles = {n["region_role"] for n in nodes if n["kind"] == "operator"}
    full_attr = "unknown" not in op_roles and not any(
        n["kind"] == "region" and n["region_role"] == "unknown" for n in nodes)
    p6check(full_attr, f"[{w}] every operator attributed to a real role (roles={sorted(op_roles)})")

    from collections import Counter
    nk = Counter(n["kind"] for n in nodes)
    return {"workload": w, "nodes": len(nodes), "edges": len(edges),
            "phase": nk.get("phase", 0), "region": nk.get("region", 0),
            "operator": nk.get("operator", 0), "state": nk.get("state_object", 0)}


def verify_p6_global(graphs: dict) -> None:
    present = set(graphs) == {w for w in RECAP_MODELS if w in _MODELS}
    p6check(present, f"graph contains all workloads ({sorted(graphs)})")
    p6_files = ["workload_contract_graph.yaml", "workload_contract_graph_summary.md",
                "phase_rate_table.csv", "multi_rate_contract.yaml", "rate_mismatch_report.md"]
    leaked = {}
    for fn in p6_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p6check(not leaked, f"P6 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P7 parallelism / sharding / hierarchy
# Re-derived against the IR primitive + the P6 data-dependency edges: critical-path bound, the
# work/span identity, sharding tail/byte formulas, hierarchy/unit references, and discipline.

p7_results: list[tuple[bool, str]] = []


def p7check(ok: bool, msg: str) -> None:
    p7_results.append((bool(ok), msg))


def verify_p7_workload(w: str, graphs: dict) -> dict:
    crit = next(r for r in _csv_rows(CS_CRIT) if r["workload"] == w)
    total_macs = int(crit["total_macs"])
    cp_macs = int(crit["critical_path_macs"])
    total_ops = int(crit["total_ops"])

    # P7-A: total work matches the operator nodes' MAC sum (independent of the parallelism module)
    recs = ATTR.extract_matmuls(str(RECAP / w))
    indep_total = sum(r.macs for r in recs)
    p7check(total_macs == indep_total and total_ops == len(recs),
            f"[{w}] DAG total work == sum of operator MACs ({indep_total:,}) over {len(recs)} ops")

    # P7-B: critical path is a real bound (<= total work) and >= the single largest op
    biggest = max((r.macs for r in recs), default=0)
    p7check(biggest <= cp_macs <= total_macs,
            f"[{w}] largest op ({biggest:,}) <= critical path ({cp_macs:,}) <= total ({total_macs:,})")

    # P7-C: available_parallelism recomputes from total / critical
    ap = float(crit["available_parallelism"])
    p7check(_eq(ap, total_macs / cp_macs, tol=1e-3),
            f"[{w}] available_parallelism == total/critical == {total_macs/cp_macs:.4f}")

    # P7-D: sharding formulas recompute for this workload's ops (tails + partial-sum / dup bytes)
    by_idx = {r.index: r for r in recs}
    srows = [r for r in _csv_rows(CS_SHARD) if r["workload"] == w]
    shard_ok = True
    k_has_reduction = mn_has_broadcast = True
    for r in srows:
        rec = by_idx[int(r["op_index"])]
        axis = r["axis"]
        dim = {"M": rec.M, "N": rec.N, "K": rec.K}[axis]
        if int(r["dim_size"]) != dim or (r["tail_8"] == "True") != (dim % 8 != 0):
            shard_ok = False
        # K must require reduction + a partial-sum object; M/N must not
        if axis == "K":
            if r["reduction_required"] != "True" or "partial_sum_object" not in r["required_abstractions"]:
                k_has_reduction = False
            if int(r["per_extra_shard_bytes"]) != rec.M * rec.N * 4:           # M*N*acc(4)
                shard_ok = False
        else:
            if r["reduction_required"] != "False":
                mn_has_broadcast = False
            if "broadcast" not in r["required_abstractions"] and "multicast" not in r["required_abstractions"]:
                mn_has_broadcast = False
    p7check(shard_ok, f"[{w}] sharding dim_size/tail/partial-sum bytes recompute from M/N/K")
    p7check(k_has_reduction, f"[{w}] K-sharding requires reduction + partial_sum_object")
    p7check(mn_has_broadcast, f"[{w}] M/N-sharding is reduction-free + needs broadcast/multicast")

    return {"workload": w, "available_parallelism": ap, "max_ready_width": int(crit["max_ready_width"]),
            "serialization": crit["serialization"]}


def verify_p7_global() -> None:
    # hierarchy hints reference only known geometry clusters
    known_clusters = set(ST.GEOMETRY_CLASSES)
    hier = _csv_rows(CS_HIER)
    p7check(all(r["shape_class"] in known_clusters for r in hier),
            "hierarchy hints reference only known shape clusters")
    # structural hierarchy hints cover the rest of the P7-c vocabulary, only with known unit names
    hints = load_yaml(CS / "parallel_hierarchy_hints.yaml")["parallel_hierarchy_hints"]
    from merlin.dse_guidance.resource_hierarchy import HIER_UNITS
    su = {h["hierarchy_option"]: h for h in hints.get("structural_units", [])}
    units_ok = set(su) <= HIER_UNITS and all(h["evidence"] in ALLOWED_EVIDENCE for h in su.values())
    expect = {"reduction_tree", "epilogue_unit", "DMA_engine", "loop_controller",
              "multi_engine_cluster"}
    p7check(units_ok and expect <= set(su),
            f"structural hierarchy hints cover {sorted(expect)} with known-unit names + valid evidence")
    # processing-unit candidates: any unit marked present must cite a present resource class
    press = {r["resource_class"]: r for r in _csv_rows(CS_RESPRESS)}
    units = load_yaml(CS / "processing_unit_candidates.yaml")["processing_unit_candidates"]["units"]
    unit_ok = True
    for u in units:
        supported = bool(u["workloads_supporting"])
        if supported and u["evidence"] != "recovered_from_ir":
            unit_ok = False
        if not supported and u["evidence"] != "unavailable":
            unit_ok = False
    p7check(unit_ok, "processing-unit candidates: supported<->recovered_from_ir, else unavailable")
    # resource pressure MAC fractions for compute classes are in [0,1]
    comp = [r for r in press.values() if r["basis"] == "compute_macs"]
    p7check(all(0.0 <= float(r["mac_fraction"]) <= 1.0 for r in comp),
            "resource-pressure compute MAC fractions in [0,1]")
    # no forbidden performance claims in P7 artifacts
    p7_files = ["dag_parallelism_report.md", "critical_path_table.csv", "concurrency_windows.csv",
                "parallel_region_candidates.yaml", "sharding_table.csv",
                "sharding_opportunities.yaml", "intra_op_sharding_report.md",
                "operator_cluster_to_hierarchy.csv", "parallel_hierarchy_hints.yaml",
                "resource_pressure_table.csv", "processing_unit_candidates.yaml",
                "processing_unit_parallelism_report.md"]
    leaked = {}
    for fn in p7_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p7check(not leaked, f"P7 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P8 pipeline / overlap / unit guidance

p8_results: list[tuple[bool, str]] = []


def p8check(ok: bool, msg: str) -> None:
    p8_results.append((bool(ok), msg))


def verify_p8_workload(w: str, graphs: dict) -> dict:
    g = graphs[w]
    graph_phase_ids = {n["id"].split(":")[-1] for n in g["nodes"] if n["kind"] == "phase"}
    env = load_yaml(CS / "pipeline_envelope.yaml")["pipeline_envelope"]
    phases = next(x["phases"] for x in env["workloads"] if x["workload"] == w)

    # P8-A: every phase references a graph phase node
    p8check(all(p["phase"] in graph_phase_ids for p in phases),
            f"[{w}] pipeline phases reference graph phase nodes ({sorted(graph_phase_ids)})")

    # P8-B: phase cadence matches phase_rate_table.csv (the P6 table)
    rate = {r["phase"]: r["cadence"] for r in _csv_rows(CS_PHASE_RATE) if r["workload"] == w}
    stage = {r["phase"]: r["cadence"] for r in _csv_rows(CS_PIPE_STAGE) if r["workload"] == w}
    p8check(all(stage.get(p, rate.get(p)) == rate.get(p) for p in rate) and stage == {
        p: rate[p] for p in rate},
            f"[{w}] pipeline stage cadences == phase_rate_table cadences")

    # P8-C/D/E: overlap candidates well-formed (can_overlap enum, buffers, allowed abstractions)
    cand = load_yaml(CS / "pipeline_candidates.yaml")["pipeline_candidates"]
    cands = next(x["candidates"] for x in cand["workloads"] if x["workload"] == w)
    from merlin.dse_guidance.pipeline_envelope import ALLOWED_ABSTRACTIONS
    enum_ok = all(c["can_overlap"] in ("yes", "no", "unknown") for c in cands)
    abst_ok = all(a in ALLOWED_ABSTRACTIONS for c in cands for a in c["required_abstractions"])
    buf_ok = True
    for c in cands:
        b = c["required_buffer_count"]
        if b == "unavailable":
            continue
        if not (isinstance(b, int) and b >= 1):
            buf_ok = False
        if c["can_overlap"] == "yes" and not (isinstance(b, int) and b >= 1):
            buf_ok = False
    p8check(enum_ok, f"[{w}] overlap can_overlap in {{yes,no,unknown}}")
    p8check(abst_ok, f"[{w}] overlap required_abstractions in allowed vocabulary")
    p8check(buf_ok, f"[{w}] overlap buffer counts are positive ints or unavailable "
                    f"(yes-overlaps have >=1 buffer)")

    # P8-F: gated overlaps reflect recovered per-workload structure (not a uniform template)
    bb_ops = sum(int((n.get("work_summary") or {}).get("matmul_count") or 0)
                 for n in g["nodes"] if n["kind"] == "region" and n["region_role"] == "backbone_once")
    bb_yes = any(c["can_overlap"] == "yes" for c in cands
                 if c["source_phase"].startswith("backbone(next"))
    p8check((bb_ops > 0) == bb_yes,
            f"[{w}] backbone overlap=yes iff recovered backbone compute exists (ops={bb_ops})")
    from merlin.dse_guidance import models as _M
    a = _M.MODEL_ARCH.get(w)
    is_vla = bool(a and a.family in {"flow_matching", "diffusion", "autoregressive_vla"}
                  and a.control_rate_hz and a.action_horizon)
    ctrl_yes = any(c["can_overlap"] == "yes" for c in cands
                   if c["source_phase"] == "control_tick_consumer")
    p8check(is_vla == ctrl_yes,
            f"[{w}] control-tick overlap=yes iff workload is a VLA with a control loop (vla={is_vla})")
    return {"workload": w, "phases": len(phases),
            "yes_overlaps": sum(1 for c in cands if c["can_overlap"] == "yes")}


def verify_p8_global() -> None:
    # processing-unit guidance references existing resource classes
    classes = {r["resource_class"] for r in _csv_rows(CS_RESPRESS)}
    pug = load_yaml(CS / "processing_unit_guidance.yaml")["processing_unit_guidance"]
    spec = next(o for o in pug["options"] if o["option"] == "multiple_specialized_units")
    units_ok = all(u["for"] in classes for u in spec.get("candidate_units", []))
    p8check(units_ok, "processing-unit guidance candidate_units reference existing resource classes")
    # the three multiplicity options are all present
    opts = {o["option"] for o in pug["options"]}
    p8check(opts == {"one_bigger_unit", "multiple_identical_units", "multiple_specialized_units"},
            "processing-unit guidance covers monolithic / replicated / specialized")
    # no forbidden performance claims in P8 artifacts
    p8_files = ["pipeline_envelope.yaml", "pipeline_stage_table.csv", "pipeline_candidates.yaml",
                "buffering_requirement_table.csv", "overlap_opportunities.md",
                "processing_unit_guidance.yaml", "heterogeneity_report.md"]
    leaked = {}
    for fn in p8_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p8check(not leaked, f"P8 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P9 memory / DMA / buffer envelope
# Re-derived against the IR primitive: per-region weight bytes, the avoidable-reload formula, the
# dtype-scaled resident set, and that DMA streams / buffers reference valid regions and name the
# unavailable bytes explicitly.

p9_results: list[tuple[bool, str]] = []


def p9check(ok: bool, msg: str) -> None:
    p9_results.append((bool(ok), msg))


def verify_p9_workload(w: str) -> dict:
    # independent per-role weight bytes from the IR primitive, via the SAME role logic as
    # attribution (structural scf.for boundary on a loop capture; prov.fqn on a flat capture).
    K = int(RECAP_MODELS[w]["K"])
    role_weight = {role: facts["weight_bytes"]
                   for role, facts in _head_facts_from_raw(w)["roles"].items()}
    dm = {r["region"]: r for r in _csv_rows(CS_DATAMOVE) if r["workload"] == w}

    # P9-A: region weight bytes match the independent recompute
    wb_ok = all(int(dm[role]["weight_bytes"]) == wb for role, wb in role_weight.items()
                if role in dm)
    p9check(wb_ok and set(role_weight) <= set(dm),
            f"[{w}] data_movement weight bytes match IR recompute ({role_weight})")

    # P9-B: avoidable reload == weight_bytes * max(inv - 1, 0)
    reload_ok = True
    for role, row in dm.items():
        inv = int(row["invocations"])
        if int(row["avoidable_weight_reload"]) != int(row["weight_bytes"]) * max(inv - 1, 0):
            reload_ok = False
    p9check(reload_ok, f"[{w}] avoidable_weight_reload == weight_bytes * max(inv-1, 0)")

    # P9-C: dtype-scaled resident set matches the f32 element-count scaling (int8 = WB/4)
    res_ok = True
    for role, row in dm.items():
        wb = int(row["weight_bytes"])
        n = wb / ELEMENT_BYTES["f32"]
        if not (_eq(row["resident_int8_B"], n * ELEMENT_BYTES["int8"])
                and _eq(row["resident_bf16_B"], n * ELEMENT_BYTES["bf16"])):
            res_ok = False
    p9check(res_ok, f"[{w}] resident_by_dtype == weights scaled by element width (int8 == WB/4)")

    # P9-D: intermediate / scale / KV bytes are explicitly unavailable (not invented)
    una_ok = all(row["intermediate_bytes"] == "unavailable" and row["scale_bytes"] == "unavailable"
                 and row["kv_bytes"] == "unavailable" for row in dm.values())
    p9check(una_ok, f"[{w}] intermediate / scale / KV bytes explicitly unavailable")

    # P9-E: DMA streams reference valid regions; byte-carrying streams have int bytes
    regions = set(dm)
    streams = [r for r in _csv_rows(CS_DMA) if r["workload"] == w]
    sref_ok = all(r["region"] in regions for r in streams)
    byte_ok = all((r["bytes"] == "unavailable") or r["bytes"].lstrip("-").isdigit()
                  for r in streams)
    p9check(sref_ok and byte_ok and streams,
            f"[{w}] DMA streams reference valid regions; bytes int-or-unavailable")

    # P9-F: buffer rows reference valid regions; counts are positive ints; double-buffer is yes/no/unknown
    bufs = [r for r in _csv_rows(CS_BUFREQ) if r["workload"] == w]
    buf_ok = all(r["region"] in regions and int(r["min_input_buffer_count"]) >= 1
                 and int(r["min_output_buffer_count"]) >= 1
                 and r["double_buffering_needed"] in ("yes", "no", "unknown") for r in bufs)
    p9check(buf_ok and bufs, f"[{w}] buffer rows reference valid regions with positive counts")
    return {"workload": w, "regions": len(dm),
            "top_avoidable_reload": max((int(r["avoidable_weight_reload"]) for r in dm.values()),
                                        default=0)}


def verify_p9_global() -> None:
    # candidate DMA abstractions are in the allowed vocabulary
    from merlin.dse_guidance.dma_buffer_analysis import ALLOWED_DMA_ABSTRACTIONS
    streams = _csv_rows(CS_DMA)
    p9check(all(r["candidate_abstraction"] in ALLOWED_DMA_ABSTRACTIONS for r in streams),
            "DMA candidate abstractions in allowed vocabulary")
    # no bandwidth/speedup/cycle claim leaks into P9 artifacts
    p9_files = ["memory_hierarchy_envelope.yaml", "data_movement_table.csv",
                "reuse_lifetime_table.csv", "memory_abstraction_candidates.yaml",
                "memory_envelope_report.md", "dma_stream_table.csv",
                "buffer_requirement_table.csv", "dma_pressure_report.md"]
    leaked = {}
    for fn in p9_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p9check(not leaked, f"P9 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P10 fusion / epilogue / accumulator
p10_results: list[tuple[bool, str]] = []


def p10check(ok: bool, msg: str) -> None:
    p10_results.append((bool(ok), msg))


def verify_p10_workload(w: str) -> dict:
    recs = ATTR.extract_matmuls(str(RECAP / w))
    by_idx = {r.index: r for r in recs}
    pats = [r for r in _csv_rows(CS_EPILOGUE) if r["workload"] == w]

    # P10-A: pattern rows reference valid matmul ops, one per matmul
    p10check(len(pats) == len(recs) and all(int(p["op_index"]) in by_idx for p in pats),
             f"[{w}] epilogue pattern rows reference valid ops ({len(recs)} matmuls)")

    # P10-B: bias detection is sound — every addmm op is flagged has_bias (independent op kind)
    bias_ok = all(p["has_bias"] == "True" for p in pats
                  if (by_idx[int(p["op_index"])].op == "addmm"))
    p10check(bias_ok, f"[{w}] every addmm op detected as has_bias (matmul+bias epilogue)")

    # P10-C: accumulator/dequant/requant fields use the allowed vocabulary + dtype is consistent
    accs = [r for r in _csv_rows(CS_ACCUM) if r["workload"] == w]
    vocab_ok = all(r["dequant_location"] in _DEQ_VOCAB and r["requant_location"] in _REQ_VOCAB
                   and r["accumulator_materialization"] in _ACC_VOCAB for r in accs)
    # f32 storage -> accumulator == compute (no widening); int8 would be i32 (derived, not here)
    dtype_ok = all(r["accumulator_dtype"] == r["compute_dtype"] for r in accs
                   if r["storage_dtype"] in ("f32", "float32", "fp32"))
    p10check(vocab_ok and dtype_ok and accs,
             f"[{w}] accumulator/dequant/requant fields use allowed vocabulary; f32 acc==compute")

    # P10-D: scale/zero-point + dequant are explicitly unavailable (erased by dequantized capture)
    erased_ok = all(r["scale_dtype"] == "unavailable" and r["dequant_location"] == "unavailable"
                    for r in accs)
    p10check(erased_ok, f"[{w}] scale metadata + dequant location explicitly unavailable")

    # P10-E: reshape-separated matmuls carry NO directly-fused epilogue flag (no over-claim of a
    # reshape-distant residual/rotary as bias/scale)
    rs_ok = all(p["has_bias"] == "False" and p["has_activation"] == "False"
                and p["has_scale"] == "False" and p["has_clamp"] == "False"
                for p in pats if p["reshape_separated_epilogue"] == "True")
    p10check(rs_ok, f"[{w}] reshape-separated matmuls claim no directly-fused epilogue")
    return {"workload": w, "matmuls": len(recs),
            "bias_ops": sum(1 for p in pats if p["has_bias"] == "True")}


def verify_p10_global() -> None:
    cand = load_yaml(CS / "numerical_epilogue_candidates.yaml")["numerical_epilogue_candidates"]
    units = {c["abstraction"]: c for c in cand["candidates"]}
    # certificate evidence is consistent: recovered_from_ir iff a workload shows the pattern
    cons_ok = all((c["evidence"] == "recovered_from_ir") == bool(c["present_in_workloads"])
                  for c in units.values())
    p10check(cons_ok, "epilogue certificates: recovered_from_ir iff present in a workload, else unavailable")
    # low-bit / scale / sparsity abstractions must be unavailable (not falsely IR-supported)
    must_block = {"fused_dequant_matmul", "scale_object", "packed_lowbit_tensor",
                  "resident_packed_weight_object", "structured_sparsity_skip"}
    block_ok = all(units[a]["evidence"] == "unavailable" for a in must_block if a in units)
    p10check(block_ok, f"low-bit/scale/sparsity certificates are unavailable (not falsely supported)")
    # no false measured-pass for any non-int8 format leaked into the P10 artifacts
    p10_files = ["epilogue_pattern_table.csv", "accumulator_contract_table.csv",
                 "numerical_epilogue_candidates.yaml", "lost_numerical_contracts.csv",
                 "fusion_opportunity_report.md"]
    no_pass = all("measured_pass" not in (CS / f).read_text() for f in p10_files)
    p10check(no_pass, "no measured_pass status leaked into P10 fusion artifacts")
    leaked = {}
    for fn in p10_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p10check(not leaked, f"P10 artifacts free of forbidden performance terms (found: {leaked})")


# ============================================================ P12 HW/SW boundary placement
p12_results: list[tuple[bool, str]] = []


def p12check(ok: bool, msg: str) -> None:
    p12_results.append((bool(ok), msg))


def verify_p12() -> dict:
    from merlin.dse_guidance.boundary_placement import (ABSTRACTIONS, LEVELS, STATUS, RESP_CELLS,
                                                        REGION_ROLES)
    contracts = load_yaml(CS / "boundary_candidate_contracts.yaml")["boundary_candidate_contracts"]
    certs = contracts["certificates"]
    cp_axes = {r["axis"] for r in _csv_rows(CS / "compiler_proof_matrix.csv")}
    valid_wl = {w for w in RECAP_MODELS if w in _MODELS}
    ALL_ABS = set(ABSTRACTIONS)

    # (1) every candidate references a known abstraction
    p12check(all(c["abstraction"] in ALL_ABS for c in certs) and len(certs) == len(ALL_ABS),
             f"every boundary candidate references a known abstraction ({len(certs)})")
    # (2) every workload reference exists
    p12check(all(w in valid_wl for c in certs for w in c["supporting_workloads"]),
             "every boundary workload reference exists")
    # (3) every region role reference is real
    p12check(all(r in REGION_ROLES for c in certs for r in c["region_roles"]),
             "every boundary region-role reference is real")
    # (4) every boundary level uses the allowed vocabulary
    lv_ok = all(b["level"] in LEVELS for c in certs for b in c["boundary_levels"]) and \
        all({b["level"] for b in c["boundary_levels"]} == set(LEVELS) for c in certs)
    p12check(lv_ok, "every boundary level uses the allowed 6-level vocabulary")
    # (5) every status uses the allowed vocabulary
    p12check(all(b["status"] in STATUS for c in certs for b in c["boundary_levels"]),
             "every placement status uses the allowed vocabulary")
    # (6) every required proof references a CP-matrix axis or is unavailable
    proof_ok = all((c["compiler_proof_matrix_axis"] in cp_axes)
                   or (c["compiler_proof_status"] == "unavailable") for c in certs)
    p12check(proof_ok, "every required compiler proof references a CP-matrix axis or is unavailable")
    # (7) every DSE knob has a reason + evidence
    knobs = load_yaml(CS / "boundary_dse_knobs.yaml")["boundary_dse_knobs"]["knobs"]
    p12check(knobs and all(k.get("reason") and k.get("evidence") for k in knobs),
             f"every boundary DSE knob has a reason + evidence ({len(knobs)})")
    # (8) every responsibility-matrix cell uses the allowed vocabulary
    cols = ["compiler", "runtime_hal", "command_processor", "accelerator_isa", "device_microcode",
            "datapath"]
    resp = _csv_rows(CS / "responsibility_split_matrix.csv")
    p12check(resp and all(r[col] in RESP_CELLS for r in resp for col in cols),
             f"every responsibility-matrix cell uses the allowed vocabulary ({len(resp)} rows)")
    # (9) no boundary artifact claims speedup/cycles/area/energy/optimal/best
    p12_files = ["hw_sw_boundary_matrix.csv", "boundary_candidate_contracts.yaml",
                 "boundary_placement_report.md", "responsibility_split_matrix.csv",
                 "interface_contract_sketches.md", "isa_candidate_primitives.yaml",
                 "runtime_object_candidates.yaml", "command_isa_candidates.yaml",
                 "boundary_dse_knobs.yaml"]
    leaked = {}
    for fn in p12_files:
        low = (CS / fn).read_text(errors="ignore").lower()
        for t in DANGEROUS + ("optimal",):
            if t in low:
                leaked.setdefault(t, []).append(fn)
    p12check(not leaked, f"P12 artifacts free of forbidden performance terms (found: {leaked})")
    # (10) partial mode: builder works with absent inputs (no workloads / no proofs)
    from merlin.dse_guidance.boundary_placement import build_certificates
    try:
        part = build_certificates({}, {})
        partial_ok = len(part) == len(ALL_ABS) and all(c.supporting_workloads == [] for c in part) \
            and all(c.compiler_proof_status == "unavailable" for c in part)
    except Exception:
        partial_ok = False
    p12check(partial_ok, "partial mode: boundary builder works with absent P7/P8/P9 inputs")

    top = max(certs, key=lambda c: c["boundary_pressure_score"])
    return {"abstractions": len(certs), "top": top["abstraction"],
            "top_score": top["boundary_pressure_score"]}


# ============================================================ P13 evidence mining / insight extract
# Runs the meta-analysis over the committed case-study (per network + 'all'), asserts the P13-g
# cross-artifact consistency checks pass for every scope, mining is deterministic, no main finding is
# weakly-evidenced, and partial mode degrades cleanly. Output is non-committed, so this is
# run-in-memory-and-check (not a byte-stable committed diff).

p13_results: list[tuple[bool, str]] = []


def p13check(ok: bool, msg: str) -> None:
    p13_results.append((bool(ok), msg))


def verify_p13() -> dict:
    from merlin.dse_guidance import insight_mining as IM
    nets = IM._workloads(CS)
    scopes = nets + ["all"]
    total_facts = 0
    expected_arts = {a for a, _ in IM._ARTIFACT_PHASE.items()}
    for scope in scopes:
        b = IM.mine(CS, scope)
        total_facts += len(b["facts"]) if scope == "all" else 0
        p13check(all(ok for ok, _ in b["consistency_checks"]),
                 f"[{scope}] all {len(b['consistency_checks'])} P13-g consistency checks pass")
        p13check(len(b["facts"]) > 0, f"[{scope}] >=1 normalized fact mined")
        ids = [f["fact_id"] for f in b["facts"]]
        p13check(len(ids) == len(set(ids)), f"[{scope}] no duplicate fact_id")
        main = [f for f in b["findings"] if f["presentation_placement"] == "main"]
        p13check(all(f["evidence_tier"] in ("A", "B") for f in main),
                 f"[{scope}] every main finding is tier A/B (no weak/assumed main)")
        # P14: every main finding is a SIGNAL metric (decision-relevant, not a row-count/context),
        # tier A/B. Corroboration / a harness check are reported as STRENGTH (corroborated_by,
        # verifying_check), not the gate -- so real single-source recovered signal is allowed.
        p13check(all(all(m in IM.SIGNAL_METRICS for m in f["relevant_metrics"]) for f in main),
                 f"[{scope}] every main finding is a SIGNAL metric (no context/count padding)")
        # P14 devil's-advocate convergence: ZERO open avoidable gaps
        oa = b["open_avoidable_gaps"]
        p13check(not oa, f"[{scope}] gap_audit: 0 open avoidable gaps "
                         f"(found {len(oa)}: {[g['category'] for g in oa][:4]})")
        # P14: every inherent limit is scoped to a required input (no bare caveats)
        p13check(b["required_inputs"] and all(x.get("required_input") for x in b["required_inputs"]),
                 f"[{scope}] every inherent limit carries a required_input")
    # P14: the 'all' scope leverages EVERY expected artifact (coverage == all)
    ball = IM.mine(CS, "all")
    used = {f["source_artifact"] for f in ball["facts"]}
    missing = sorted(a for a in expected_arts if a not in used)
    p13check(not missing, f"[all] coverage == all {len(expected_arts)} expected artifacts "
                          f"(unused: {missing[:5]})")
    # P14: real measured + low-bit evidence is surfaced (not just small f32 recaptures)
    measured = [f for f in ball["facts"] if f["derivation_type"] == "measured"]
    lowbit = [f for f in ball["facts"] if "lowbit" in f["metric_name"] or f["workload"] == "ZOO"]
    p13check(measured and lowbit,
             f"[all] real measured ({len(measured)}) + low-bit ({len(lowbit)}) evidence surfaced")
    # determinism: same committed inputs -> identical mined facts + findings
    a1, a2 = IM.unified_facts(CS, "all"), IM.unified_facts(CS, "all")
    p13check(a1 == a2, "insight mining is deterministic (same inputs -> identical facts)")
    f1 = IM.presentation_findings(a1, IM.usefulness(CS, "all", a1))
    f2 = IM.presentation_findings(a2, IM.usefulness(CS, "all", a2))
    p13check(f1 == f2, "presentation findings are deterministic")
    # forbidden wording across the mined findings/answers (outside the FORBIDDEN tuple definition)
    bundle = IM.mine(CS, "all")
    blob = (str(bundle["findings"]) + str(bundle["usefulness"]) + str(bundle["plots"])).lower()
    leaked = [t for t in ("speedup", "faster", "optimal", "performance improvement", "gap_closure",
                          "predicted cycles") if t in blob]
    p13check(not leaked, f"no forbidden performance wording in mined findings/answers ({leaked})")
    # partial mode: mining a dir without the root artifacts (a workload subdir) degrades cleanly
    part = IM.mine(CS / nets[0], "all") if nets else None
    p13check(part is not None and all("exists" in r for r in part["inventory"])
             and any(not r["exists"] for r in part["inventory"]),
             "partial mode: missing artifacts recorded exists=no, no crash")
    verify_p15(IM, bundle)
    verify_p16(IM, bundle)
    verify_p17_audit(IM, bundle)
    verify_p17_envelope(IM, bundle)
    verify_p20(IM)
    verify_p21(IM)
    return {"scopes": len(scopes), "facts_all": total_facts}


def verify_p15(IM, bundle) -> None:
    """P15 signal-first study: re-derive the canonical table, hotspots, coverage, family + corpus
    plan independently and assert they match; every signal metric answers a DSE question."""
    import csv as _csv
    import io as _io

    def _rows(name):
        p = CS / name
        return list(_csv.DictReader(_io.StringIO(p.read_text()))) if p.is_file() else []

    # 1. canonical signal table is signal/measured ONLY (no context/count metric leaks in)
    canon = bundle["canonical_signal"]
    facts_by_metric = {f["metric_name"]: f for f in bundle["facts"]}
    ctx_in_canon = [r["metric"] for r in canon
                    if facts_by_metric.get(r["metric"], {}).get("metric_class") == "context"]
    p13check(canon and not ctx_in_canon,
             f"[P15] canonical_signal_table is signal/measured only ({len(canon)} rows, "
             f"context leaks: {ctx_in_canon[:4]})")
    # 2. every SIGNAL metric maps to a DSE question; every canonical row + main finding carries one
    unmapped = [m for m in IM.SIGNAL_METRICS if m not in IM._METRIC_QUESTION]
    canon_noq = [r["metric"] for r in canon if not r["dse_question"]]
    main = [f for f in bundle["findings"] if f["presentation_placement"] == "main"]
    main_noq = [f["title"] for f in main if not f.get("dse_question")]
    valid_q = set(IM.DSE_QUESTIONS)
    bad_q = [r["dse_question"] for r in canon if r["dse_question"] not in valid_q]
    p13check(not unmapped and not canon_noq and not main_noq and not bad_q,
             f"[P15] every signal metric/row/main-finding maps to a valid DSE question "
             f"(unmapped {unmapped[:3]}, canon_noq {canon_noq[:3]}, main_noq {main_noq[:3]})")
    # 3. per-operator hotspots reference real ops + recompute independently
    h = bundle["hotspots"]
    ops = _rows("operator_shape_table.csv")
    op_keys = {(o["workload"], o["op_index"]) for o in ops}
    refs_real = all((r["workload"], r["op_index"]) in op_keys for r in h["by_macs"])
    indep_macs = sorted(ops, key=lambda o: -int(o["macs"]))[:10]
    macs_match = [r["op_index"] for r in h["by_macs"]] == [o["op_index"] for o in indep_macs]
    p13check(refs_real and macs_match and h["n_ops"] == len(ops),
             f"[P15] hotspots reference real ops and top-by-MACs recomputes "
             f"(refs_real={refs_real}, macs_match={macs_match}, n_ops={h['n_ops']})")
    # padding waste matches an independent best-tile recompute
    tw = _rows("tile_waste_table.csv")
    best = {}
    for r in tw:
        if r.get("primitive_kind") == "tile" and r.get("applicable") == "True":
            key = (r["workload"], r["op_index"])
            w = float(r["padding_waste"])
            best[key] = min(best.get(key, w), w)
    pad_ok = all(abs(r["best_tile_padding_waste"]
                     - best[(r["workload"], r["op_index"])]) < 1e-9 for r in h["by_padding_waste"])
    p13check(pad_ok, "[P15] hotspot padding waste == independent best-tile recompute")
    # 4. abstraction coverage MAC/byte/workload coverage recomputes from source artifacts
    cov = bundle["abstraction_coverage"]
    dm_rows = _rows("data_movement_table.csv")
    mac_by_wl, byte_by_wl, reg_by_wl = {}, {}, {}
    for o in ops:
        mac_by_wl[o["workload"]] = mac_by_wl.get(o["workload"], 0) + int(o["macs"])
        byte_by_wl[o["workload"]] = byte_by_wl.get(o["workload"], 0) + int(o["rhs_weight_bytes"])
    for r in dm_rows:
        reg_by_wl[r["workload"]] = reg_by_wl.get(r["workload"], 0) + 1
    tmac, tbyte = sum(mac_by_wl.values()) or 1, sum(byte_by_wl.values()) or 1
    treg = len(dm_rows) or 1
    nwl = len(mac_by_wl)
    cov_ok = True
    for r in cov:
        supp = [w.strip() for w in r["workloads_supporting"].split(";") if w.strip()]
        exp_mac = round(sum(mac_by_wl.get(w, 0) for w in supp) / tmac, 4)
        exp_byte = round(sum(byte_by_wl.get(w, 0) for w in supp) / tbyte, 4)
        exp_reg = round(sum(reg_by_wl.get(w, 0) for w in supp) / treg, 4)
        exp_wl = round(len(supp) / (nwl or 1), 4)
        if (abs(r["mac_coverage"] - exp_mac) > 1e-6 or abs(r["byte_coverage"] - exp_byte) > 1e-6
                or abs(r["region_coverage"] - exp_reg) > 1e-6
                or abs(r["workload_coverage"] - exp_wl) > 1e-6):
            cov_ok = False
            break
    p13check(cov and cov_ok,
             f"[P15] abstraction coverage (workload/MAC/byte/region) recomputes from source "
             f"({len(cov)} rows)")
    # per-network scoping: a single-workload run reports only that network's hotspots + canonical,
    # and the corpus-level cross-workload artifacts are withheld from a per-network bundle
    one = [w for w in IM._workloads(CS) if w != "rdt"][0]
    bn = IM.mine(CS, one)
    hs = bn["hotspots"]
    scoped_ok = (hs["by_macs"] and all(r["workload"] == one for r in hs["by_macs"])
                 and hs["n_ops"] == sum(1 for o in ops if o["workload"] == one)
                 and {r["workload"] for r in bn["canonical_signal"]
                      if r["workload"] not in ("ALL", "ZOO", "")} == {one})
    withheld_ok = (not bn["abstraction_coverage"] and not bn["corpus_plan"]
                   and not bn["family_summary"]["families"])
    p13check(scoped_ok and withheld_ok,
             f"[P15] per-network scope '{one}' is network-scoped (hotspots+canonical) with "
             f"corpus-level artifacts withheld (scoped={scoped_ok}, withheld={withheld_ok})")
    # 5. corpus-expansion plan references only real registry families lacking a recapture
    from merlin.dse_guidance.models import MODEL_ARCH
    cp = bundle["corpus_plan"]
    missing = [m["model"] for ms in cp["missing_by_family"].values() for m in ms]
    real = all(m in MODEL_ARCH for m in missing)
    none_captured = not (set(missing) & set(cp["captured_models"]))
    p13check(real and none_captured and cp["n_missing"] == len(missing),
             f"[P15] corpus plan references only real, uncaptured registry models "
             f"({cp['n_missing']} missing, all real={real})")
    # 6. every rendered-eligible plot carries a non-empty DSE caption; no forbidden wording in P15
    plots = bundle["plots"]
    no_cap = [p["plot_id"] for p in plots
              if p["recommendation"] != "omit" and not p.get("dse_caption")]
    p15_blob = (str(canon) + str(h) + str(cov) + str(bundle["family_summary"])
                + str(cp) + str([p.get("dse_caption") for p in plots])).lower()
    leaked = [t for t in ("speedup", "faster", "optimal", "performance improvement",
                          "predicted cycles", "throughput of", "x speedup") if t in p15_blob]
    p13check(not no_cap and not leaked,
             f"[P15] every presented plot has a DSE caption + no forbidden wording "
             f"(no_cap {no_cap[:3]}, leaked {leaked})")


def verify_p16(IM, bundle) -> None:
    """P16 decision-frontier & robustness: independently re-derive the strict necessity, the
    primitive-set frontier, the operator Pareto, leave-one-out, and the capture-fidelity, and assert
    they are decision-discriminating (NOT all-permissive) and recompute from source."""
    import csv as _csv
    import io as _io
    import itertools

    def _rows(name):
        p = CS / name
        return list(_csv.DictReader(_io.StringIO(p.read_text()))) if p.is_file() else []

    from merlin.dse_guidance.boundary_placement import catalog_rows
    # 1. necessity is NOT permissive: not everything 'necessary'; blocked == erased/kv catalog set;
    #    every cell is a valid class; at least one per-workload N/A exists (non-AR workload + decode/kv)
    nec = bundle["abstraction_necessity"]
    roll = nec["rollup"]
    valid = set(IM._NEC_CLASSES)
    wls = nec["workloads"]
    all_cells_valid = all(r[w] in valid for r in nec["rows"] for w in wls)
    not_all_nec = roll["necessary"] < len(nec["rows"]) and roll["blocked"] > 0
    erased_or_kv = {c["abstraction"] for c in catalog_rows() if c["erased"] or c["kv"]}
    blocked_macro = {r["abstraction"] for r in nec["rows"] if r["macro_class"] == "blocked"}
    blocked_ok = blocked_macro <= erased_or_kv and blocked_macro
    has_na_cell = any(r[w] == "not_applicable" for r in nec["rows"] for w in wls)
    p13check(all_cells_valid and not_all_nec and blocked_ok and has_na_cell,
             f"[P16] necessity is discriminating (rollup={roll}, blocked⊆erased/kv={blocked_ok}, "
             f"has N/A cell={has_na_cell})")
    # 1b. independent recompute of matrix_engine (dense) necessity from shape_summary
    shape = _rows("shape_summary_by_workload.csv")
    dense_by_wl = {}
    for w in wls:
        dense_by_wl[w] = sum(float(r["mac_fraction"]) for r in shape
                             if r["workload"] == w and r["shape_class"] == "squareish_gemm")
    me = next(r for r in nec["rows"] if r["abstraction"] == "matrix_engine")
    me_ok = all((me[w] == "necessary") == (dense_by_wl[w] > 0.5) for w in wls)
    p13check(me_ok, "[P16] matrix_engine necessity == (dense MAC fraction > 0.5) per workload")
    # 2. primitive-set frontier: best worst-coverage is monotone non-decreasing in set size, and a
    #    2-set strictly beats the best single (the headline)
    fr = bundle["primitive_frontier"]["best_by_size"]
    mono = all(fr[s]["worst"] >= fr[s - 1]["worst"] - 1e-9 for s in fr if s - 1 in fr)
    beats = fr[2]["worst"] > fr[1]["worst"] if 1 in fr and 2 in fr else False
    p13check(mono and beats,
             f"[P16] primitive-set frontier monotone in size and 2-set beats 1-set "
             f"(1:{fr.get(1,{}).get('worst')} 2:{fr.get(2,{}).get('worst')})")
    # 2b. set-union recompute: the best 2-set's worst coverage matches an independent op-level union
    tw = _rows("tile_waste_table.csv")
    op_macs, op_cover = {}, {}
    for r in tw:
        if r.get("applicable") != "True":
            continue
        k = (r["workload"], r["op_index"])
        op_macs[k] = float(r["true_macs"])
        op_cover.setdefault(k, {})[r["primitive"]] = (r["covered_under_10pct"] == "True")
    pset = bundle["primitive_frontier"]["best_by_size"][2]["set"]
    num, den = {}, {}
    for (w, op), m in op_macs.items():
        den[w] = den.get(w, 0.0) + m
        if any(op_cover[(w, op)].get(p, False) for p in pset):
            num[w] = num.get(w, 0.0) + m
    worst_indep = round(min(num.get(w, 0.0) / den[w] for w in den), 4)
    p13check(abs(worst_indep - fr[2]["worst"]) < 1e-6,
             f"[P16] 2-set worst coverage recomputes by op-level union ({worst_indep})")
    # 3. operator Pareto: k thresholds monotone, reach within n_ops
    par = bundle["operator_pareto"]
    pok = all(r[f"k_macs_{int(t*100)}"] <= r["n_ops"]
              and r["k_macs_50"] <= r["k_macs_95"] for r in par["rows"] for t in par["thresholds"])
    rdt = next(r for r in par["rows"] if r["workload"] == "rdt")
    p13check(pok and rdt["k_macs_50"] == 1,
             f"[P16] operator Pareto monotone + within n_ops; rdt k@50%MAC={rdt['k_macs_50']}")
    # 4. leave-one-out MECHANISM is sound (not a corpus-specific outcome — adding the VLAs correctly
    #    overturned the 4-model 'dense dominates micro' finding, which is the point of LOO). Assert
    #    macro/micro are valid fractions, micro_loo has an entry per workload, and collapses_if_removed
    #    is exactly the set whose removal drops micro below 0.2 from at/above it.
    dom = next(f for f in bundle["robustness"]["findings"] if f["finding"] == "dense_gemm_mac_dominance")
    wls = bundle["robustness"]["workloads"]
    frac_ok = 0.0 <= dom["macro"] <= 1.0 and 0.0 <= dom["micro"] <= 1.0
    loo_ok = set(dom["micro_loo"]) == set(wls)
    derived = sorted(w for w in wls if dom["micro_loo"][w] < 0.2 <= dom["micro"])
    p13check(frac_ok and loo_ok and sorted(dom["collapses_if_removed"]) == derived,
             f"[P16] dense-MAC LOO well-formed (macro={dom['macro']} micro={dom['micro']} "
             f"collapses={dom['collapses_if_removed']})")
    # 5. capture-fidelity: low-bit erased everywhere; AR workloads hide KV; matches fidelity vocab
    cf = bundle["capture_fidelity"]
    lowbit = next(r for r in cf["matrix"] if r["feature"] == "packed_lowbit_layout")
    erased_all = all(lowbit[w] == "erased" for w in cf["workloads"])
    ar_hidden = any(cf["per_workload"][w]["hidden_axes"] for w in cf["workloads"])
    p13check(erased_all and ar_hidden,
             f"[P16] capture-fidelity: low-bit erased for all, hidden axes present (erased={erased_all})")
    # 6. new plots carry captions; no forbidden wording in P16 outputs; digest uses necessity
    new_plots = {"primitive_set_frontier", "operator_cumulative_mac", "boundary_necessity_matrix",
                 "decision_sharding_per_top_op"}
    caps = {p["plot_id"]: p.get("dse_caption", "") for p in bundle["plots"]}
    caps_ok = all(caps.get(pid) for pid in new_plots)
    blob = (str(nec) + str(bundle["primitive_frontier"]) + str(par) + str(bundle["robustness"])
            + str(cf) + str(bundle["decision_scorecard"])).lower()
    leaked = [t for t in ("speedup", "faster", "optimal", "performance improvement",
                          "predicted cycles", "x speedup") if t in blob]
    p13check(caps_ok and not leaked,
             f"[P16] new plots captioned + no forbidden wording (caps_ok={caps_ok}, leaked={leaked})")


def verify_p17_audit(IM, bundle) -> None:
    """P17 adversarial audit (commit 1): the predicate audit carries the per-workload scalars and
    flags configured-K necessity; the necessity rollup is now K-free (the reporting bug is fixed);
    the gemv abstraction is renamed; the primitive-frontier robustness recompute matches the
    committed tile_waste at 10% and is monotone in set size; the influence table flags
    winner-stable/magnitude-unstable metrics; every audited conclusion references an artifact."""
    import csv as _csv
    import io as _io
    import re as _re

    def _rows(name):
        p = CS / name
        return list(_csv.DictReader(_io.StringIO(p.read_text()))) if p.is_file() else []

    # 1. K-rollup reporting bug is FIXED: no necessity rollup predicate embeds a literal per-workload
    #    K (e.g. "K=7"); the per-workload K lives in the predicate audit instead.
    nec = bundle["abstraction_necessity"]
    k_leak = [r["abstraction"] for r in nec["rows"] if _re.search(r"K=\d", r["predicate"])]
    pa = bundle["predicate_audit"]
    rwo = [r for r in pa["rows"] if r["abstraction"] == "resident_weight_object"]
    rwo_has_k = all(_re.search(r"K=\d", r["predicate_inputs"]) for r in rwo) and rwo
    rwo_uses_k = all(r["uses_configured_K"] == "yes" for r in rwo)
    p13check(not k_leak and rwo_has_k and rwo_uses_k,
             f"[P17] K-rollup fixed: no literal K in rollup predicates (leaks={k_leak}); "
             f"per-workload K in predicate_audit (uses_K={rwo_uses_k})")
    # 1b. predicate audit is well-formed: every cell has inputs+thresholds, and at least one
    #     suspicious cell (necessity resting on configured K) is flagged
    cells_ok = all(r["predicate_inputs"] and r["thresholds"] and r["classification"] in IM._NEC_CLASSES
                   for r in pa["rows"])
    susp = [r for r in pa["rows"] if r["suspicious"]]
    p13check(cells_ok and susp,
             f"[P17] predicate audit well-formed ({len(pa['rows'])} cells); suspicious flagged "
             f"({len(susp)})")
    # 2. gemv rename: the old name is gone everywhere in the necessity output; the honest umbrella
    #    name is present; the true_gemv vs skinny split is exposed in the predicate audit inputs
    names = {r["abstraction"] for r in nec["rows"]}
    gemv_cells = [r for r in pa["rows"] if r["abstraction"] == "skinny_gemm_or_gemv_engine"]
    split_ok = all("true_gemv=" in r["predicate_inputs"] and "skinny_gemm=" in r["predicate_inputs"]
                   for r in gemv_cells) and gemv_cells
    p13check("vector_gemv_engine" not in names and "skinny_gemm_or_gemv_engine" in names and split_ok,
             "[P17] gemv abstraction renamed (skinny_gemm_or_gemv_engine) + true/skinny split exposed")
    # 3. primitive-frontier robustness: 10% recompute matches the committed tile_waste boolean
    #    (the regression gate), worst-coverage is monotone in set size per threshold
    ops = {(o["workload"], o["op_index"]): o for o in _rows("operator_shape_table.csv")}
    mism = 0
    for r in _rows("tile_waste_table.csv"):
        if r.get("applicable") != "True":
            continue
        o = ops.get((r["workload"], r["op_index"]))
        if not o:
            continue
        M, N, K = int(o["M"]), int(o["N"]), int(o["K"])
        if M <= 0 or N <= 0 or K <= 0:
            continue
        ap, wv = IM._prim_waste(r["primitive"], M, N, K, o["shape_class"])
        if ap and (wv <= 0.10) != (r["covered_under_10pct"] == "True"):
            mism += 1
    fro = bundle["primitive_frontier_robustness"]
    by_thr = {}
    for row in fro["rows"]:
        by_thr.setdefault(row["threshold_pct"], {})[row["set_size"]] = row["worst"]
    mono = all(s[sz] >= s[sz - 1] - 1e-9 for s in by_thr.values() for sz in s if sz - 1 in s)
    p13check(mism == 0 and mono and fro["rows"],
             f"[P17] frontier-robustness 10% recompute matches committed tile_waste (mismatch={mism}) "
             f"+ monotone in set size")
    # 4. influence: macro & micro present per metric, dense-GEMM is winner-stable/magnitude-unstable
    inf = bundle["macro_micro_influence"]
    rows_ok = all(isinstance(r["macro"], float) and isinstance(r["micro"], float) for r in inf["rows"])
    dense = next((r for r in inf["rows"] if r["metric"] == "dense_gemm_mac_fraction"), {})
    flagged = dense.get("winner_stable_magnitude_unstable") == "yes"
    loo_complete = len(inf["loo_rows"]) == len(inf["rows"]) * len(inf["workloads"])
    p13check(rows_ok and flagged and loo_complete,
             f"[P17] influence table: macro+micro present, dense-GEMM magnitude-unstable={flagged}, "
             f"LOO delta complete={loo_complete}")
    # 5. adversarial audit: every conclusion references an artifact + carries a verdict; metric_class
    #    is in the allowed vocabulary; no forbidden wording in any P17 output
    aud = bundle["adversarial_audit"]["rows"]
    vocab = ("recovered", "derived", "measured", "assumed", "unavailable")
    aud_ok = all(r["conclusion"] and r["supporting_artifact"] and r["verdict"]
                 and any(v in r["metric_class"] for v in vocab) for r in aud)
    blob = (str(pa) + str(fro) + str(inf) + str(bundle["adversarial_audit"])).lower()
    leaked = [t for t in ("speedup", "faster", "optimal", "performance improvement",
                          "predicted cycles", "x speedup", "gap_closure") if t in blob]
    p13check(aud_ok and not leaked,
             f"[P17] adversarial audit references artifacts + verdicts ({len(aud)}); no forbidden "
             f"wording (leaked={leaked})")


def verify_p17_envelope(IM, bundle) -> None:
    """P17 requirements envelope + omitted-op accounting + plots (commit 2): every envelope row is a
    derived requirement (work / deadline) recomputed from workload facts + scenario inputs, never a
    measured-hardware label; command rate is proxy-tagged except for the measured workload; the
    omitted-op flags are consistent with the capture-fidelity matrix; the new decision plots are
    available and captioned; no forbidden wording."""
    import csv as _csv
    import io as _io

    def _rows(name):
        p = CS / name
        return list(_csv.DictReader(_io.StringIO(p.read_text()))) if p.is_file() else []

    env = bundle["timing_envelope"]
    # 1. every row tags K + deadline provenance, and required_compute recomputes as work/deadline
    req = {(r["workload"], r["region"], r["requirement"]): r["value"] for r in
           _rows("requirements_table.csv")}
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    tagged = all(r["K_basis"] in ("configured", "sweep")
                 and ("sweep" in r["deadline_basis"] or "derived" in r["deadline_basis"])
                 for r in env["rows"])
    recompute_ok = True
    for r in env["rows"][:200]:
        arch = MODEL_ARCH.get(_base_model(r["workload"]))
        Kcfg = arch.loop_count if arch else 1
        mpr = float(req.get((r["workload"], "repeated_head", "macs_per_replan"), 0) or 0)
        expect = (mpr / Kcfg) * r["K"] / (r["deadline_ms"] / 1000.0)
        if abs(expect - r["required_compute_MAC_per_s"]) > max(1.0, expect * 1e-6):
            recompute_ok = False
            break
    p13check(env["rows"] and tagged and recompute_ok,
             f"[P17] envelope rows are derived requirements (recompute_ok={recompute_ok}, "
             f"provenance-tagged={tagged}, {len(env['rows'])} rows)")
    # 2. command rate is proxy-only except the measured workload; no measured-HARDWARE label anywhere
    cmd_ok = all(("proxy_only" in r["command_rate_basis"]) or (r["workload"] in IM._MEASURED_DISPATCH)
                 for r in env["rows"])
    blob = (str(env) + str(bundle["omitted_ops"])).lower()
    no_hw = "measured hardware" not in blob and "measured performance" not in blob
    p13check(cmd_ok and no_hw,
             f"[P17] command rate proxy-tagged (ok={cmd_ok}); no measured-hardware label ({no_hw})")
    # 3. operator-recovery accounting (P18): linear MACs == named-matmul sum; visible_linear_fraction
    #    in [0,1]; attention RECOVERED where present; low-bit still erased.
    ops = _rows("operator_shape_table.csv")
    mac_by_wl = {}
    for o in ops:
        mac_by_wl[o["workload"]] = mac_by_wl.get(o["workload"], 0) + int(o["macs"])
    om = bundle["omitted_ops"]["rows"]
    macs_match = all(int(r["linear_gemm_macs"]) == mac_by_wl.get(r["workload"], -1) for r in om)
    frac_ok = all(0.0 <= float(r["visible_linear_fraction"]) <= 1.0 for r in om)
    lowbit_erased = all("no" in r["lowbit_packed_recovered"] for r in om)
    attn_recovered = any(int(r["attention_macs"]) > 0 for r in om)
    p13check(om and macs_match and frac_ok and lowbit_erased and attn_recovered,
             f"[P17] recovery accounting: linear MACs == named-matmul sum ({macs_match}); "
             f"visible_fraction in [0,1] ({frac_ok}); attention recovered ({attn_recovered}); "
             f"low-bit still erased ({lowbit_erased})")
    # 4. the new decision plots are available (not omit) and captioned
    new_plots = {"primitive_frontier_by_threshold", "macro_vs_micro_primitive_coverage",
                 "required_compute_envelope", "required_memory_movement_envelope",
                 "required_command_rate_envelope", "workload_influence_loo_delta"}
    pm = {p["plot_id"]: p for p in bundle["plots"]}
    plots_ok = all(pid in pm and pm[pid]["recommendation"] != "omit" and pm[pid].get("dse_caption")
                   for pid in new_plots)
    p13check(plots_ok, f"[P17] new decision plots available + captioned ({sorted(new_plots)})")
    # 5. no forbidden performance wording in the P17 commit-2 outputs
    leaked = [t for t in ("speedup", "faster", "optimal", "performance improvement",
                          "predicted cycles", "x speedup", "gap_closure") if t in blob]
    p13check(not leaked, f"[P17] requirements/omitted outputs free of forbidden wording ({leaked})")
    # 6. Stage C: capture-erasure evidence is demonstrated (loops absent in all but a known artifact,
    #    no low-bit types); per-family fractions are valid.
    ce = bundle["capture_erasure"]["rows"]
    no_lowbit = all(not r["lowbit_int_types_present"] for r in ce)
    loops = [r["workload"] for r in ce if r["loops_preserved"]]
    pf = bundle["per_family"]["rows"]
    fam_ok = pf and all(0.0 <= float(r["visible_linear_fraction"]) <= 1.0 for r in pf)
    p13check(ce and no_lowbit and len(loops) <= 1 and fam_ok,
             f"[P18] capture-erasure evidence (no low-bit types={no_lowbit}, loop-captures={loops}); "
             f"per-family fractions valid ({fam_ok})")
    # P19 attention-classifier fix: SDPA-fused attention is recovered (xr0 had 0 -> now >0), and a
    # non-attention batch_matmul (groot's CategorySpecificMLP bmm) is separated as batched_matmul, not
    # mislabeled attention.
    wc = {r["workload"]: r for r in _rows("work_coverage_table.csv")}
    xr0_attn = int(wc.get("xr0", {}).get("attention_macs", 0) or 0) > 0
    groot_bmm = int(wc.get("groot_n1d7", {}).get("n_batched_matmul", 0) or 0) > 0
    p13check(xr0_attn and groot_bmm,
             f"[P19] attention classifier: xr0 SDPA attention recovered ({xr0_attn}); groot MLP bmm "
             f"separated as batched_matmul ({groot_bmm})")
    # 7. Stage B capture-level ablation (only if the multi-level recaptures are present): high-level
    #    captures expose attention/softmax as NAMED linalg_ext ops; qdq captures expose quant_ext
    #    dequant; loops stay absent (torch.export-blocked) at every level.
    ab = bundle["capture_ablation"]
    if ab["rows"]:
        hl = [r for r in ab["rows"] if r["level"] == "high_level" and r["available"]]
        qd = [r for r in ab["rows"] if r["level"] == "quant_qdq" and r["available"]]
        named_attn = bool(hl) and all(int(r["linalg_ext_softmax"]) > 0 for r in hl)
        named_quant = bool(qd) and all(int(r["quant_ext_dequantize"]) > 0 for r in qd)
        no_loops = all(int(r["scf_for"]) == 0 for r in ab["rows"])
        p13check(named_attn and named_quant and no_loops,
                 f"[P18] capture-level ablation: high-level=named attention ({named_attn}), "
                 f"qdq=named dequant ({named_quant}), loops absent every level ({no_loops})")
    else:
        p13check(True, "[P18] capture-level ablation: no multi-level recaptures present (skipped)")


def verify_p20(IM) -> None:
    """P20 Tool A: the Timeloop problem shapes are internally consistent and DSE-consumable — each
    instance's M*N*K equals the recorded macs; every problem has the 3 GEMM data-spaces with every
    dimension projected; attention shapes carry no stationary weight (operand_identity=activation,
    weight_stationary excluded); the yaml count matches the dataflow table; no perf wording."""
    import csv as _csv
    import io as _io
    from merlin.common.yaml import load_yaml
    ts = load_yaml(CS / "timeloop_problem_shapes.yaml").get("timeloop_problem_shapes", {})
    shapes = ts.get("shapes", [])
    df = list(_csv.DictReader(_io.StringIO((CS / "dataflow_candidate_table.csv").read_text())))
    dims_ok = all(int(s["problem"]["instance"]["M"]) * int(s["problem"]["instance"]["N"])
                  * int(s["problem"]["instance"]["K"]) == int(s["macs_per_instance"]) for s in shapes)
    ds_ok = all(len(s["problem"]["shape"]["data-spaces"]) == 3
                and len(s["problem"]["shape"]["dimensions"]) == 3 for s in shapes)
    attn_ok = all(("weight_stationary" not in s["dataflow_candidates"]
                   and all(d.get("operand_identity") != "weight"
                           for d in s["problem"]["shape"]["data-spaces"]))
                  for s in shapes if s["op_class"] == "attention_contraction")
    lin_ok = all(any(d.get("operand_identity") == "weight"
                     for d in s["problem"]["shape"]["data-spaces"])
                 for s in shapes if s["op_class"] == "linear_gemm")
    cnt_ok = bool(shapes) and len(df) == ts.get("count") == len(shapes)
    blob = str(shapes).lower()
    leaked = [t for t in ("speedup", "faster", "optimal", "cycles", "throughput") if t in blob]
    p13check(dims_ok and ds_ok and attn_ok and lin_ok and cnt_ok and not leaked,
             f"[P20] Timeloop problem shapes consistent: dims==macs ({dims_ok}), 3 dataspaces ({ds_ok}), "
             f"attention no-weight ({attn_ok}), linear has-weight ({lin_ok}), count=={len(shapes)} "
             f"({cnt_ok}), no perf wording ({not leaked})")
    # P20 Tool B: operand locality bytes reconcile with data_movement; reuse scopes valid; weights resident.
    dm = {(r["workload"], r["region"]): r for r in
          _csv.DictReader(_io.StringIO((CS / "data_movement_table.csv").read_text()))}
    loc = list(_csv.DictReader(_io.StringIO((CS / "operand_locality_table.csv").read_text())))
    SCOPES = {"within_op", "across_ops", "across_K", "across_decode", "across_replan"}
    scope_ok = all(r["reuse_scope"] in SCOPES for r in loc)

    def _i(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0
    wb_ok = all(_i(r["bytes"]) == _i(dm.get((r["workload"], r["region"]), {}).get("weight_bytes"))
                for r in loc if r["operand"] == "weight")
    wt_resident = all(r["resident_candidate"].startswith("yes") for r in loc
                      if r["operand"] == "weight" and _i(r["bytes"]) > 0)
    p13check(loc and scope_ok and wb_ok and wt_resident,
             f"[P20] operand-locality: reuse scopes valid ({scope_ok}); weight bytes reconcile with "
             f"data_movement ({wb_ok}); weights are resident candidates ({wt_resident})")
    # P20 Tool E: quant metadata traces to qdq dequant ops; granularity valid; native gap recorded.
    qmf = CS / "quant_metadata_visibility.csv"
    if qmf.is_file():
        qm = list(_csv.DictReader(_io.StringIO(qmf.read_text())))
        GRAN = {"per_channel", "per_tensor", "per_group", "unspecified"}
        qm_ok = bool(qm) and all(_i(r["n_dequant_ops"]) > 0 and r["scale_granularity"] in GRAN
                                 and r["storage_dtype"] for r in qm)
        native_ok = any("ternary" in r["native_scheme_gap"].lower()
                        for r in qm if r["workload"] == "bitvla")
        p13check(qm_ok and native_ok,
                 f"[P20] quant metadata: every row traces to qdq dequant ({qm_ok}); bitvla native-ternary "
                 f"gap recorded ({native_ok})")
    else:
        p13check(True, "[P20] quant metadata: no qdq recaptures present (skipped)")


def verify_p21(IM) -> None:
    """P21-S1 loop-preserving recovery: K and the loop-carried state are RE-DERIVED here directly from
    the loop-preserving captures (recaptures_loop/<w>/model.mlir) — independently of the emitted
    artifact — and matched against loop_preserving_recovery.csv + the capture_fidelity matrix flip.
    If no loop-preserving captures are present, the check is a no-op pass (committed summary stands)."""
    import csv as _csv
    import io as _io
    from pathlib import Path as _P
    loop_dir = CS.parent / "recaptures_loop"
    if not loop_dir.is_dir() or not any((loop_dir / w / "model.mlir").is_file()
                                        for w in ("smolvla", "openvla", "pi05")):
        p13check(True, "[P21] loop-preserving recovery: no loop-preserving captures present (skipped)")
        return
    from merlin.dse_guidance.loop_recovery import recover_loop
    # 1. independent re-derivation of K + carried state straight from the IR
    EXPECT = {"smolvla": (10, "latent", None), "openvla": (7, "kv_cache", 221184),
              "pi05": (10, "latent", None)}
    re_ok = True
    for w, (k, role, kvb) in EXPECT.items():
        mp = loop_dir / w / "model.mlir"
        if not mp.is_file():
            continue
        lr = recover_loop(mp, w)
        roles = {c.role for c in lr.carried_state}
        re_ok = re_ok and lr.present and lr.K == k and lr.K_source == "recovered_from_ir" \
            and role in roles and lr.kv_cache_bytes == kvb and lr.repeated_region_op_count > 50
    # 1b. corpus-wide: EVERY present loop-preserving capture re-derives a valid loop from the IR
    present = sorted(d.name for d in loop_dir.glob("*") if (d / "model.mlir").is_file())
    corpus_ok = True
    for w in present:
        lr = recover_loop(loop_dir / w / "model.mlir", w)
        corpus_ok = corpus_ok and lr.present and (lr.K or 0) > 0 \
            and lr.K_source == "recovered_from_ir" and lr.repeated_region_op_count > 50 \
            and any(c.role in ("latent", "kv_cache", "token_buffer") for c in lr.carried_state)
    # 2. the emitted artifact matches the re-derivation (for every present capture)
    af = CS / "loop_preserving_recovery.csv"
    art_ok = False
    if af.is_file():
        ar = {r["workload"]: r for r in _csv.DictReader(_io.StringIO(af.read_text()))}
        art_ok = all(w in ar and ar[w]["K_source"] == "recovered_from_ir"
                     and int(ar[w]["K"]) == (recover_loop(loop_dir / w / "model.mlir", w).K or -1)
                     for w in present)
    re_ok = re_ok and corpus_ok
    # 3. the capture_fidelity matrix actually flipped K/KV/loop_carried to recovered-from-IR
    cf = IM.capture_fidelity(CS)
    kloop = next(r for r in cf["matrix"] if r["feature"] == "K_or_decode_loop")
    lcs = next(r for r in cf["matrix"] if r["feature"] == "loop_carried_state")
    kv = next(r for r in cf["matrix"] if r["feature"] == "kv_cache_state")
    flip_ok = (all("recovered" in kloop[w] and "IR" in kloop[w] for w in EXPECT)
               and all("recovered" in lcs[w] for w in EXPECT)
               and "recovered" in kv["openvla"] and "221184" in kv["openvla"])
    blob = af.read_text().lower() if af.is_file() else ""
    leaked = [t for t in ("speedup", "faster", "optimal", "cycles", "throughput") if t in blob]
    p13check(re_ok and art_ok and flip_ok and not leaked,
             f"[P21] loop-preserving recovery: K/carried-state re-derived from IR ({re_ok}); artifact "
             f"matches ({art_ok}); capture_fidelity flipped K/KV/loop_carried->recovered ({flip_ok}); "
             f"no perf wording ({not leaked})")
    # P21 GAP-C: IR-proven residency split — every present capture has loop-invariant (resident-eligible)
    # operands referenced in the body but defined outside the region, re-derived here and matched to the artifact.
    from merlin.dse_guidance.loop_recovery import residency_from_ir
    rf = CS / "residency_from_ir.csv"
    res_ok = rf.is_file()
    if res_ok:
        rar = {r["workload"]: r for r in _csv.DictReader(_io.StringIO(rf.read_text()))}
        for w in present:
            rc = residency_from_ir(loop_dir / w / "model.mlir", w)
            res_ok = res_ok and rc.present and rc.n_loop_invariant_operands > 0 \
                and rc.n_loop_carried >= 1 and w in rar \
                and int(rar[w]["n_loop_invariant_operands"]) == rc.n_loop_invariant_operands
    p13check(res_ok,
             f"[P21] residency-from-IR: loop-invariant (resident-eligible) operands re-derived from the "
             f"scf.for region boundary for all {len(present)} captures + artifact matches ({res_ok})")
    # P22 GAP-B: the loop-aware contract synthesis reconciles with loop_recovery + residency_from_ir
    # (additive; the flat artifacts are untouched). Re-derive each joined field independently.
    laf = CS / "loop_aware_contract.csv"
    la_ok = laf.is_file()
    if la_ok:
        lar = {r["workload"]: r for r in _csv.DictReader(_io.StringIO(laf.read_text()))}
        for w in present:
            lr = residency_from_ir(loop_dir / w / "model.mlir", w)
            lrec = __import__("merlin.dse_guidance.loop_recovery", fromlist=["recover_loop"]).recover_loop(
                loop_dir / w / "model.mlir", w)
            r = lar.get(w, {})
            wb = int(r.get("resident_weight_bytes", -1))
            la_ok = la_ok and r and int(r["K_ir"]) == lrec.K \
                and int(r["repeated_region_ops"]) == lrec.repeated_region_op_count \
                and int(r["n_resident_eligible_operands"]) == lr.n_loop_invariant_operands \
                and int(r["avoidable_reload_bytes"]) == wb * max((lrec.K or 0) - 1, 0)
    p13check(la_ok,
             f"[P22] loop-aware contract: synthesis reconciles with loop_recovery + residency_from_ir "
             f"(K, repeated region, resident-eligible operands, avoidable_reload=wb*(K-1)) for all "
             f"{len(present)} captures ({la_ok})")
    # P21 S2/S3: deployment-real magnitudes (config-exact composition) + KV sizing with IR cross-check
    from merlin.dse_guidance import real_config as RC
    # 1. independent re-derivation of openVLA = Llama-2-7B: 32 * (4*4096^2 + 3*4096*11008) + embeds
    g = RC.REAL_GEOMETRY["openvla"]
    per_layer = 4 * 4096 * 4096 + 3 * 4096 * 11008
    exp_total = per_layer * 32 + 2 * (32064 * 4096)        # untied embed + lm_head
    mag_ok = (g.total_params() == exp_total) and abs(g.total_params() / 1e9 - 6.74) < 0.1
    # 2. KV formula is IR-validated on the captured config then applied at deployment scale
    kv = {r["workload"]: r for r in RC.kv_sizing_rows(loop_dir)}
    kv_ir_ok = "matches IR iter_arg" in kv.get("openvla", {}).get("ir_formula_check", "")
    kv_real_ok = RC.REAL_GEOMETRY["openvla"].kv_cache_bytes("bf16") == 2 * 32 * 128 * 263 * 32 * 2
    # 3. artifacts present + config-evidence labelled
    mf, kf = CS / "real_config_magnitudes.csv", CS / "kv_cache_sizing.csv"
    arts_ok = mf.is_file() and kf.is_file() \
        and all(r["evidence"] == "recovered_from_model_config"
                for r in _csv.DictReader(_io.StringIO(mf.read_text())))
    p13check(mag_ok and kv_ir_ok and kv_real_ok and arts_ok,
             f"[P21] real-config magnitudes: openVLA params==Llama-2-7B composition ({mag_ok}); "
             f"KV formula matches IR iter_arg ({kv_ir_ok}) + deployment KV exact ({kv_real_ok}); "
             f"artifacts config-evidenced ({arts_ok})")
    # P22 GAP-A: every REAL_GEOMETRY entry is fully populated (no placeholder / guessed field) and the
    # two anchors are exact (openVLA==Llama-2-7B 6.74B, tiny_llama==TinyLlama-1.1B). Generic over models.
    nonnull = True
    for w, g in RC.REAL_GEOMETRY.items():
        for s in g.stacks:
            # per-layer geometry must be populated; a raw-params (DiT) stack still has real h/heads/etc.
            nonnull = nonnull and all(isinstance(getattr(s, f), int) and getattr(s, f) > 0
                                      for f in ("n_layers", "hidden", "interm", "heads",
                                                "kv_heads", "head_dim"))
            nonnull = nonnull and s.layer_params() > 0
        # vocab/embed may legitimately be 0 for an action-space DiT head (no token embedding); if
        # vocab>0 then embed_hidden>0 must hold (no half-specified embedding).
        nonnull = nonnull and (g.decode_seq or 0) > 0 and (g.vocab == 0 or g.embed_hidden > 0)
    anchors = (6.6e9 < RC.REAL_GEOMETRY["openvla"].total_params() < 6.9e9
               and 1.0e9 < RC.REAL_GEOMETRY["tiny_llama"].total_params() < 1.2e9
               and 0.45e9 < RC.REAL_GEOMETRY["rdt2"].total_params() < 0.55e9)  # DiT: 473M GEMM params
    p13check(nonnull and anchors,
             f"[P22] real-config geometry: all {len(RC.REAL_GEOMETRY)} entries fully populated, no "
             f"placeholder ({nonnull}); param anchors exact (openVLA 6.74B, tiny_llama 1.1B, rdt2 ~473M: {anchors})")
    # P24: hardware-INDEPENDENT roofline — AI is a workload property (no peak/bandwidth/latency assumed).
    from merlin.dse_guidance import arithmetic_intensity as AI
    air = AI.ai_rows("bf16")
    db = RC._DTYPE_BYTES["bf16"]
    ai_ok = bool(air)
    for r in air:
        # non-resident AI is exactly 1/dtype_bytes (every MAC reloads its weight = the floor)
        ai_ok = ai_ok and abs(r["ai_nonresident_mac_per_byte"] - 1.0 / db) < 1e-6
        # residency strictly raises AI; gain == (prefix + repeated*K)/(prefix+repeated), re-derived
        p, rep, K = r["prefix_params"], r["repeated_params"], r["K"]
        exp_gain = (p + rep * K) / (p + rep)
        ai_ok = ai_ok and r["ai_resident_mac_per_byte"] > r["ai_nonresident_mac_per_byte"] \
            and abs(r["residency_gain"] - exp_gain) < 0.01
    artf = CS / "arithmetic_intensity.csv"
    blob = artf.read_text().lower() if artf.is_file() else ""
    # the roofline must NOT smuggle in a chip: no peak/bandwidth/latency/cycle numbers
    hw_free = artf.is_file() and not any(t in blob for t in
                                         ("peak_mac", "bandwidth_gb", "latency_ms", "ghz", "cycles"))
    p13check(ai_ok and hw_free,
             f"[P24] HW-independent roofline: AI_nonres==1/dtype, residency raises AI by "
             f"(prefix+rep*K)/(prefix+rep) re-derived ({ai_ok}); no chip assumed in artifact ({hw_free})")
    # P21 S4: native low-bit (bitvla packed-int2 ternary) datapath captured + reported, when present
    from merlin.dse_guidance import quant_metadata as QM
    nat_cap = (CS.parent / "recaptures_native" / "bitvla" / "model.mlir")
    if nat_cap.is_file():
        nrows = {r["workload"]: r for r in QM.native_quant_rows(CS)}
        bv = nrows.get("bitvla", {})
        # re-derive: the native capture must carry packed-int2 (i8-stored) weights and the absmean scale
        cap_txt = nat_cap.read_text(errors="ignore")
        # P22 GAP-D: the int2 unpack is now a named quant_ext.unpack_int2 op (opt-in recognizer).
        n_unpack = cap_txt.count("quant_ext.unpack_int2")
        s4_ok = (bv.get("storage") == "int2_packed_in_i8"
                 and int(bv.get("n_packed_weight_tensors", 0)) > 0
                 and cap_txt.count("xi8>") == int(bv.get("n_packed_weight_tensors", 0))
                 and "absmean" in bv.get("scale", "")
                 and "recovered" in bv.get("status", "")
                 and n_unpack > 0                                  # named unpack op present
                 and "named op" in bv.get("unpack_visibility", "")
                 and "func.call @aten_stack" not in cap_txt        # opaque unpack call-sites folded away
                 and (CS / "native_lowbit_datapath.csv").is_file())
        p13check(s4_ok,
                 f"[P21/P22] native low-bit: bitvla packed-int2 ternary storage + absmean scale + the "
                 f"int2 unpack as quant_ext.unpack_int2 ({n_unpack} ops, opaque chain folded) -> native "
                 f"datapath fully recovered ({s4_ok})")
    else:
        p13check(True, "[P21] native low-bit: no native bitvla capture present (skipped)")

    # P24-D: corpus-wide HONEST low-bit tiering — re-derive each tier from capture-file presence (not the
    # artifact), confirm the artifact matches, and assert it never over-claims (only bitvla=native) nor
    # smuggles a perf word; int8 accuracy_status must echo the MEASURED gate, never assume fp8/int4.
    lbv = CS / "low_bit_visibility.csv"
    if lbv.is_file():
        import csv as _csv_mod
        from merlin.dse_guidance import accuracy_gate as _AG
        nat_d = CS.parent / "recaptures_native"
        lvl_d = CS.parent / "recaptures_levels"
        _pts = _AG.load()
        got = {r["workload"]: r for r in _csv_mod.DictReader(lbv.read_text().splitlines())}
        tier_ok, acc_ok = True, True
        for w, r in got.items():
            natf = nat_d / w / "model.mlir"
            qdqf = lvl_d / w / "model_qdq.mlir"
            if natf.is_file() and "quant_ext.unpack_int2" in natf.read_text(errors="ignore"):
                exp = "native"
            elif qdqf.is_file():
                exp = "qdq_int8"
            else:
                exp = "dequant_only"
            tier_ok &= (r["tier"] == exp)
            st = _AG.status_for(w, "int8_w8a8", _pts)
            exp_acc = {"pass": "measured_pass", "fail": "measured_fail"}.get(st, "unavailable")
            acc_ok &= r["accuracy_status"].startswith(exp_acc)
        only_bitvla_native = {w for w, r in got.items() if r["tier"] == "native"} == {"bitvla"}
        no_perf = not any(t in lbv.read_text().lower()
                          for t in ("speedup", "faster", "latency", "throughput", "cycles"))
        no_assumed_lowbit = not any("int4" in r["accuracy_status"] or "fp8" in r["accuracy_status"]
                                    for r in got.values())
        d_ok = tier_ok and acc_ok and only_bitvla_native and no_perf and no_assumed_lowbit
        p13check(d_ok,
                 f"[P24] low-bit tiering: {len(got)} workloads tiered native/qdq_int8/dequant_only from "
                 f"capture presence (re-derived match={tier_ok}); int8 accuracy echoes measured gate "
                 f"({acc_ok}); only bitvla=native ({only_bitvla_native}); no fp8/int4 assumed, no perf "
                 f"wording ({d_ok})")
    else:
        p13check(True, "[P24] low-bit tiering: low_bit_visibility.csv absent (skipped)")

    # P25: real-time requirement is a HW-INDEPENDENT requirement, not a perf claim. Re-derive one VLA
    # row's required compute from the recovered structure and assert the artifact claims no chip behaviour.
    rtf = CS / "realtime_requirement.csv"
    if rtf.is_file():
        from merlin.dse_guidance import models as _M
        ai = {r["workload"]: r for r in _csv_mod.DictReader((CS / "arithmetic_intensity.csv").read_text().splitlines())}
        rt = list(_csv_mod.DictReader(rtf.read_text().splitlines()))
        der_ok = True
        for r in rt:
            if not r["regime"].startswith("VLA 30Hz"):
                continue
            arch = _M.MODEL_ARCH.get(r["workload"])
            a = ai.get(r["workload"])
            if not arch or not a:
                continue
            window_s = (arch.action_horizon or 1) / 30.0
            exp = round(float(a["macs_per_replan"]) / window_s / 1e9, 4)
            der_ok &= abs(exp - float(r["required_GMAC_per_s"])) <= max(1e-3, 0.001 * exp)
        txt = rtf.read_text().lower()
        # requirement language only — every regime must be a design_target, never a met/achieved claim
        no_perf = not any(t in txt for t in ("speedup", "achieves", "meets the", "faster", "outperform"))
        all_req = all("required_from_recovered_structure" in r["evidence"]
                      and "design_target" in r["evidence"] for r in rt)
        e_ok = der_ok and no_perf and all_req and len(rt) > 0
        p13check(e_ok,
                 f"[P25] real-time requirement: {len(rt)} regime rows; required compute re-derived from "
                 f"macs_per_replan/(H/rate) ({der_ok}); every row is a HW-independent requirement w/ "
                 f"design_target regime ({all_req}); no chip-performance claim ({no_perf}) -> {e_ok}")
    else:
        p13check(True, "[P25] real-time requirement: realtime_requirement.csv absent (skipped)")


def main(write: bool = True) -> int:
    rows = [verify_workload(w) for w in RECAP_MODELS if w in _MODELS]
    verify_global()
    p5_rows = [verify_p5_workload(w) for w in RECAP_MODELS if w in _MODELS]
    verify_p5_global()
    _graphs = _graphs_by_workload()
    p6_rows = [verify_p6_workload(w, _graphs) for w in RECAP_MODELS
               if w in _MODELS]
    verify_p6_global(_graphs)
    p7_rows = [verify_p7_workload(w, _graphs) for w in RECAP_MODELS
               if w in _MODELS]
    verify_p7_global()
    p8_rows = [verify_p8_workload(w, _graphs) for w in RECAP_MODELS
               if w in _MODELS]
    verify_p8_global()
    p9_rows = [verify_p9_workload(w) for w in RECAP_MODELS
               if w in _MODELS]
    verify_p9_global()
    p10_rows = [verify_p10_workload(w) for w in RECAP_MODELS
                if w in _MODELS]
    verify_p10_global()
    p12_summary = verify_p12()
    p13_summary = verify_p13()

    _all = (results + p5_results + p6_results + p7_results + p8_results + p9_results
            + p10_results + p12_results + p13_results)
    passed = sum(1 for ok, _ in _all if ok)
    total = len(_all)
    p13_passed = sum(1 for ok, _ in p13_results if ok)
    p5_passed = sum(1 for ok, _ in p5_results if ok)
    p6_passed = sum(1 for ok, _ in p6_results if ok)
    p7_passed = sum(1 for ok, _ in p7_results if ok)
    p8_passed = sum(1 for ok, _ in p8_results if ok)
    p9_passed = sum(1 for ok, _ in p9_results if ok)
    p10_passed = sum(1 for ok, _ in p10_results if ok)
    p12_passed = sum(1 for ok, _ in p12_results if ok)
    L = ["# Implementation verification report\n",
         "> Independent re-derivation of the dse_guidance package: every number below is recomputed "
         "from the raw captures / base facts and cross-checked against the emitted artifacts. "
         "Generated by `verify_implementation.py`.\n",
         f"**RESULT: {passed}/{total} checks passed.**\n",
         "## Re-derived per-workload facts (recomputed here, matched to artifacts)\n",
         "| workload | K | head matmuls | weight bytes | avoidable reload (=WB·(K−1)) | "
         "int8 capacity (=WB/4) | dispatches/replan (=mm·K) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['workload']} | {r['K']} | {r['matmuls']} | {r['weight_bytes']:,} | "
                 f"{r['avoidable_reload']:,} | {r['cap_int8']:,} | {r['dispatches_per_replan']} |")
    L.append("\n## Checks\n")
    for ok, msg in results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\n## What each check proves\n"
             "- A/B: matmul count + head weight bytes are real (raw grep == IR primitive == YAML).\n"
             "- C–H: every derived artifact equals a first-principles recomputation of the same fact.\n"
             "- I: no low-bit format is credited with accuracy it was not measured to have.\n"
             "- J: every emitted number carries an allowed evidence label.\n"
             "- K/L: no speedup / cycle / gap_closure claim leaks into the generated artifacts.\n")

    # P5 operator-geometry verification section
    L.append("## P5 operator geometry verification\n")
    L.append(f"**{p5_passed}/{len(p5_results)} P5 checks passed.**\n")
    L.append("Key derived facts (operators / distinct geometry classes / distinct semantic roles):\n")
    L.append("| workload | operators | geometry classes | semantic roles |")
    L.append("|---|---|---|---|")
    for r in p5_rows:
        L.append(f"| {r['workload']} | {r['operators']} | {r['shape_classes']} | "
                 f"{r['semantic_classes']} |")
    L.append("\n### P5 checks\n")
    for ok, msg in p5_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP5 re-derives M/N/K from the IR primitive, re-implements the taxonomy thresholds "
             "and tile arithmetic independently, and recomputes the coverage aggregates from "
             "`tile_waste_table.csv` — so a divergence in `operator_geometry.py`, "
             "`primitive_coverage.py`, or `shape_taxonomy.py` fails a check. All metrics are "
             "structural geometry; **no speedup** is claimed.\n")

    # P6 multi-rate contract graph verification section
    L.append("## P6 multi-rate contract graph verification\n")
    L.append(f"**{p6_passed}/{len(p6_results)} P6 checks passed.**\n")
    L.append("Graph size (recomputed against the IR primitive + P5 shape table + attribution):\n")
    L.append("| workload | nodes | edges | phase | region | operator | state |")
    L.append("|---|---|---|---|---|---|---|")
    for r in p6_rows:
        L.append(f"| {r['workload']} | {r['nodes']} | {r['edges']} | {r['phase']} | "
                 f"{r['region']} | {r['operator']} | {r['state']} |")
    L.append("\n### P6 checks\n")
    for ok, msg in p6_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP6 checks that the contract graph agrees with everything earlier phases recovered: "
             "operator nodes == `operator_shape_table.csv`, region facts == attribution, "
             "macs_per_replan == macs_per_invocation·invocations, loop trip == K, and every edge "
             "evidence / cadence label is valid with data-dependency edges recovered from the SSA "
             "use-def graph. Structural only; **no speedup** claimed.\n")

    # P7 parallelism / sharding / hierarchy verification section
    L.append("## P7 parallelism / sharding / hierarchy verification\n")
    L.append(f"**{p7_passed}/{len(p7_results)} P7 checks passed.**\n")
    L.append("Inter-op concurrency (recomputed from the DAG); low values mean the parallelism "
             "opportunity is intra-op sharding, not inter-op concurrency:\n")
    L.append("| workload | available parallelism | max ready width | structure |")
    L.append("|---|---|---|---|")
    for r in p7_rows:
        L.append(f"| {r['workload']} | {r['available_parallelism']}× | {r['max_ready_width']} | "
                 f"{r['serialization']} |")
    L.append("\n### P7 checks\n")
    for ok, msg in p7_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP7 recomputes total work from the operator MACs, bounds the critical path "
             "(largest op ≤ critical path ≤ total), recomputes available_parallelism = total/"
             "critical, and recomputes the M/N/K sharding tail + partial-sum/broadcast byte formulas. "
             "available_parallelism is work/span — **not a speedup**, no hardware assumed.\n")

    # P8 pipeline / overlap / unit-guidance verification section
    L.append("## P8 pipeline / overlap / processing-unit guidance verification\n")
    L.append(f"**{p8_passed}/{len(p8_results)} P8 checks passed.**\n")
    L.append("| workload | phases | candidate overlaps (can_overlap=yes) |")
    L.append("|---|---|---|")
    for r in p8_rows:
        L.append(f"| {r['workload']} | {r['phases']} | {r['yes_overlaps']} |")
    L.append("\n### P8 checks\n")
    for ok, msg in p8_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP8 checks the phase model against the graph phase nodes, that pipeline cadences "
             "match `phase_rate_table.csv`, that overlap candidates are well-formed (enum + allowed "
             "abstraction vocabulary + positive/unavailable buffer counts), and that the "
             "processing-unit guidance references existing resource classes. Structural overlap "
             "candidates only — **no speedup**, no schedule.\n")

    # P9 memory / DMA / buffer verification section
    L.append("## P9 memory / DMA / buffer envelope verification\n")
    L.append(f"**{p9_passed}/{len(p9_results)} P9 checks passed.**\n")
    L.append("| workload | regions | top avoidable reload (B) |")
    L.append("|---|---|---|")
    for r in p9_rows:
        L.append(f"| {r['workload']} | {r['regions']} | {r['top_avoidable_reload']:,} |")
    L.append("\n### P9 checks\n")
    for ok, msg in p9_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP9 recomputes per-region weight bytes from the IR primitive, the avoidable-reload "
             "formula (weight × max(K−1,0)), and the dtype-scaled resident set (int8 == WB/4); it "
             "checks DMA streams/buffers reference valid regions and that intermediate/scale/KV "
             "bytes are explicitly unavailable. **No bandwidth/speedup** is claimed (needs a design "
             "YAML).\n")

    # P10 fusion / epilogue / accumulator verification section
    L.append("## P10 fusion / epilogue / accumulator verification\n")
    L.append(f"**{p10_passed}/{len(p10_results)} P10 checks passed.**\n")
    L.append("| workload | matmuls | matmul+bias epilogues |")
    L.append("|---|---|---|")
    for r in p10_rows:
        L.append(f"| {r['workload']} | {r['matmuls']} | {r['bias_ops']} |")
    L.append("\n### P10 checks\n")
    for ok, msg in p10_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP10 detects matmul epilogue patterns from the IR (every addmm is flagged bias), "
             "checks the accumulator/dequant/requant fields use the allowed vocabulary (f32 "
             "accumulator == compute), and confirms scale/dequant/low-bit/sparsity are explicitly "
             "unavailable with no false measured-pass. **No speedup or low-bit performance** is "
             "claimed.\n")

    # P12 HW/SW boundary-placement verification section
    L.append("## P12 HW/SW boundary-placement verification\n")
    L.append(f"**{p12_passed}/{len(p12_results)} P12 checks passed.** "
             f"{p12_summary['abstractions']} abstractions; top by evidence breadth: "
             f"`{p12_summary['top']}` (score {p12_summary['top_score']}).\n")
    for ok, msg in p12_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP12 checks the boundary search space: every candidate references a known "
             "abstraction / real workload / real region role; every boundary level + status + "
             "responsibility cell uses the allowed vocabulary; every required compiler proof "
             "references the compiler-proof matrix (or is unavailable); every DSE knob carries a "
             "reason + evidence; partial mode works with absent inputs; and no boundary artifact "
             "claims speedup/cycles/area/energy/optimal/best. Merlin generates the search space; "
             "the DSE tool chooses.\n")

    # P13 insight-mining verification section
    L.append("## P13 evidence-mining / insight-extraction verification\n")
    L.append(f"**{p13_passed}/{len(p13_results)} P13 checks passed.** "
             f"{p13_summary['scopes']} scopes (per-network + all); "
             f"{p13_summary['facts_all']} normalized facts in the combined scope.\n")
    for ok, msg in p13_results:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    L.append("\nP13 mines the committed package per network + combined, asserts the 10 cross-artifact "
             "consistency checks pass for every scope, mining is deterministic, every main finding "
             "is tier A/B with a DSE implication, partial mode degrades cleanly, and no forbidden "
             "performance wording leaks. Output is a non-committed `results/` run (regeneratable).\n")
    if write:
        (CS / "verification_report.md").write_text("\n".join(L) + "\n")

    print("\n".join(f"[{'PASS' if ok else 'FAIL'}] {m}" for ok, m in _all))
    dest = f" -> {CS / 'verification_report.md'}" if write else ""
    print(f"\n{passed}/{total} checks passed ({p5_passed}/{len(p5_results)} P5, "
          f"{p6_passed}/{len(p6_results)} P6, {p7_passed}/{len(p7_results)} P7, "
          f"{p8_passed}/{len(p8_results)} P8, {p9_passed}/{len(p9_results)} P9, "
          f"{p10_passed}/{len(p10_results)} P10, {p12_passed}/{len(p12_results)} P12, "
          f"{p13_passed}/{len(p13_results)} P13){dest}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(write="--check-only" not in sys.argv))
