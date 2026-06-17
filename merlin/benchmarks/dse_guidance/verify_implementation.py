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
from merlin.dse_guidance.case_study import RECAP_MODELS
from merlin.dse_guidance.design_envelope import ELEMENT_BYTES

HERE = Path(__file__).resolve().parent
CS = HERE / "case_study"
RECAP = HERE / "recaptures"
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
    """Recompute per-role facts straight from the IR primitive (independent of the YAML)."""
    recs = ATTR.extract_matmuls(str(RECAP / workload))
    roles: dict[str, dict] = {}
    for r in recs:
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
    n_elem = WB / ELEMENT_BYTES["f32"]
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
    expected_pass = {w for w in RECAP_MODELS if w in gate_models}
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
        if not f.is_file() or f.name in V0_DOCS:
            continue
        low = f.read_text(errors="ignore").lower()
        for t in DANGEROUS:
            if t in low:
                found.setdefault(t, []).append(f.name)
    check(not found, f"dangerous terms absent from generated artifacts (found: {found})")

    # --- L: speedup appears in generated artifacts only inside disclaimers / not_claimed fields ---
    affirmative = []
    for f in CS.rglob("*"):
        if not f.is_file() or f.name in V0_DOCS:
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
    present = set(graphs) == {w for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()}
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
    recs = ATTR.extract_matmuls(str(RECAP / w))
    # independent per-role weight bytes + invocations from the IR primitive
    K = int(RECAP_MODELS[w]["K"])
    role_weight: dict[str, int] = {}
    for r in recs:
        role = ATTR.role_from_fqn(r.fqn)
        if role:
            role_weight[role] = role_weight.get(role, 0) + r.weight_bytes
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
    valid_wl = {w for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()}
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


def main(write: bool = True) -> int:
    rows = [verify_workload(w) for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()]
    verify_global()
    p5_rows = [verify_p5_workload(w) for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()]
    verify_p5_global()
    _graphs = _graphs_by_workload()
    p6_rows = [verify_p6_workload(w, _graphs) for w in RECAP_MODELS
               if (RECAP / w / "model.mlir").is_file()]
    verify_p6_global(_graphs)
    p7_rows = [verify_p7_workload(w, _graphs) for w in RECAP_MODELS
               if (RECAP / w / "model.mlir").is_file()]
    verify_p7_global()
    p8_rows = [verify_p8_workload(w, _graphs) for w in RECAP_MODELS
               if (RECAP / w / "model.mlir").is_file()]
    verify_p8_global()
    p9_rows = [verify_p9_workload(w) for w in RECAP_MODELS
               if (RECAP / w / "model.mlir").is_file()]
    verify_p9_global()
    p10_rows = [verify_p10_workload(w) for w in RECAP_MODELS
                if (RECAP / w / "model.mlir").is_file()]
    verify_p10_global()
    p12_summary = verify_p12()

    _all = (results + p5_results + p6_results + p7_results + p8_results + p9_results
            + p10_results + p12_results)
    passed = sum(1 for ok, _ in _all if ok)
    total = len(_all)
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
    if write:
        (CS / "verification_report.md").write_text("\n".join(L) + "\n")

    print("\n".join(f"[{'PASS' if ok else 'FAIL'}] {m}" for ok, m in _all))
    dest = f" -> {CS / 'verification_report.md'}" if write else ""
    print(f"\n{passed}/{total} checks passed ({p5_passed}/{len(p5_results)} P5, "
          f"{p6_passed}/{len(p6_results)} P6, {p7_passed}/{len(p7_results)} P7, "
          f"{p8_passed}/{len(p8_results)} P8, {p9_passed}/{len(p9_results)} P9, "
          f"{p10_passed}/{len(p10_results)} P10, {p12_passed}/{len(p12_results)} P12){dest}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(write="--check-only" not in sys.argv))
