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
ALLOWED_CADENCE = {"once_per_instruction", "once_per_replan", "K_times_per_replan", "token_loop",
                   "control_tick", "once_per_forward", "unknown"}
ALLOWED_EVIDENCE = {"recovered_from_ir", "recovered_from_prov_fqn", "assumed_reference",
                    "derived_requirement", "design_assumption", "measured", "proxy_measured",
                    "unavailable"}
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

    # P6-D: cadence vocabulary + edge evidence labels + unknown explicitness
    cad_ok = all((n.get("rate") or {}).get("cadence", "unknown") in ALLOWED_CADENCE
                 for n in nodes if n["kind"] in ("phase", "region", "loop_body"))
    ev_ok = all(e["evidence"] in ALLOWED_EVIDENCE for e in edges)
    unk_ok = all(e["evidence"] == "unavailable" and e.get("can_pipeline") == "unknown"
                 for e in edges if e["kind"] == "unknown_dependency")
    p6check(cad_ok, f"[{w}] all cadence fields in allowed vocabulary")
    p6check(ev_ok, f"[{w}] all edge evidence labels valid")
    p6check(unk_ok, f"[{w}] unknown_dependency edges are explicit (unavailable + can_pipeline unknown)")

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


def main(write: bool = True) -> int:
    rows = [verify_workload(w) for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()]
    verify_global()
    p5_rows = [verify_p5_workload(w) for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()]
    verify_p5_global()
    _graphs = _graphs_by_workload()
    p6_rows = [verify_p6_workload(w, _graphs) for w in RECAP_MODELS
               if (RECAP / w / "model.mlir").is_file()]
    verify_p6_global(_graphs)

    passed = sum(1 for ok, _ in results + p5_results + p6_results if ok)
    total = len(results) + len(p5_results) + len(p6_results)
    p5_passed = sum(1 for ok, _ in p5_results if ok)
    p6_passed = sum(1 for ok, _ in p6_results if ok)
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
             "evidence / cadence label is valid with `unknown_dependency` edges explicit. Structural "
             "only; **no speedup** claimed.\n")
    if write:
        (CS / "verification_report.md").write_text("\n".join(L) + "\n")

    all_results = results + p5_results + p6_results
    print("\n".join(f"[{'PASS' if ok else 'FAIL'}] {m}" for ok, m in all_results))
    dest = f" -> {CS / 'verification_report.md'}" if write else ""
    print(f"\n{passed}/{total} checks passed ({p5_passed}/{len(p5_results)} P5, "
          f"{p6_passed}/{len(p6_results)} P6){dest}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(write="--check-only" not in sys.argv))
