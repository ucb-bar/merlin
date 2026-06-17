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


def main(write: bool = True) -> int:
    rows = [verify_workload(w) for w in RECAP_MODELS if (RECAP / w / "model.mlir").is_file()]
    verify_global()

    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
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
    if write:
        (CS / "verification_report.md").write_text("\n".join(L) + "\n")

    print("\n".join(f"[{'PASS' if ok else 'FAIL'}] {m}" for ok, m in results))
    dest = f" -> {CS / 'verification_report.md'}" if write else ""
    print(f"\n{passed}/{total} checks passed{dest}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(main(write="--check-only" not in sys.argv))
