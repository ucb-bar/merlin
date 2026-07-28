#!/usr/bin/env python3
"""Full-suite A/B report: per-arm pass over ALL capsules, per-capsule cycle matrix, and active-vs-
sim-wait time. Separate from gen_reports.py (which the raw-baseline session owns) to avoid edit
conflicts; reads the same run records read-only.

Sources per run dir runs/<arm>/<run_id>/:
  - run_manifest.yaml         arm/model/public_dev/hidden/process/oracle_mode/integrity
  - qa_loop_summary.yaml      n_rounds + agent_wall_total_seconds / sim_wait_total_seconds (full-suite)
  - grading_{public,hidden}/runs/gemmini-capsule-bench/<capsule>/capsule_result.json  per-tier cycles

Outputs (reports/):
  - fullsuite_comparison.md   one row per run: pass N/M, tier, integrity, tokens, cost, rounds,
                              active(agent) vs sim-wait time
  - cycles_by_capsule.md      per-capsule status + L2(spike)/L3(verilator) cycles, per run + a
                              cross-run L3-cycle matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

import _common as C

ORDER = (  # canonical capsule order for the matrix
    [f"A{i}" for i in range(8)] + [f"B{i}" for i in range(5)] + [f"C{i}" for i in range(7)]
    + [f"H{i}" for i in range(5)])


def _passed_frac(s: str | None) -> tuple[int, int]:
    if not s or "/" not in str(s):
        return (0, 0)
    a, b = str(s).split("/")[:2]
    try:
        return (int(a), int(b))
    except ValueError:
        return (0, 0)


def _is_pass(pub: str | None, hid: str | None) -> bool:
    pa, pb = _passed_frac(pub)
    ha, hb = _passed_frac(hid)
    return pb > 0 and pa == pb and hb > 0 and ha == hb


def _capsule_results(run_dir: Path) -> dict[str, dict]:
    """capsule -> {status, L2, L3, label} merged across the public + hidden grading work trees."""
    out: dict[str, dict] = {}
    for phase in ("grading_public", "grading_hidden"):
        for cr in sorted((run_dir / phase).glob("runs/*/*/capsule_result.json")):
            try:
                r = json.loads(cr.read_text())
            except Exception:
                continue
            name = r.get("capsule", cr.parent.name)
            tiers = r.get("tiers") or {}
            out[name] = {
                "status": r.get("status"),
                "L2": (tiers.get("L2") or {}).get("cycles"),
                "L3": (tiers.get("L3") or {}).get("cycles"),
                "phase": "hidden" if phase == "grading_hidden" else "public",
            }
    return out


def _short(name: str) -> str:
    return name.split("_")[0]  # A2_single_tile_matmul -> A2


def main() -> int:
    runs = []
    for mf in sorted(C.RUNS.glob("*/*/run_manifest.yaml")):
        rd = mf.parent
        try:
            m = yaml.safe_load(mf.read_text()) or {}
        except Exception:
            continue
        ql = {}
        qf = rd / "qa_loop_summary.yaml"
        if qf.exists():
            ql = yaml.safe_load(qf.read_text()) or {}
        side = {}
        sf = rd / "fullsuite_agent_sim_timing.yaml"
        if sf.exists():
            side = yaml.safe_load(sf.read_text()) or {}
        env = {}
        ef = rd / "environment.yaml"
        if ef.exists():
            env = yaml.safe_load(ef.read_text()) or {}
        runs.append({"dir": rd, "m": m, "ql": ql, "side": side, "env": env,
                     "caps": _capsule_results(rd)})

    # ---------------- fullsuite_comparison.md ----------------
    L = ["# Full-suite comparison (gemmini_capsule_bench_v0)", "",
         "Per-arm pass over **all** capsules (dynamic n/n, not a hardcoded pilot count). "
         "Time split (cumulative across quota-resumes): `active` = doing work (agent+oracle, from the "
         "driver), `quota_wait` = slept waiting on the 5h limit; within active, `agent`/`sim` = agent "
         "subprocess vs oracle (spike+verilator) wall. Cert tier = highest REQUIRED tier reached "
         "(L3 = real cycle-accurate RTL). Cycles are diagnostic-only and never gate.", "",
         "| arm | run_id | suite | public | hidden | pass | tier | integrity | rounds | tokens | "
         "cost$ | active(s) | quota_wait(s) | agent(s) | sim(s) | wall(s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m, ql, env, side = r["m"], r["ql"], r["env"], r["side"]
        pub = (m.get("public_dev") or {}).get("passed")
        hid = (m.get("hidden") or {}).get("passed")
        proc = m.get("process") or {}
        suite = env.get("suite", "pilot")
        tm = ql.get("timing") or {}
        active = tm.get("active_wall_s", "—")
        qwait = tm.get("rate_limit_wait_s", "—")
        agent_s = side.get("agent_active_s", "—")
        sim_s = side.get("sim_wait_s", "—")
        L.append(
            f"| {m.get('arm')} | {m.get('run_id')} | {suite} | {pub} | {hid} | "
            f"{'PASS' if _is_pass(pub, hid) else 'no'} | {(m.get('public_dev') or {}).get('highest_tier')} | "
            f"{m.get('integrity_status')} | {ql.get('n_rounds', m.get('iterations'))} | "
            f"{proc.get('tokens_total')} | {proc.get('estimated_cost_usd')} | {active} | {qwait} | "
            f"{agent_s} | {sim_s} | {proc.get('wall_time_seconds')} |")
    L += ["", "_`active`+`quota_wait` = `wall` (cumulative across resume invocations). `agent`+`sim` "
          "split `active` (the rest of active is harness/finalize overhead). `—` = a run predating "
          "this instrumentation (e.g. pilot runs launched before run_fullsuite.py)._"]
    (C.REPORTS / "fullsuite_comparison.md").write_text("\n".join(L) + "\n")

    # ---------------- cycles_by_capsule.md ----------------
    full_runs = [r for r in runs if r["caps"]]
    M = ["# Cycles by capsule (gemmini_capsule_bench_v0)", "",
         "Per-capsule status + **L2 spike** / **L3 verilator (cycle-accurate RTL)** cycle counts, from "
         "each run's `capsule_result.json`. L5 FireSim columns are added once the FPGA backfill runs.", ""]
    # cross-run L3-cycle matrix (capsule rows x run columns)
    cols = [f"{r['m'].get('arm','?')[:6]}/{r['m'].get('run_id')}" for r in full_runs]
    M += ["## L3 (verilator) cycle matrix", "",
          "| capsule | " + " | ".join(cols) + " |",
          "|---|" + "|".join(["---"] * len(cols)) + "|"]
    seen = {}
    for r in full_runs:
        for name, d in r["caps"].items():
            seen.setdefault(_short(name), name)
    for sh in ORDER:
        if sh not in seen:
            continue
        full = seen[sh]
        cells = []
        for r in full_runs:
            d = r["caps"].get(full) or next((v for k, v in r["caps"].items() if _short(k) == sh), None)
            if not d:
                cells.append("·")
            else:
                mark = "" if d["status"] == "pass" else "✗"
                cells.append(f"{d['L3']}{mark}" if d["L3"] is not None else (d["status"] or "·"))
        M.append(f"| {full} | " + " | ".join(str(c) for c in cells) + " |")
    # per-run detail (L2 + L3)
    for r in full_runs:
        m = r["m"]
        M += ["", f"## {m.get('arm')}/{m.get('run_id')} — per-capsule L2/L3", "",
              "| capsule | phase | status | L2 spike | L3 verilator |", "|---|---|---|---|---|"]
        for name in sorted(r["caps"], key=lambda n: (ORDER.index(_short(n)) if _short(n) in ORDER else 99, n)):
            d = r["caps"][name]
            M.append(f"| {name} | {d['phase']} | {d['status']} | {d['L2']} | {d['L3']} |")
    (C.REPORTS / "cycles_by_capsule.md").write_text("\n".join(M) + "\n")

    print(f"wrote fullsuite_comparison.md + cycles_by_capsule.md ({len(runs)} runs, "
          f"{len(full_runs)} with per-capsule cycles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
