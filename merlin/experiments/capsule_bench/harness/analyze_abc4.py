#!/usr/bin/env python3
"""Deep analysis of the abc4 three-arm run (baseline C++ / merlin xDSL / merlin+CIRCT).

Read-only. Produces reports/abc4_analysis/{struggle_matrix.json, circt_vs_verilator.json, trajectory.json}.
Answers:
  A. struggle/strength per arm per capsule (rounds-to-pass + failure plane)
  B. progression: per-round + cumulative cost/tokens/tool-calls/self-checks (active wall only)
  C. ★ CIRCT-replaces-sim thesis: replay the SAME CIRCT screen (rtl_check_runner.prescreen) over every
     arm×round×capsule emitted artifact, correlate the CIRCT verdict vs the actual spike(L2)/verilator(L3)
     outcome -> confusion matrix; the load-bearing number is FALSE-CLEAN (CIRCT-ok but sim-FAIL): if 0,
     CIRCT is a safe correctness gate.
"""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402 — active target (descriptor-driven), bootstraps merlin/python

EXP = C.EXP
REPORTS = C.REPORTS
from merlin.targetgen import rtl_check_runner as RCR  # noqa: E402 — the EXACT live screen
import yaml  # noqa: E402

OUT = REPORTS / "abc4_analysis"
ARMS = {"rb_abc4": ("raw_baseline", "baseline-C++"),
        "merlin_abc4": ("merlin_assisted", "merlin-xDSL"),
        "merlincirct_abc4": ("merlin_assisted", "merlin+CIRCT")}


def _cls(name: str) -> str:
    n = name.lower()
    for k in ("conv", "mvin", "mvout", "movement", "attention", "mlp", "relu", "acc_scale", "padding",
              "resident", "k_accum", "matmul", "config", "linear", "quantized"):
        if k in n:
            return {"mvin": "movement", "mvout": "movement"}.get(k, k)
    return "other"


def round_dirs(rid: str):
    """All per-round capsule dirs for an arm (has generated/ + capsule_result.json), sorted by round."""
    sd = ARMS[rid][0]
    base = EXP / "runs" / sd / rid / "_qa_work"
    out = []
    for cr in base.glob(f"runs_*/runs/{C.TARGET}-capsule-bench/*/capsule_result.json"):
        rnd = int(str(cr).split("runs_")[1][:2])
        out.append((rnd, cr.parent))
    return sorted(out)


def struggle_matrix() -> dict:
    """A: per arm, per capsule, per round: status + failure plane. + rounds-to-first-pass per op-class."""
    M = {}
    for rid, (_, label) in ARMS.items():
        per_cap = {}
        for rnd, d in round_dirs(rid):
            try:
                r = json.loads((d / "capsule_result.json").read_text())
            except Exception:
                continue
            cap = r.get("capsule", d.name)
            plane = (r.get("failure") or {}).get("plane")
            per_cap.setdefault(cap, {"class": _cls(cap), "by_round": {}})
            per_cap[cap]["by_round"][rnd] = {"status": r.get("status"), "plane": plane,
                                             "L2": (r.get("tiers", {}).get("L2") or {}).get("status")}
        # rounds-to-first-pass per capsule
        for cap, info in per_cap.items():
            firstpass = next((rn for rn in sorted(info["by_round"]) if info["by_round"][rn]["status"] == "pass"), None)
            info["first_pass_round"] = firstpass
            info["n_rounds_failing"] = sum(1 for rn in info["by_round"] if info["by_round"][rn]["status"] != "pass")
        M[label] = per_cap
    return M


def circt_vs_sim() -> dict:
    """C: replay the CIRCT screen on every arm×round×capsule; confusion vs that round's spike(L2) outcome.
    The decisive metric is false-clean = CIRCT-ok (verdict in {ok,warn}) AND sim status == fail."""
    rows, conf = [], {"true_neg": 0, "true_pos": 0, "false_clean": 0, "false_alarm": 0}
    false_clean_cases, by_check = [], {}
    for rid, (_, label) in ARMS.items():
        for rnd, d in round_dirs(rid):
            try:
                cr = json.loads((d / "capsule_result.json").read_text())
            except Exception:
                continue
            if not (d / "generated" / "instruction_trace.json").is_file():
                continue
            try:
                cv = RCR.prescreen(d)  # SAME screen the live arm used
            except Exception as e:
                cv = None
            if cv is None:
                continue
            circt_ok = cv["verdict"] in ("ok", "warn")     # 'ok'/'warn' = would NOT skip the sim; reject = catches
            # sim outcome this round: capsule overall status (L0/L1/L2 spike); 'pass' = functionally correct
            sim_pass = cr.get("status") == "pass"
            cap = cr.get("capsule", d.name)
            cell = ("true_neg" if (circt_ok and sim_pass) else
                    "true_pos" if (not circt_ok and not sim_pass) else
                    "false_clean" if (circt_ok and not sim_pass) else "false_alarm")
            conf[cell] += 1
            if cell == "false_clean":
                false_clean_cases.append({"arm": label, "round": rnd, "capsule": cap,
                                          "plane": (cr.get("failure") or {}).get("plane"),
                                          "circt_verdict": cv["verdict"]})
            rows.append({"arm": label, "round": rnd, "capsule": cap, "class": _cls(cap),
                         "circt_verdict": cv["verdict"], "circt_ok": circt_ok, "sim_pass": sim_pass,
                         "cell": cell})
            # per-check attribution (which checks fired on the rejects)
            for f in (cv.get("screen", {}) or {}).get("checks", []) if isinstance(cv.get("screen"), dict) else []:
                pass
    n = sum(conf.values())
    return {"confusion": conf, "n_points": n, "false_clean_count": conf["false_clean"],
            "false_clean_cases": false_clean_cases,
            "circt_safe_gate": conf["false_clean"] == 0 and n > 0,
            "rows": rows}


def trajectory() -> dict:
    """B: per-arm effort to correct (active wall only) + self-check intensity + per-round cost."""
    T = {}
    for rid, (sd, label) in ARMS.items():
        d = EXP / "runs" / sd / rid
        st = yaml.safe_load((d / "qa_loop_state.yaml").read_text()) if (d / "qa_loop_state.yaml").is_file() else {}
        rs = st.get("rounds") or []
        cum = st.get("cumulative") or {}
        scf = d / "selfcheck_log.jsonl"
        scs = [json.loads(l) for l in scf.read_text().splitlines()] if scf.is_file() else []
        T[label] = {
            "converged": st.get("converged"), "n_rounds": len(rs),
            "cost_usd": round(sum(r.get("estimated_cost_usd") or 0 for r in rs), 2),
            "tokens_total": sum(r.get("tokens_total") or 0 for r in rs),
            "tool_calls": sum(r.get("tool_calls") or 0 for r in rs),
            "active_wall_min": round(cum.get("active_wall_s", 0) / 60, 1),
            "rate_limit_wait_h": round(cum.get("rate_limit_wait_s", 0) / 3600, 2),
            "n_self_checks": len(scs),
            "self_check_sims": {s: sum(1 for x in scs if x.get("sim") == s) for s in ("spike", "verilator", "vcs")},
            "per_round": [{"round": r.get("round"), "n_passed": r.get("n_passed"),
                           "cost_usd": r.get("estimated_cost_usd"), "tool_calls": r.get("tool_calls")} for r in rs],
        }
    return T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== A. struggle matrix =="); M = struggle_matrix()
    (OUT / "struggle_matrix.json").write_text(json.dumps(M, indent=2))
    for label, caps in M.items():
        hard = sorted([(c, i["n_rounds_failing"]) for c, i in caps.items() if i["n_rounds_failing"] > 0],
                      key=lambda x: -x[1])
        print(f"  {label}: {len(caps)} capsules seen; hardest: {hard[:5]}")
    print("== B. trajectory =="); T = trajectory()
    (OUT / "trajectory.json").write_text(json.dumps(T, indent=2))
    for label, t in T.items():
        print(f"  {label}: ${t['cost_usd']} {t['tokens_total']/1e6:.1f}M {t['tool_calls']}tc "
              f"{t['n_self_checks']}self-checks(sims={t['self_check_sims']}) active={t['active_wall_min']}min conv={t['converged']}")
    print("== C. CIRCT vs sim (the thesis) =="); cs = circt_vs_sim()
    (OUT / "circt_vs_verilator.json").write_text(json.dumps(cs, indent=2))
    print(f"  N points (arm×round×capsule): {cs['n_points']}")
    print(f"  confusion: {cs['confusion']}")
    print(f"  ★ FALSE-CLEAN (CIRCT-ok but sim-FAIL): {cs['false_clean_count']}  -> "
          f"CIRCT safe as correctness gate: {cs['circt_safe_gate']}")
    if cs['false_clean_cases']:
        print("  false-clean cases:")
        for fc in cs['false_clean_cases']:
            print(f"    {fc}")
    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
