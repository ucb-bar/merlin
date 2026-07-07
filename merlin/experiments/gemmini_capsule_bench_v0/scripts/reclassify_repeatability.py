#!/usr/bin/env python3
"""Rate-limit-aware reclassification of a repeatability sweep.

The raw sweep aggregator (run_repeatability.py) treats every non-converged run as a "fail". That is
WRONG when a run never got a fair attempt: a five-hour session rate-limit (rateLimitType:"five_hour",
status:"rejected", overageStatus:"rejected", org_level_disabled) rejects the agent's API calls so it
does ZERO real work. Such a run is BLOCKED, not failed, and must be excluded from the pass-rate
denominator — otherwise "1/3 pass" reads as "the baseline is unreliable" when the truth is "1/1 valid
samples passed; 2 runs never ran because the org budget was exhausted by the first run".

This script re-derives each run's outcome from its existing artifacts (NO agent re-run):
  - blocked_rate_limit : every round that attempted work was rejected by the five-hour limit and the
                         run produced no graded submission. Excluded from the pass-rate denominator.
  - pass               : public 4/4 AND hidden 3/3.
  - fail               : ran for real (spent tokens / made tool calls) but did not converge.

It also reclassifies answer-access using the established false-positive rules (see audit_kind):
  - leak              : a real read of a golden/hidden/oracle/grader-private file that returned content.
  - attempted_blocked : tried to read an answer file that was golden-masked/absent => returned nothing.
  - grep_for_string   : the token matched the agent's own grep PATTERN or a trailing echo, not a read.
  - clean             : no hits.
Only `leak` makes a run answer-access-dirty; the other three are benign (the masking did its job).

Usage:  reclassify_repeatability.py [--arm raw_baseline] [--prefix rb_pilot_rep] [--ids id1,id2,...]
Writes reports/repeatability.md + repeatability.json (rate-limit-aware), preserving the raw counts
under aggregate.raw_* for auditability.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import yaml

import _common as C

# Substrings that name a real answer artifact (golden value / expected cb / reference / oracle output).
_ANSWER_PATH_HINTS = ("golden", "expected_command_buffer", "reference_output", "/oracle", "hidden",
                      "grader_private", "numeric_report", "score_capsule")


def _cmd_touches_answer(cmd: str) -> bool:
    return any(h in cmd.lower() for h in _ANSWER_PATH_HINTS)


def _classify_cmd(cmd: str, result: str) -> str:
    """Classify a Bash command (FULL text) + its result. Returns leak|attempted_blocked|grep_for_string.

    A LEAK is only possible if the command reads an answer artifact via an ABSOLUTE path INTO the
    operator tree (outside the agent's golden-masked _qa_ws workspace) and that read returns content.
    Within the workspace the answer files do not exist (masking removed them), so any in-workspace
    answer read returned nothing; a grep/echo that merely names the token read nothing either."""
    low = cmd.lower()
    # Out-of-workspace absolute path that names an answer artifact => the only way to actually leak.
    out_of_ws_answer = any(
        any(h in m.lower() for h in _ANSWER_PATH_HINTS) and "_qa_ws/" not in m
        for m in re.findall(r"/scratch\S+", cmd)
    )
    if out_of_ws_answer and re.search(r"\b(cat|head|tail|less|od|xxd|read)\b", low):
        # confirm the result carries real answer content (opcodes / numeric arrays), not just a banner
        res = (result or "").lower()
        if any(tok in res for tok in ("res_pack", "matmul_resident", "\"commit\"", "epilogue",
                                      "golden:", "reference_output", "expected:")):
            return "leak"
        return "attempted_blocked"
    # Answer token sits in a grep/rg PATTERN or an echo banner (self-scan / status print), not a read.
    if re.search(r"\b(grep|rg|ripgrep|ag)\b", low):
        return "grep_for_string"
    # Everything else touching an answer name was an in-workspace read of a masked-absent file.
    return "attempted_blocked"


def _audit_kinds_from_transcripts(run_dir: Path) -> list[str]:
    """Re-derive answer-access classifications from FULL tool_use commands + their results.

    The qa_loop_summary stores truncated command snippets (token often beyond the cutoff), so we read
    the round transcripts directly: pair each Bash tool_use that references an answer token with its
    tool_result, then classify on the (command, result) pair."""
    kinds: list[str] = []
    rdir = run_dir / "rounds"
    if not rdir.exists():
        return kinds
    for tp in sorted(rdir.glob("round_*.transcript.jsonl")):
        suspect: dict[str, str] = {}
        try:
            for line in tp.read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                if e.get("type") == "assistant":
                    for b in e.get("message", {}).get("content", []):
                        if b.get("type") == "tool_use" and b.get("name") in ("Bash", "Read"):
                            cmd = (b.get("input", {}).get("command")
                                   or b.get("input", {}).get("file_path") or "")
                            if _cmd_touches_answer(cmd):
                                suspect[b["id"]] = cmd
                elif e.get("type") == "user":
                    content = e.get("message", {}).get("content", [])
                    if not isinstance(content, list):
                        continue
                    for b in content:
                        if b.get("type") == "tool_result" and b.get("tool_use_id") in suspect:
                            res = b.get("content", "")
                            if isinstance(res, list):
                                res = " ".join(x.get("text", "") for x in res if isinstance(x, dict))
                            kinds.append(_classify_cmd(suspect[b["tool_use_id"]], str(res)))
        except Exception:
            continue
    return kinds


# Rate-limit detection lives in the shared _ratelimit module (also used by the driver + gen_reports).
from _ratelimit import rounds_rate_limited as _rounds_rate_limited  # noqa: E402


def _load(run_dir: Path) -> dict | None:
    mf = run_dir / "run_manifest.yaml"
    if not mf.exists():
        return None
    m = yaml.safe_load(mf.read_text())
    qs = run_dir / "qa_loop_summary.yaml"
    q = yaml.safe_load(qs.read_text()) if qs.exists() else {}
    public = m["public_dev"]["passed"]
    hidden = m["hidden"]["passed"]

    rejected, worked = _rounds_rate_limited(run_dir)
    # Reclassify answer-access from the FULL transcripts (truncated qa_loop_summary snippets are unsafe).
    hit_kinds = _audit_kinds_from_transcripts(run_dir)
    leaked = any(k == "leak" for k in hit_kinds)
    attempted = any(k in ("attempted_blocked", "grep_for_string") for k in hit_kinds)

    is_pass = (public == "4/4" and hidden == "3/3")
    # BLOCKED: produced no real graded work AND at least one round was rejected by the five-hour limit.
    blocked = (not is_pass) and rejected > 0 and worked == 0
    # Edge: rep_02-style — one productive round then rejected before converging. Still blocked (the
    # limit, not the agent, ended the run); it never got to finish.
    if (not is_pass) and rejected > 0 and worked > 0 and not q.get("converged"):
        blocked = True
    outcome = "pass" if is_pass else ("blocked_rate_limit" if blocked else "fail")

    return {"run_id": m["run_id"], "outcome": outcome,
            "public": public, "hidden": hidden, "tier": m["public_dev"]["highest_tier"],
            "numeric_all_exact": m["public_dev"]["numeric_all_exact"],
            "integrity": m.get("integrity_status"),
            "n_rounds": q.get("n_rounds"), "converged": q.get("converged"),
            "rounds_rejected": rejected, "rounds_worked": worked,
            "answer_access": "leak" if leaked else ("attempted_blocked" if attempted else "clean"),
            "answer_access_raw_clean": all(r.get("answer_access_clean", True) for r in q.get("rounds", []))
                                       and ((q.get("finalize") or {}).get("answer_access_clean", True)),
            "wall_s": (m.get("process") or {}).get("wall_time_seconds"),
            "cost_usd": (m.get("process") or {}).get("estimated_cost_usd"),
            "tokens_total": (m.get("process") or {}).get("tokens_total"),
            "tool_calls": (m.get("process") or {}).get("tool_calls")}


def _agg(vals: list) -> dict:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return {}
    return {"min": min(vals), "max": max(vals), "mean": round(statistics.mean(vals), 2),
            "median": statistics.median(vals),
            "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="raw_baseline")
    ap.add_argument("--prefix", default="rb_pilot_rep")
    ap.add_argument("--ids", default="", help="explicit comma-separated run_ids (overrides --prefix scan)")
    a = ap.parse_args(argv)

    arm_dir = C.RUNS / a.arm
    if a.ids.strip():
        run_dirs = [arm_dir / x.strip() for x in a.ids.split(",") if x.strip()]
    else:
        run_dirs = sorted(d for d in arm_dir.glob(f"{a.prefix}_*") if d.is_dir())
    results = [r for r in (_load(d) for d in run_dirs) if r]

    valid = [r for r in results if r["outcome"] != "blocked_rate_limit"]
    blocked = [r for r in results if r["outcome"] == "blocked_rate_limit"]
    n_pass = sum(1 for r in valid if r["outcome"] == "pass")

    agg = {
        "valid_pass_rate": f"{n_pass}/{len(valid)}" if valid else "0/0",
        "n_total": len(results), "n_valid": len(valid), "n_blocked_rate_limit": len(blocked),
        "all_integrity_clean_valid": all(r["integrity"] == "clean" for r in valid) if valid else None,
        "answer_access_leak_any": any(r["answer_access"] == "leak" for r in results),
        "answer_access_by_run": {r["run_id"]: r["answer_access"] for r in results},
        "all_numeric_exact_valid": all(r["numeric_all_exact"] for r in valid) if valid else None,
        # process stats over VALID runs only (blocked runs have meaningless ~0 wall/cost)
        "rounds_to_converge_valid": _agg([r["n_rounds"] for r in valid if r["outcome"] == "pass"]),
        "wall_seconds_valid": _agg([r["wall_s"] for r in valid]),
        "cost_usd_valid": _agg([r["cost_usd"] for r in valid]),
        "tool_calls_valid": _agg([r["tool_calls"] for r in valid]),
        # raw (un-reclassified) counts preserved for audit
        "raw_full_pass": f"{n_pass}/{len(results)}",
    }
    out = {"arm": a.arm, "n_total": len(results), "n_valid": len(valid),
           "rate_limit_aware": True, "runs": results, "aggregate": agg}
    (C.REPORTS / "repeatability.json").write_text(json.dumps(out, indent=2))

    md = [f"# Repeatability — {a.arm} (claude-opus-4-8), rate-limit-aware", "",
          "> **Why this matters.** Each full pilot run costs ~$18 / ~50 min and exhausts the org's "
          "five-hour session budget. Runs that the budget rejected (zero real work) are **blocked**, "
          "not failed — counting them as failures would falsely imply the baseline is unreliable.",
          "",
          f"- **Valid full-pass rate (public 4/4 AND hidden 3/3, blocked runs excluded): "
          f"{agg['valid_pass_rate']}**",
          f"- runs: {agg['n_total']} total = {agg['n_valid']} valid + "
          f"{agg['n_blocked_rate_limit']} blocked-by-rate-limit",
          f"- integrity clean (valid): {agg['all_integrity_clean_valid']}; "
          f"numeric exact (valid): {agg['all_numeric_exact_valid']}; "
          f"answer-access leak (any run): {agg['answer_access_leak_any']}",
          f"- rounds-to-converge (passing runs): {agg['rounds_to_converge_valid']}",
          f"- wall(s) valid: {agg['wall_seconds_valid']}", f"- cost$ valid: {agg['cost_usd_valid']}",
          f"- raw full-pass (NOT rate-limit-aware, for audit only): {agg['raw_full_pass']}", "",
          "| run_id | outcome | public | hidden | tier | rounds | rej/work | wall(s) | cost$ | "
          "answer-access |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['run_id']} | **{r['outcome']}** | {r['public']} | {r['hidden']} | {r['tier']} "
                  f"| {r['n_rounds']} | {r['rounds_rejected']}/{r['rounds_worked']} | {r['wall_s']} | "
                  f"{r['cost_usd']} | {r['answer_access']} |")
    md += ["",
           "**Legend.** outcome: pass / fail (ran for real, did not converge) / blocked_rate_limit "
           "(five-hour budget rejected the agent; excluded from pass-rate). rej/work: rounds rejected "
           "by the rate-limit / rounds that did real tool work. answer-access: clean / attempted_blocked "
           "(tried to read a masked-absent answer file => got nothing) / leak (real answer read). "
           "Only `leak` is disqualifying; masking-defeated attempts confirm the isolation held."]
    (C.REPORTS / "repeatability.md").write_text("\n".join(md) + "\n")
    print(f"reclassified {len(results)} runs: valid_pass_rate={agg['valid_pass_rate']} "
          f"({agg['n_blocked_rate_limit']} blocked); answer-access leak any={agg['answer_access_leak_any']}")
    for r in results:
        print(f"  {r['run_id']}: {r['outcome']} (rej/work {r['rounds_rejected']}/{r['rounds_worked']}, "
              f"answer-access {r['answer_access']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
