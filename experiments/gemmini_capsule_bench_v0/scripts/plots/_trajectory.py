#!/usr/bin/env python3
"""Extraction layer for the experiment plots.

Turns a run dir (the QA-loop output) + the full-suite audit into TIDY, plot-ready JSON. Everything
here is read from artifacts ALREADY on disk — no agent re-run, no grading — so it works on a finished
run, a frozen pilot, OR a run that is still mid-flight (it just sees fewer rounds).

Per run it surfaces, round by round:
  - the grader outcome (n_passed / n_capsules) and per-round effort (tokens, cost, tool-calls, thinking)
  - the per-capsule status + L3 verilator cycles + first-failure-plane  (from _qa_work/runs_NN/.../
    capsule_result.json — one snapshot PER ROUND, so we know exactly when each capsule flipped green)
  - a per-event TIMELINE across the whole transcript (thinking / assistant-text / tool:Read|Edit|Bash /
    tool-result / rate-limit), each stamped with its position through the transcript (cumulative output
    tokens when usage is present, else event ordinal) — this is what the reference-styled trajectory
    plot sweeps along its x-axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

# event kinds we track along the transcript (the "activity mix")
KIND_THINK = "thinking"
KIND_TEXT = "assistant_text"
KIND_READ = "tool:Read"
KIND_EDIT = "tool:Edit"
KIND_BASH = "tool:Bash"
KIND_TOOL_OTHER = "tool:Other"
KIND_RESULT = "tool_result"      # environment hands a tool result back (a "user" turn)
KIND_RATELIMIT = "rate_limit"

_TOOL_KIND = {"Read": KIND_READ, "Edit": KIND_EDIT, "Write": KIND_EDIT, "Bash": KIND_BASH}


def _capsule_class(name: str) -> str:
    """Coarse workload class from the capsule name (mirrors full_suite_audit's tagging, name-only)."""
    n = name.lower()
    if "conv" in n:
        return "conv"
    if "mvin" in n or "mvout" in n or "movement" in n:
        return "movement"
    if "attention" in n:
        return "attention"
    if "mlp" in n:
        return "mlp"
    if "relu" in n:
        return "relu"
    if "acc_scale" in n or "quantized" in n:
        return "acc_scale"
    return "matmul"


def parse_transcript_timeline(path: Path) -> list[dict]:
    """One row per content event, in order, with position-through-transcript stamps.

    Position uses cumulative OUTPUT tokens (decode) when the transcript carries usage metadata, else a
    plain event ordinal. Both are returned so the plotter can pick; both are normalised later.
    """
    events: list[dict] = []
    cum_out = 0
    ordinal = 0
    if not path.exists():
        return events
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        if t == "rate_limit_event":
            ordinal += 1
            events.append({"kind": KIND_RATELIMIT, "ordinal": ordinal, "cum_out": cum_out,
                           "rejected": (o.get("rate_limit_info", {}) or {}).get("status") == "rejected"})
            continue
        if t == "assistant":
            msg = o.get("message", {}) or {}
            usage = msg.get("usage", {}) or {}
            out_tok = int(usage.get("output_tokens", 0) or 0)
            for blk in msg.get("content", []) or []:
                bt = blk.get("type")
                ordinal += 1
                if bt == "thinking":
                    kind = KIND_THINK
                elif bt == "text":
                    kind = KIND_TEXT
                elif bt == "tool_use":
                    kind = _TOOL_KIND.get(blk.get("name", ""), KIND_TOOL_OTHER)
                else:
                    continue
                # spread this message's output tokens evenly across its blocks for a monotone x
                events.append({"kind": kind, "ordinal": ordinal, "cum_out": cum_out,
                               "tool": blk.get("name") if bt == "tool_use" else None})
            cum_out += out_tok
        elif t == "user":
            # tool results coming back from the environment
            msg = o.get("message", {}) or {}
            content = msg.get("content", [])
            n_res = sum(1 for b in (content if isinstance(content, list) else [])
                        if isinstance(b, dict) and b.get("type") == "tool_result") or 1
            for _ in range(n_res):
                ordinal += 1
                events.append({"kind": KIND_RESULT, "ordinal": ordinal, "cum_out": cum_out})
    return events


def _round_capsule_snapshot(run_dir: Path, rnd: int) -> dict:
    """Per-capsule {status, l3_cycles, plane, class} for one round, from _qa_work/runs_NN."""
    snap: dict[str, dict] = {}
    base = run_dir / "_qa_work" / f"runs_{rnd:02d}" / "runs" / "gemmini-capsule-bench"
    if not base.exists():
        return snap
    for cr in sorted(base.glob("*/capsule_result.json")):
        try:
            d = json.loads(cr.read_text())
        except Exception:
            continue
        name = d.get("capsule") or cr.parent.name
        tiers = d.get("tiers", {}) or {}
        l3 = tiers.get("L3", {}) or {}
        plane = (d.get("failure") or {}).get("plane")
        snap[name] = {"status": d.get("status"),
                      "l3_cycles": l3.get("cycles"),
                      "plane": plane,
                      "class": _capsule_class(name)}
    return snap


def extract_run(run_dir: Path) -> dict:
    """Tidy, plot-ready dict for one run (works on partial/live runs)."""
    run_dir = Path(run_dir)
    out: dict = {"run_id": run_dir.name, "arm": run_dir.parent.name,
                 "rounds": [], "capsule_rounds": {}, "timeline": []}

    state = run_dir / "qa_loop_state.yaml"
    rounds_meta = []
    if state.exists():
        st = yaml.safe_load(state.read_text()) or {}
        rounds_meta = st.get("rounds", []) or []
        out["converged"] = st.get("converged")
        out["model"] = st.get("model")
    # whole-run process telemetry (for cost/token economics + Pareto)
    cty = run_dir / "cost_time_toolcalls.yaml"
    if cty.exists():
        out["process"] = yaml.safe_load(cty.read_text()) or {}

    # per-round outcome + effort + per-capsule snapshot + that round's transcript timeline
    n_rounds_seen = 0
    rdir = run_dir / "rounds"
    n_files = len(list(rdir.glob("round_*.transcript.jsonl"))) if rdir.exists() else 0
    n_rounds = max(len(rounds_meta), n_files)
    for r in range(n_rounds):
        meta = rounds_meta[r] if r < len(rounds_meta) else {}
        snap = _round_capsule_snapshot(run_dir, r)
        planes: dict[str, int] = {}
        for cap, info in snap.items():
            out["capsule_rounds"].setdefault(cap, {"class": info["class"], "status": [], "cycles": []})
            out["capsule_rounds"][cap]["status"].append(info["status"])
            out["capsule_rounds"][cap]["cycles"].append(info["l3_cycles"])
            if info["status"] != "pass" and info["plane"]:
                planes[info["plane"]] = planes.get(info["plane"], 0) + 1
        out["rounds"].append({
            "round": r,
            "n_passed": meta.get("n_passed", sum(1 for s in snap.values() if s["status"] == "pass") or None),
            "n_capsules": meta.get("n_capsules", len(snap) or None),
            "tool_calls": meta.get("tool_calls"),
            "tokens_total": meta.get("tokens_total"),
            "tokens_output": meta.get("tokens_output"),
            "tokens_cached": meta.get("tokens_cached"),
            "estimated_cost_usd": meta.get("estimated_cost_usd"),
            "thinking_blocks": meta.get("thinking_blocks"),
            "failure_planes": planes,
        })
        # transcript timeline for this round (tagged with the round index so the plotter can band them)
        tl = parse_transcript_timeline(rdir / f"round_{r:02d}.transcript.jsonl") if rdir.exists() else []
        for e in tl:
            e["round"] = r
        out["timeline"].extend(tl)
        n_rounds_seen += 1
    out["n_rounds_seen"] = n_rounds_seen
    return out


def load_full_suite(reports_dir: Path) -> dict:
    p = Path(reports_dir) / "full_suite_audit.json"
    return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    print(json.dumps(extract_run(Path(a.run_dir)), indent=2)[:4000])
