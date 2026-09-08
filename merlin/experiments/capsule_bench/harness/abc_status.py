"""Live status dashboard for a running A/B batch. Read-only; run anytime.

Usage:
  python abc_status.py            # auto-detects the newest batch (e.g. abc9)
  python abc_status.py --tag abc9
  watch -n 60 'python abc_status.py'   # refresh every minute
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from pathlib import Path

import _common as C  # noqa: E402 — active target (descriptor-driven)
EXP = C.EXP
RUNS = C.RUNS
# Every rung the launcher can stamp, so a batch that includes one is not invisible here. The C++
# infra arm (arm-2, prefix rbinfra) was missing: `launch_ab_batch --arms baseline,cpp_merlininfra,merlin`
# ran three arms and this dashboard showed two, with nothing saying the third was omitted rather than
# dead. Keep this list in step with launch_ab_batch.ARMS.
#: Per-arm, the tokens whose appearance in a Bash tool call means THAT RUNG'S GRANTED TOOLING FIRED.
#: This column used to count the CIRCT generators for arm-4 and print "-" for every other arm, which
#: is the wrong shape for a ladder: an arm whose one treatment is a tool set is uninterpretable until
#: you know the agent actually invoked it, and a null result then cannot be told apart from a tool the
#: agent never noticed. Nested exactly like tool_registry.ARM_TOOLS, so the count is comparable.
_INFRA = ("mlir_scaffold", "target_repo", "llvm_plan")
_ASSISTED = ("isa_tools", "cca_contract", "action_catalog")
_RTL = ("gen_isa_module", "gen_rtl_digest", "gen_numeric_facts")
ARMS = [("baseline", "raw_baseline", "rb", ()),
        ("cpp+infra", "cpp_merlininfra", "rbinfra", _INFRA),
        ("merlin", "merlin_assisted", "merlin", _ASSISTED),
        ("merlin+CIRCT", "merlin_assisted", "merlincirct", _ASSISTED + _RTL),
        ("merlin+eqsat", "merlin_assisted", "merlineqsat", _ASSISTED + ("eqsat",))]


def _alive(runid):
    try:
        return bool(subprocess.run(["pgrep", "-f", f"run-id {runid}"], capture_output=True).stdout.strip())
    except Exception:
        return False


def _runid(prefix, tag):
    return f"{prefix}_{tag}"


def _newest_tag():
    tags = set()
    for j in RUNS.glob("ab_batch_*.json"):
        m = re.match(r"ab_batch_(.+)\.json", j.name)
        if m:
            tags.add((j.stat().st_mtime, m.group(1)))
    return sorted(tags)[-1][1] if tags else None


def _run_dir(sub, runid):
    return RUNS / sub / runid


def _tool_call_count(d, tokens):
    """How many Bash tool calls in this run invoked one of ``tokens`` (an arm's granted tooling).

    Counts CALLS, not token occurrences. The predecessor summed ``cmd.count(token)`` over every token,
    so a single command that dumped three granted scripts at once scored 3 -- inflating exactly the
    number a reader uses to decide whether a rung's treatment fired at all. Measured on this batch: one
    arm-3 command named ``cca_contract``, ``action_catalog`` and ``isa_tools`` in one line.
    """
    n = 0
    if not tokens:
        return "n/a"
    # round_*.transcript.jsonl ONLY, matching every other consumer. `rounds/` also holds each round's
    # untouched raw stdout tee (round_NN.stream.raw.jsonl) and the codex raw/timestamped streams, and a
    # bare "*.jsonl" counted the same tool call once per copy.
    for tp in sorted((d / "rounds").glob("round_*.transcript.jsonl")) if (d / "rounds").is_dir() else []:
        for ln in tp.read_text(errors="ignore").splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") != "assistant":
                continue
            for b in o.get("message", {}).get("content", []) or []:
                if b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmd = b.get("input", {}).get("command", "")
                    n += 1 if any(g in cmd for g in tokens) else 0
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    tag = a.tag or _newest_tag()
    if not tag:
        print("no batch found"); return 1
    now = time.time()
    print(f"=== A/B batch '{tag}'  ({time.strftime('%H:%M:%S')}) ===")
    print(f"{'arm':14s} {'alive':5s} {'round':7s} {'last':>6s} {'grades (spike L0-L2)':28s} "
          f"{'L3 cert':14s} tools")
    for label, sub, prefix, tokens in ARMS:
        rid = _runid(prefix, tag)
        d = _run_dir(sub, rid)
        if not d.is_dir():
            print(f"{label:14s} (no run dir)"); continue
        rounds = sorted((d / "rounds").glob("round_*.transcript.jsonl"))
        cur = rounds[-1].stem.split(".")[0].replace("round_", "r") if rounds else "init"
        age = f"{int(now - rounds[-1].stat().st_mtime)}s" if rounds else "-"
        grades = []
        for v in sorted((d / "qa_history").glob("verdict_round_*.json")) if (d / "qa_history").is_dir() else []:
            try:
                j = json.loads(v.read_text())
                grades.append(f"r{v.stem.split('_')[-1]}={j.get('n_passed')}/{j.get('n_capsules')}")
            except Exception:
                pass
        l3 = "-"
        vcp = d / "verilator_checkpoints.json"
        if vcp.is_file():
            try:
                cj = json.loads(vcp.read_text())
                last = cj.get("attempts", [{}])[-1] if cj.get("attempts") else {}
                l3 = f"{last.get('n_passed','?')}/{last.get('n_capsules','?')} pass={cj.get('final_all_pass')}"
            except Exception:
                pass
        tooled = _tool_call_count(d, tokens)
        alive = "yes" if _alive(rid) else "DEAD"
        print(f"{label:14s} {alive:5s} {cur:7s} {age:>6s} {str(grades) or 'none yet':28.28s} {l3:14s} {tooled}")
    print("\n  grades = spike L0/L1/trace/L2 per graded round; L3 = cycle-accurate verilator cert "
          "(non-terminal). 'converged' when L3 all-pass.")
    print("  tools  = Bash calls invoking THIS arm's granted tooling ('n/a' = the rung grants none, "
          "so there is nothing to fire). A rung sitting at 0 has an untreated treatment: its score is "
          "not evidence about the tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
