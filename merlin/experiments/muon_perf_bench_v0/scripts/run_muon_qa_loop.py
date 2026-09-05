#!/usr/bin/env python
"""Single-arm Muon merlin+CIRCT agentic QA loop.

The Muon analog of run_rtlchecks_qa_loop.py — but a single arm (the best/merlin+CIRCT setting) and far
simpler, because cyclotron (~1 s/kernel) is BOTH the fast iterate oracle and the cycle-accurate perf
oracle, so there is no verilator/simjob/broker. Each round: a fresh sandboxed `claude` agent authors
`submission/` (a self-contained Muon SIMT backend) from the answer-free workspace + the RTL-grounded
MUON_DIGEST (the "+CIRCT" advisory); the operator then grades it on cyclotron and hands back a redacted
verdict for the next round. Stops when all public capsules pass; then drives cycles toward the FP peak.

Isolation is sandbox=none (copy/symlink the answer-free materials; the Bun `claude` binary SIGILLs under
bwrap) + a submission integrity scan + the documented no-import-merlin rule.

Usage:
  run_muon_qa_loop.py --run-id muon_circt_0001 --model claude-opus-4-8 [--effort high]
                      [--max-rounds 6] [--round-timeout 3600]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _c in (_HERE, *_HERE.parents):
    if (_c / "merlin" / "python").is_dir():
        _REPO = _c
        sys.path.insert(0, str(_c / "merlin" / "python"))
        break
sys.path.insert(0, str(_HERE))
import agent_selfcheck as SC                      # noqa: E402  (grade())
from merlin.common import arrival_stamp as AS  # noqa: E402  (one arrival-time convention)
from merlin.benchharness import runs_root          # noqa: E402  (canonical out/runs root)
from merlin.targetgen import experiment_tokens as ET   # noqa: E402  (canonical transcript parser)
from merlin.targetgen.rtl.facts import rtl_facts_path   # noqa: E402  (target-agnostic facts dir)

EXP = _REPO / "merlin/experiments/muon_perf_bench_v0"
KERNELS = EXP / "kernels"
BUNDLE = EXP / "input_bundles/muon_rtlchecks_public_v0"
# muon has no curated merlin/targets/muon; the resolver routes to artifacts/targets/muon.
FACTS = rtl_facts_path("muon").parent
TASK = EXP / "task" / "TASK_muon.md"
REFERENCE = _REPO / "out/artifacts/targets/muon/reference_v0"

# answer-bearing names never staged into the agent's workspace
_DROP = {"golden.yaml", "expected_command_buffer_g0.json"}


def assemble_workspace(ws: Path) -> None:
    """Stage the answer-free materials: contract (symlink), public capsule INPUTS (minus golden),
    the RTL digest + backend guide, and the self-check tool. The reference backend, the Merlin
    reference/simulator, and goldens are never staged."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    # the contract is answer-free (goldens live with the capsules, not in the contract)
    _link(ws / "merlin/contract", _REPO / "merlin/contract")
    # public capsule inputs, minus golden.yaml
    caps = ws / "capsules"
    caps.mkdir(exist_ok=True)
    for capdir in sorted(KERNELS.iterdir()):
        if not (capdir / "capsule.yaml").is_file():
            continue
        d = caps / capdir.name
        d.mkdir(exist_ok=True)
        for f in capdir.iterdir():
            if f.name in _DROP:
                continue
            if f.suffix in (".yaml", ".mlir"):
                shutil.copy(f, d / f.name)
    # docs (the +CIRCT advisory + the how-to) and the self-check tool — advisories are optional;
    # skip any that aren't present (e.g. MUON_DIGEST.md is produced by muon_introspect, may be absent)
    for src, dst in ((FACTS / "MUON_DIGEST.md", "MUON_DIGEST.md"),
                     (BUNDLE / "MUON_BACKEND_GUIDE.md", "MUON_BACKEND_GUIDE.md")):
        if src.is_file():
            shutil.copy(src, ws / dst)
        else:
            print(f"  note: advisory {src.name} absent ({src}) — skipping", file=sys.stderr)
    shutil.copy(EXP / "scripts" / "agent_selfcheck.py", ws / "agent_selfcheck.py")


def _link(dst: Path, src: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def build_task(ws: Path, run_dir: Path) -> None:
    task = TASK.read_text(encoding="utf-8")
    (ws / "TASK.md").write_text(task, encoding="utf-8")
    shutil.copy(ws / "TASK.md", run_dir / "TASK.md")


def launch_agent(ws: Path, run_dir: Path, model: str, effort: str, rnd: int, timeout: int) -> int:
    if rnd == 0:
        build_task(ws, run_dir)
    ws_task = ws / "TASK.md"
    inner = (f"claude --print --model {model} --effort {effort} "
             f"--permission-mode bypassPermissions --add-dir {ws} "
             f"--output-format stream-json --verbose < {ws_task}")
    (run_dir / "rounds").mkdir(parents=True, exist_ok=True)
    tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
    epath = run_dir / "rounds" / f"round_{rnd:02d}.stderr.log"
    # Streamed, not redirected: a straight stdout redirect leaves no process able to observe a line,
    # so the transcript carries no per-event wall time and a trajectory has to synthesise its axis.
    # arrival_stamp appends `arrived_at` to every event, in the same shape every other driver writes.
    return AS.stream_stamped(["bash", "-c", inner], cwd=ws, transcript=tpath, stderr_path=epath,
                             timeout=timeout,
                             raw_path=run_dir / "rounds" / f"round_{rnd:02d}.stream.raw.jsonl")


def grade_round(ws: Path, run_dir: Path, rnd: int, timeout: int) -> dict:
    runs = run_dir / "_qa_work" / f"runs_{rnd:02d}"
    if not (ws / "submission" / "manifest.yaml").exists():
        verdict = {"all_pass": False, "n_passed": 0, "n_capsules": 0,
                   "package_failure": {"plane": "schema", "detail": "no submission/manifest.yaml"},
                   "per_capsule": []}
    else:
        verdict = SC.grade(str(ws / "submission"), str(ws / "capsules"), str(runs), timeout)
    (run_dir / "qa_history").mkdir(parents=True, exist_ok=True)
    (run_dir / "qa_history" / f"verdict_round_{rnd:02d}.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    (ws / "qa").mkdir(exist_ok=True)
    (ws / "qa" / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def _transcript_usage(tpath: Path) -> dict:
    """Cost/token sum via the canonical `experiment_tokens.parse_transcript` (shared with the gemmini
    QA loop), mapped to the keys finalize_report reads. Honest zeros if no usage metadata."""
    s = ET.parse_transcript(tpath)
    if not s.get("available"):
        return {"cost_usd": 0, "input_tokens": 0, "output_tokens": 0}
    return {"cost_usd": s.get("estimated_cost_usd", 0) or 0,
            "input_tokens": s.get("tokens_input", 0) or 0,
            "output_tokens": s.get("tokens_output", 0) or 0}


def finalize_report(run_dir: Path, model: str, effort: str, rounds: list[dict],
                    verdict: dict, wall_s: float) -> None:
    last = verdict
    lines = [f"# Muon merlin+CIRCT agentic run — {run_dir.name}", "",
             f"- model: `{model}`  effort: `{effort}`  rounds: {len(rounds)}  "
             f"wall: {wall_s/60:.1f} min",
             f"- result: **{last['n_passed']}/{last['n_capsules']} public capsules pass** "
             f"({'ALL PASS' if last['all_pass'] else 'incomplete'})",
             f"- FP peak = 32 GFLOP/s (64 flop/cycle @ 500 MHz)", "",
             "## Per-capsule (final round)", "",
             "| capsule | status | cycles | % FP peak |", "|---|---|---:|---:|"]
    for r in last.get("per_capsule", []):
        lines.append(f"| {r['capsule']} | {r['status']} | {r.get('cycles') or '-'} | "
                     f"{(str(r.get('pct_fp_peak'))+'%') if r.get('pct_fp_peak') is not None else '-'} |")
    lines += ["", "## Effort per round", "", "| round | cost $ | in tok | out tok | passed |",
              "|---|---:|---:|---:|---:|"]
    total_cost = 0.0
    for i, rr in enumerate(rounds):
        u = rr["usage"]
        total_cost += u["cost_usd"] or 0
        lines.append(f"| {i} | {u['cost_usd']:.2f} | {u['input_tokens']} | {u['output_tokens']} | "
                     f"{rr['verdict']['n_passed']}/{rr['verdict']['n_capsules']} |")
    lines.append("")
    lines.append(f"**total cost ≈ ${total_cost:.2f}**")
    (run_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(
        {"run_id": run_dir.name, "model": model, "effort": effort, "rounds": len(rounds),
         "wall_s": wall_s, "all_pass": last["all_pass"], "final_verdict": last,
         "total_cost_usd": total_cost}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Muon merlin+CIRCT single-arm agentic QA loop")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--round-timeout", type=int, default=14400)  # 4h; a tight cap forces more rounds (net-detrimental)
    ap.add_argument("--grade-timeout", type=int, default=300)
    ap.add_argument("--runs-root", default=str(runs_root("muon", "perf-bench")))
    ap.add_argument("--keep-ws", action="store_true")
    a = ap.parse_args(argv)

    run_dir = Path(a.runs_root) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ws = run_dir / "workspace"
    assemble_workspace(ws)

    t0 = time.time()
    rounds: list[dict] = []
    verdict = {"all_pass": False, "n_passed": 0, "n_capsules": 0, "per_capsule": []}
    for rnd in range(a.max_rounds):
        print(f"\n=== round {rnd} (model={a.model}, effort={a.effort}) ===", flush=True)
        try:
            rc = launch_agent(ws, run_dir, a.model, a.effort, rnd, a.round_timeout)
        except subprocess.TimeoutExpired:
            print(f"  round {rnd} agent timed out after {a.round_timeout}s", flush=True)
            rc = -1
        verdict = grade_round(ws, run_dir, rnd, a.grade_timeout)
        usage = _transcript_usage(run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl")
        rounds.append({"round": rnd, "rc": rc, "verdict": verdict, "usage": usage})
        print(f"  -> {verdict['n_passed']}/{verdict['n_capsules']} pass "
              f"(cost ${usage['cost_usd']:.2f})", flush=True)
        if verdict["all_pass"]:
            print(f"  ALL PASS at round {rnd}", flush=True)
            break

    finalize_report(run_dir, a.model, a.effort, rounds, verdict, time.time() - t0)
    if not a.keep_ws:
        # keep the submission, drop the bulky symlinked workspace scaffold
        sub = ws / "submission"
        if sub.exists():
            shutil.copytree(sub, run_dir / "submission", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "build"))
    print(f"\nwrote {run_dir/'final_report.md'}")
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
