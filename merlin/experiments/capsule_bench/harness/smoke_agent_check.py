"""Small LIVE-agent smoke test: launch a real claude agent inside the bwrap sandbox (the actual driver
path, bwrap_cmd) and have it try to reach + USE every tool it might need, then report. This validates
what static checks can't: claude itself runs under bwrap (auth + network + stdin), and the agent can
actually invoke each tool — and still cannot read any answer.

Cheap by design: one short claude call (haiku), a fixed checklist prompt, ~1-2 min. No backend authored.
Usage: smoke_agent_check.py [--arm merlin_rtlchecks|merlin|baseline] [--model claude-haiku-4-5]
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C
import run_agent_experiment as RX
import run_baseline_qa_loop as QA
import yaml

ARM_BUNDLE = {
    "baseline": "raw_baseline_hwbringup_v0",
    "merlin": "merlin_assisted_hwbringup_v0",
    "merlin_rtlchecks": "merlin_assisted_rtlchecks_hwbringup_v0",
}

def _smoke_prompt(arm: str, is_rtlchecks: bool, golden_rel: str, facts_rel: str, capsule: str) -> str:
    """Build a TARGET-AWARE smoke checklist. Tests what the agent ACTUALLY uses, not a gemmini-fixed list:
    the merlin arms build their dialect with the xDSL kit and COPY the oot_starterkit — they must NOT
    ``import merlin`` (the integrity contract forbids it; the sandbox blocks it), so we check ``import
    xdsl`` + a readable starterkit, never an ``import merlin`` / ``python3 -m merlin...``. The sim tools
    are the target's own (``--sim {sim_via}``), the RTL surface is the arm-4 rtl_facts DIR (readable, not a
    RoCC generator run), and the answer-isolation cat targets an EXISTING masked golden of THIS target."""
    T: list[str] = []
    if arm != "baseline":
        T.append('`python3 -c "import xdsl; from xdsl.ir import Dialect; print(\'xdsl_ok\')"`  (the xDSL spine; do NOT import merlin — that is forbidden and blocked)')
        T.append('`ls oot_starterkit >/dev/null 2>&1 && echo starterkit_ok`  (the scaffold you COPY from)')
    T += ['`mlir-opt --version | head -1`', '`cmake --version | head -1`', '`ninja --version`',
          '`g++ --version | head -1`']
    if is_rtlchecks and facts_rel:
        T.append(f'`ls {facts_rel} >/dev/null 2>&1 && echo rtlfacts_ok`  (the arm-4 derived RTL facts you read)')
    n = len(T)
    lines = ["You are running inside an isolated sandbox for a smoke test. Do EXACTLY this, then stop.", "",
             "Run each command with the Bash tool and record whether it succeeds (each echoes an _ok marker).",
             "", "TOOLS (must succeed):"]
    lines += [f"{i}. {cmd}" for i, cmd in enumerate(T, 1)]
    lines += [
        f"{n+1}. SELF-CHECK via the broker: `python3 agent_selfcheck.py --capsules {capsule} --timeout 120`"
        " — returns a REDACTED JSON verdict (no submission yet, so an error/fail verdict is fine; we only"
        " need it to RETURN a verdict). If you got JSON back, echo `selfcheck_ok`.",
        f"{n+2}. ASYNC ORACLE: `python3 simjob.py submit --capsules {capsule}` — prints JSON with a"
        " `job_id` immediately. Capture it, then `python3 simjob.py poll --job-id <id>` (repeat if"
        " queued/running). If poll returned JSON, echo `simjob_ok`.",
        "", "ANSWER (must be UNREADABLE — confirm you are blocked):",
        f"{n+3}. `cat {golden_rel} 2>&1 | head -c 40` (expect empty/error — you MUST NOT be able to read it)",
        "", "Finally print ONE line of valid JSON, nothing else after it:",
        '{"tools_ok": ["..."], "tools_failed": ["..."], "answers_readable": true_or_false}']
    return "\n".join(lines) + "\n"


def _bedrock_ping(model: str, region: str, profile: str, timeout: int) -> int:
    """Confirm the agent's `claude` CLI can reach Bedrock (creds + model access) with one tiny call —
    before committing a real run to Bedrock. Scopes the Bedrock env to THIS process only; the sandbox's
    claude_runtime_binds binds ~/.aws because CLAUDE_CODE_USE_BEDROCK=1 is set here."""
    env = dict(os.environ)
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    env["AWS_REGION"] = region
    env.setdefault("AWS_DEFAULT_REGION", region)
    if profile:
        env["AWS_PROFILE"] = profile
    elif not env.get("AWS_BEARER_TOKEN_BEDROCK"):
        # Bearer-token auth lives in the gitignored .env (read-only loader won't export it).
        from merlin.common.paths import env as _dotenv
        _bearer = _dotenv("AWS_BEARER_TOKEN_BEDROCK")
        if _bearer:
            env["AWS_BEARER_TOKEN_BEDROCK"] = _bearer
    print(f"=== Bedrock ping — model={model} region={region} "
          f"{'profile=' + profile if profile else 'env-var creds'} ===")
    try:
        r = subprocess.run(["claude", "--print", "--model", model, "Reply with exactly: BEDROCK_OK"],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT — no response from Bedrock"); return 1
    ok = "BEDROCK_OK" in (r.stdout or "")
    print(f"  reply: {(r.stdout or '').strip()[:120] or '(none)'}")
    if not ok:
        print("  stderr tail:", (r.stderr or "")[-300:])
    print(f"\n  bedrock verdict: reachable={ok} (rc={r.returncode})")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="merlin_rtlchecks", choices=list(ARM_BUNDLE))
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription",
                    help="bedrock: run a minimal Bedrock reachability ping instead of the full tool smoke")
    ap.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""))
    a = ap.parse_args(argv)
    if a.provider == "bedrock":
        return _bedrock_ping(a.model, a.aws_region, a.aws_profile, a.timeout)
    bundle = yaml.safe_load((C.BUNDLES / ARM_BUNDLE[a.arm] / "input_bundle_manifest.yaml").read_text())
    # DERIVE the target's own facts so the checklist is target-aware (not gemmini-fixed): its sim engine,
    # an EXISTING masked golden to probe isolation against, and (arm-4) its RTL-facts dir.
    from merlin.targetgen.target_experiment import load_target_experiment
    te = load_target_experiment(C.EXP / "target_experiment.yaml")
    sim = te.sim_via or ""
    is_rtlchecks = a.arm == "merlin_rtlchecks"
    # assemble_workspace flattens each grant to its BASENAME at the workspace root, so checks/paths must be
    # workspace-relative basenames, not repo paths. The isolation cat targets an EXISTING (but masked) golden
    # of THIS target's corpus so "unreadable" means MASKED, not missing.
    corpus = C.REPO / (te.capsule_corpus or "merlin/contract/capsules")
    goldens = sorted(corpus.rglob("golden.yaml")) if corpus.is_dir() else []
    capsule = goldens[0].parent.name if goldens else "all"   # a single real capsule (simple for a haiku agent)
    golden_rel = f"{corpus.name}/{capsule}/golden.yaml" if goldens else corpus.name
    # arm-4 RTL facts are granted as basename 'rtl_facts' ONLY if the target actually has derived facts (a
    # SIMT target has none — n/a, not a failure); gate on the host grant source existing.
    facts_rel = "rtl_facts" if (C.REPO / f"merlin/targets/{te.target}/contracts/rtl_facts").is_dir() else ""
    smoke = _smoke_prompt(a.arm, is_rtlchecks, golden_rel, facts_rel, capsule)
    print(f"=== LIVE smoke agent — arm={a.arm} target={te.target} sim={sim or 'n/a'} model={a.model} "
          f"(sandbox=bwrap, real driver path) ===")
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        ws = Path(td) / "workspace"
        RX.assemble_workspace(bundle, ws)
        (ws / "SMOKE.md").write_text(smoke)
        # EXACTLY the driver's flags (bypassPermissions so Bash runs; --add-dir for the workspace)
        inner = (f'claude --print --model {a.model} --permission-mode bypassPermissions --add-dir {ws} '
                 f'--output-format stream-json --verbose < {ws / "SMOKE.md"}')
        cmd = QA.bwrap_cmd(inner, ws, bundle)   # the REAL driver sandbox path
        broker = QA._start_selfcheck_broker(ws)   # same as launch_agent: shim + driver-side broker
        print(f"  launching claude inside bwrap + broker (timeout {a.timeout}s)...")
        try:
            r = subprocess.run(["bash", "-c", cmd], cwd=str(ws), capture_output=True,
                               text=True, timeout=a.timeout)
        except subprocess.TimeoutExpired:
            QA._stop_selfcheck_broker(ws, broker)
            print("  TIMEOUT — agent did not finish"); return 1
        QA._stop_selfcheck_broker(ws, broker)

        # parse stream-json: collect tool_result OUTPUTS (not command echoes) + count permission denials
        import json
        tool_outputs, denials, final_text = [], 0, ""
        for ln in r.stdout.splitlines():
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("type") == "result":
                denials += len(o.get("permission_denials") or [])
                final_text = o.get("result", "") or final_text
            for b in (o.get("message", {}) or {}).get("content", []) or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    c = b.get("content")
                    tool_outputs.append(c if isinstance(c, str) else json.dumps(c))
        blob = "\n".join(tool_outputs)
        # the agent self-reports answers_readable in its final JSON — the isolation invariant
        answers_readable = None
        for cand in reversed(final_text.splitlines()):
            s = cand.strip().strip("`").strip()
            if s.startswith("{") and "answers_readable" in s:
                try:
                    answers_readable = bool(json.loads(s).get("answers_readable"))
                except Exception:  # noqa: BLE001 — malformed line, leave as unknown
                    pass
                break
        # TARGET-AWARE markers (must appear in tool RESULTS). No gemmini-fixed gcc/spike/merlin: the merlin
        # arms use the xDSL kit and MUST NOT import merlin; the sim + RTL surface are the target's own.
        merlin_arm = a.arm != "baseline"
        markers = {"mlir/cmake/ninja/g++ ran": ("LLVM" in blob or "cmake version" in blob),
                   "async simjob oracle returns job+poll": ("simjob_ok" in blob or '"job_id"' in blob),
                   # a RETURNED verdict is the signal — incl. the expected no-submission verdict
                   # ({"error": "...build your package first"}), which proves the selfcheck path works.
                   "self-check via broker returns verdict": ("selfcheck_ok" in blob or '"n_passed"' in blob
                                                             or '"all_pass"' in blob
                                                             or "build your package first" in blob),
                   "answers masked (isolation)": answers_readable is False}
        if merlin_arm:
            markers["xdsl spine importable"] = "xdsl_ok" in blob
            markers["oot_starterkit readable"] = "starterkit_ok" in blob
        if is_rtlchecks and facts_rel:
            markers["arm-4 rtl_facts readable"] = "rtlfacts_ok" in blob
        print("\n--- agent final line ---\n  " + (final_text[:300] if final_text else "(none)"))
        print("  tool-result markers:", {k: ("✓" if v else "✗") for k, v in markers.items()})
        print(f"  permission_denials: {denials}  (want 0 — bypassPermissions)")
        ok = all(markers.values()) and denials == 0
        print(f"\n  smoke verdict: tools reached+USED={ok}  (rc={r.returncode})")
        if not ok and r.returncode != 0:
            print("  stderr tail:", (r.stderr or "")[-300:])
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
