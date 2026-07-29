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

SMOKE = r"""You are running inside an isolated sandbox for a smoke test. Do EXACTLY this, then stop.

Run each command with the Bash tool and record whether it succeeds. Then try to read the answer files
(these MUST be unreadable/empty — that is correct, not a failure).

TOOLS (must succeed):
1. `python3 -c "import merlin.targetgen.oot_starterkit; import xdsl; print('merlin_ok')"`  (skip if baseline)
2. `mlir-opt --version | head -1`
3. `g++ --version | head -1`
4. `cmake --version | head -1`
5. `ninja --version`
6. `printf 'int main(){return 0;}' > /tmp/s.c && riscv64-unknown-elf-gcc -I "$MERLIN_GEMMINI_HARNESS_DIR/include" -c /tmp/s.c -o /tmp/s.o && echo gcc_ok`
7. `spike --help >/dev/null 2>&1 && echo spike_ok`
8. (only if merlin_rtlchecks) `python3 -m merlin.targetgen.rtl.gen_isa_module --out /tmp/isa.py >/dev/null 2>&1 && echo circt_ok`
8b. SELF-CHECK via the broker: `python3 agent_selfcheck.py --sim spike --capsules A1_mvin_mvout --timeout 120` — it returns a REDACTED JSON verdict (you have no submission yet, so expect an error/fail verdict — that's fine; we only need it to RETURN a verdict, proving the self-check works). If you got any JSON back, echo `selfcheck_ok`.
8c. ASYNC ORACLE: `python3 simjob.py submit --sim spike --capsules A1_mvin_mvout` — it prints JSON with a `job_id` IMMEDIATELY (async). Capture that job_id, then run `python3 simjob.py poll --job-id <id>` (repeat a few times if state is queued/running). If you got a JSON response back from poll, echo `simjob_ok`.

ANSWERS (must be UNREADABLE — confirm you are blocked):
9. `cat merlin/contract/capsules/isa/A4_acc_scale_i8/golden.yaml 2>&1 | head -c 40` (expect empty/error)

Finally print ONE line of valid JSON, nothing else after it:
{"tools_ok": ["..."], "tools_failed": ["..."], "answers_readable": true_or_false}
"""


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
    print(f"=== LIVE smoke agent — arm={a.arm} model={a.model} (sandbox=bwrap, real driver path) ===")
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        ws = Path(td) / "workspace"
        RX.assemble_workspace(bundle, ws)
        (ws / "SMOKE.md").write_text(SMOKE)
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
        # markers must appear in tool RESULTS (echoed by the commands actually executing)
        markers = {"gcc_ok": "gcc_ok" in blob, "spike_ok": "spike_ok" in blob,
                   "merlin_ok": ("merlin_ok" in blob) or a.arm == "baseline",
                   "mlir/cmake/ninja ran": ("LLVM" in blob or "cmake version" in blob),
                   "async simjob oracle returns job+poll": ("simjob_ok" in blob or '"job_id"' in blob),
                   "self-check via broker returns verdict": ("selfcheck_ok" in blob or "manifest" in blob
                                                             or '"n_passed"' in blob)}
        if a.arm == "merlin_rtlchecks":
            # accept either the echoed marker OR the generator's own success line (a cheap haiku run may
            # &&-chain step 8 differently); the generator is also covered deterministically by readiness.
            markers["circt_ok"] = ("circt_ok" in blob or "gen_isa_module" in blob or "wrote /tmp/isa.py" in blob
                                   or "isa.py" in blob)
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
