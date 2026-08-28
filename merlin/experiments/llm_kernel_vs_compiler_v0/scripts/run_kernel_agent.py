#!/usr/bin/env python3
"""Drive one agent through generate -> compile -> REDACTED feedback -> repair, on one task.

The feedback-driven loop the study measures, not one-shot prompting. Each round the agent writes a
kernel, the shared evaluator grades it through the real oracle, and only the redacted view of that
verdict goes back into the workspace. The agent sees what a developer profiling their own kernel
would see -- compiler errors, mismatch counts, error magnitudes, cycles -- and never an expected
value.

Each agent CLI is invoked DIRECTLY with the task card this module writes, rather than through the
capsule-bench driver wrappers. Those wrappers inject a system prompt for a different job -- "build the
target backend under submission/ (manifest.yaml + the entrypoint tool)" -- which contradicts a task
that asks for one kernel file. An agent given both would be scored on whether it guessed which
instruction to obey, and that failure would look like a weak model. Driving the CLIs directly keeps
the task identical across arms, so a score difference is attributable to the model.

The harness still differs between arms and cannot be made not to: the Codex seat is reachable only
through `codex`, while Bedrock and Google are reached through `opencode`. That is recorded per run
rather than papered over -- the harness is a known first-order variable.

Every round is recorded whether it passed or not. A study that keeps only winners cannot report the
cost of the failures, and the failures are most of the cost.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

#: Where the agent must leave its kernel. Named in TASK.md and read by the shim.
KERNEL_REL = "submission/kernel.llvm.mlir"


def _repo_root() -> Path:
    for anc in HERE.parents:
        if (anc / "merlin" / "python" / "merlin").is_dir():
            return anc
    raise SystemExit("could not locate the repo root from the script location")


def _load_method(name: str) -> dict:
    import yaml
    p = EXP / "methods" / name / "method.yaml"
    if not p.is_file():
        raise SystemExit(f"no method config at {p}")
    return yaml.safe_load(p.read_text()) or {}


def _task_card(capsule_dir: Path, iface: str, rnd: int) -> str:
    """What the agent is told. Contains the interface and the ABI -- never a reference answer."""
    return f"""# Task: write a compute kernel for this accelerator

You are writing ONE compute kernel for a SIMT accelerator, as **LLVM-dialect MLIR**.

Write it to exactly this path, and nothing else:

    {KERNEL_REL}

## The operation

This is the capsule interface you must implement. It names the operands, their shapes and dtypes,
and the order of operations:

```mlir
{iface}
```

## The kernel ABI

Emit a single `llvm.func` named `radiance_kernel`. Its parameters are plain pointers, in this order:

    [weight tensors] ++ [input tensors in command order] ++ [output tensors in command order]

Read the operands through `llvm.getelementptr` + `llvm.load`, compute, and `llvm.store` into the
output pointers.

**Every operand is 32-bit float in memory, whatever dtype the interface declares.** The harness
decodes a declared f16 / bf16 / fp8 operand to its numeric value and stores it as one `f32` per
element, so a 16x32 f16 tensor arrives as 512 consecutive `f32` values -- NOT as packed halves.
Load `f32`, compute in `f32`, store `f32`. The declared dtype tells you the numeric PRECISION the
values were rounded to; it does not describe the memory layout. Plain scalar compute is correct and sufficient -- warps, barriers and scheduling are
supplied by the runtime, so do not emit them. Do not emit `.insn`, raw `.word`, DRAM addresses, or a
halt.

## How you are graded

Your kernel is compiled with a stock toolchain and run on a cycle-accurate simulator. Its outputs are
compared against an independent reference you cannot see. You will receive, each round, only what a
developer profiling their own kernel would have: compiler and linker errors, whether it ran, how many
elements mismatched and by how much, and the cycle count.

Correctness first. A kernel that is fast and wrong scores nothing.

## Rules

- Your kernel may not import the grader, read a golden file, or shell out. This is scanned.
- Do not modify anything outside `{KERNEL_REL}`.

Round {rnd}. If `qa/verdict.json` exists, it is the grader's verdict on your previous attempt --
read it first and fix what it reports.
"""


def _evaluate(kernel: Path, capsule_dir: Path, *, shim_pkg: Path, runs_root: Path,
              task_id: str, config_id: str, target: str, method: str, timeout: int) -> dict:
    """Grade the agent's kernel through the shim, and return (full, redacted) as plain dicts.

    Run in a SUBPROCESS: the oracle pulls in the target backend and a simulator, and a crash there
    must fail this one evaluation rather than take the whole campaign down with it.
    """
    repo = _repo_root()
    code = f'''
import json, pathlib, sys
from merlin.runtime.backends.base import get_backend
from merlin.benchharness import evaluation as EV
MR = get_backend("muon").muon_capsule_runner
d = pathlib.Path({str(capsule_dir)!r})
try:
    r = MR.run_capsule(MR.load_capsule(d), {str(shim_pkg)!r},
                       runs_root={str(runs_root)!r}, target={target!r}, timeout={timeout})
except Exception as e:
    r = {{"status": "error", "tiers": {{}}, "numeric": {{}},
          "failure": {{"plane": "oracle", "category": "tool_crash", "detail": f"{{type(e).__name__}}: {{e}}"}}}}
ev = EV.from_capsule_result(r, task_id={task_id!r}, config_id={config_id!r}, target={target!r},
                            certifying_tiers=frozenset({{"L2"}}), method={method!r},
                            artifact_provenance="agent_kernel_in_harness_shim")
print("<<<KVC>>>" + json.dumps({{"full": ev.to_dict(), "redacted": ev.redact()}}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "merlin" / "python")
    env["MERLIN_KVC_KERNEL_FILE"] = str(kernel)
    env["MERLIN_MUON_SKIP_RTL_L3"] = env.get("MERLIN_MUON_SKIP_RTL_L3", "1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=str(repo), timeout=timeout + 300)
    for line in r.stdout.splitlines():
        if line.startswith("<<<KVC>>>"):
            return json.loads(line[len("<<<KVC>>>"):])
    return {"full": {"status": "error", "failure_detail": (r.stderr or r.stdout)[-600:]},
            "redacted": {"status": "error", "correct": False,
                         "failure_detail": (r.stderr or r.stdout)[-600:]}}



def _run_agent(cfg: dict, ws: Path, run_dir: Path, rnd: int, prompt: str,
               timeout: int) -> tuple[int, Path | None]:
    """Invoke the agent CLI for this method, with OUR prompt. Returns (rc, raw-output path).

    The full stdout is kept per round regardless of outcome -- a failed round is part of the cost
    distribution and its tokens still count.
    """
    rounds = run_dir / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    out = rounds / f"round_{rnd:02d}.agent.jsonl"
    driver = cfg["driver"]

    env = dict(os.environ)
    if cfg.get("provider") == "google":
        # opencode's Google provider reads a DIFFERENT variable name than the key is stored under;
        # without the alias it fails with ProviderAuthError in ~2s, which reads as a weak agent.
        env.setdefault("GOOGLE_GENERATIVE_AI_API_KEY", env.get("GOOGLE_API_KEY", ""))
    if cfg.get("aws_region"):
        env["AWS_REGION"] = str(cfg["aws_region"])

    if driver == "codex":
        argv = ["codex", "exec", "--json", "--skip-git-repo-check",
                "-c", "approval_policy=never", "--sandbox", "workspace-write", "--cd", str(ws)]
        stdin = prompt
    elif driver == "opencode":
        model = cfg["model"]
        if cfg.get("provider") == "bedrock" and "/" not in model:
            model = f"amazon-bedrock/{model}"
        argv = ["opencode", "run", "--format", "json", "-m", model, "--auto", prompt]
        stdin = None
    else:
        raise SystemExit(f"no CLI wiring for driver {driver!r}")

    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                           timeout=timeout, cwd=str(ws), env=env)
        out.write_text(r.stdout + ("\n" + r.stderr if r.stderr else ""))
        return r.returncode, out
    except subprocess.TimeoutExpired:
        out.write_text(f"TIMEOUT after {timeout}s")
        return 124, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, help="a dir under methods/ (e.g. codex_kernel)")
    ap.add_argument("--capsule", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--target", default="radiance")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--round-timeout", type=int, default=1800)
    ap.add_argument("--eval-timeout", type=int, default=900)
    ap.add_argument("--runs-root", type=Path,
                    default=Path("/scratch/agustin/tmp/kvc-runs"))
    ap.add_argument("--sandbox", default="none")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)

    repo = _repo_root()
    cfg = _load_method(a.method)

    # The routing policy is a launch gate, not advice: refuse to spend on an unapproved route.
    sys.path.insert(0, str(HERE))
    import check_method_models as policy
    bad = policy.violations(cfg, where=a.method)
    if bad:
        for b in bad:
            print(f"  !! {b}")
        raise SystemExit("routing policy violation -- refusing to launch")

    capsule_dir = a.capsule if a.capsule.is_absolute() else repo / a.capsule
    iface = (capsule_dir / "capsule.interface.mlir").read_text()

    run_dir = a.runs_root / "agents" / f"{a.run_id}"
    ws = run_dir / "ws"
    (ws / "submission").mkdir(parents=True, exist_ok=True)
    (ws / "qa").mkdir(parents=True, exist_ok=True)
    (run_dir / "rounds").mkdir(parents=True, exist_ok=True)

    # Stage the shim package for this run.
    shim_pkg = run_dir / "shim"
    shim_pkg.mkdir(parents=True, exist_ok=True)
    for f in ("kernelshim", "manifest.yaml"):
        shutil.copy2(EXP / "shim" / f, shim_pkg / f)
    os.environ.setdefault(
        "MERLIN_KVC_REFERENCE_TOOL",
        str(repo / "out" / "artifacts" / "targets" / "muon" / "reference_v0" / "muon-opt"))


    kernel = ws / KERNEL_REL
    history: list[dict] = []
    t_start = time.time()

    for rnd in range(a.rounds):
        card = _task_card(capsule_dir, iface, rnd)
        prev = (ws / "qa" / "verdict.json")
        if rnd and prev.is_file():
            card += ("\n## The grader's verdict on your previous attempt\n\n```json\n"
                     + prev.read_text()[:4000] + "\n```\n")
        (ws / "TASK.md").write_text(card)
        t0 = time.time()
        rc, transcript = _run_agent(cfg, ws, run_dir, rnd, card, a.round_timeout)
        agent_s = time.time() - t0

        rec: dict = {"round": rnd, "rc": rc, "agent_seconds": round(agent_s, 1),
                     "transcript": str(transcript) if transcript else None,
                     "kernel_written": kernel.is_file()}
        if kernel.is_file():
            t1 = time.time()
            res = _evaluate(kernel, capsule_dir, shim_pkg=shim_pkg, runs_root=a.runs_root,
                            task_id=capsule_dir.name, config_id="C0", target=a.target,
                            method=a.method, timeout=a.eval_timeout)
            rec["eval_seconds"] = round(time.time() - t1, 1)
            rec["evaluation"] = res["full"]
            # ONLY the redacted view reaches the workspace the agent reads.
            (ws / "qa" / "verdict.json").write_text(json.dumps(res["redacted"], indent=2))
            rec["correct"] = bool(res["full"].get("verdict") == "match"
                                  and res["full"].get("certifying_tier"))
        else:
            rec["correct"] = False
            (ws / "qa" / "verdict.json").write_text(json.dumps(
                {"status": "error", "correct": False,
                 "failure_detail": f"no kernel found at {KERNEL_REL}"}, indent=2))

        history.append(rec)
        (run_dir / "rounds" / f"round_{rnd:02d}.json").write_text(json.dumps(rec, indent=2))
        print(f"  round {rnd}: rc={rc} kernel={rec['kernel_written']} "
              f"correct={rec['correct']} agent={rec['agent_seconds']}s", flush=True)
        if rec["correct"]:
            break                                  # functional stage satisfied

    summary = {
        "run_id": a.run_id, "method": a.method, "driver": cfg["driver"],
        "provider": cfg.get("provider"), "model": cfg["model"],
        "billing_mode": cfg.get("billing_mode"), "seed": a.seed,
        "task_id": capsule_dir.name, "target": a.target,
        "rounds_run": len(history), "solved": any(r["correct"] for r in history),
        "wall_seconds": round(time.time() - t_start, 1),
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {a.method}/{capsule_dir.name}: solved={summary['solved']} "
          f"rounds={summary['rounds_run']} wall={summary['wall_seconds']}s")
    print(f"  -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
