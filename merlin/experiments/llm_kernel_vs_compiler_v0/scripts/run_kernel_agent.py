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


def _task_card(capsule_dir: Path, iface: str, rnd: int, *, kernel_path: Path,
               stage: str = "functional", best: dict | None = None) -> str:
    """What the agent is told. Contains the interface and the ABI -- never a reference answer.

    Two stages, because they ask for different things. The functional stage asks for a correct kernel
    and stops when it has one. The optimization stage keeps a correct kernel in hand and asks for a
    faster one, which is only answerable if the agent is told where the time went -- so that stage
    carries the profiler counters and names the axes explicitly.
    """
    return f"""# Task: write a compute kernel for this accelerator

You are writing ONE compute kernel for a SIMT accelerator, as **LLVM-dialect MLIR**.

Write it to exactly this ABSOLUTE path, and nothing else:

    {kernel_path}

(Absolute, because an agent CLI may resolve a relative path against a project root that is not your
working directory -- a kernel written to the wrong root is not graded.)

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
- Do not modify anything outside `{kernel_path}`.

Round {rnd}.
""" + (_optimization_brief(best) if stage == "optimization" else
       f"If `{kernel_path.parent.parent / 'qa' / 'verdict.json'}` exists, it is the grader's verdict "
       "on your previous attempt -- read it first and fix what it reports.\n")


def _optimization_brief(best: dict | None) -> str:
    """The optimization stage's instructions: keep it correct, make it faster AND better utilized."""
    b = best or {}
    util = b.get("utilization") or {}
    lines = [
        "\n## You already have a CORRECT kernel. Now make it faster.",
        "",
        f"Your best correct kernel so far runs in **{b.get('cycles')} cycles** "
        f"({b.get('pct_fp_peak')} of FP peak).",
        "",
        "Where its time actually goes, as fractions of the run:",
    ]
    for k, v in util.items():
        lines.append(f"  - {k}: {'unmeasured' if v is None else f'{v:.4f}'}")
    lines += [
        "",
        "Read those before changing anything. Low FP utilization with healthy warp occupancy means "
        "the wrong work is being issued -- more arithmetic per loop iteration, fewer redundant loads. "
        "High utilization with low occupancy means the machine is starved and wants more parallel "
        "work in flight. A unit sitting at 0 is one you are not using at all.",
        "",
        "Rewrite the kernel to lower cycles AND raise utilization. Correctness is not negotiable: a "
        "faster kernel that computes the wrong answer scores nothing, and the previous correct "
        "version is kept if you regress.",
    ]
    return "\n".join(lines) + "\n"


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



def _find_misrouted_kernel(transcript: Path | None, *, expected: Path) -> Path | None:
    """A kernel the agent wrote somewhere other than where the harness looks.

    Exists because the failure it catches is indistinguishable, in the score, from a model that could
    not write a kernel -- and that mistake has now been made once. The transcript records the path the
    agent actually wrote, so the harness asks it rather than guessing.

    Parses each line as JSON and walks it for `filePath` values, rather than matching the text: a
    string match would depend on the emitter's spacing, and would go quietly blind the day a CLI
    pretty-printed its output -- reinstating the exact silent failure this exists to catch.
    """
    if transcript is None or not transcript.is_file():
        return None

    def paths(node) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "filePath" and isinstance(v, str):
                    found.append(v)
                else:
                    found.extend(paths(v))
        elif isinstance(node, list):
            for v in node:
                found.extend(paths(v))
        return found

    want, best = expected.name, None
    for line in transcript.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue                      # a partial or non-JSON line is not evidence either way
        for cand in paths(rec):
            q = Path(cand)
            if q.name == want and q != expected and q.is_file():
                best = q
    return best


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
    ap.add_argument("--rounds", type=int, default=2,
                    help="functional-stage budget: rounds allowed to reach a CORRECT kernel")
    ap.add_argument("--opt-rounds", type=int, default=0,
                    help="optimization-stage budget: rounds spent making a correct kernel faster")
    ap.add_argument("--seed-kernel", type=Path, default=None,
                    help="start from this correct kernel and go straight to optimization -- the "
                         "like-for-like setting against a search that requires a working seed")
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


    # An agent CLI resolves relative paths against the PROJECT ROOT IT DETECTS, not the cwd it was
    # given -- and it detects that root by walking up for a `.git`. With the workspace nested under a
    # checkout, opencode rooted at the repo and wrote every kernel to <repo>/submission/, where the
    # harness never looked. Three arms were scored 0/3 for kernels they had actually written.
    # Making the workspace its own repository puts the detected root back inside it.
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=str(ws), capture_output=True)

    kernel = ws / KERNEL_REL
    history: list[dict] = []
    t_start = time.time()

    best: dict | None = None          # best CORRECT candidate so far, by cycles
    best_src: str | None = None
    solved_round: int | None = None

    # A supplied seed skips the functional stage: the kernel is already correct, so the run measures
    # optimization alone. This is the setting that compares like-for-like against a search which
    # REQUIRES a working seed and therefore never pays a generation cost at all.
    if a.seed_kernel:
        seed = a.seed_kernel.read_text()
        kernel.write_text(seed)
        res0 = _evaluate(kernel, capsule_dir, shim_pkg=shim_pkg, runs_root=a.runs_root,
                         task_id=capsule_dir.name, config_id="C0", target=a.target,
                         method=a.method, timeout=a.eval_timeout)
        full0 = res0["full"]
        if not (full0.get("verdict") == "match" and full0.get("certifying_tier")):
            # Refuse rather than silently start the optimization stage from a broken kernel, which
            # would make every later round look like the agent breaking something it never had.
            raise SystemExit(f"seed kernel is not correct under this oracle: "
                             f"{full0.get('failure_detail') or full0.get('verdict')}")
        best = {k: full0.get(k) for k in ("cycles", "pct_fp_peak", "gflops", "utilization")}
        best_src = seed
        solved_round = -1                      # correct before the agent was asked anything
        (run_dir / "best_kernel.llvm.mlir").write_text(seed)
        (ws / "qa" / "verdict.json").write_text(json.dumps(res0["redacted"], indent=2))
        history.append({"round": -1, "stage": "seed", "rc": 0, "agent_seconds": 0.0,
                        "kernel_written": True, "correct": True, "evaluation": full0,
                        "best_cycles_so_far": best.get("cycles")})
        print(f"  seed: correct, {best.get('cycles')} cycles -- optimization stage only", flush=True)

    for rnd in range(a.rounds + a.opt_rounds):
        stage = "functional" if best is None else "optimization"
        if stage == "functional" and rnd >= a.rounds:
            break                          # never became correct; the optimization budget is unusable
        card = _task_card(capsule_dir, iface, rnd, kernel_path=kernel, stage=stage, best=best)
        prev = (ws / "qa" / "verdict.json")
        if rnd and prev.is_file():
            card += ("\n## The grader's verdict on your previous attempt\n\n```json\n"
                     + prev.read_text()[:4000] + "\n```\n")
        (ws / "TASK.md").write_text(card)
        t0 = time.time()
        rc, transcript = _run_agent(cfg, ws, run_dir, rnd, card, a.round_timeout)
        agent_s = time.time() - t0

        misroute = None
        if not kernel.is_file():
            misroute = _find_misrouted_kernel(transcript, expected=kernel)

        rec: dict = {"round": rnd, "stage": stage, "rc": rc,
                     "agent_seconds": round(agent_s, 1),
                     "transcript": str(transcript) if transcript else None,
                     "kernel_written": kernel.is_file() or bool(misroute)}
        if misroute:
            # The agent DID the work; the harness looked in the wrong place. Recording this as an
            # incorrect kernel is how a harness defect gets published as a model result, so it is
            # named as a harness fault and the kernel is graded where it actually landed.
            rec["harness_misroute"] = {"found_at": str(misroute), "expected_at": str(kernel)}
            print(f"  !! HARNESS MISROUTE: agent wrote {misroute}, harness expected {kernel}. "
                  f"Grading the kernel it wrote; the run is flagged.", flush=True)
            kernel.parent.mkdir(parents=True, exist_ok=True)
            kernel.write_text(Path(misroute).read_text())
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

        # Keep the best CORRECT candidate. A regression does not discard it -- the agent is asked
        # for a faster kernel, not a riskier one, and every candidate is preserved either way.
        if rec["correct"]:
            cyc = (rec.get("evaluation") or {}).get("cycles")
            if solved_round is None:
                solved_round = rnd
            if cyc is not None and (best is None or cyc < (best.get("cycles") or 1 << 62)):
                best = {k: (rec["evaluation"] or {}).get(k)
                        for k in ("cycles", "pct_fp_peak", "gflops", "utilization")}
                best_src = kernel.read_text()
                (run_dir / "best_kernel.llvm.mlir").write_text(best_src)
        elif best_src is not None:
            # Restore the best correct kernel so the next optimization round starts from something
            # that works rather than from the regression it just produced.
            kernel.write_text(best_src)

        rec["best_cycles_so_far"] = (best or {}).get("cycles")
        history.append(rec)
        (run_dir / "rounds" / f"round_{rnd:02d}.json").write_text(json.dumps(rec, indent=2))
        print(f"  round {rnd} [{stage}]: rc={rc} kernel={rec['kernel_written']} "
              f"correct={rec['correct']} cycles={(rec.get('evaluation') or {}).get('cycles')} "
              f"best={(best or {}).get('cycles')} agent={rec['agent_seconds']}s", flush=True)

    summary = {
        "run_id": a.run_id, "method": a.method, "driver": cfg["driver"],
        "provider": cfg.get("provider"), "model": cfg["model"],
        "billing_mode": cfg.get("billing_mode"), "seed": a.seed,
        "task_id": capsule_dir.name, "target": a.target,
        "rounds_run": len(history),
        "functional_rounds": sum(1 for r in history if r["stage"] == "functional"),
        "optimization_rounds": sum(1 for r in history if r["stage"] == "optimization"),
        "solved": any(r["correct"] for r in history),
        "solved_at_round": solved_round,
        "started_from": ("supplied seed kernel (optimization only)" if a.seed_kernel
                         else "the specification (generation then optimization)"),
        "best_cycles": (best or {}).get("cycles"),
        "best_utilization": (best or {}).get("utilization"),
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
