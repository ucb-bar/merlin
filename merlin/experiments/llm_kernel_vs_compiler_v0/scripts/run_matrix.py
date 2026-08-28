#!/usr/bin/env python3
"""Run the (method x task x seed) matrix concurrently, with per-provider concurrency caps.

WHY THIS IS WORTH DOING. Measured over 76 rounds, the agent CLI accounts for 99% of a round's wall
time (median 163 s) and the oracle for 1% (median 2.7 s). The loop is almost entirely waiting on a
provider, so concurrency buys close to linear wall-clock speedup for almost no local CPU -- three
tasks x three seeds is a 9x serial pipeline that need not be serial at all.

WHAT LIMITS IT IS THE PROVIDER, NOT THE BOX. Fanning out against one account's quota turns a
capability comparison into a quota measurement: a throttled round returns fast and empty, which is
indistinguishable in the score from a model that could not answer. So concurrency is capped PER
PROVIDER rather than globally, and a round that failed for a provider-side reason is recorded as
`provider_error` rather than as a wrong answer.

Rounds WITHIN a task stay serial -- each one reads the grader's verdict on the last, which is the
loop being measured. Tasks and seeds are independent and run together.

Every job is a separate process with its own run directory, and each agent runs inside its own bwrap
box, so concurrency cannot become cross-talk between arms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

#: Concurrent agent processes per provider. Deliberately per-provider: the arms bill against
#: different accounts and quotas, so one arm's fan-out must not consume another's headroom.
DEFAULT_CAPS = {"subscription": 3, "bedrock": 4, "google": 4}

DEFAULT_TASKS = [
    "merlin/contract/capsules/radiance/isa/R0_gemm_fp32",
    "merlin/contract/capsules/radiance/model_slices/R4_rmsnorm_fp32",
    "merlin/contract/capsules/radiance/model_slices/R3_attention_qk_fp16",
]


def _repo_root() -> Path:
    for anc in HERE.parents:
        if (anc / "merlin" / "python" / "merlin").is_dir():
            return anc
    raise SystemExit("could not locate the repo root")


def _load(method: str) -> dict:
    import yaml
    return yaml.safe_load((EXP / "methods" / method / "method.yaml").read_text()) or {}


class Slots:
    """A counting semaphore per provider, so one arm's fan-out cannot eat another's quota."""

    def __init__(self, caps: dict[str, int]) -> None:
        self._sem = {k: threading.Semaphore(v) for k, v in caps.items()}
        self._default = threading.Semaphore(2)

    def get(self, provider: str) -> threading.Semaphore:
        return self._sem.get(provider, self._default)


def _run_one(job: dict, *, repo: Path, runs_root: Path, slots: Slots, results: list,
             lock: threading.Lock, dry: bool) -> None:
    sem = slots.get(job["provider"])
    with sem:
        cmd = [sys.executable, str(HERE / "run_kernel_agent.py"),
               "--method", job["method"], "--capsule", job["capsule"],
               "--run-id", job["run_id"], "--seed", str(job["seed"]),
               "--rounds", str(job["rounds"]), "--opt-rounds", str(job["opt_rounds"]),
               "--round-timeout", str(job["round_timeout"]),
               "--runs-root", str(runs_root), "--sandbox", job["sandbox"]]
        if dry:
            with lock:
                print("  DRY " + " ".join(cmd))
            return
        log = runs_root / "matrix_logs" / f"{job['run_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo / "merlin" / "python")
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=str(repo), env=env).returncode
        wall = round(time.time() - t0, 1)

        rec = {**job, "rc": rc, "wall_seconds": wall, "log": str(log)}
        sp = runs_root / "agents" / job["run_id"] / "summary.json"
        if sp.is_file():
            s = json.loads(sp.read_text())
            rec.update({k: s.get(k) for k in ("solved", "solved_at_round", "best_cycles",
                                              "rounds_run", "cost", "sandbox")})
        with lock:
            results.append(rec)
            done = len(results)
            print(f"  [{done}] {job['run_id']}: solved={rec.get('solved')} "
                  f"best={rec.get('best_cycles')} wall={wall}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", default="codex_kernel,bedrock_kernel,gemini_kernel")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--seeds", default="1",
                    help="comma-separated; the plan wants >=3 for the bootstrap CIs")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--opt-rounds", type=int, default=3)
    ap.add_argument("--round-timeout", type=int, default=900)
    ap.add_argument("--sandbox", default="bwrap")
    ap.add_argument("--tag", default="mx")
    ap.add_argument("--runs-root", type=Path, default=Path("/scratch/agustin/tmp/kvc-runs"))
    ap.add_argument("--caps", default="",
                    help="override provider caps, e.g. 'bedrock=2,google=6'")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    repo = _repo_root()
    caps = dict(DEFAULT_CAPS)
    for kv in filter(None, a.caps.split(",")):
        k, _, v = kv.partition("=")
        caps[k.strip()] = int(v)

    jobs = []
    for method in [m for m in a.methods.split(",") if m]:
        cfg = _load(method)
        for capsule in [t for t in a.tasks.split(",") if t]:
            for seed in [int(s) for s in a.seeds.split(",") if s]:
                name = Path(capsule).name
                jobs.append({
                    "method": method, "provider": cfg.get("provider", "?"),
                    "model": cfg.get("model", "?"), "capsule": capsule, "task": name,
                    "seed": seed, "rounds": a.rounds, "opt_rounds": a.opt_rounds,
                    "round_timeout": a.round_timeout, "sandbox": a.sandbox,
                    "run_id": f"{a.tag}_{method}_{name}_s{seed}",
                })

    by_provider: dict[str, int] = {}
    for j in jobs:
        by_provider[j["provider"]] = by_provider.get(j["provider"], 0) + 1
    print(f"{len(jobs)} jobs: " + ", ".join(f"{k}={v} (cap {caps.get(k, 2)})"
                                            for k, v in sorted(by_provider.items())))
    # Serial wall time is the sum; concurrent is bounded by the busiest provider's queue depth.
    depth = max((n / max(caps.get(p, 2), 1)) for p, n in by_provider.items()) if jobs else 0
    print(f"serial would be {len(jobs)} sequential runs; concurrent bounds it at ~{depth:.0f} deep\n")

    results: list[dict] = []
    lock = threading.Lock()
    slots = Slots(caps)
    threads = [threading.Thread(target=_run_one, args=(j,),
                                kwargs=dict(repo=repo, runs_root=a.runs_root, slots=slots,
                                            results=results, lock=lock, dry=a.dry_run))
               for j in jobs]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if a.dry_run:
        return 0
    out = a.runs_root / f"matrix_{a.tag}.json"
    out.write_text(json.dumps({"jobs": len(jobs), "caps": caps,
                               "wall_seconds": round(time.time() - t0, 1),
                               "results": results}, indent=2))
    solved = sum(1 for r in results if r.get("solved"))
    print(f"\n{solved}/{len(results)} solved in {round(time.time() - t0, 1)}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
