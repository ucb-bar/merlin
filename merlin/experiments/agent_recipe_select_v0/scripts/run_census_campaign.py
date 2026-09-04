"""Run one agent arm across every census shape, with bounded concurrency.

WHY A DRIVER AND NOT A SHELL LOOP. Three things have to hold across ~30 shapes and they are easy to
get wrong by hand: the host is SHARED (another session's perf-bench campaign runs on it), each shape
spawns its own cycle-accurate simulator, and a run that dies to a SIGHUP when a parent session ends
takes its whole shape with it. So: a slot limit, `setsid` per child, and a manifest written as each
shape finishes rather than at the end.

Shapes are independent -- nothing is shared between them but read-only inputs and the minted compiler
-- so the only reason to bound concurrency is the machine, not correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                        # noqa: E402

T.assert_right_merlin()
HERE = Path(__file__).resolve().parent

#: AutoComp's own interpreter. Named explicitly for the reason in the launch site below.
AUTOCOMP_PY = Path(os.environ.get("MERLIN_EXT_AUTOCOMP",
                                  "/scratch/agustin/projects/autocomp")) / ".venv" / "bin" / "python"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workloads", required=True, help="directory of .mlir workloads")
    ap.add_argument("--arm", choices=("recipe", "autocomp"), default="recipe")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--slots", type=int, default=6, help="shapes in flight at once")
    ap.add_argument("--include", default="", help="substring filter over workload names")
    ap.add_argument("--exclude", default="", help="substring filter, applied after --include")
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--method", default="")
    ap.add_argument("--resume", action="store_true",
                    help="skip shapes a prior campaign of this ARM already finished cleanly")
    ap.add_argument("--resume-from", action="append", default=[], metavar="DIR",
                    help="explicit prior campaign dir to read; repeatable. Implies --resume")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    wdir = Path(a.workloads)
    shapes = sorted(p for p in wdir.glob("*.mlir"))
    if a.include:
        shapes = [p for p in shapes if a.include in p.name]
    if a.exclude:
        shapes = [p for p in shapes if a.exclude not in p.name]
    if not shapes:
        raise SystemExit(f"no workloads matched under {wdir}")

    logs = Path(a.log_dir)
    logs.mkdir(parents=True, exist_ok=True)

    # RESUME. A shape is expensive (30-150 min of cycle-accurate simulation) and independent, so a
    # campaign that was interrupted should cost only what was actually in flight. Only rc==0 rows
    # count: a shape whose driver died mid-flight has no row at all and is correctly re-run.
    #
    # ⚠️ The arm is part of the key. Both arms write into the same parent directory and a recipe
    # rerun must not skip a shape that only the AUTOCOMP campaign finished -- they are different
    # measurements of the same shape. Rows written from here carry `arm`; older rows are attributed
    # from their campaign directory name, which is where the arm was already encoded.
    completed: dict[str, str] = {}
    if a.resume or a.resume_from:
        prior = [Path(d) for d in a.resume_from]
        if not prior:
            prior = sorted(q for q in logs.parent.glob("campaign_*") if q.is_dir() and q != logs)
        for d in prior:
            mf = d / "campaign.jsonl"
            if not mf.exists():
                continue
            for line in mf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                arm = rec.get("arm") or ("recipe" if "campaign_recipe" in d.name else
                                         "autocomp" if "campaign_autocomp" in d.name else "")
                if arm != a.arm or rec.get("rc") != 0:
                    continue
                completed[rec["workload"]] = str(d)
    if completed:
        skipped = [p for p in shapes if p.name in completed]
        shapes = [p for p in shapes if p.name not in completed]
        print(f"resume: skipping {len(skipped)} shape(s) already completed on arm={a.arm}")
        for p in skipped:
            print(f"  skip {p.name}  <- {completed[p.name]}")
        if not shapes:
            print("nothing left to run")
            return 0
    method = a.method or (f"recipe_agent_census" if a.arm == "recipe" else "autocomp_census")

    print(f"{len(shapes)} shapes, arm={a.arm}, budget={a.budget}, slots={a.slots}")
    for p in shapes:
        print(f"  {p.name}")
    if a.dry_run:
        return 0

    running: list[tuple[subprocess.Popen, Path, float]] = []
    done: list[dict] = []
    manifest = logs / "campaign.jsonl"

    def _reap(block: bool) -> bool:
        """Retire finished children. Returns True once at least one slot is free.

        ⚠️ The obvious loop shape is wrong and was measured to be: reaping one child and then
        CONTINUING to loop while `running` is non-empty drains the pool to zero before the caller
        gets its slot back, which turns a rolling window into fixed batches and makes every batch as
        slow as its slowest shape. Return as soon as a slot opens.
        """
        while True:
            for i, (proc, path, t0) in enumerate(list(running)):
                if proc.poll() is not None:
                    rec = {"workload": path.name, "arm": a.arm, "rc": proc.returncode,
                           "wall_s": round(time.time() - t0, 1),
                           "log": str(logs / f"{path.stem}.log")}
                    done.append(rec)
                    with manifest.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"[{len(done)}/{len(shapes)}] {path.name} rc={proc.returncode} "
                          f"{rec['wall_s']:.0f}s", flush=True)
                    running.pop(i)
                    return True
            if not block or not running:
                return bool(running) is False
            time.sleep(5)

    for path in shapes:
        while len(running) >= a.slots:
            _reap(block=True)
        if a.arm == "recipe":
            argv_ = [sys.executable, str(HERE / "run_recipe_agent.py"),
                     "--workload", str(path), "--budget", str(a.budget), "--method", method]
        else:
            # AutoComp imports `autocomp` at module scope, so its arm runs under ITS venv while the
            # oracle crosses back to merlin's interpreter. Inheriting sys.executable here once made a
            # known-good reference kernel read as wrong, so the interpreter is named, not assumed.
            argv_ = [str(AUTOCOMP_PY), str(HERE / "run_autocomp_gemmini.py"),
                     "--workload-mlir", str(path), "--budget-candidates", str(a.budget),
                     "--method", method]
        env = T.py_env()
        log = (logs / f"{path.stem}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(argv_, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        running.append((proc, path, time.time()))
        print(f"launched {path.name} (pid {proc.pid}, {len(running)} in flight)", flush=True)

    while running:
        _reap(block=True)


    ok = sum(1 for d in done if d["rc"] == 0)
    print(f"\n{ok}/{len(done)} shapes completed cleanly; manifest {manifest}")
    return 0 if ok == len(done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
