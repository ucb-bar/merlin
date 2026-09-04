#!/usr/bin/env python3
"""ONE unattended chain: the functional arm-4 run, then the performance stage — with honest gates.

Written to be started before going to sleep and read on waking. Every stage records what it did, and a
stage that CANNOT run refuses loudly and records why rather than running something adjacent and letting
the report imply the real thing happened. That failure mode is the reason this file is shaped the way it
is: this repo has repeatedly shipped a check that could not run and reported success.

    STAGE 0  preflight        the shipped GO/NO-GO gate, the prohibition gates, the coverage gate,
                              and the two external things a run needs (a sandbox and an agent CLI)
    STAGE 1  functional       arm-4 (the RTL-checks track) on the codex seat, in the enforced sandbox
    STAGE 2  grade + freeze    aggregate the verdicts, then FREEZE the submission -- that frozen
                              compiler is the fork point the performance phase is defined against
    STAGE 3  calibration      measure the mechanisms a performance model needs (occupancy, the
                              composition operator, realised overlap) on the frozen submission
    STAGE 4  performance      the optimization run -- GATED, and it refuses when the performance
                              capsule families do not exist, which is the state at the time of writing

⚠️ STAGE 4 IS EXPECTED TO REFUSE TODAY. The performance capsule families (tile utilisation, staging,
issue-vs-wait, loop order, queue pressure, fusion, inter-op residency, boundary break-even, quantisation,
mixed micro-model) have not been authored. Without them an "optimization run" is a second functional run
whose numbers mean nothing, so this script will not start one. Stage 3 is the part of the performance
phase that IS ready, and the plan requires it first regardless: the analytical model the optimization
loop scores candidates with is calibrated from L3 mechanism measurements taken BEFORE the loop starts.

Usage:
    .venv/bin/python merlin/experiments/capsule_bench/harness/run_overnight.py --tag n1 --dry-run
    nohup .venv/bin/python merlin/experiments/capsule_bench/harness/run_overnight.py --tag n1 \
        > /dev/null 2>&1 &

Read on waking:  out/artifacts/capsule-bench/<target>/overnight_<tag>/REPORT.md
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402 — bootstraps merlin/python and resolves the active target

from merlin.common.artifacts import artifacts_dir  # noqa: E402
from merlin.common.paths import repo_root  # noqa: E402

REPO = repo_root()
PY = sys.executable


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Journal:
    """The record a sleeping operator reads on waking. Written after EVERY stage, not at the end, so a
    chain that dies mid-way still explains how far it got and what it was doing."""

    def __init__(self, root: Path, tag: str, target: str):
        self.root = root
        self.path = root / "journal.json"
        self.report = root / "REPORT.md"
        self.doc = {"tag": tag, "target": target, "started_utc": _utc(),
                    "git_sha": _sh_out(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip(),
                    "stages": []}
        self.flush()

    def stage(self, name: str, status: str, detail: str, **extra):
        entry = {"stage": name, "status": status, "detail": detail,
                 "at_utc": _utc(), **extra}
        self.doc["stages"].append(entry)
        self.flush()
        marker = {"ok": "  ok  ", "refused": "REFUSED", "failed": "FAILED ", "skipped": "SKIPPED"}
        print(f"[{marker.get(status, status):7s}] {name}: {detail}", flush=True)
        return entry

    def flush(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.doc, indent=2), encoding="utf-8")
        self.report.write_text(self._render(), encoding="utf-8")

    def _render(self) -> str:
        lines = [f"# Overnight chain — {self.doc['tag']} ({self.doc['target']})", "",
                 f"started `{self.doc['started_utc']}` at `{self.doc['git_sha'][:12]}`", ""]
        # THE VERDICT FIRST. Someone reading this half-awake must not have to infer it from the table.
        bad = [s for s in self.doc["stages"] if s["status"] in ("failed", "refused")]
        if not self.doc["stages"]:
            lines += ["**status: nothing ran yet.**", ""]
        elif bad:
            lines += [f"**status: {len(bad)} stage(s) did not complete — "
                      f"{', '.join(s['stage'] for s in bad)}.**", ""]
        else:
            lines += ["**status: every stage that ran completed.**", ""]
        lines += ["| stage | status | detail |", "|---|---|---|"]
        for s in self.doc["stages"]:
            lines.append(f"| {s['stage']} | {s['status']} | {s['detail'].replace('|', '/')} |")
        lines += ["", "Full machine-readable record: `journal.json`.", ""]
        for s in self.doc["stages"]:
            if s.get("note"):
                lines += [f"## {s['stage']}", "", s["note"], ""]
        return "\n".join(lines)


def _sh_out(cmd, cwd=None, timeout=120) -> str:
    try:
        p = subprocess.run(cmd, cwd=cwd or str(REPO), capture_output=True, text=True, timeout=timeout)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:                                     # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def _run(cmd, log: Path, timeout: int | None = None) -> tuple[int, str]:
    """Run a command to completion, tee'ing to ``log``. Returns (rc, tail).

    Synchronous ON PURPOSE. The batch launcher backgrounds its sessions, and polling a state file for
    completion cannot distinguish "still working" from "died without writing". Owning the child means
    process exit IS the completion signal, and a timeout kills exactly this child rather than matching a
    pattern -- this is a shared host and a broad pattern kill would take other people's work with it.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            rc = 124
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    return rc, tail


# ---------------------------------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------------------------------

def stage_preflight(j: Journal, logs: Path, *, strict: bool) -> bool:
    """Everything that must hold before an unattended run is worth starting.

    A soft gate reports and does not block; a hard gate blocks. The split is deliberate: the coverage
    gate carries ratcheted debt by design and should not stop a run, whereas a missing sandbox or a
    missing agent CLI means the run cannot do what it claims.
    """
    ok = True

    # SOFT: the harness's report directory. It is mode 000 on this checkout, and the operator's answer
    # is that the experiment's own sandbox handles the write path -- so this RECORDS rather than blocks.
    # It stays here because it is cheap and because if a report step ever does fail on it hours into an
    # unattended run, the journal should already name the reason instead of leaving someone to guess.
    import os as _os
    if not _os.access(C.REPORTS, _os.W_OK):
        j.stage("preflight/reports-writable", "skipped",
                f"{C.REPORTS} is not writable (mode "
                f"{oct(C.REPORTS.stat().st_mode & 0o777) if C.REPORTS.exists() else 'absent'})",
                note=f"The harness writes every aggregate and report here. Left as found rather than\n"
                     f"changed: a directory deliberately set to 000 is somebody's decision, and this\n"
                     f"repo uses exactly that pattern to protect answer surfaces. Unlock it with\n\n"
                     f"    chmod u+rwx {C.REPORTS}\n\n"
                     f"Not treated as a blocker: the experiment sandbox provides the write path. If a\n"
                     f"report stage later fails, this is the first thing to check.")
    else:
        j.stage("preflight/reports-writable", "ok", f"{C.REPORTS} writable")

    # HARD: the two external things. A run without the sandbox is not the experiment -- the isolation is
    # what makes the result a measurement of the agent rather than of the filesystem it could reach.
    for tool, why in (("bwrap", "the enforced sandbox"), ("codex", "the agent CLI")):
        found = shutil.which(tool)
        if not found:
            j.stage("preflight/tools", "failed", f"{tool!r} not on PATH — {why} is required")
            ok = False
        else:
            j.stage("preflight/tools", "ok", f"{tool} at {found}")

    rc, tail = _run([PY, str(HERE / "readiness_check.py")], logs / "readiness.log", timeout=3600)
    j.stage("preflight/readiness", "ok" if rc == 0 else "failed",
            f"readiness_check.py exit {rc} (GO)" if rc == 0 else f"readiness_check.py exit {rc} (NO-GO)",
            note=None if rc == 0 else f"```\n{tail}\n```")
    if rc != 0:
        ok = False

    for name, cmd in (
        ("no-target-name", [PY, "build_tools/scripts/check_no_target_name.py"]),
        ("no-regex", [PY, "build_tools/scripts/check_no_regex.py"]),
        ("contract-copies", [PY, "build_tools/scripts/check_contract_copies.py"]),
        ("answer-keys", [PY, "build_tools/scripts/check_no_answer_keys.py"]),
    ):
        rc, tail = _run(cmd, logs / f"gate_{name}.log", timeout=900)
        j.stage(f"preflight/{name}", "ok" if rc == 0 else "failed", f"exit {rc}",
                note=None if rc == 0 else f"```\n{tail}\n```")
        if rc != 0:
            ok = False

    # SOFT: coverage carries ratcheted debt by design, so it reports and never blocks.
    rc, tail = _run([PY, "build_tools/scripts/check_conformance_coverage.py",
                     "--target", C.TARGET], logs / "coverage.log", timeout=3600)
    j.stage("preflight/coverage", "ok", f"recorded (exit {rc}); see coverage.log",
            note=f"```\n{tail}\n```")

    return ok or not strict


def stage_functional(j: Journal, logs: Path, a) -> Path | None:
    """Arm-4 — the RTL-checks track — on the codex seat, inside the sandbox."""
    run_id = f"merlincirct_{a.tag}"
    run_dir = C.RUNS / "merlin_assisted" / run_id
    driver = (REPO / "merlin" / "experiments" / "capsule_bench" / "targets" / C.TARGET
              / "scripts" / "run_rtlchecks_qa_loop.py")
    if not driver.is_file():
        j.stage("functional", "failed", f"no arm-4 driver at {driver}")
        return None
    cmd = [PY, str(driver), "--run-id", run_id, "--model", a.model, "--effort", a.effort,
           "--max-rounds", str(a.max_rounds), "--max-rate-limit-waits", str(a.max_rate_limit_waits),
           "--round-timeout", str(a.round_timeout), "--driver", "codex", "--sandbox", "bwrap"]
    if a.dry_run:
        j.stage("functional", "skipped", "dry-run: not launched",
                note="```\n$ " + " ".join(cmd) + "\n```")
        return run_dir
    t0 = time.time()
    rc, tail = _run(cmd, logs / "functional.log", timeout=a.functional_budget_s)
    mins = (time.time() - t0) / 60.0
    status = "ok" if rc == 0 else ("failed" if rc != 124 else "failed")
    j.stage("functional", status,
            f"arm-4 exit {rc} after {mins:.0f} min -> {run_dir}",
            run_dir=str(run_dir), note=f"```\n{tail}\n```")
    return run_dir if rc == 0 else None


def stage_grade_and_freeze(j: Journal, logs: Path, run_dir: Path, a) -> bool:
    if a.dry_run:
        j.stage("grade+freeze", "skipped", "dry-run")
        return True
    rc, tail = _run([PY, str(HERE / "agg_ab_results.py")], logs / "aggregate.log", timeout=3600)
    j.stage("grade", "ok" if rc == 0 else "failed", f"agg_ab_results exit {rc}",
            note=f"```\n{tail}\n```")
    # THE FREEZE IS THE FORK POINT. The performance phase is defined as an optimization of a SPECIFIC
    # functional compiler; without a freeze there is nothing to say a later candidate did not weaken.
    rc2, tail2 = _run([PY, str(HERE / "freeze_run.py"), "--run-dir", str(run_dir)],
                      logs / "freeze.log", timeout=1800)
    j.stage("freeze", "ok" if rc2 == 0 else "failed",
            f"freeze_run exit {rc2} — this submission is the performance fork point",
            note=f"```\n{tail2}\n```")
    return rc == 0 and rc2 == 0


def stage_calibration(j: Journal, logs: Path, a) -> bool:
    """Measure the mechanisms a performance model is scored against.

    The plan puts this BEFORE any optimization loop and it is the half of the performance phase that is
    ready: the analytical model a candidate schedule is ranked by has to be calibrated from real
    measurements, and the falsifier on this archetype is realised overlap rather than correctness.
    """
    script = HERE / "perf_calibrate.py"
    if not script.is_file():
        j.stage("calibration", "refused",
                "no perf_calibrate.py in the harness — calibration is not wired yet",
                note="The measurement libraries exist (`merlin.perf.occupancy`, `merlin.perf.headroom`,\n"
                     "`merlin.perf.differential`) but nothing drives them over a frozen submission.\n"
                     "Refusing rather than reporting a calibration that did not happen.")
        return False
    if a.dry_run:
        j.stage("calibration", "skipped", "dry-run")
        return True
    rc, tail = _run([PY, str(script)], logs / "calibration.log", timeout=a.calibration_budget_s)
    j.stage("calibration", "ok" if rc == 0 else "failed", f"perf_calibrate exit {rc}",
            note=f"```\n{tail}\n```")
    return rc == 0


def _perf_capsules_exist() -> tuple[bool, str]:
    """Does a performance capsule family exist at all?

    A performance capsule is an A/B on IDENTICAL work — same bytes, same golden, different schedule — so
    it declares a comparison group with at least two members. A corpus with none of those cannot support
    an optimization run, however many functional capsules it has.
    """
    import yaml

    # SCOPED TO THE ACTIVE TARGET'S GRADED ROOTS, and glob RECURSIVELY. The first version did neither
    # and was wrong twice over: `*/*/capsule.yaml` missed a nesting level, so it under-reported; and
    # scanning the whole tree would have found ANOTHER target's performance capsules and let this run
    # proceed on them. Measured: 14 capsules declare `performance:` here and every one belongs to a
    # different target, whose fusion groups say nothing about the one being launched. Counting them
    # would have started a performance run with nothing to grade -- the failure this gate exists to stop,
    # arrived at through the gate itself.
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        desc = (REPO / "merlin" / "experiments" / "capsule_bench" / "targets" / C.TARGET
                / "target_experiment.yaml")
        roots = [Path(r) for r in load_target_experiment(desc).graded_roots()]
    except Exception as e:                                     # noqa: BLE001
        return False, f"cannot resolve the graded roots for {C.TARGET!r}: {type(e).__name__}: {e}"
    groups: dict[str, int] = {}
    for cy in sorted(c for r in roots for c in r.rglob("capsule.yaml")):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        g = doc.get("comparison_group")
        name = g.get("name") if isinstance(g, dict) else g
        if name:
            groups[str(name)] = groups.get(str(name), 0) + 1
    usable = {g: n for g, n in groups.items() if n >= 2}
    if usable:
        return True, f"{len(usable)} comparison group(s) with >= 2 members: {sorted(usable)}"
    return False, (f"no comparison group in {C.TARGET}'s graded roots has two members (groups seen: "
                   f"{sorted(groups) or 'none'}); a "
                   f"performance capsule is an A/B on identical work, so a group of one cannot be "
                   f"compared to anything")


def stage_performance(j: Journal, logs: Path, a) -> bool:
    ready, why = _perf_capsules_exist()
    if not ready:
        j.stage("performance", "refused", "the performance capsule families do not exist",
                note=f"{why}\n\n"
                     f"Starting an optimization run without them would grade a second functional run\n"
                     f"and report it as a performance result. The families the plan calls for — tile\n"
                     f"utilisation, staging/residency, issue-vs-wait, loop order, queue pressure,\n"
                     f"fusion, inter-op residency, boundary break-even, quantisation, mixed\n"
                     f"micro-model — have not been authored.\n\n"
                     f"On this archetype the falsifier must be realised overlap, NOT bit-exactness:\n"
                     f"hazards resolve in hardware, so every reordering is correct by construction and\n"
                     f"a correctness-gated capsule passes every candidate schedule.")
        return False
    j.stage("performance", "refused",
            "capsules exist but the optimization driver is not wired in this script",
            note=why)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="batch label, e.g. n1")
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400)
    ap.add_argument("--functional-budget-s", type=int, default=12 * 3600)
    ap.add_argument("--calibration-budget-s", type=int, default=3 * 3600)
    ap.add_argument("--no-strict-preflight", action="store_true",
                    help="record preflight failures and continue anyway (default: a hard gate stops)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    # The journal goes somewhere WRITABLE even when the harness's own report dir is not, because the
    # journal is how a sleeping operator finds out that the report dir was the problem.
    root = artifacts_dir() / "capsule-bench" / C.TARGET / f"overnight_{a.tag}"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = artifacts_dir() / "overnight" / f"{C.TARGET}_{a.tag}"
    logs = root / "logs"
    j = Journal(root, a.tag, C.TARGET)
    print(f"journal: {j.path}\nreport : {j.report}\n", flush=True)

    if not stage_preflight(j, logs, strict=not a.no_strict_preflight):
        j.stage("chain", "failed", "preflight did not pass; nothing was launched")
        return 2

    run_dir = stage_functional(j, logs, a)
    if run_dir is None:
        j.stage("chain", "failed", "functional stage did not complete; not freezing or calibrating")
        return 3

    stage_grade_and_freeze(j, logs, run_dir, a)
    stage_calibration(j, logs, a)
    stage_performance(j, logs, a)

    j.stage("chain", "ok", "chain finished; read REPORT.md for what ran and what refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
