#!/usr/bin/env python3
"""Target-agnostic auto-resume watchdog for an A/B/C capsule-bench batch.

Guards one or more arms of a running batch and keeps them going across interruptions WITHOUT ever
restarting from round 0:

  * resume-on-death — if an arm's process has exited but its ``qa_loop_state.yaml`` is not
    ``converged: true`` (hard crash / OOM / host reboot / rate-limit waits exhausted), relaunch it with
    ``--resume``, which the QA loop honours by continuing from the checkpointed ``next_round``.
  * login-kick (opt-in, ``--login-kick``) — when the Claude credentials file changes (a fresh ``/login``
    with a new 5h window) and a quick probe shows headroom, restart any arm currently *sleeping* on a
    rate-limit wait so it picks up the fresh limit instead of idling.

Everything is derived from the active target descriptor (``MERLIN_TARGET_EXPERIMENT`` via ``_common``)
and the launcher's own arm table — there is NO per-target path or arm hardcoded here, so the same
watchdog guards gemmini, atlas, radiance, … unchanged. It reuses ``launch_ab_batch``'s ``ARMS`` /
``_run_id`` / ``_run_dir`` / ``_arm_cmd`` so the resume command is byte-identical to the original launch
(plus ``--resume``).

Usage (target comes from the env, exactly like the launcher):
  MERLIN_TARGET_EXPERIMENT=.../targets/atlas/target_experiment.yaml \
    python watchdog.py --tag atlas_arm4 --arms merlin_rtlchecks
  # ...same knobs as launch_ab_batch (--model/--effort/--max-rounds/--round-timeout/--sandbox/...)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _common as C            # noqa: E402  (honors MERLIN_TARGET_EXPERIMENT)
import launch_ab_batch as LB   # noqa: E402  (ARMS + _run_id/_run_dir/_arm_cmd — the single source of truth)


def _converged(run_dir: Path) -> bool:
    st = run_dir / "qa_loop_state.yaml"
    try:
        for line in st.read_text().splitlines():
            if line.strip().startswith("converged:"):
                return line.split(":", 1)[1].strip() == "true"
    except OSError:
        pass
    return False


def _alive(run_id: str) -> bool:
    return subprocess.run(["pgrep", "-f", f"run-id {run_id}"],
                          capture_output=True).returncode == 0


def _sleeping(run_id: str, run_dir: Path) -> bool:
    """Alive AND its most recent log line is a rate-limit sleep (so a login-kick can move it)."""
    if not _alive(run_id):
        return False
    logs = sorted((p for p in (run_dir.parent.glob(f"{run_id}.resume.log"),
                               run_dir.parent.glob(f"{run_id}.launch.log")) for p in p),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for lg in logs:
        try:
            tail = lg.read_text().splitlines()[-3:]
            if any("RATE-LIMITED" in ln for ln in tail):
                return True
        except OSError:
            continue
    return False


def _resume(arm: str, run_id: str, a, env: dict) -> None:
    run_dir = LB._run_dir(arm, run_id)
    cmd = LB._arm_cmd(arm, run_id, a) + ["--resume"]
    rlog = run_dir.parent / f"{run_id}.resume.log"
    rlog.parent.mkdir(parents=True, exist_ok=True)
    print(f"[wd {time.strftime('%H:%M:%S')}] resume {run_id}: {' '.join(cmd)}", flush=True)
    with open(rlog, "ab") as fh:
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(C.HARNESS))


def _probe_headroom(model: str) -> bool:
    try:
        r = subprocess.run(["claude", "--print", "--model", model, "Reply: OK"],
                           capture_output=True, text=True, timeout=70)
        return '"status":"allowed"' in (r.stdout + r.stderr)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="batch tag (same one passed to launch_ab_batch)")
    ap.add_argument("--arms", default=",".join(LB.ARMS),
                    help="comma-separated arms to guard (default: all)")
    ap.add_argument("--poll", type=int, default=120, help="seconds between checks")
    ap.add_argument("--once", action="store_true", help="report status once and exit (no resume)")
    ap.add_argument("--login-kick", action="store_true",
                    help="on a new /login with headroom, restart arms sleeping on a rate-limit wait")
    # mirror launch_ab_batch's run knobs so the resume command matches the original launch exactly
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--max-rate-limit-waits", type=int, default=8)
    ap.add_argument("--round-timeout", type=int, default=14400)
    ap.add_argument("--experiment", choices=["full", "realistic"], default="full")
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    ap.add_argument("--skip-hidden", action="store_true")
    # Agent driver + provider — mirrored so _arm_cmd builds the SAME command on the initial launch AND on
    # every --resume (else a resume would silently drop to driver=auto / a different provider than the run
    # was started with). Names + defaults match launch_ab_batch exactly.
    ap.add_argument("--driver", choices=["auto", "converse", "claudecode", "opencode"], default="auto",
                    help="agent driver for the guarded arm(s) (auto|converse|claudecode|opencode)")
    ap.add_argument("--subagent-model", default="")
    ap.add_argument("--background-model", default="")
    ap.add_argument("--provider", choices=["subscription", "bedrock"], default="subscription")
    ap.add_argument("--aws-region", default="us-east-1", help="AWS region for --provider bedrock")
    ap.add_argument("--aws-profile", default="", help="AWS profile (~/.aws) for --provider bedrock")
    a = ap.parse_args()

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    bad = [x for x in arms if x not in LB.ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; valid: {list(LB.ARMS)}")

    env = dict(os.environ)
    env["MERLIN_TARGET_EXPERIMENT"] = str(C.EXP / "target_experiment.yaml")
    ids = {arm: LB._run_id(arm, a.tag) for arm in arms}

    print(f"[wd {time.strftime('%H:%M:%S')}] target={C.TARGET} tag={a.tag} "
          f"arms={arms} runs={C.RUNS}", flush=True)

    if a.once:
        for arm in arms:
            rid, rd = ids[arm], LB._run_dir(arm, LB._run_id(arm, a.tag))
            print(f"  {rid}: alive={_alive(rid)} converged={_converged(rd)} dir={rd}", flush=True)
        return 0

    cred = Path(os.environ.get("CLAUDE_CRED", Path.home() / ".claude" / ".credentials.json"))
    last_cred = cred.stat().st_mtime if cred.exists() else 0.0

    while True:
        alldone = True
        for arm in arms:
            rid = ids[arm]
            rd = LB._run_dir(arm, rid)
            if _converged(rd):
                continue
            alldone = False
            if not _alive(rid):
                print(f"[wd {time.strftime('%H:%M:%S')}] {rid} not alive + not converged", flush=True)
                _resume(arm, rid, a, env)
                time.sleep(10)
        if alldone:
            print(f"[wd {time.strftime('%H:%M:%S')}] all arms converged — exiting", flush=True)
            break
        if a.login_kick and cred.exists():
            m = cred.stat().st_mtime
            if m != last_cred:
                last_cred = m
                print(f"[wd {time.strftime('%H:%M:%S')}] new login detected — probing headroom", flush=True)
                if _probe_headroom(a.model):
                    for arm in arms:
                        rid, rd = ids[arm], LB._run_dir(arm, ids[arm])
                        if not _converged(rd) and _sleeping(rid, rd):
                            print(f"[wd] {rid} sleeping + fresh headroom -> kick", flush=True)
                            subprocess.run(["pkill", "-9", "-f", f"run-id {rid}"],
                                           capture_output=True)  # scoped to THIS run-id only
                            time.sleep(3)
                            _resume(arm, rid, a, env)
                            time.sleep(10)
        time.sleep(a.poll)

    print(f"=== {a.tag} FINAL ===", flush=True)
    for arm in arms:
        rd = LB._run_dir(arm, ids[arm])
        st = rd / "qa_loop_state.yaml"
        line = ""
        if st.exists():
            line = " ".join(l.strip() for l in st.read_text().splitlines()
                            if l.strip().startswith(("converged:", "next_round:")))
        print(f"  {ids[arm]}: {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
