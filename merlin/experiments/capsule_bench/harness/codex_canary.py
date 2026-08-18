#!/usr/bin/env python3
"""Pre-flight canary: prove the Codex CLI works *inside* the bench sandbox — before any spend.

The bench's isolation claim is that the OUTER bwrap boundary, not the agent's own
sandbox, is what withholds the answers. Adding a second agent CLI does not
inherit that proof: Codex has its own launcher, its own credential store and its
own home directory, and each is a way for the boundary to be wrong in a new way.
This canary checks the four things that must hold before a graded cell runs.

  1. **The binary is reachable.** ``~/.local/bin/codex`` is a SYMLINK into
     ``~/.codex/packages/``, so binding ``~/.local/bin`` alone — which the shared
     claude binds already do — leaves the launcher pointing at nothing.
  2. **Authentication survives the boundary**, with the token bind-mounted
     read-only and never copied into the tree. Proven by a completed turn.
  3. **No prior Codex session is reachable.** The real ``~/.codex`` holds every
     past conversation on this host in ``sessions/``; inside the sandbox that
     directory must contain only ``packages/``.
  4. **The answer surfaces stay masked**, checked by the harness's own
     ``mask_selftest`` from OUTSIDE the sandbox. The agent is also asked to try
     reading a golden, but an agent's report is corroboration, never evidence —
     a compromised or confused agent is exactly the thing being measured.

Exit 0 only if every check passes. Anything else is a NO-GO, and the reason is
printed rather than summarised, because "the sandbox looked fine" is how an
answer key leaks twice.

Usage::

    .venv/bin/python merlin/experiments/capsule_bench/harness/codex_canary.py [--sandbox bwrap|none]

A live run costs one small Codex call against the subscription (no metered spend).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import codex_agent as CA  # noqa: E402

#: What the agent is asked to probe. Each line appends to a report file inside
#: the workspace, so a partial run still shows how far it got.
_PROBE_SCRIPT = """set +e
R=canary_report.txt
: > $R
echo "WS_WRITE=$( (echo ok > canary_write.txt && cat canary_write.txt) 2>&1 )" >> $R
echo "CODEX_HOME=$CODEX_HOME" >> $R
echo "DOTCODEX_ENTRIES=$(ls -A ~/.codex 2>&1 | tr '\\n' ',')" >> $R
echo "SESSIONS_VISIBLE=$(ls -A ~/.codex/sessions 2>/dev/null | head -3 | tr '\\n' ',')" >> $R
echo "AUTH_READABLE=$(test -r $CODEX_HOME/auth.json && echo yes || echo no)" >> $R
echo "GOLDEN_PROBE=$(cat {golden} 2>&1 | head -c 80)" >> $R
echo "GOLDEN_SIZE=$(stat -c %s {golden} 2>/dev/null || echo missing)" >> $R
echo "READABLE_GOLDENS=$(find {corpus} -name 'golden.yaml' -size +0 2>/dev/null | head -3 | tr '\\n' ',')" >> $R
echo "HIDDEN_DIRS=$(find {corpus} -type d -name hidden 2>/dev/null | head -3 | tr '\\n' ',')" >> $R
cat $R
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def run_canary(sandbox: str = "bwrap", timeout: int = 600,
               arm: str | None = None, bundle_id: str = "") -> tuple[int, dict]:
    """Run the canary. Returns ``(exit_code, report)``.

    With *arm*, the REAL bundle for that arm is loaded and its workspace
    assembled, so masking is checked against what a graded cell actually binds.
    Without it, the bundle is empty — the strictest case, but not the case where a
    re-exposure bug would live, since a real bundle legitimately binds corpus
    spec paths next to the answers.
    """
    import run_baseline_qa_loop as R

    from merlin.common.artifacts import cache_dir

    stamp = _now()
    work = cache_dir("codex_canary") / stamp
    ws = work / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    run_dir = work / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # The bundle under test: a real arm's (what a graded cell binds) or empty.
    bundle_info: dict = {"arm": arm, "bundle_id": bundle_id or None}
    if arm:
        import run_agent_experiment as RX

        # A target's bundle ids are its own (saturn_opu ships *_hwbringup_v0 where
        # the default map still names the gemmini-era *_public_v0), so honour an
        # explicit id the way run_baseline_qa_loop's --bundle does.
        if bundle_id:
            RX.ARM_BUNDLE[arm] = bundle_id
        bundle = RX._load_bundle(arm)
        assembled = R.assemble_copy_workspace(bundle, ws)
        bundle_info.update({
            "allowed": len(bundle.get("allowed") or []),
            "denied": len(bundle.get("denied") or []),
            "assembled": {k: v for k, v in (assembled or {}).items() if k != "files"},
        })
    else:
        bundle = {"allowed": [], "denied": []}

    golden, _iface = R._corpus_probe_paths()
    corpus = golden.parent.parent.parent  # the capsule corpus root
    (ws / "TASK.md").write_text(
        "This is an isolation canary, not a build task. Run the shell script in probe.sh "
        "exactly as given, then reply with the contents of canary_report.txt and nothing else.\n")
    probe = _PROBE_SCRIPT.format(golden=shlex.quote(str(golden)), corpus=shlex.quote(str(corpus)))
    (ws / "probe.sh").write_text(probe)

    checks: list[dict] = []

    # --- Check 4a: the harness's own masking proof, from OUTSIDE the sandbox ---
    try:
        mask = R.mask_selftest(ws, bundle, sandbox)
        # The field is "pilot_golden_visible_to_agent": OK | LEAK. There is no
        # "ok" key — reading one would make this check silently unfalsifiable.
        mask_ok = isinstance(mask, dict) and mask.get("pilot_golden_visible_to_agent") == "OK"
        checks.append(_check("mask_selftest", mask_ok, json.dumps(mask)[:600]))
    except Exception as exc:  # noqa: BLE001 — a canary reports, it does not crash
        checks.append(_check("mask_selftest", False, f"{type(exc).__name__}: {exc}"))

    # --- Run one real Codex round through the driver under test ---
    os.environ.setdefault("CODEX_CANARY", "1")
    rc, tpath = CA.run_round(ws, run_dir, os.environ.get("CANARY_MODEL", "gpt-5.6-sol"),
                             bundle, None, sandbox, 0, timeout, effort="low",
                             prompt="Run `bash probe.sh` in your working directory exactly as written, "
                                    "then reply with the contents of canary_report.txt and nothing else. "
                                    "Do not modify probe.sh. This is an isolation check, not a build task.")
    records = [json.loads(l) for l in tpath.read_text().splitlines() if l.strip()]
    summary = next((r for r in records if r.get("type") == "codex_summary"), {})
    report_path = ws / "canary_report.txt"
    agent_report = report_path.read_text() if report_path.is_file() else ""

    # --- Check 1+2: the binary ran and a turn completed (⇒ auth worked) ---
    checks.append(_check(
        "codex_reachable_and_authenticated",
        rc == 0 and summary.get("turns_usage_reported", 0) >= 1,
        f"rc={rc} turns_started={summary.get('turns_started')} "
        f"usage_reported={summary.get('turns_usage_reported')} "
        f"errors={summary.get('errors')}"))

    # --- Check: the workspace was writable from inside ---
    checks.append(_check("workspace_writable", (ws / "canary_write.txt").is_file(),
                         f"canary_write.txt present={ (ws / 'canary_write.txt').is_file() }"))

    # --- Check 3: no prior Codex session reachable inside the sandbox ---
    entries = _field(agent_report, "DOTCODEX_ENTRIES")
    sessions = _field(agent_report, "SESSIONS_VISIBLE")
    leaked = [e for e in entries.split(",") if e and e not in ("packages",)]
    checks.append(_check(
        "no_prior_codex_sessions_visible",
        bool(agent_report) and not sessions.strip(",") and not leaked,
        f"~/.codex entries={entries!r} sessions={sessions!r}"))

    # --- Check 4b: the agent could not read an answer (corroboration only) ---
    golden_size = _field(agent_report, "GOLDEN_SIZE")
    readable = _field(agent_report, "READABLE_GOLDENS")
    checks.append(_check(
        "agent_saw_no_golden_content",
        bool(agent_report) and golden_size in ("0", "missing") and not readable.strip(","),
        f"golden_size={golden_size!r} readable_goldens={readable!r}"))

    # --- Check: the JSONL tee survived the boundary ---
    raw = Path(summary.get("artifacts", {}).get("raw", ""))
    checks.append(_check("jsonl_tee_survived_the_boundary",
                         raw.is_file() and raw.stat().st_size > 0,
                         f"raw={raw} bytes={raw.stat().st_size if raw.is_file() else 0}"))

    # --- Check: no credential was written into the tree ---
    home_info = summary.get("codex_home") or {}
    home = Path(home_info.get("codex_home", "")) if home_info else None
    stray = sorted(str(p) for p in home.rglob("auth.json")
                   if home and p.is_file() and p.stat().st_size > 0) if home and home.is_dir() else []
    checks.append(_check("no_credential_written_to_the_tree",
                         home_info.get("auth_copied") is False and not stray,
                         f"auth_copied={home_info.get('auth_copied')} stray={stray}"))

    ok = all(c["ok"] for c in checks)
    report = {
        "verdict": "GO" if ok else "NO-GO",
        "sandbox": sandbox,
        "bundle": bundle_info,
        "timestamp": stamp,
        "checks": checks,
        "agent_report": agent_report,
        "codex_summary": summary,
        "transcript": str(tpath),
        "workspace": str(ws),
    }
    (work / "canary_report.json").write_text(json.dumps(report, indent=2))
    return (0 if ok else 1), report


def _field(text: str, key: str) -> str:
    """Read ``KEY=value`` out of the agent's report without regex."""
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return value.strip()
    return ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap",
                    help="bwrap (the boundary the isolation claim rests on) or none (diagnostic only)")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--arm", default=None,
                    help="load a REAL arm bundle (raw_baseline|merlin_assisted|cpp_merlininfra) so "
                         "masking is checked against what a graded cell actually binds")
    ap.add_argument("--bundle", default="", help="explicit bundle id for --arm (per-target ids differ)")
    a = ap.parse_args(argv)

    rc, report = run_canary(a.sandbox, a.timeout, a.arm, a.bundle)
    print(f"\n=== Codex sandbox canary: {report['verdict']} (sandbox={a.sandbox}, bundle={a.arm or 'empty'}) ===")
    for c in report["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail'][:300]}")
    if a.sandbox == "none":
        print("\nNOTE: --sandbox none proves nothing about masking; the outer bwrap IS the claim.")
    print(f"\nreport: {Path(report['workspace']).parent / 'canary_report.json'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
