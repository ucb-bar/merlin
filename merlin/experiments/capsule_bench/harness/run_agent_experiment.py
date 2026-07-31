#!/usr/bin/env python3
"""Launch ONE isolated, instrumented target-generation experiment, then grade it.

The session that BUILT this harness is not an experiment run. A valid run is created only here: a
fresh run dir, a workspace assembled from exactly one declared input bundle (denied paths masked),
a non-interactive agent launch with full transcript + token/cost capture, a frozen submission, and a
separate full-access grading phase.

Agent launch:
  - real:  claude --print --model <m> --output-format stream-json --verbose < TASK.md   (in workspace)
  - dummy: --dummy-agent copies a known-good reference submission (pipeline validation, no LLM)

Sandbox:
  - bwrap (real runs): bind allowed paths RO, fresh writable workspace, /scratch* hidden.
  - none (dummy/dev): copy-based workspace; denied paths asserted absent.

Usage:
  run_agent_experiment.py --arm raw_baseline --run-id rb_0001 --model opus [--sandbox bwrap]
  run_agent_experiment.py --arm raw_baseline --run-id dry_0001 --dummy-agent \
      --dummy-submission out/artifacts/targets/<target>/agent_spec_v1_mlir_oot --no-oracle-grade \
      --grade-capsules /tmp/grade_subset
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _common as C

sys.path.insert(0, str(C.REPO / "merlin" / "python"))
from merlin.targetgen import experiment_tokens as ET  # noqa: E402

ARM_BUNDLE = {"raw_baseline": "raw_baseline_public_v0", "merlin_assisted": "merlin_assisted_public_v0",
              "cpp_merlininfra": "cpp_merlininfra_hwbringup_v0"}


def _load_bundle(arm: str) -> dict:
    b = C.BUNDLES / ARM_BUNDLE[arm] / "input_bundle_manifest.yaml"
    return yaml.safe_load(b.read_text())


def assemble_workspace(bundle: dict, ws: Path) -> list[str]:
    """Symlink each allowed path into the workspace (RO intent); return the denied basenames to verify."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "submission").mkdir(exist_ok=True)
    for entry in bundle.get("allowed", []):
        # Grants are repo-root-relative with the documented ``experiments/... resolves under merlin/``
        # shorthand — resolve exactly as the sandbox binder does so both stay in lockstep.
        src = C.REPO / entry["path"]
        if not src.exists():
            src = C.REPO / "merlin" / entry["path"]
        if not src.exists():
            continue
        # Honor an explicit ``as:`` alias (e.g. hwbringup set mounted as ``<target>/``) so the workspace
        # entry matches the name the prompt tells the agent to read; else fall back to the basename.
        dst = ws / (entry.get("as") or Path(entry["path"]).name)
        if dst.exists() or dst.is_symlink():
            dst = ws / entry["path"].replace("/", "_").rstrip("_")
        try:
            dst.symlink_to(src)
        except FileExistsError:
            pass
    return [Path(d["path"]).name for d in bundle.get("denied", [])]


def assert_isolation(ws: Path, bundle: dict) -> list[str]:
    """Confirm no denied path is reachable as a workspace entry. Returns violations (empty = ok)."""
    viol = []
    for d in bundle.get("denied", []):
        target = (C.REPO / d["path"]).resolve()
        for entry in ws.iterdir():
            try:
                if entry.is_symlink() and entry.resolve() == target:
                    viol.append(f"denied path reachable: {entry.name} -> {d['path']}")
                elif entry.resolve() == target:
                    viol.append(f"denied path present: {d['path']}")
            except Exception:
                pass
    return viol


def _dummy_transcript(path: Path) -> None:
    """A minimal stream-json transcript so token/cost extraction is exercised on the dummy path."""
    evs = [
        {"type": "assistant", "message": {"id": "dummy1", "model": "claude-opus-4-8",
            "usage": {"input_tokens": 4200, "cache_creation_input_tokens": 800,
                      "cache_read_input_tokens": 15000, "output_tokens": 950},
            "content": [{"type": "thinking", "text": "..."},
                        {"type": "tool_use", "name": "Write"},
                        {"type": "tool_use", "name": "Bash"}]}},
        {"type": "result", "subtype": "success"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in evs) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARM_BUNDLE))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--task", default=str(C.EXP / "task" / "TASK.md"))
    ap.add_argument("--sandbox", choices=["bwrap", "none"], default="bwrap")
    ap.add_argument("--allow-unsandboxed", action="store_true",
                    help="permit a real (non-dummy) run with --sandbox none (NOT for trusted results)")
    ap.add_argument("--dummy-agent", action="store_true",
                    help="copy a known-good submission instead of launching an LLM (pipeline test)")
    ap.add_argument("--dummy-submission",
                    default=str(C.REPO / "out/artifacts/targets" / C.TARGET / "agent_spec_v1_mlir_oot"))
    ap.add_argument("--no-oracle-grade", action="store_true")
    ap.add_argument("--grade-capsules", default=str(C.REPO / "merlin/contract" / "capsules"))
    ap.add_argument("--skip-hidden", action="store_true")
    a = ap.parse_args(argv)

    # A real (non-dummy) run MUST be sandboxed: without bwrap the agent can read any absolute path
    # (incl. denied /scratch* dirs), so workspace assembly alone does not isolate.
    if not a.dummy_agent and a.sandbox != "bwrap" and not a.allow_unsandboxed:
        print("REFUSING: a real run requires --sandbox bwrap (or explicit --allow-unsandboxed). "
              "Workspace assembly alone does not hide denied absolute paths.", file=sys.stderr)
        return 4

    bundle = _load_bundle(a.arm)
    run_dir = C.RUNS / a.arm / a.run_id
    if run_dir.exists():
        print(f"run dir exists, refusing to overwrite: {run_dir}", file=sys.stderr)
        return 2
    ws = run_dir / "workspace"
    run_dir.mkdir(parents=True)

    # provenance + bundle copy + task
    shutil.copy(C.BUNDLES / ARM_BUNDLE[a.arm] / "input_bundle_manifest.yaml",
                run_dir / "input_bundle_manifest.yaml")
    shutil.copy(a.task, run_dir / "TASK.md")
    denied_names = assemble_workspace(bundle, ws)
    viol = assert_isolation(ws, bundle)
    (run_dir / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": a.run_id, "arm": a.arm, "model": a.model, "effort": a.effort,
        "sandbox": a.sandbox, "dummy_agent": a.dummy_agent,
        "repo_sha": C.repo_sha(), "bundle_id": bundle["bundle_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "isolation_violations": viol,
        "denied_paths_checked": denied_names,
    }, sort_keys=False))
    if viol:
        print(f"ISOLATION FAILURE: {viol}", file=sys.stderr)
        return 3

    transcript = run_dir / "transcript.jsonl"
    t0 = time.time()
    exit_code = 0
    if a.dummy_agent:
        # harness stand-in: emit a known-good submission + a synthetic transcript (no LLM)
        src = C.REPO / a.dummy_submission
        dst = run_dir / "submission"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build", "__pycache__"))
        _dummy_transcript(transcript)
        (run_dir / "claude_stdout.log").write_text("[dummy-agent] copied known-good submission\n")
        (run_dir / "claude_stderr.log").write_text("")
    else:
        cmd = (f'claude --print --model {a.model} --effort {a.effort} '
               f'--permission-mode bypassPermissions --output-format stream-json --verbose '
               f'< {run_dir / "TASK.md"}')
        if a.sandbox == "bwrap":
            cmd = _bwrap_wrap(cmd, ws, bundle)
        with open(transcript, "w") as tf, open(run_dir / "claude_stderr.log", "w") as ef:
            proc = subprocess.run(["bash", "-c", cmd], cwd=str(ws), stdout=tf, stderr=ef)
            exit_code = proc.returncode
        # the agent writes its package into workspace/submission; capture it
        wsub = ws / "submission"
        if wsub.exists():
            dst = run_dir / "submission"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(wsub, dst, ignore=shutil.ignore_patterns("build", "__pycache__"))
    wall = round(time.time() - t0, 3)

    # process metrics
    summ = ET.parse_transcript(transcript)
    ET.write_cost_yaml(summ, run_dir / "cost_time_toolcalls.yaml",
                       wall_time_seconds=wall, model=a.model, exit_code=exit_code)

    # separate full-access grading phase
    grade_cmd = [sys.executable, str(C.EXP / "scripts" / "grade_agent_run.py"),
                 "--run-dir", str(run_dir), "--arm", a.arm, "--model", a.model,
                 "--capsules", a.grade_capsules]
    if a.no_oracle_grade:
        grade_cmd.append("--no-oracle")
    if a.skip_hidden:
        grade_cmd.append("--skip-hidden")
    subprocess.run(grade_cmd, cwd=str(C.REPO))
    print(f"run complete: {run_dir}")
    return 0


def bwrap_argv(ws: Path, bundle: dict) -> list[str]:
    """Deny-by-default bwrap argv prefix (delegates to the shared, target-agnostic sandbox base): system
    RO, /scratch* tmpfs-hidden, ONLY the bundle's allowed paths bound RO, denied sub-paths re-masked,
    workspace writable+last. Reused by the launcher (real runs) and the pre-flight canary probe."""
    from merlin.targetgen.sandbox import bwrap as _BW
    return _BW.base_argv(ws, bundle, repo=C.REPO)


def _bwrap_wrap(cmd: str, ws: Path, bundle: dict) -> str:
    return " ".join(bwrap_argv(ws, bundle)) + f" bash -c '{cmd}'"


if __name__ == "__main__":
    raise SystemExit(main())
