"""Repair-iteration recorder: make the path-to-green first-class evidence.

Each call to :func:`record_iteration` writes an ``iteration_{NNN}/`` directory capturing what the
backend looked like before/after, the first-failure plane, the numeric/trace diffs, a git patch of
what changed, and free-form notes. :func:`freeze` pins the repo commit + toolchain so the hidden set
can be run once against a frozen artifact (hidden capsules are only run after a freeze exists).

All git access is read-only (``git diff`` / ``git rev-parse``); the recorder never commits.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _git(args: list[str], cwd: str | Path | None = None) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True, timeout=60)
        return p.stdout if p.returncode == 0 else ""
    except Exception:
        return ""


def repo_head(repo: str | Path = ".") -> str:
    return _git(["rev-parse", "HEAD"], repo).strip() or "unknown"


def working_tree_diff(repo: str | Path = ".", paths: list[str] | None = None) -> str:
    """Read-only diff of the working tree (optionally restricted to paths)."""
    args = ["diff"]
    if paths:
        args += ["--", *paths]
    return _git(args, repo)


def first_failure(results: list[dict]) -> dict | None:
    """The lowest-tier / earliest plane failure across the suite (ordering ~ ladder order)."""
    plane_order = ["schema", "parse", "interface_to_target", "target_to_command_buffer",
                   "command_buffer_schema", "command_buffer_reference", "command_buffer_simulate",
                   "numeric_golden", "target_to_instruction_trace", "trace_check",
                   "target_to_llvm", "llvm_compile", "runtime_link", "spike", "verilator",
                   "vcs", "firesim", "oracle_unavailable", "runner_internal"]

    def rank(p):
        return plane_order.index(p) if p in plane_order else len(plane_order)

    fails = [(r["capsule"], r["failure"]) for r in results
             if r.get("status") not in ("pass",) and r.get("failure")]
    if not fails:
        return None
    cap, f = min(fails, key=lambda cf: rank(cf[1].get("plane", "")))
    return {"capsule": cap, **f}


def status_map(results: list[dict]) -> dict[str, str]:
    return {r["capsule"]: r["status"] for r in results}


def regressions(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return [c for c in after if before.get(c) == "pass" and after[c] != "pass"]


def record_iteration(index: int, *, run_dir: str | Path,
                     before: dict[str, str], after: dict[str, str],
                     results: list[dict], repo: str | Path = ".",
                     changed_paths: list[str] | None = None,
                     phase: str = "repair", notes: str = "",
                     numeric_diff: dict | None = None,
                     trace_diff: dict | None = None,
                     profile: dict | None = None,
                     cost_time_toolcalls: dict | None = None,
                     started_at: str | None = None, ended_at: str | None = None,
                     frozen: bool = False, contract: str | Path | None = None) -> Path:
    """Write iteration_{index:03d}/ with the full repair evidence; returns the dir."""
    from .contract import schemas
    d = Path(run_dir) / f"iteration_{index:03d}"
    d.mkdir(parents=True, exist_ok=True)

    patch = working_tree_diff(repo, changed_paths)
    (d / "files_changed.patch").write_text(patch, encoding="utf-8")
    (d / "capsule_status_before.yaml").write_text(yaml.safe_dump(before, sort_keys=True))
    (d / "capsule_status_after.yaml").write_text(yaml.safe_dump(after, sort_keys=True))
    ff = first_failure(results)
    (d / "first_failure.yaml").write_text(yaml.safe_dump(ff or {}, sort_keys=False))
    (d / "numeric_diff.yaml").write_text(yaml.safe_dump(numeric_diff or {}, sort_keys=False))
    (d / "instruction_trace_diff.yaml").write_text(yaml.safe_dump(trace_diff or {}, sort_keys=False))
    (d / "profile.yaml").write_text(yaml.safe_dump(profile or {}, sort_keys=False))
    if cost_time_toolcalls is not None:
        (d / "cost_time_toolcalls.yaml").write_text(
            yaml.safe_dump(cost_time_toolcalls, sort_keys=False))
    if notes:
        (d / "notes.md").write_text(notes, encoding="utf-8")

    record = {
        "index": index, "phase": phase, "started_at": started_at, "ended_at": ended_at,
        "status_before": before, "status_after": after, "first_failure": ff,
        "files_changed": "files_changed.patch",
        "numeric_diff": numeric_diff or {}, "instruction_trace_diff": trace_diff or {},
        "profile": profile or {}, "cost_time_toolcalls": cost_time_toolcalls,
        "regressions": regressions(before, after), "notes": notes or None, "frozen": frozen,
    }
    try:
        schemas.validate(record, "iteration", contract=contract)
    except schemas.ContractViolation as e:
        import sys
        sys.stderr.write(f"WARNING: iteration record self-validation failed: {e}\n")
    (d / "iteration.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return d


def freeze(run_dir: str | Path, *, results: list[dict], repo: str | Path = ".",
           toolchain_shas: dict | None = None, hidden_eval_version: str = "v0") -> Path:
    """Write freeze.json pinning the repo commit + toolchain; gates hidden-set execution."""
    from datetime import datetime, timezone
    allpass = all(r.get("status") == "pass" for r in results)
    rec = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_head(repo),
        "public_dev_status": "all_pass" if allpass else "NOT_all_pass",
        "n_capsules": len(results),
        "toolchain_shas": toolchain_shas or {},
        "hidden_eval_version": hidden_eval_version,
    }
    p = Path(run_dir) / "freeze.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return p


def is_frozen(run_dir: str | Path) -> bool:
    return (Path(run_dir) / "freeze.json").is_file()
