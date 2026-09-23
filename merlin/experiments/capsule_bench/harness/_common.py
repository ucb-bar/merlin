"""Shared helpers for the target-neutral capsule-bench isolation harness.

Thin shim over ``merlin.benchharness`` (the shared harness primitives). This module is imported by
harness scripts BEFORE they add merlin/python to sys.path, so it bootstraps the repo root itself
(git first, parents[] fallback), puts merlin/python on the path, then re-exports the shared helpers.
Public symbols (REPO/HARNESS/EXP/RUNS/REPORTS/BUNDLES/sh/hash_tree/repo_sha) are preserved for callers.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Self-contained bootstrap (runs before merlin is importable).
_HERE = Path(__file__).resolve()
_root = os.environ.get("MERLIN_REPO_ROOT", "").strip()
if not _root:
    _root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=str(_HERE.parent), capture_output=True, text=True
    ).stdout.strip()
REPO = Path(_root).expanduser().resolve() if _root else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin_experiments.phase1 import context as _context  # noqa: E402

from merlin.benchharness import hash_tree, repo_sha, reports_root, runs_root, sh  # noqa: E402, F401

# The harness lives in its own TARGET-NEUTRAL home (experiments/capsule_bench/harness); HARNESS is that
# home. DESCRIPTOR selects the target definition; EXP is its declared resource directory,
# which need not contain the descriptor. Native discovery retains its historical default;
# shared context initialization resolves explicit resource ownership below.
HARNESS = _HERE.parent
_override = os.environ.get("MERLIN_TARGET_EXPERIMENT", "").strip()
if _override:
    _desc = Path(_override).expanduser()
    if not _desc.is_file():
        raise SystemExit(f"MERLIN_TARGET_EXPERIMENT={_override!r} is not a readable descriptor file")
    EXP = _desc.resolve().parent
    # Normalize the env var to an ABSOLUTE path so child processes (host-side brokers, the sandboxed
    # agent's tools) that inherit it resolve the descriptor regardless of their cwd. A relative override
    # resolves here (main process runs from the repo root) but breaks a broker chdir'd elsewhere.
    os.environ["MERLIN_TARGET_EXPERIMENT"] = str(_desc.resolve())
else:
    _desc = REPO / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"  # default target
    EXP = _desc.parent


def _source_experiment_env(exp_dir: Path) -> list[str]:
    """Compatibility entrypoint; the descriptor-aware library owns the sidecar parser."""
    from merlin.targetgen.corpora import source_experiment_env

    return source_experiment_env(descriptor=exp_dir / "target_experiment.yaml")


def _legacy_target(descriptor: Path) -> str:
    """Only this native edge retains permissive descriptor/dir-name fallback."""
    try:
        import yaml

        target = (yaml.safe_load(descriptor.read_text()) or {}).get("target") if descriptor.is_file() else None
    except Exception:  # noqa: BLE001
        target = None
    return target or EXP.name.split("_")[0]


# The package owner sources tooling before target parsing and output routing. Native discovery,
# default selection and environment normalization above intentionally remain compatibility-only.
CONTEXT = _context.load_context(_desc, repo=REPO, harness=HARNESS, _legacy_target_reader=_legacy_target)
EXP = CONTEXT.experiment
SOURCED_EXPERIMENT_ENV = list(CONTEXT.sourced_environment)
TARGET = CONTEXT.target
#: The SELECTED target's descriptor file itself. Exported so a harness script can load the descriptor
#: (answer surfaces, oracle routing) instead of re-deriving its path from EXP and guessing the filename.
DESCRIPTOR = _desc
RUNS = CONTEXT.runs
REPORTS = CONTEXT.reports
BUNDLES = CONTEXT.bundles


def require_scaffolding() -> None:
    """Use current legacy globals so existing launcher overrides remain observable."""
    _context.require_scaffolding(EXP, TARGET)


def experiment_conditions() -> list[str]:
    """Use the package owner while retaining patchable native bundle selection."""
    return _context.experiment_conditions(BUNDLES)


__all__ = [
    "REPO",
    "EXP",
    "HARNESS",
    "TARGET",
    "RUNS",
    "REPORTS",
    "BUNDLES",
    "sh",
    "hash_tree",
    "repo_sha",
    "require_scaffolding",
    "experiment_conditions",
    "SOURCED_EXPERIMENT_ENV",
]
