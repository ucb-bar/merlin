"""Imperative build customization (project metadata lives in ``pyproject.toml``).

The read-only data the SDK loads at runtime — the ``*.schema.yaml`` schemas and the agent prompts —
lives at the repo top level (``merlin/schemas``, ``merlin/prompts``) as curated input corpora with
their own AGENT.md, OUTSIDE the importable package root (``merlin/python/merlin``). setuptools can
only ship data that sits INSIDE a package dir, so at build time we copy those trees into a bundled,
gitignored ``merlin/python/merlin/_data/`` which ``[tool.setuptools.package-data]`` then includes in
the wheel. The canonical copies stay put (in-repo tooling + ``data_path()``'s checkout branch read
them directly); the bundle is what ``pip install merlin`` resolves via ``importlib.resources`` when
no checkout is present. Single source of truth, refreshed on every build — no committed duplicate.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_ROOT = Path(__file__).resolve().parent
_PKG = _ROOT / "merlin" / "python" / "merlin"
# canonical (top-level) -> bundled (inside the package)
_BUNDLE = {kind: _PKG / "_data" / kind
           for kind in ("schemas", "prompts", "benchmarks", "contract", "targets", "runtime")}

# Per-tree exclusions from the bundle:
#  - benchmarks: the heavy capture corpora (``recaptures*`` — model.mlir/safetensors, tens of MB and
#    regenerable); a wheel user reaches them via MERLIN_BENCH_DIR at a checkout.
#  - targets: ``rtl_facts`` (per-target RTL-cert data — regenerable via the RTL flow, needs a
#    checkout); the wheel ships the target *contracts* (dialect_plan/target_contract), not cert data.
# Plus build cruft everywhere.
#  - contract: the benchmark ANSWER SURFACES. `check_no_answer_keys.py` keeps every golden, held-out
#    subtree and holdout sidecar UNTRACKED, and it is green because they are untracked -- but this
#    bundle copies from DISK, not from git, so the wheel shipped exactly what that gate exists to keep
#    unpublished. Two green checks with the gap between them publishing the answers; the gate that
#    inspects the WHEEL (check_standalone_install.py) is wired to nothing and does not complete.
#    `check_no_answer_keys._is_answer_key` remains the authority on what an answer surface IS;
#    merlin/tests/infra/test_wheel_excludes_answer_keys.py holds these two in agreement against the
#    real tree, so this list cannot silently fall behind it.
_EXCLUDE = {"benchmarks": ("recaptures",), "targets": ("rtl_facts",),
            "contract": ("hidden", "golden")}
#: Name SUFFIXES excluded per tree. Needed because the holdout SPECIFICATION sidecar is named for its
#: target (`radiance.hidden.yaml`), so no prefix rule reaches it -- and a holdout's op, dtype and exact
#: shape is itself an answer. `expected_instruction_coverage.yaml` is here for the same reason: the
#: required instruction classes ARE the answer. The agreement test found this one; do not hand-extend
#: this list without re-running it.
_EXCLUDE_SUFFIXES = {"contract": (".hidden.yaml", "expected_instruction_coverage.yaml")}
# The bundle ships read-only DATA only — never code. The corpora carry dev/repro helper scripts
# (e.g. benchmarks/dse_guidance/verify_implementation.py) the SDK never loads; keep them out of the
# wheel (smaller, and no stray code under the importable package for the repo linters to scan).
_CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".sh")


def _ignore_for(kind: str):
    prefixes = _EXCLUDE.get(kind, ())
    suffixes = _EXCLUDE_SUFFIXES.get(kind, ())

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names
                if n == "__pycache__"
                or n.endswith(_CODE_SUFFIXES)
                or any(n.startswith(p) for p in prefixes)
                or any(n.endswith(x) for x in suffixes)}

    return _ignore


def _sync_bundled_data() -> None:
    for kind, dst in _BUNDLE.items():
        src = _ROOT / "merlin" / kind
        if not src.is_dir():  # e.g. building from an sdist that already vendored _data
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=_ignore_for(kind))


class BuildPyWithData(build_py):
    def run(self) -> None:  # noqa: D102
        _sync_bundled_data()
        super().run()


setup(cmdclass={"build_py": BuildPyWithData})
