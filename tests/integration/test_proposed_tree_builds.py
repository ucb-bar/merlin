"""End-to-end: stage-mutation -> proposed_tree -> ./merlin build.

This is the **load-bearing** Tier 2 test. For each host-friendly Merlin
build profile, it:

  1. Resolves the matching capability spec.
  2. Runs ``./merlin targetgen stage-mutation`` to produce ``proposed_tree``.
  3. Creates a fresh ``git worktree`` of the repo at HEAD.
  4. Copies the ``proposed_tree`` over the worktree (overlay-style) and
     ``git add`` s the new files so CMake-discovered subdirectories see them.
  5. Runs ``./merlin build --profile <X>`` inside the worktree.
  6. Asserts exit 0.
  7. Tears the worktree down via ``git worktree remove --force``.

Today's scaffold templates emit ``TODO``-only stubs that **will not**
build cleanly without the LLM filling them in. So the realistic Tier 2
contract is: *the unmodified existing repo* still builds with each
profile, after a no-op stage-mutation overlay (i.e., the overlay introduces
nothing that breaks the existing build). The test pivots on this:

  * ``test_baseline_build_per_profile`` — runs ``./merlin build`` against
    a clean worktree at HEAD with **no** overlay. Proves the build env
    works and the host-friendly profiles still configure + compile.
  * ``test_stage_mutation_overlay_does_not_break_existing_build`` —
    overlays only the *non-source* scaffold artefacts (Markdown notes,
    ``proposal_brief.md``, etc.) onto the worktree and re-runs the
    build. Asserts the overlay does not regress the build.

The "scaffold compiles end-to-end" test is intentionally Tier 3
(LLM-in-loop) — it requires Claude Code to fill in the TODOs first.

Markers: ``integration``, ``slow``. Skips cleanly if ``./merlin``
cannot reach a healthy Python env (broken ``.venv``, missing conda env).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "target_specs" / "examples"

# Build profiles vetted as host-friendly (no cross-toolchain required).
HOST_BUILD_PROFILES: tuple[tuple[str, str], ...] = (
    # (profile_name, capability_target_name_for_overlay)
    ("vanilla", "saturn_opu_v128"),
    ("gemmini", "gemmini_mx"),
    ("npu", "npu_ucb"),
    ("radiance", "radiance_muon"),
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _conda_python_works() -> bool:
    """The Merlin build needs a Python interpreter accessible via uv. Locally,
    the ``.venv`` symlink can break (root-owned). We probe whether
    ``./merlin`` can resolve at all."""
    proc = subprocess.run(
        ["./merlin", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return proc.returncode == 0


def _make_worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "worktree"
    branch = f"targetgen-buildtest-{tmp_path.name}"
    proc = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"git worktree add failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return worktree


def _remove_worktree(worktree: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _merlin_build(worktree: Path, profile: str, timeout_s: int = 1800) -> subprocess.CompletedProcess:
    """Drive ``./merlin build --profile X`` against a worktree.

    Always uses ``--reconfigure`` so the cache from a prior run inside the
    fresh tmp build dir does not leak between profiles.
    """
    return subprocess.run(
        ["./merlin", "build", "--profile", profile, "--reconfigure"],
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


@pytest.fixture(scope="session")
def build_env_available() -> bool:
    if not _conda_python_works():
        pytest.skip(
            "`./merlin --help` failed in this environment "
            "(commonly caused by a root-owned .venv that uv cannot recreate). "
            "Skipping the live-build tier; the structural tier covers this."
        )
    return True


@pytest.mark.parametrize("profile,target", HOST_BUILD_PROFILES, ids=lambda p: p if isinstance(p, str) else None)
def test_baseline_build_per_profile(build_env_available: bool, tmp_path: Path, profile: str, target: str) -> None:
    """Sanity: the unmodified repo at HEAD still builds with each
    host-friendly profile. This is the test that catches "I broke
    `./merlin build`" before any TargetGen overlay.

    A passing baseline is a precondition for the overlay test.
    """
    worktree = _make_worktree(tmp_path)
    try:
        rc = _merlin_build(worktree, profile)
        assert rc.returncode == 0, (
            f"baseline `./merlin build --profile {profile}` failed:\n"
            f"STDOUT (last 3000):\n{rc.stdout[-3000:]}\n"
            f"STDERR (last 1500):\n{rc.stderr[-1500:]}"
        )
    finally:
        _remove_worktree(worktree)


def _stage_mutation_into(worktree: Path, target: str, out_dir: Path) -> Path:
    cap = REPO_ROOT / "target_specs" / "examples" / target / "capability.yaml"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "merlin.py"),
            "targetgen",
            "stage-mutation",
            str(cap),
            "--out-dir",
            str(out_dir),
        ],
        cwd=worktree,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"stage-mutation failed for {target}:\n{proc.stderr}\n{proc.stdout}"
    return out_dir / target / "mutation" / "proposed_tree"


def _overlay_non_source_scaffold(proposed_tree: Path, worktree: Path) -> list[Path]:
    """Copy only Markdown / TXT scaffold artefacts onto the worktree.

    Source-bearing scaffolds (.cpp/.h/.cmake) are deliberately *not*
    overlaid — they contain TODOs and would break the build until an LLM
    fills them in. The overlay test only proves that adding the
    non-source advisory artefacts does not regress configure/build.
    """
    copied: list[Path] = []
    advisory_suffixes = {".md", ".txt", ".rst"}
    for src in proposed_tree.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in advisory_suffixes:
            continue
        rel = src.relative_to(proposed_tree)
        dst = worktree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


@pytest.mark.parametrize("profile,target", HOST_BUILD_PROFILES, ids=lambda p: p if isinstance(p, str) else None)
def test_stage_mutation_overlay_does_not_break_build(
    build_env_available: bool, tmp_path: Path, profile: str, target: str
) -> None:
    """After overlaying TargetGen's *non-source* scaffold artefacts onto a
    clean worktree, ``./merlin build --profile X`` still succeeds.

    This is the load-bearing assertion: TargetGen's scaffold output does
    not silently break the upstream build. If a Markdown note ends up in
    a directory CMake `glob()`s for sources, this test is the alarm.
    """
    worktree = _make_worktree(tmp_path)
    try:
        proposed = _stage_mutation_into(worktree, target, tmp_path / "stage_out")
        copied = _overlay_non_source_scaffold(proposed, worktree)
        assert copied, (
            f"stage-mutation produced no advisory artefacts for {target}; "
            f"the overlay test does not exercise the planner's output"
        )
        rc = _merlin_build(worktree, profile)
        assert rc.returncode == 0, (
            f"`./merlin build --profile {profile}` regressed after overlay "
            f"of {len(copied)} advisory file(s):\n"
            f"STDOUT (last 3000):\n{rc.stdout[-3000:]}\n"
            f"STDERR (last 1500):\n{rc.stderr[-1500:]}"
        )
    finally:
        _remove_worktree(worktree)
