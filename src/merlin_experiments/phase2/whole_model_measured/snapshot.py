"""The frozen source a measured run executes: taken from a COMMIT, checked before launch, never collided.

WHY A COMMIT, NOT THE TREE.  A plain copy of a live checkout cannot tell a reviewed byte from another
session's uncommitted, in-progress edit sitting in the same tree; one such edit once rode into a
campaign's frozen snapshot and changed what the whole-model builder did, unreviewed, for every round.
:func:`committed_source` refuses a dirty tracked tree (or a stray untracked ``.py`` under the snapshot's
source roots), naming every offending path, and otherwise archives exactly ``HEAD`` into a scratch
checkout.  Host-local, never-tracked inputs -- ``.env``, the ``.venv`` and ``third_party`` links -- come
from the LIVE tree, because a commit has none of them and a relaunch that silently lost ``.env`` failed
its first external-path lookup.  The sealed copy itself is made by :mod:`merlin_experiments.source_snapshot`.

WHY A PREFLIGHT.  A frozen snapshot whose own compiler does not resolve, or whose declared screen
capsules are absent, fails a night later rather than at launch.  :func:`preflight` runs INSIDE the
snapshot, with its own import root, and refuses the launch loudly.

WHY :func:`claim_dir`.  A per-iteration analysis snapshot indexed by a count of RECORDED iterations
collides with the leftover of an interrupted one; a collision once ended a run outright.  The leftover
is moved aside under its own name, never deleted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Host-local files a commit never carries, taken from the live tree.
LIVE_FILES = (".env", ".env.example")
#: Host-local directories a commit never carries, linked from the live tree.
LIVE_LINKS = (".venv", "third_party")
RECEIPT_SCHEMA = "merlin.phase2.whole_model_measured.commit_snapshot.v1"


class SnapshotError(RuntimeError):
    """A frozen source snapshot cannot be taken, or cannot run what the launch declares."""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def dirty_paths(repo_root: Path, *, source_roots: Sequence[str]) -> list[str]:
    """Every path under ``source_roots`` a commit snapshot cannot account for: a TRACKED file with an
    uncommitted change against ``HEAD`` (staged or not), or an UNTRACKED ``.py`` file."""
    roots = [str(r) for r in source_roots]
    modified = _git(Path(repo_root), "diff", "--name-only", "HEAD", "--", *roots).split()
    untracked = _git(Path(repo_root), "ls-files", "--others", "--exclude-standard", "--", *roots).split()
    return sorted(set(modified) | {p for p in untracked if p.endswith(".py")})


def committed_source(repo_root: Path, tmp_root: Path, *, source_roots: Sequence[str]) -> tuple[Path, str]:
    """A clean checkout of exactly ``HEAD`` under ``tmp_root`` (and the commit sha), with the live
    tree's host-local files and links beside it.  The caller removes the checkout once the sealed
    copy is made."""
    repo_root = Path(repo_root).resolve()
    dirty = dirty_paths(repo_root, source_roots=source_roots)
    if dirty:
        named = ", ".join(dirty[:20]) + (", ..." if len(dirty) > 20 else "")
        raise SnapshotError(f"the tracked tree is dirty; a frozen snapshot must come from a commit ({named})")
    sha = _git(repo_root, "rev-parse", "HEAD").strip()
    checkout = Path(tmp_root) / f"commit-{sha}-{os.getpid()}"
    if checkout.exists():
        raise SnapshotError(f"commit checkout destination already exists: {checkout}")
    checkout.mkdir(parents=True, mode=0o700)
    archive = subprocess.Popen(["git", "archive", sha], cwd=repo_root, stdout=subprocess.PIPE)
    try:
        extract = subprocess.run(["tar", "-x", "-C", str(checkout)], stdin=archive.stdout, check=False)
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
        archive_rc = archive.wait()
    if archive_rc != 0 or extract.returncode != 0:
        shutil.rmtree(checkout, ignore_errors=True)
        raise SnapshotError(f"git archive {sha} | tar -x failed (git rc={archive_rc}, tar rc={extract.returncode})")
    for name in LIVE_FILES:
        if (repo_root / name).is_file():
            shutil.copy2(repo_root / name, checkout / name)
    for name in LIVE_LINKS:
        if (repo_root / name).exists() and not (checkout / name).exists():
            (checkout / name).symlink_to((repo_root / name).resolve(), target_is_directory=True)
    return checkout, sha


def create_from_commit(
    repo_root: Path, destination: Path, *, tmp_root: Path, source_roots: Sequence[str], **create_options: Any
) -> dict[str, Any]:
    """Seal ``destination`` (via :func:`merlin_experiments.source_snapshot.create`) from the commit at
    ``HEAD``, and write the commit it came from beside it (``<destination>.commit.json``)."""
    from merlin_experiments import source_snapshot

    checkout, sha = committed_source(repo_root, tmp_root, source_roots=source_roots)
    try:
        receipt = source_snapshot.create(
            checkout, Path(destination), source_roots=tuple(source_roots), **create_options
        )
    finally:
        shutil.rmtree(checkout, ignore_errors=True)
    record = {
        "schema": RECEIPT_SCHEMA,
        "commit_sha": sha,
        "repo_root": str(Path(repo_root).resolve()),
        "snapshot": str(destination),
        "seal": str(receipt),
        "live_files": [n for n in LIVE_FILES if (Path(repo_root) / n).is_file()],
        "at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    }
    Path(str(destination) + ".commit.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def claim_dir(path: Path) -> Path:
    """``path``, free to be written: an existing leftover is moved aside under its own name, never deleted."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        path.rename(path.with_name(f"{path.name}.interrupted_{time.time_ns()}"))
    return path


#: Run INSIDE the snapshot with its own python and import root, so what it reports is what a real
#: screen would see.  It names no target and no capsule; both are arguments.
PREFLIGHT_SCRIPT = """
import json, sys
from pathlib import Path
result = {"ok": False, "reason": None, "clang": None, "missing_capsules": []}
try:
    from merlin.llvmlower import toolchain
    clang = toolchain.clang()
    result["clang"] = str(clang)
    if not Path(clang).is_file():
        result["reason"] = f"the resolved compiler {clang} is not a file"
    else:
        catalog = Path(sys.argv[1]) if sys.argv[1] else None
        wanted = [c.strip() for c in sys.argv[2].split(",") if c.strip()]
        names = {p.parent.name for p in catalog.rglob("capsule.yaml")} if catalog and catalog.is_dir() else set()
        missing = [c for c in wanted if c not in names]
        result["missing_capsules"] = missing
        if missing:
            result["reason"] = f"the capsule catalog {catalog} does not have {missing}"
        else:
            result["ok"] = True
except Exception as exc:
    result["reason"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
"""


def preflight(
    source: Path,
    *,
    python: str,
    import_roots: Sequence[str],
    capsule_catalog: Path | None,
    required_capsules: Sequence[str],
    timeout_s: float = 300,
) -> dict[str, Any]:
    """Refuse to launch when the frozen snapshot at ``source`` cannot run its own declared screen: its
    compiler resolves to a real file and the declared screen capsules are in the declared catalog."""
    argv = [python, "-c", PREFLIGHT_SCRIPT, str(capsule_catalog or ""), ",".join(required_capsules)]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(Path(source) / root) for root in import_roots),
        "MERLIN_REPO_ROOT": str(source),
    }
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(source))
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError(f"snapshot preflight timed out after {timeout_s:g}s") from exc
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if not lines:
        raise SnapshotError(f"snapshot preflight produced no report (exit {done.returncode}): {done.stderr[-2000:]}")
    try:
        report = json.loads(lines[-1])
    except ValueError as exc:
        raise SnapshotError(f"snapshot preflight report is not JSON: {lines[-1]!r}") from exc
    if not report.get("ok"):
        raise SnapshotError(f"snapshot preflight refused the launch: {report.get('reason')}")
    return report


__all__ = ["SnapshotError", "claim_dir", "committed_source", "create_from_commit", "dirty_paths", "preflight"]
