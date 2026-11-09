#!/usr/bin/env python3
"""Link into a git worktree the gitignored inputs it needs to produce a true answer.

WHY THIS EXISTS. A `git worktree` of this repo carries only tracked files. Everything a run actually
needs beyond that -- the per-machine `.env`, the venv, the LLVM toolchain, the derived fact caches, the
capsule answer keys and model weights, the hidden cohort -- is gitignored, and NOT ONE of them fails
loudly when absent. Each turns into a plausible-looking wrong answer: a corpus that reads as empty, a
capability that reads as undeclared, a module that skips rather than fails. Measured 2026-09-23:
standing up one runnable worktree took eight such artifacts, found one at a time over several hours,
and three wrong conclusions were drawn from trees missing one.

The roster is `build_tools/scripts/worktree_provisioning.yaml` -- data, reviewed, each entry carrying
the CONSEQUENCE of its absence, because the symptom is never "file not found".

    python build_tools/scripts/provision_worktree.py <worktree>           # link, then verify
    python build_tools/scripts/provision_worktree.py <worktree> --verify  # report only, change nothing
    python build_tools/scripts/provision_worktree.py <worktree> --from <checkout>

Links rather than copies: the artifacts are large and read-mostly, and a copy drifts from the source
silently -- which is the failure this script exists to prevent, one level up. An entry may ask for a
HARD link when its consumer refuses a symlink.

FAILS CLOSED. An artifact absent from the SOURCE is reported with its consequence and exits non-zero;
it is never skipped quietly. A roster that provisions nothing is an error, not a clean run. It also
reports which `merlin` the worktree's python imports, because the linked venv may name another
checkout's library.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "build_tools" / "scripts" / "worktree_provisioning.yaml"


class ProvisionError(RuntimeError):
    """The roster, the source or the destination is not in a state that can be provisioned."""


def _entries(roster: Path | None = None) -> list[dict]:
    # Resolved at CALL time, not bound into the signature: a default evaluated at import could never be
    # overridden, so every test of this tool would have to read the real roster.
    roster = roster or ROSTER
    doc = yaml.safe_load(roster.read_text(encoding="utf-8")) or {}
    rows = doc.get("artifacts") or []
    if not rows:
        raise ProvisionError(f"{roster} declares no artifacts; a roster that provisions nothing is a defect")
    for row in rows:
        if not str(row.get("consequence") or "").strip():
            raise ProvisionError(f"roster entry {row} states no consequence, so a reader cannot recognise its symptom")
    return rows


def _targets(entry: dict, source: Path) -> list[Path]:
    """The concrete relative paths an entry names, resolved against the SOURCE checkout."""
    if "path" in entry:
        return [Path(entry["path"])]
    pattern = entry.get("path_glob")
    if not pattern:
        raise ProvisionError(f"roster entry names neither `path` nor `path_glob`: {entry}")
    return sorted(p.relative_to(source) for p in source.glob(pattern))


def import_roots(worktree: Path) -> list[Path]:
    """The source roots a process in ``worktree`` must import from: the core and every distribution."""
    roots = [package.parent for package in _source_layout.source_packages(worktree)]
    return list(dict.fromkeys(root for root in roots if root.is_dir()))


def resolved_library(worktree: Path) -> dict[str, str]:
    """Which ``merlin`` this worktree's python actually imports, and the PYTHONPATH that fixes it.

    The linked venv has merlin installed in DEVELOPMENT mode against whichever checkout last installed
    it, so a worktree that links it imports THAT library while running its own scripts. The test suite
    pins its own checkout (``tests/conftest.py``); a script, a ``python -c`` and every child
    process a library spawns do not. Measured: a 30-minute sweep against the wrong library, and a phase-1
    run "pinned" to a commit that imported its library from the source tree for its entire life.
    Provisioning is the moment to say so, because provisioning is what creates the link.
    """
    expected = os.pathsep.join(str(root) for root in import_roots(worktree))
    python = worktree / ".venv/bin/python"
    if not python.exists():
        return {"status": "no venv", "resolves": "", "expected": expected}
    try:
        out = subprocess.run(
            [str(python), "-c", "import merlin; print(merlin.__file__)"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(worktree),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": f"unreadable: {type(exc).__name__}", "resolves": "", "expected": expected}
    lines = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not lines:
        return {"status": f"unreadable: exit {out.returncode}", "resolves": "", "expected": expected}
    resolved = lines[-1]
    own = Path(resolved).resolve().is_relative_to(worktree.resolve())
    return {"status": "own" if own else "SHADOWED", "resolves": resolved, "expected": expected}


def _place(src: Path, dst: Path, entry: dict) -> None:
    """Put ``src`` at ``dst`` by the link kind the entry declares.

    Default is a symlink. ``link: hard`` is for a consumer that refuses a symlink -- the model capsule
    loader rejects one explicitly ("missing or a symlink"), so a symlink there provisions nothing while
    reporting success. A hard link satisfies that check and costs no bytes; it falls back to a copy only
    across filesystems, where a hard link cannot exist.
    """
    if entry.get("link") != "hard":
        dst.symlink_to(src)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def provision(worktree: Path, source: Path, *, verify_only: bool = False) -> dict:
    """Link each rostered artifact into ``worktree``. Returns what was linked, present and missing."""
    worktree, source = Path(worktree).resolve(), Path(source).resolve()
    if worktree == source:
        raise ProvisionError("worktree and source are the same checkout; there is nothing to provision")
    if not (worktree / ".git").exists():
        raise ProvisionError(f"{worktree} is not a git checkout or worktree")

    linked: list[str] = []
    present: list[str] = []
    missing: list[tuple[str, str]] = []
    entries = _entries()

    for entry in entries:
        consequence = " ".join(str(entry["consequence"]).split())
        for rel in _targets(entry, source):
            src, dst = source / rel, worktree / rel
            if not src.exists():
                missing.append((str(rel), consequence))
                continue
            # A TRACKED DIRECTORY HOLDING UNTRACKED FILES IS "PRESENT" AND STILL UNPROVISIONED: the
            # capsule directories come from git while their weights and answer keys are gitignored, so
            # an entry declaring `merge: per_file` is compared FILE BY FILE, narrowed by `files_glob`.
            if entry.get("merge") == "per_file":
                for f_src in sorted(src.glob(entry.get("files_glob") or "**/*")):
                    if not f_src.is_file():
                        continue
                    f_rel = rel / f_src.relative_to(src)
                    f_dst = worktree / f_rel
                    if f_dst.exists():
                        present.append(str(f_rel))
                        continue
                    if verify_only:
                        missing.append((str(f_rel), consequence))
                        continue
                    f_dst.parent.mkdir(parents=True, exist_ok=True)
                    _place(f_src, f_dst, entry)
                    linked.append(str(f_rel))
                continue
            if dst.is_symlink() and not dst.exists():
                missing.append((str(rel), f"{dst} is a link to nothing; {consequence}"))
                continue
            if dst.exists():
                present.append(str(rel))
                continue
            if verify_only:
                missing.append((str(rel), consequence))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            _place(src, dst, entry)
            linked.append(str(rel))

    # A destination that has nothing is a legitimate VERIFY answer -- it is the report. But provisioning
    # that placed nothing and found nothing already there means `--from` names a checkout without these
    # artifacts, and reporting that as a clean run is the exact shape this roster exists to stop.
    if not verify_only and not (linked or present):
        raise ProvisionError("nothing was provisioned and nothing was already present -- check --from")
    if verify_only and not any((source / rel).exists() for entry in entries for rel in _targets(entry, source)):
        raise ProvisionError(f"the source checkout {source} holds none of these artifacts -- check --from")
    return {"linked": linked, "present": present, "missing": missing, "library": resolved_library(worktree)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("worktree", type=Path)
    ap.add_argument("--from", dest="source", type=Path, default=ROOT, help="checkout to link from (default: this one)")
    ap.add_argument("--verify", action="store_true", help="report only; change nothing")
    a = ap.parse_args(argv)

    try:
        result = provision(a.worktree, a.source, verify_only=a.verify)
    except ProvisionError as exc:
        print(f"[FAIL] {exc}")
        return 1

    for rel in result["linked"]:
        print(f"  linked   {rel}")
    for rel in result["present"]:
        print(f"  present  {rel}")
    for rel, why in result["missing"]:
        print(f"  MISSING  {rel}\n           {why}")

    lib = result["library"]
    if lib["status"] == "SHADOWED":
        print(
            "\n[WARN] this worktree's python imports merlin from ANOTHER checkout:\n"
            f"         resolves: {lib['resolves']}\n"
            "       The test suite pins its own checkout; scripts and child processes do not. Every\n"
            f"       other invocation here needs:  PYTHONPATH={lib['expected']}\n"
            "       A result taken without it is a result about a different checkout."
        )
    elif lib["status"] != "own":
        print(f"\n[note] could not determine which merlin this tree imports: {lib['status']}")

    if result["missing"]:
        print(f"\n[FAIL] {len(result['missing'])} artifact(s) unprovisioned. Each degrades SILENTLY -- see the")
        print("       consequence above rather than assuming the tree is usable.")
        return 1
    print(f"\n[  ok] worktree provisioned: {len(result['linked'])} linked, {len(result['present'])} already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
