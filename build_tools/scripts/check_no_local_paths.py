#!/usr/bin/env python3
"""Gate: a tracked file may not name one person's home or scratch directory.

This repo is public. An absolute path like ``/scratch/<someone>/projects/model2MLIR`` is both a
disclosure and a defect: it is a default no other clone can follow, so the feature it configures is
silently unavailable everywhere else. Measured 2026-09-16, before this gate: 35 tracked files outside
``out/`` carried one. Among them a compiler wrapper that ``exec``'d a clang inside ONE worktree of
ONE checkout, two library modules pointing at a directory that no longer exists on this machine
either, and a test whose temp root sat outside any git repository -- which made the gate it was
exercising die with CalledProcessError instead of asserting anything.

The fix is almost always already in the tree: ``merlin.common.paths.env`` / ``ext_path`` read the
gitignored ``.env`` (documented in ``.env.example``), external checkouts resolve as SIBLINGS of the
repo the way ``pyproject.toml`` reaches ``aet``, and scratch belongs under ``$TMPDIR``.

What stays, and why it is a ledger rather than an allowlist: a measurement PROVENANCE record
legitimately says where bytes were read from -- a frozen ``python_executable`` beside its sha256, a
hardware pin naming the build it certified. Those are facts about one run and cannot be
generalised without becoming false. They are listed in ``no_local_paths_ratchet.txt``, which may
only shrink (``check_ratchets_shrink.py`` holds it).

Usage:
  check_no_local_paths.py            # every tracked file
  check_no_local_paths.py --staged   # only what is staged (pre-commit)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RATCHET = ROOT / "build_tools" / "scripts" / "no_local_paths_ratchet.txt"

#: Roots whose FIRST segment is a per-account directory on the machines this repo is developed on.
_USER_ROOTS = ("/home/", "/scratch/", "/scratch2/")

#: Segments that are deliberately not a person: documentation placeholders, the synthetic names
#: tests use when a machine-shaped path is the input under test, and SHARED service directories
#: (``/scratch/firesim_queue`` is a queue every session submits to, not anyone's home).
_GENERIC = frozenset(
    {
        "<user>",
        "$USER",
        "${USER}",
        "path",
        "to",
        "builder",
        "someone",
        "answer",
        "runner",
        "user",
        "...",
        "firesim_queue",
        "chipyard",
        "\\w+",  # a redaction PATTERN in transcript_tooling_audit.py, not a path
    }
)

#: An account name is at least this long. Below it the segment is a test placeholder -- ``/home/x``
#: in a fabricated environment dict, ``/scratch/u/repo/...`` in a leak fixture -- and flagging those
#: would fill the ledger with noise, which is how a ledger stops being read.
_MIN_ACCOUNT_LEN = 3

#: Characters that may precede an ABSOLUTE path. Without this, the relative fixture path
#: ``out/scratch/whatever.json`` matched ``/scratch/`` in the middle of a word.
_PATH_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./")

#: Binary-ish payloads a text scan would only produce noise on.
_SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".o",
    ".a",
    ".so",
    ".bin",
    ".elf",
    ".safetensors",
    ".tar",
    ".gz",
    ".bz2",
    ".zip",
    ".npz",
    ".pt",
    ".pyc",
)


def _tracked(staged: bool) -> list[str]:
    """`check=True`: an empty list because git failed is not an empty list because nothing matched."""
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _ratcheted() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {
        ln.strip() for ln in RATCHET.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")
    }


def _offending_segments(text: str) -> set[str]:
    """Account-shaped segments named in this text. Structural: no regex, no pattern to mis-spell."""
    found: set[str] = set()
    for root in _USER_ROOTS:
        start = 0
        while True:
            at = text.find(root, start)
            if at == -1:
                break
            start = at + len(root)
            if at > 0 and text[at - 1] in _PATH_CHARS:
                continue  # not the start of an absolute path (e.g. out/scratch/x.json)
            rest = text[start:]
            # The segment ends at the next separator of any kind the surrounding syntax may use.
            cut = len(rest)
            for stop in ("/", '"', "'", "`", " ", "\n", "\t", ")", ",", ";", ":", "<", ">"):
                got = rest.find(stop)
                if got != -1:
                    cut = min(cut, got)
            seg = rest[:cut]
            if len(seg) >= _MIN_ACCOUNT_LEN and seg not in _GENERIC:
                found.add(root + seg)
    return found


def check(staged: bool) -> list[str]:
    ratcheted = _ratcheted()
    violations: list[str] = []
    for rel in _tracked(staged):
        if rel in ratcheted or rel.endswith(_SKIP_SUFFIXES):
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        # Compatibility links can expose the same immutable, ratcheted provenance file at
        # its former location. Apply the existing exception to the file's canonical in-repo
        # owner, not to arbitrary links outside the checkout.
        if p.is_symlink():
            resolved = p.resolve(strict=False)
            if resolved.is_relative_to(ROOT) and resolved.relative_to(ROOT).as_posix() in ratcheted:
                continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        named = _offending_segments(text)
        if named:
            violations.append(f"{rel}: names a personal directory: {', '.join(sorted(named))}")
    return violations


def main(argv: list[str]) -> int:
    violations = check("--staged" in argv)
    if violations:
        sys.stderr.write(f"local paths FAILED -- {len(violations)} tracked file(s) name a personal directory:\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        sys.stderr.write(
            "Read it from .env (merlin.common.paths.env / ext_path), resolve it as a "
            "sibling of the repo, or put it under $TMPDIR. A measurement provenance "
            "record that cannot be generalised goes in no_local_paths_ratchet.txt.\n"
        )
        return 1
    print(f"local paths: OK ({len(_ratcheted())} provenance record(s) ratcheted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
