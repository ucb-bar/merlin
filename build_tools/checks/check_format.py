#!/usr/bin/env python3
"""Gate: Python a commit adds or changes must already be ruff-formatted.

The tree was brought to ruff format in four commits (see .git-blame-ignore-revs), but ~410 files were
deliberately left alone because other work had them checked out. Reformatting those wholesale would
have made every in-flight diff unmergeable; leaving them with no rule would let the formatted part
decay. So the rule is incremental: whatever you touch, you format. The untouched remainder converges as
it is edited, and `ruff format --check .` can go blocking once it reaches zero.

It checks the STAGED bytes, not the working tree -- a partially staged file is judged by what will be
committed -- by materializing them into a temporary mirror beside the repo's pyproject.toml, so the
formatter's exclusions (generated benchmark inputs, marker-bearing files, Markdown) apply exactly as
they do in CI. One ruff invocation per commit, not one per file.

Fail-closed: if the pinned ruff cannot run, the commit is refused with the command to install it.

Usage:
  check_format.py --staged          # pre-commit
  check_format.py --base <rev>      # CI: every .py changed since <rev>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUFF = ["uvx", "ruff@0.16.8"]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _changed(staged: bool, base: str | None) -> list[str]:
    if staged:
        out = _git("diff", "--cached", "--name-status", "-z", "--find-renames", "--diff-filter=ACMR")
    else:
        out = _git("diff", "--name-status", "-z", "--find-renames", "--diff-filter=ACMR", f"{base}...HEAD")
    fields = iter(out.rstrip("\0").split("\0")) if out else iter(())
    changed = []
    for status in fields:
        path = next(fields)
        if status.startswith(("R", "C")):
            path = next(fields)  # the destination owns the staged/HEAD blob
        # Git proves R100 has identical bytes. Preserve untouched formatting debt through a pure
        # relocation, but never let an edited rename avoid the incremental formatter.
        if status != "R100" and path.endswith(".py"):
            changed.append(path)
    metadata = _git("ls-files", "--stage", "-z") if staged else _git("ls-tree", "-r", "-z", "HEAD")
    links = {record.partition("\t")[2] for record in metadata.split("\0") if record.startswith("120000 ")}
    # A Git symlink blob is its target pathname, not Python source. The canonical target is checked
    # independently when it changes; never pass a link's path text to the Python formatter.
    return [path for path in changed if path not in links]


def _content(path: str, staged: bool) -> bytes:
    spec = f":{path}" if staged else f"HEAD:{path}"
    return subprocess.run(["git", "show", spec], cwd=ROOT, capture_output=True, check=True).stdout


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    base = argv[argv.index("--base") + 1] if "--base" in argv else None
    if not staged and base is None:
        print(__doc__)
        return 2
    try:
        files = _changed(staged, base)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"format: could not list changed files ({exc}); refusing rather than passing\n")
        return 1
    if not files:
        print("format: OK (no Python in this change)")
        return 0
    # Judge the bytes being committed, but IN PLACE: import sorting decides first- vs third-party by
    # looking at sibling modules, so a mirror holding only the changed files re-sorted correct imports
    # (measured: a harness importing its own `tracking` package read as unsorted). A file whose staged
    # blob equals its working copy -- the usual case -- is checked on disk in one batch; only a
    # partially staged file is fed through stdin under its real path.
    on_disk, via_stdin = [], []
    for rel in files:
        wt = ROOT / rel
        (on_disk if wt.is_file() and wt.read_bytes() == _content(rel, staged) else via_stdin).append(rel)
    bad_fmt: set[str] = set()
    bad_imp: set[str] = set()
    try:
        if on_disk:
            fmt = subprocess.run(
                [*RUFF, "format", "--check", "--force-exclude", *on_disk], cwd=ROOT, capture_output=True, text=True
            )
            imp = subprocess.run(
                [*RUFF, "check", "--select", "I", "--force-exclude", "--output-format", "concise", *on_disk],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if fmt.returncode not in (0, 1) or imp.returncode not in (0, 1):
                sys.stderr.write(f"format: ruff failed to run:\n{fmt.stderr[-800:]}{imp.stderr[-800:]}")
                return 1
            bad_fmt |= {
                ln.split("-->", 1)[1].strip().rsplit(":", 2)[0] for ln in fmt.stdout.splitlines() if "-->" in ln
            }
            bad_imp |= {
                ln.split(":", 1)[0]
                for ln in imp.stdout.splitlines()
                if ":" in ln and ln.split(":", 1)[0].endswith(".py")
            }
        for rel in via_stdin:
            blob = _content(rel, staged)
            fmt = subprocess.run(
                [*RUFF, "format", "--check", "--force-exclude", "--stdin-filename", rel, "-"],
                cwd=ROOT,
                input=blob,
                capture_output=True,
            )
            imp = subprocess.run(
                [*RUFF, "check", "--select", "I", "--force-exclude", "--stdin-filename", rel, "-"],
                cwd=ROOT,
                input=blob,
                capture_output=True,
            )
            if fmt.returncode == 1:
                bad_fmt.add(rel)
            if imp.returncode == 1:
                bad_imp.add(rel)
            if fmt.returncode not in (0, 1) or imp.returncode not in (0, 1):
                sys.stderr.write(f"format: ruff failed on {rel}\n")
                return 1
    except FileNotFoundError:
        sys.stderr.write(
            "format: `uvx` is not available, so ruff 0.16.8 cannot run; install uv "
            "(https://docs.astral.sh/uv/) -- refusing rather than passing\n"
        )
        return 1
    if bad_fmt or bad_imp:
        sys.stderr.write("format FAILED -- changed Python is not ruff-formatted:\n")
        for f in sorted(set(bad_fmt) | set(bad_imp)):
            sys.stderr.write(f"  - {f}\n")
        sys.stderr.write(
            "Fix with:  uvx ruff@0.16.8 check --select I --fix <files> && "
            "uvx ruff@0.16.8 format <files>\n"
            "then check a `# target-ok:`-style marker still sits on the line it excuses "
            "(pin a single-line statement with `  # fmt: skip`).\n"
        )
        return 1
    print(f"format: OK ({len(files)} changed Python file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
