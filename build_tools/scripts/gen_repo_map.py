#!/usr/bin/env python3
"""Living repo map: derive docs/reference/repo_map.md from what is actually tracked.

The question "what is in this repo, directory by directory" had no answer that could be trusted.
`reference/repo_structure.md` describes the intended shape and `module_index.md` lists importable
packages, but neither says how much is where, and both are hand-maintained in the parts that matter
most -- so a directory could appear, grow to a thousand files, and never show up in either.

This generator reads the tracked tree instead: `git ls-files` for the counts, each directory's own
`AGENT.md` "## Purpose" for what it is for, and the experiments' declared `Status:`. Nothing is
written down twice, so nothing can drift.

Usage:
  gen_repo_map.py            # (re)generate docs/reference/repo_map.md
  gen_repo_map.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "docs" / "reference" / "repo_map.md"

#: Trees listed one level deeper, because their children are the unit a reader looks for: one
#: importable package, one experiment, one target.
DEEP = (
    "src",
    "src/merlin",
    "packages",
    "merlin/python",
    "merlin/python/merlin",
    "merlin/experiments",
    "merlin/targets",
    "merlin/tests",
)

#: Vendored upstream; its size says nothing about this repo and its AGENT.md files are ours.
SKIP_TOP = ("third_party",)


def _approx(n: int) -> str:
    """A count coarse enough to commit.

    Exact counts made this doc stale constantly, for two reasons. Every commit that adds a file
    changes one -- and check_docs.py runs in pre-commit, so a map nobody was editing would start
    failing other sessions' commits several times an hour. Worse, `git commit --only` runs the hook
    against a TEMPORARY INDEX holding just the committed paths, so the same tree counted differently
    inside the hook than outside it whenever another session had something staged.

    Two significant figures, floor of ten, answers what the map is for -- how much is where -- and
    a single file cannot move it.
    """
    if n < 10:
        return str(n)
    step = max(10, 10 ** (len(str(n)) - 2))
    rounded = round(n / step) * step
    return f"~{rounded / 1000:g}k" if rounded >= 1000 else f"~{rounded}"


def _tracked() -> list[str]:
    """The files in HEAD -- deliberately not `git ls-files`, which reads the INDEX.

    `git commit --only <paths>` runs pre-commit hooks against a temporary index holding only the
    paths being committed, so an index-based count answered differently inside the hook than outside
    it whenever another session had something staged: three staged files under merlin/perf/ were
    enough to move a bucket and fail the docs gate on a commit that had not touched that tree. HEAD
    is the same revision in both places, so the map is a function of the last commit and nothing else.

    `check=True`: an empty file list because git failed reads exactly like an empty repo.
    """
    out = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def _purpose(rel: str) -> str:
    """The directory's own one-line answer, from its AGENT.md ``## Purpose`` section."""
    agent = ROOT / rel / "AGENT.md"
    if not agent.is_file():
        return ""
    lines = agent.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("## purpose"):
            body: list[str] = []
            for nxt in lines[i + 1 :]:
                if nxt.startswith("#"):
                    break
                if nxt.strip():
                    body.append(nxt.strip())
                elif body:
                    break
            text = " ".join(body)
            # Deliberately NOT split into sentences: the first attempt cut `scripts/` at
            # "Maintenance/validation scripts (e.g." because "e.g." ends in a period. A length cap
            # at a word boundary keeps the table readable without inventing grammar rules.
            if len(text) > 200:
                cut = text.rfind(" ", 0, 200)
                text = text[: cut if cut > 0 else 200].rstrip(" ,;:") + "…"
            return text
    return ""


def _status(rel: str) -> str:
    """An experiment declares `Status: active|frozen|reference` in its AGENT.md (merlin/experiments)."""
    agent = ROOT / rel / "AGENT.md"
    if not agent.is_file():
        return ""
    for line in agent.read_text(encoding="utf-8").splitlines()[:25]:
        low = line.strip().lower()
        if low.startswith("status:") or low.startswith("**status:**"):
            return line.split(":", 1)[1].strip().strip("*").split()[0] if ":" in line else ""
    return ""


def _rows(files: list[str]) -> list[tuple[str, int, str, str]]:
    """(relative dir, tracked file count, purpose, status), depth-ordered."""
    counts: Counter[str] = Counter()
    for rel in files:
        parts = rel.split("/")
        for depth in range(1, len(parts)):
            counts["/".join(parts[:depth])] += 1

    wanted: list[str] = []
    for d in sorted(counts):
        parts = d.split("/")
        if parts[0] in SKIP_TOP:
            continue
        if len(parts) == 1:
            wanted.append(d)
        elif len(parts) == 2 and parts[0] in ("merlin", "src", "packages", "build_tools", "docs", ".claude", ".github"):
            wanted.append(d)
        elif d.rsplit("/", 1)[0] in DEEP:
            wanted.append(d)
        elif parts[0] == "packages" and len(parts) in (3, 4) and parts[2] == "src":
            wanted.append(d)
    return [(d, counts[d], _purpose(d), _status(d)) for d in wanted]


def render() -> str:
    files = _tracked()
    rows = _rows(files)
    root_files = sorted(f for f in files if "/" not in f)

    out = [
        "# Repository map (AUTO-GENERATED)",
        "",
        "Generated by `build_tools/scripts/gen_repo_map.py` from the tracked tree and each directory's",
        "own `AGENT.md`. **Do not edit by hand** — rerun the generator. It is a SNAPSHOT, deliberately not",
        "gated: a map of the tree goes stale the moment anyone adds a file, and failing the next unrelated",
        "commit for that cost other sessions a regenerate commit each.",
        "",
        "`files` counts TRACKED files at or below that directory, so a parent's count includes its",
        "children. Generated output under `out/` is mostly gitignored; what is tracked there is the",
        "skeleton plus the curated reports the layout convention keeps. `third_party/` is vendored",
        "upstream and is not counted.",
        "",
        f"Totals: **{_approx(len(files))} tracked files**, {len(root_files)} of them at the "
        "repository root. Counts of ten or more are rounded to two significant figures, so "
        "a single added file cannot make this map stale.",
        "",
        "## Directories",
        "",
        "| directory | files | status | purpose |",
        "|---|---:|---|---|",
    ]
    for d, n, purpose, status in rows:
        depth = d.count("/")
        name = ("&nbsp;&nbsp;" * depth) + "`" + d.split("/")[-1] + ("/`" if depth else "/`")
        out.append(f"| {name} | {_approx(n)} | {status or ''} | {purpose.replace(chr(124), chr(92) + chr(124))} |")

    out += ["", "## Root files", "", " ".join(f"`{f}`" for f in root_files), ""]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    body = render()
    if "--check" in argv:
        if not MAP.is_file() or MAP.read_text(encoding="utf-8") != body:
            sys.stderr.write(f"{MAP.relative_to(ROOT)} is stale — run: python build_tools/scripts/gen_repo_map.py\n")
            return 1
        print(f"{MAP.relative_to(ROOT)}: up to date")
        return 0
    MAP.parent.mkdir(parents=True, exist_ok=True)
    MAP.write_text(body, encoding="utf-8")
    print(f"wrote {MAP.relative_to(ROOT)} ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
