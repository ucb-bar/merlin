#!/usr/bin/env python3
"""Fast docs anti-drift aggregator — the doc subset of check_structure, for hooks/CI.

Runs the doc generators + linters in --check mode (no writes) and reports any staleness:
  - gen_cli_docs.py       (docs/reference/cli.md vs pyproject scripts)
  - gen_package_docs.py   (docs/reference/module_index.md + package AGENT.md)
  - gen_schema_docs.py    (docs/reference/schemas.md vs merlin/schemas/)
  - gen_docs_index.py     (docs/README.md hub vs front-matter)
  - check_docs_freshness.py (front-matter schema validity)
  - check_doc_paths.py    (no retired paths in docs/ + AGENT.md)
  - root scaffold-phrase guard (README.md / AGENT.md)

Usage:
  check_docs.py             # plain text; exit 1 on any staleness (pre-commit / CI / manual)
  check_docs.py --stop-hook # emit Claude Code Stop-hook JSON instead (exit 0; signal via JSON)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "build_tools" / "scripts"

CHECKS = [
    ("cli docs", "gen_cli_docs.py"),
    ("package docs", "gen_package_docs.py"),
    ("schema docs", "gen_schema_docs.py"),
    ("docs index", "gen_docs_index.py"),
    ("docs freshness", "check_docs_freshness.py"),
    ("doc paths", "check_doc_paths.py"),
]
ROOT_STALE_PHRASES = ("placeholder modules", "not working compiler",
                      "do not implement major algorithms", "currently a scaffold",
                      "status: **scaffold**")


def collect() -> list[str]:
    problems: list[str] = []
    for label, script in CHECKS:
        r = subprocess.run([sys.executable, str(SCRIPTS / script), "--check"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or "").strip().splitlines()
            first = next((ln.strip() for ln in detail if ln.strip()), f"{label}: stale")
            problems.append(f"{label}: {first}")
    for name in ("README.md", "AGENT.md"):
        p = ROOT / name
        if p.is_file():
            low = p.read_text(encoding="utf-8").lower()
            for ph in ROOT_STALE_PHRASES:
                if ph in low:
                    problems.append(f"root docs: {name} contains stale scaffold-era phrase {ph!r}")
    return problems


def main(argv: list[str]) -> int:
    problems = collect()
    if "--stop-hook" in argv:
        if problems:
            print(json.dumps({"decision": "block",
                              "reason": "Docs are stale — regenerate/fix:\n- " + "\n- ".join(problems)}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code
    if problems:
        sys.stderr.write("docs check FAILED:\n")
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
        return 1
    print("docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
