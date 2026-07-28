#!/usr/bin/env python3
"""Linter: enforce the single-root artifact convention (out/{runs,artifacts,build}).

Fails (exit 1) if any of:
  * a TRACKED file lives under a retired top-level generated root (runs/, artifacts/, build/ —
    now consolidated under out/) or an old/forbidden root (output/, results/, selfcheck_out/,
    mined_knowledge/, docs/presentation/) or a GENERATED output dir inside merlin/
    (experiments/*/runs/, experiments/*/reports/, benchmarks/*/case_study/) — those belong
    under out/artifacts/ (see the docs/experiment migrations);
  * a TRACKED file has a generated extension (.png/.svg/.pdf/.zip/.jsonl) outside out/artifacts/
    (and not allowlisted);
  * a versioned product dir out/artifacts/<topic>/v*/<leaf>/ is missing manifest.yaml;
  * a `latest` symlink under out/artifacts/ is absolute or dangling.

Usage:
  check_artifact_layout.py             # full working tree (tracked files)
  check_artifact_layout.py --staged    # only staged files (pre-commit)
  check_artifact_layout.py --stop-hook  # emit Claude Code Stop-hook JSON instead of plain text
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

GENERATED_EXTS = {".png", ".svg", ".pdf", ".zip", ".jsonl"}
# Retired top-level generated roots (consolidated under out/) + legacy forbidden locations.
FORBIDDEN_ROOTS = ("runs/", "artifacts/", "build/", "output/", "results/",
                   "selfcheck_out/", "mined_knowledge/", "docs/presentation/")
# Generated OUTPUT dirs that must NOT live inside the source tree (runs/ reports/ = experiment
# output; case_study/ = dse-guidance generated analysis). They belong under artifacts/ or runs/.
# (Curated INPUT corpora — benchmarks/*/recaptures*, region_maps, methods/, observability/,
# input_bundles/ — are not matched and legitimately stay in-tree.)
def _is_forbidden_gen_dir(rel: str) -> bool:
    """True if ``rel`` contains a generated-output dir that must not live in-tree: an
    ``experiments/<name>/{runs,reports}/`` or ``benchmarks/<name>/case_study/`` component
    (structural check on path parts — the ``<name>`` is exactly one segment)."""
    parts = PurePosixPath(rel).parts
    for i in range(len(parts) - 2):
        top, leaf = parts[i], parts[i + 2]
        if top == "experiments" and leaf in ("runs", "reports"):
            return True
        if top == "benchmarks" and leaf == "case_study":
            return True
    return False
# genuinely-tracked source images, if any (verify with `git ls-files '*.png'` before adding).
ALLOW_TRACKED_GEN: set[str] = {"docs/assets/merlin_transparent.png"}  # project logo (branding, not generated)
# No blanket exemptions: every tracked path (incl. targetgen_evals) obeys the three-root rule.
# targetgen_evals runs/reports now live under the single out/ root (out/runs/targetgen-evals/ and
# out/artifacts/targetgen-evals/, both gitignored).
SKIP_PREFIXES: tuple[str, ...] = ()


# --- stale generated-root path LITERALS in code -----------------------------------------------
# The out/ consolidation moved generated output under out/{runs,artifacts,build}. Writers were
# updated via merlin.common.paths, but relative-path *reader* literals ("artifacts/…", "runs/…")
# resolve against repo_root() -> the RETIRED top-level root, which no longer exists. This lints for
# such literals in code so the class can't silently reappear. Only quoted literals with a leading
# quote are matched (so subscripts like x["artifacts"] are not) and only the retired prefixes.
_STALE_LITERALS = ('"artifacts/', "'artifacts/", '"runs/', "'runs/",
                   '"build/generated', "'build/generated")
# Lines that legitimately name a retired-root-shaped literal (NOT a repo-root-relative read):
#   - run-dir/CWD-relative uses inside experiment sandboxes (shell `runs/${…}`, `.glob("runs/*")`);
#   - the artifact-layout deny-test, which asserts the retired roots are rejected.
# Keyed by "<relpath>:<substring that must be on the flagged line>".
_STALE_LITERAL_ALLOW = {
    "merlin/tests/infra/test_artifact_layout.py:artifacts/plots/foo.png",   # deny-test fixture
    "merlin/experiments/capsule_bench/harness/gen_fullsuite_report.py:.glob(\"runs/",
    "merlin/experiments/capsule_bench/harness/abc_watchdog.sh:runs/${",
}


def _stale_path_literals(root: Path, tracked: list[str]) -> list[str]:
    out: list[str] = []
    for rel in tracked:
        if not (rel.endswith(".py") or rel.endswith(".sh")):
            continue
        if rel.endswith(("check_artifact_layout.py", "check_doc_paths.py")) or rel.endswith("common/paths.py"):
            continue
        p = root / rel
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for lit in _STALE_LITERALS:
                if lit in line and ("out/" + lit[1:]) not in line:
                    if any(a.startswith(rel + ":") and a.split(":", 1)[1] in line
                           for a in _STALE_LITERAL_ALLOW):
                        continue
                    out.append(f"stale generated-root literal {lit[1:]!r} (use out/ prefix): {rel}:{i}")
                    break
    return out


def _repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True).stdout.strip()
    return Path(out) if out else Path.cwd()


def _tracked(root: Path, staged: bool) -> list[str]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def check(root: Path, staged: bool) -> list[str]:
    violations: list[str] = []
    tracked = _tracked(root, staged)
    violations.extend(_stale_path_literals(root, tracked))
    for rel in tracked:
        if any(rel.startswith(s) for s in SKIP_PREFIXES):
            continue
        p = Path(rel)
        if any(rel.startswith(r) for r in FORBIDDEN_ROOTS) or _is_forbidden_gen_dir(rel):
            violations.append(f"tracked file under a forbidden root: {rel}")
            continue
        if (p.suffix.lower() in GENERATED_EXTS and not rel.startswith("out/artifacts/")
                and rel not in ALLOW_TRACKED_GEN):
            violations.append(f"generated file '{rel}' tracked outside out/artifacts/")
    art = root / "out" / "artifacts"
    if art.is_dir():
        for v in art.glob("*/v*"):
            if not v.is_dir():
                continue
            latest = v / "latest"
            if latest.is_symlink():
                tgt = os.readlink(latest)
                if os.path.isabs(tgt):
                    violations.append(f"absolute `latest` symlink (bwrap-unsafe): {latest.relative_to(root)}")
                elif not (v / tgt).exists():
                    violations.append(f"dangling `latest` symlink: {latest.relative_to(root)}")
            for leaf in v.iterdir():
                if leaf.is_dir() and not leaf.name.startswith(".") and leaf.name != "latest":
                    if not (leaf / "manifest.yaml").exists():
                        violations.append(f"product dir missing manifest.yaml: {leaf.relative_to(root)}")
    return violations


def main(argv: list[str]) -> int:
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    root = _repo_root()
    violations = check(root, staged)
    if stop_hook:
        if violations:
            print(json.dumps({"decision": "block",
                              "reason": "Artifact-layout violations:\n- " + "\n- ".join(violations)}))
        else:
            print(json.dumps({}))
        return 0  # stop-hook signals via JSON, not exit code
    if violations:
        sys.stderr.write("Artifact-layout check FAILED:\n")
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        sys.stderr.write("\nSee CLAUDE.md 'Generated-output convention' / .claude/skills/artifact-layout.\n")
        return 1
    print("artifact-layout: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
