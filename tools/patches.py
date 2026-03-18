#!/usr/bin/env python3
# tools/patches.py
#
# Merlin patch management tool.
#
# Merlin maintains downstream IREE changes as commits on ucb-bar/main.
# This tool verifies the submodule state and assists with upstream bumps.

import argparse
import subprocess
import sys

import utils

MANIFEST_ENV = utils.REPO_ROOT / "build_tools" / "patches" / "manifest.env"
IREE_REPO = utils.REPO_ROOT / "third_party" / "iree_bar"


def _load_manifest() -> dict[str, str]:
    """Parse manifest.env into a dict."""
    result = {}
    if not MANIFEST_ENV.exists():
        return result
    for line in MANIFEST_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"')
    return result


def _git(*args: str, cwd: str | None = None) -> tuple[int, str]:
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd or str(IREE_REPO),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip()


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify the submodule is a clean rebase of the pinned upstream base."""
    manifest = _load_manifest()
    upstream_base = manifest.get("IREE_UPSTREAM_BASE")

    if not IREE_REPO.is_dir():
        utils.eprint(f"Submodule not found: {IREE_REPO}")
        utils.eprint("  Run: git submodule update --init third_party/iree_bar")
        return 1

    errors = 0

    # Check submodule HEAD
    rc, head_sha = _git("rev-parse", "HEAD")
    rc2, head_short = _git("rev-parse", "--short", "HEAD")
    print(f"Submodule HEAD: {head_short}")

    # Check ancestry
    if upstream_base:
        rc, _ = _git("merge-base", "--is-ancestor", upstream_base, "HEAD")
        if rc == 0:
            print(f"  OK  based on upstream {upstream_base[:10]}")
        else:
            utils.eprint(f"  FAIL  HEAD is not a descendant of upstream base {upstream_base[:10]}")
            utils.eprint("    This means ucb-bar/main needs to be rebased onto upstream.")
            errors += 1

    # Count commits on top of upstream base
    if upstream_base:
        rc, count = _git("rev-list", "--count", f"{upstream_base}..HEAD")
        if rc == 0:
            print(f"  OK  {count} Merlin commit(s) on top of upstream")

    # Check for uncommitted changes in the submodule
    rc, status = _git("status", "--porcelain")
    if status:
        utils.eprint("  WARNING  submodule has uncommitted changes:")
        for line in status.splitlines()[:10]:
            utils.eprint(f"    {line}")
        errors += 1
    else:
        print("  OK  working tree clean")

    if errors:
        utils.eprint(f"\n{errors} issue(s) found.")
        return 1
    print("\nSubmodule verification passed.")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    """Show the Merlin-specific commits on top of upstream."""
    manifest = _load_manifest()
    upstream_base = manifest.get("IREE_UPSTREAM_BASE")

    if not upstream_base:
        utils.eprint("No IREE_UPSTREAM_BASE in manifest.env")
        return 1

    rc, log = _git(
        "log",
        "--oneline",
        "--reverse",
        f"{upstream_base}..HEAD",
    )
    if rc != 0:
        utils.eprint("Failed to read git log from submodule")
        return 1

    if not log:
        print("No Merlin commits on top of upstream base.")
        return 0

    print(f"Merlin commits on ucb-bar/main (above {upstream_base[:10]}):\n")
    for line in log.splitlines():
        print(f"  {line}")
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Check how far behind upstream the current base is."""
    manifest = _load_manifest()
    upstream_base = manifest.get("IREE_UPSTREAM_BASE")
    upstream_repo = manifest.get("IREE_UPSTREAM_REPO")

    if not upstream_base:
        utils.eprint("No IREE_UPSTREAM_BASE in manifest.env")
        return 1

    print("Fetching upstream to check drift...")
    rc, _ = _git("fetch", upstream_repo or "https://github.com/iree-org/iree", "main", "--depth=100")
    if rc != 0:
        utils.eprint("Failed to fetch upstream. Check network connectivity.")
        return 1

    rc, count = _git("rev-list", "--count", f"{upstream_base}..FETCH_HEAD")
    if rc == 0 and count:
        n = int(count)
        if n == 0:
            print("  Up to date with upstream main.")
        else:
            print(f"  {n} upstream commit(s) ahead of our base ({upstream_base[:10]}).")
            print("  Consider rebasing: cd third_party/iree_bar && git rebase FETCH_HEAD")
    return 0


def cmd_export_upstream(args: argparse.Namespace) -> int:
    """Export a specific commit as a format-patch for upstream PR preparation."""
    manifest = _load_manifest()
    upstream_base = manifest.get("IREE_UPSTREAM_BASE")

    if not upstream_base:
        utils.eprint("No IREE_UPSTREAM_BASE in manifest.env")
        return 1

    out_dir = utils.REPO_ROOT / "build_tools" / "patches" / "upstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = args.commit
    rc, _ = _git("format-patch", "-1", commit, "-o", str(out_dir))
    if rc != 0:
        utils.eprint(f"Failed to export commit {commit}")
        return 1

    print(f"Exported to {out_dir}/")
    return 0


def setup_parser(parser: argparse.ArgumentParser):
    sub = parser.add_subparsers(dest="patches_action", required=True)

    sub.add_parser("verify", help="Verify submodule is a clean rebase of pinned upstream")
    sub.add_parser("log", help="Show Merlin commits on top of upstream base")
    sub.add_parser("drift", help="Check how far behind upstream the base is")

    sp = sub.add_parser("export-upstream", help="Export a commit as format-patch for upstream PR")
    sp.add_argument("commit", help="Commit hash to export")


def main(args: argparse.Namespace) -> int:
    dispatch = {
        "verify": cmd_verify,
        "log": cmd_log,
        "drift": cmd_drift,
        "export-upstream": cmd_export_upstream,
    }
    handler = dispatch.get(args.patches_action)
    if not handler:
        utils.eprint(f"Unknown patches action: {args.patches_action}")
        return 2
    return handler(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Patch Manager")
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
