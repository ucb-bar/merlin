#!/usr/bin/env python3
# tools/ci.py

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence

import utils

UPSTREAM_TRACKING_CONFIG = utils.REPO_ROOT / "config" / "upstream_tracking.yaml"
SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def setup_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Lint
    subparsers.add_parser("lint", help="Run linters (shellcheck, python)")

    # Patch Gate
    subparsers.add_parser("patch-gate", help="CI gate: apply, verify, drift check")

    # Release Status
    release = subparsers.add_parser("release-status", help="Check upstream IREE versions")
    release.add_argument("--tracking-file", default=str(UPSTREAM_TRACKING_CONFIG))
    release.add_argument("--offline", action="store_true")
    release.add_argument("--json", action="store_true")


# --- Helpers ---


def find_files_with_suffixes(roots: Sequence[pathlib.Path], suffixes: Sequence[str]) -> list[pathlib.Path]:
    found = []
    suffix_set = set(suffixes)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in suffix_set:
                found.append(path)
    return sorted(found)


def parse_semver_tag(tag: str):
    m = SEMVER_TAG_RE.match(tag)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def fetch_latest_release_tag(repo_slug: str) -> str:
    remote_url = f"https://github.com/{repo_slug}.git"
    res = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", remote_url], capture_output=True, text=True, timeout=30
    )
    if res.returncode != 0:
        raise RuntimeError(f"git ls-remote failed: {res.stderr}")

    latest_tag, latest_semver = None, None
    for line in res.stdout.splitlines():
        ref = line.split()[1]
        tag = ref.replace("refs/tags/", "")
        semver = parse_semver_tag(tag)
        if semver and (latest_semver is None or semver > latest_semver):
            latest_semver = semver
            latest_tag = tag
    return latest_tag


# --- Commands ---


def run_lint(args) -> int:
    sh_files = find_files_with_suffixes([utils.REPO_ROOT / "scripts", utils.REPO_ROOT / "patches"], [".sh"])
    py_files = find_files_with_suffixes([utils.REPO_ROOT / "tools"], [".py"])

    # Python Syntax Check
    if py_files:
        if utils.run(["python3", "-m", "py_compile", *[str(p) for p in py_files]], dry_run=args.dry_run) != 0:
            return 1

    # Shellcheck
    if shutil.which("shellcheck"):
        if utils.run(["shellcheck", *[str(p) for p in sh_files]], dry_run=args.dry_run) != 0:
            return 1
    else:
        print("Warning: shellcheck not found")

    return 0


def run_patch_gate(args) -> int:
    steps = [
        "patches/tools/apply_all.sh",
        "patches/tools/verify_clean.sh",
        "patches/tools/check_upstream_drift.sh",
    ]
    for script in steps:
        if utils.run_repo_script(script, [], args.dry_run) != 0:
            return 1
    return 0


def run_release_status(args) -> int:
    # Simplified YAML loader for the config
    def load_simple_yaml(path):
        data = {}
        with open(path) as f:
            for line in f:
                if ":" in line and not line.strip().startswith("#"):
                    k, v = line.split(":", 1)
                    data[k.strip()] = v.strip().replace('"', "")
        return data

    try:
        config = load_simple_yaml(args.tracking_file)
    except Exception as e:
        utils.eprint(f"Failed to load config: {e}")
        return 1

    tracked = config.get("tracked_release_tag")
    repo = config.get("upstream_repo", "iree-org/iree")

    latest = "offline"
    if not args.offline:
        try:
            latest = fetch_latest_release_tag(repo)
        except Exception as e:
            latest = f"error: {e}"

    if args.json:
        import json

        print(json.dumps({"tracked": tracked, "latest": latest}))
    else:
        print(f"Tracked: {tracked}")
        print(f"Latest:  {latest}")
    return 0


def main(args: argparse.Namespace) -> int:
    if args.subcommand == "lint":
        return run_lint(args)
    if args.subcommand == "patch-gate":
        return run_patch_gate(args)
    if args.subcommand == "release-status":
        return run_release_status(args)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin CI Tools")
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
