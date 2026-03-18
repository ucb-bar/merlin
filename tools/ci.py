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

_TRACKING_CONFIG_CANDIDATES = (
    utils.REPO_ROOT / ".github" / "upstream_tracking.yaml",
    utils.REPO_ROOT / "config" / "upstream_tracking.yaml",
)
UPSTREAM_TRACKING_CONFIG = next(
    (candidate for candidate in _TRACKING_CONFIG_CANDIDATES if candidate.exists()),
    _TRACKING_CONFIG_CANDIDATES[0],
)
SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def setup_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Lint
    subparsers.add_parser("lint", help="Run linters (shellcheck, python)")
    subparsers.add_parser("cli-docs-drift", help="Regenerate docs/reference/cli.md and fail on drift")

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
    sh_files = find_files_with_suffixes([utils.REPO_ROOT / "scripts", utils.REPO_ROOT / "build_tools"], [".sh"])
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


def run_cli_docs_drift(args) -> int:
    repo_root = utils.REPO_ROOT.resolve()
    docs_root = (repo_root / "docs").resolve()
    hooks_module_path = str(docs_root)
    inserted = False

    if hooks_module_path not in sys.path:
        sys.path.insert(0, hooks_module_path)
        inserted = True
    try:
        import hooks  # type: ignore
    except Exception as e:
        utils.eprint(f"❌ Failed to import docs/hooks.py: {e}")
        return 1
    finally:
        if inserted:
            try:
                sys.path.remove(hooks_module_path)
            except ValueError:
                pass

    try:
        hooks._build_cli_reference(repo_root, docs_root)
    except Exception as e:
        utils.eprint(f"❌ Failed to regenerate docs/reference/cli.md: {e}")
        return 1

    diff_cmd = ["git", "diff", "--exit-code", "--", "docs/reference/cli.md"]
    if utils.run(diff_cmd, dry_run=args.dry_run) != 0:
        utils.eprint("❌ docs/reference/cli.md is out of date. Regenerate and commit the result.")
        return 1

    print("✅ docs/reference/cli.md is up to date.")
    return 0


def run_patch_gate(args) -> int:
    """CI gate: verify the iree_bar submodule is a clean rebase of the pinned upstream."""
    manifest = _load_manifest_env()
    upstream_base = manifest.get("IREE_UPSTREAM_BASE")
    iree_repo = utils.REPO_ROOT / "third_party" / "iree_bar"

    if not iree_repo.is_dir():
        utils.eprint("FAIL: third_party/iree_bar not found. Run: git submodule update --init")
        return 1

    errors = 0

    # 1. Check ancestry: submodule HEAD must descend from pinned upstream base
    if upstream_base:
        rc = subprocess.run(
            ["git", "-C", str(iree_repo), "merge-base", "--is-ancestor", upstream_base, "HEAD"],
            capture_output=True,
        ).returncode
        if rc == 0:
            print(f"OK  submodule is based on upstream {upstream_base[:10]}")
        else:
            utils.eprint(f"FAIL  submodule HEAD is not a descendant of IREE_UPSTREAM_BASE={upstream_base[:10]}")
            utils.eprint("  The ucb-bar/main branch needs to be rebased onto upstream.")
            errors += 1
    else:
        utils.eprint("WARNING: IREE_UPSTREAM_BASE not set in manifest.env")

    # 2. Check for uncommitted changes in the submodule
    status_out = subprocess.run(
        ["git", "-C", str(iree_repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status_out:
        utils.eprint("FAIL  submodule has uncommitted changes:")
        for line in status_out.splitlines()[:10]:
            utils.eprint(f"  {line}")
        errors += 1
    else:
        print("OK  submodule working tree clean")

    # 3. Show Merlin commit count for visibility
    if upstream_base:
        count_out = subprocess.run(
            ["git", "-C", str(iree_repo), "rev-list", "--count", f"{upstream_base}..HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if count_out:
            print(f"OK  {count_out} Merlin commit(s) on top of upstream")

    if errors:
        utils.eprint(f"\npatch-gate: {errors} check(s) failed")
        return 1
    print("\npatch-gate: all checks passed")
    return 0


def _load_manifest_env() -> dict[str, str]:
    """Parse build_tools/patches/manifest.env into a dict."""
    manifest_path = utils.REPO_ROOT / "build_tools" / "patches" / "manifest.env"
    result = {}
    if not manifest_path.exists():
        return result
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"')
    return result


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

    has_valid_latest = bool(latest) and latest != "offline" and not str(latest).startswith("error:")
    update_available = bool(tracked) and has_valid_latest and latest != tracked

    if args.json:
        import json

        print(
            json.dumps(
                {
                    "tracked": tracked,
                    "latest": latest,
                    "tracked_release_tag": tracked,
                    "latest_upstream_release_tag": latest if has_valid_latest else "",
                    "upstream_repo": repo,
                    "update_available": update_available,
                }
            )
        )
    else:
        print(f"Tracked: {tracked}")
        print(f"Latest:  {latest}")
        print(f"Update available: {update_available}")
    return 0


def main(args: argparse.Namespace) -> int:
    if args.subcommand == "lint":
        return run_lint(args)
    if args.subcommand == "cli-docs-drift":
        return run_cli_docs_drift(args)
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
