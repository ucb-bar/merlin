#!/usr/bin/env python3
"""Merlin repository entrypoint for common maintenance workflows."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS_CONFIG = REPO_ROOT / "config" / "targets.json"
UPSTREAM_TRACKING_CONFIG = REPO_ROOT / "config" / "upstream_tracking.yaml"
SEMVER_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[pathlib.Path] = None,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    cmd_str = " ".join(_shell_quote(x) for x in cmd)
    print(f"+ {cmd_str}")
    if dry_run:
        return 0
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        list(cmd), cwd=str(cwd or REPO_ROOT), env=merged_env, check=False
    )
    return completed.returncode


def _shell_quote(text: str) -> str:
    if text == "":
        return "''"
    if all(ch.isalnum() or ch in "._-/:=+" for ch in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def load_targets_config() -> dict:
    with TARGETS_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_inline_comment(raw_line: str) -> str:
    # Strips shell-style inline comments while preserving URLs and tokens.
    # We only treat " <whitespace>#..." as comments.
    parts = re.split(r"\s+#", raw_line, maxsplit=1)
    return parts[0].rstrip()


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value.startswith('"') and value.endswith('"') and len(value) >= 2:
        return value[1:-1]
    if value.startswith("'") and value.endswith("'") and len(value) >= 2:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return value


def load_simple_yaml(path: pathlib.Path) -> dict:
    """Loads a simple indentation-based YAML map.

    This parser intentionally supports the subset used by config files in this
    repository (nested dictionaries with scalar values).
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    root: dict = {}
    # Maps indentation level to dict node.
    nodes_by_indent: Dict[int, dict] = {-2: root}

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped_line = raw.rstrip("\n")
            if not stripped_line.strip():
                continue
            if stripped_line.lstrip().startswith("#"):
                continue

            content = _strip_inline_comment(stripped_line)
            if not content.strip():
                continue

            indent = len(content) - len(content.lstrip(" "))
            if indent % 2 != 0:
                raise ValueError(
                    f"{path}:{line_no}: indentation must be multiples of 2 spaces"
                )

            token = content.lstrip(" ")
            if ":" not in token:
                raise ValueError(f"{path}:{line_no}: invalid mapping entry: {token!r}")

            key, value = token.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"{path}:{line_no}: empty key")

            parent = nodes_by_indent.get(indent - 2)
            if parent is None:
                raise ValueError(
                    f"{path}:{line_no}: cannot resolve parent mapping for indent {indent}"
                )

            if value == "":
                child: dict = {}
                parent[key] = child
                nodes_by_indent[indent] = child
            else:
                parent[key] = _parse_scalar(value)

    return root


def parse_semver_tag(tag: str) -> Optional[tuple]:
    m = SEMVER_TAG_RE.match(tag)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def fetch_latest_release_tag(repo_slug: str, timeout_sec: int = 60) -> str:
    remote_url = f"https://github.com/{repo_slug}.git"
    completed = subprocess.run(
        ["git", "ls-remote", "--tags", "--refs", remote_url],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git ls-remote failed for {remote_url}: {completed.stderr.strip()}"
        )

    latest_tag = None
    latest_semver = None
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        ref = fields[1]
        if not ref.startswith("refs/tags/"):
            continue
        tag = ref[len("refs/tags/") :]
        semver = parse_semver_tag(tag)
        if semver is None:
            continue
        if latest_semver is None or semver > latest_semver:
            latest_semver = semver
            latest_tag = tag

    if latest_tag is None:
        raise RuntimeError(
            f"No stable semver tag (vX.Y.Z) found in {remote_url} tag list."
        )
    return latest_tag


def resolve_repo_path(relative: str) -> pathlib.Path:
    return (REPO_ROOT / relative).resolve()


def run_repo_script(
    relative_script: str, script_args: Sequence[str], dry_run: bool
) -> int:
    script = resolve_repo_path(relative_script)
    if not script.exists():
        eprint(f"Script not found: {script}")
        return 2
    return run(["bash", str(script), *script_args], dry_run=dry_run)


def cmd_targets(args: argparse.Namespace) -> int:
    config = load_targets_config()
    if args.subcommand == "list":
        print("Build Profiles:")
        for name, entry in sorted(config.get("build_profiles", {}).items()):
            print(f"  {name:24} {entry.get('description', '')}")
        print("\nBenchmark Targets:")
        for name, entry in sorted(config.get("benchmark_targets", {}).items()):
            print(f"  {name:24} {entry.get('description', '')}")
        return 0

    if args.subcommand == "show":
        category = args.category
        name = args.name
        section = config.get(category, {})
        if name not in section:
            eprint(f"{name!r} not found in {category!r}")
            return 2
        print(json.dumps(section[name], indent=2, sort_keys=True))
        return 0

    eprint(f"Unsupported targets subcommand: {args.subcommand}")
    return 2


def cmd_patches(args: argparse.Namespace) -> int:
    script_by_action = {
        "apply": "scripts/patches/apply_all.sh",
        "verify": "scripts/patches/verify_clean.sh",
        "refresh": "scripts/patches/refresh_all.sh",
        "drift": "scripts/drift/check_upstream_drift.sh",
    }
    script = script_by_action.get(args.action)
    if not script:
        eprint(f"Unsupported patches action: {args.action}")
        return 2
    return run_repo_script(script, [], args.dry_run)


def cmd_build(args: argparse.Namespace) -> int:
    config = load_targets_config()
    profiles = config.get("build_profiles", {})
    profile = profiles.get(args.profile)
    if not profile:
        eprint(f"Unknown build profile: {args.profile}")
        return 2
    script = profile.get("script")
    if not script:
        eprint(f"Missing script for profile: {args.profile}")
        return 2
    return run_repo_script(script, args.extra_args, args.dry_run)


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = load_targets_config()
    targets = config.get("benchmark_targets", {})
    target = targets.get(args.target)
    if not target:
        eprint(f"Unknown benchmark target: {args.target}")
        return 2

    if args.action == "compile-dual-vmfb":
        script = target.get("compile_vmfb_script")
        if not script:
            eprint(f"Target {args.target} has no compile_vmfb_script")
            return 2
        return run_repo_script(script, args.extra_args, args.dry_run)

    if args.action == "run-dual-remote":
        script = target.get("run_remote_script")
        if not script:
            eprint(f"Target {args.target} has no run_remote_script")
            return 2
        return run_repo_script(script, args.extra_args, args.dry_run)

    eprint(f"Unsupported benchmark action: {args.action}")
    return 2


def find_files_with_suffixes(
    roots: Sequence[pathlib.Path], suffixes: Sequence[str]
) -> List[pathlib.Path]:
    found: List[pathlib.Path] = []
    suffix_set = set(suffixes)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in suffix_set:
                found.append(path)
    return sorted(found)


def cmd_ci(args: argparse.Namespace) -> int:
    if args.action == "lint":
        return ci_lint(args.dry_run)
    if args.action == "patch-gate":
        steps = [
            ("scripts/patches/apply_all.sh", []),
            ("scripts/patches/verify_clean.sh", []),
            ("scripts/drift/check_upstream_drift.sh", []),
        ]
        for script, script_args in steps:
            rc = run_repo_script(script, script_args, args.dry_run)
            if rc != 0:
                return rc
        return 0
    eprint(f"Unsupported ci action: {args.action}")
    return 2


def cmd_release_status(args: argparse.Namespace) -> int:
    tracking_path = resolve_repo_path(args.tracking_file)
    try:
        tracking = load_simple_yaml(tracking_path)
    except Exception as exc:  # pylint: disable=broad-except
        eprint(f"Failed to load tracking file {tracking_path}: {exc}")
        return 2

    iree = tracking.get("iree", {})
    llvm = tracking.get("llvm", {})
    tracked_tag = str(iree.get("tracked_release_tag", "")).strip()
    upstream_repo = str(iree.get("upstream_repo", "iree-org/iree")).strip()

    if not tracked_tag:
        eprint(
            f"Tracking file {tracking_path} missing iree.tracked_release_tag."
        )
        return 2
    if parse_semver_tag(tracked_tag) is None:
        eprint(
            f"Tracked tag {tracked_tag!r} is not in stable semver format vX.Y.Z."
        )
        return 2

    latest_tag: Optional[str] = None
    update_available: Optional[bool] = None
    query_error: Optional[str] = None

    if not args.offline:
        try:
            latest_tag = fetch_latest_release_tag(upstream_repo)
            tracked_semver = parse_semver_tag(tracked_tag)
            latest_semver = parse_semver_tag(latest_tag)
            # Safe because fetch_latest_release_tag only returns semver.
            update_available = bool(
                tracked_semver and latest_semver and latest_semver > tracked_semver
            )
        except Exception as exc:  # pylint: disable=broad-except
            query_error = str(exc)
            if not args.allow_network_failure:
                eprint(f"Failed to query upstream releases: {query_error}")
                return 2

    result = {
        "tracking_file": str(tracking_path),
        "iree_upstream_repo": upstream_repo,
        "tracked_release_tag": tracked_tag,
        "latest_upstream_release_tag": latest_tag,
        "update_available": update_available,
        "query_error": query_error,
        "tracked_iree_branch": iree.get("tracked_branch"),
        "tracked_iree_commit": iree.get("tracked_commit"),
        "tracked_llvm_branch": llvm.get("tracked_branch"),
        "tracked_llvm_commit": llvm.get("tracked_commit"),
        "llvm_upstream_repo": llvm.get("llvm_upstream_repo"),
        "iree_llvm_upstream_repo": llvm.get("iree_llvm_upstream_repo"),
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Tracking file: {tracking_path}")
        print(f"IREE upstream: {upstream_repo}")
        print(f"Tracked IREE release: {tracked_tag}")
        if latest_tag:
            print(f"Latest upstream release: {latest_tag}")
            print(f"Update available: {'yes' if update_available else 'no'}")
        elif args.offline:
            print("Latest upstream release: (offline mode)")
        else:
            print(f"Latest upstream release: (query failed: {query_error})")
        print(
            "Tracked branches: "
            f"iree={iree.get('tracked_branch')} "
            f"llvm={llvm.get('tracked_branch')}"
        )
        print(
            "Tracked commits: "
            f"iree={iree.get('tracked_commit')} "
            f"llvm={llvm.get('tracked_commit')}"
        )

    if args.fail_if_update_available and update_available:
        return 3
    return 0


def ci_lint(dry_run: bool) -> int:
    sh_roots = [
        REPO_ROOT / "scripts",
        REPO_ROOT / "benchmark" / "target",
    ]
    py_roots = [
        REPO_ROOT / "tools",
        REPO_ROOT / "benchmark" / "target",
    ]
    sh_files = find_files_with_suffixes(sh_roots, [".sh"])
    py_files = find_files_with_suffixes(py_roots, [".py"])

    if sh_files:
        rc = run(
            ["bash", "-n", *[str(p) for p in sh_files]],
            dry_run=dry_run,
        )
        if rc != 0:
            return rc

    for py_file in py_files:
        rc = run(["python3", "-m", "py_compile", str(py_file)], dry_run=dry_run)
        if rc != 0:
            return rc

    if shutil_which("shellcheck"):
        rc = run(
            ["shellcheck", *[str(p) for p in sh_files]],
            dry_run=dry_run,
        )
        if rc != 0:
            return rc
    else:
        print("shellcheck not found; skipping shellcheck.")

    return 0


def shutil_which(cmd: str) -> Optional[str]:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = pathlib.Path(path) / cmd
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merlin maintenance/developer entrypoint"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    targets = subparsers.add_parser("targets", help="Show configured profiles")
    targets_sub = targets.add_subparsers(dest="subcommand", required=True)
    targets_sub.add_parser("list", help="List build/benchmark targets")
    show = targets_sub.add_parser("show", help="Show one configured target")
    show.add_argument(
        "category", choices=["build_profiles", "benchmark_targets"], help="Section name"
    )
    show.add_argument("name", help="Entry key")
    targets.set_defaults(func=cmd_targets)

    patches = subparsers.add_parser("patches", help="Patch-stack workflows")
    patches.add_argument("action", choices=["apply", "verify", "refresh", "drift"])
    patches.set_defaults(func=cmd_patches)

    build = subparsers.add_parser("build", help="Run a build profile script")
    build.add_argument("profile", help="Profile name from config/targets.json")
    build.add_argument("extra_args", nargs=argparse.REMAINDER)
    build.set_defaults(func=cmd_build)

    bench = subparsers.add_parser("benchmark", help="Benchmark/deploy helpers")
    bench.add_argument("target", help="Target name from config/targets.json")
    bench.add_argument("action", choices=["compile-dual-vmfb", "run-dual-remote"])
    bench.add_argument("extra_args", nargs=argparse.REMAINDER)
    bench.set_defaults(func=cmd_benchmark)

    ci = subparsers.add_parser("ci", help="CI helper commands")
    ci.add_argument("action", choices=["lint", "patch-gate"])
    ci.set_defaults(func=cmd_ci)

    release = subparsers.add_parser(
        "release-status", help="Show tracked IREE release vs latest upstream tag"
    )
    release.add_argument(
        "--tracking-file",
        default=str(UPSTREAM_TRACKING_CONFIG.relative_to(REPO_ROOT)),
        help="Path to tracking YAML file (default: config/upstream_tracking.yaml)",
    )
    release.add_argument(
        "--offline",
        action="store_true",
        help="Skip network query and only print tracked state",
    )
    release.add_argument(
        "--allow-network-failure",
        action="store_true",
        help="Do not fail command if remote tag query fails",
    )
    release.add_argument(
        "--fail-if-update-available",
        action="store_true",
        help="Exit non-zero if a newer upstream release tag is found",
    )
    release.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    release.set_defaults(func=cmd_release_status)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
