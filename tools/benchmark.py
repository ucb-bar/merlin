#!/usr/bin/env python3
# tools/benchmark.py

import argparse
import sys

import utils


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("target", help="Target name from config/targets.json")
    parser.add_argument("action", choices=["compile-dual-vmfb", "run-dual-remote"], help="Benchmark action")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)


def main(args: argparse.Namespace) -> int:
    config = utils.load_targets_config()
    targets = config.get("benchmark_targets", {})
    target_info = targets.get(args.target)

    if not target_info:
        utils.eprint(f"Unknown benchmark target: {args.target}")
        print("Available targets:", ", ".join(targets.keys()))
        return 2

    script = None
    if args.action == "compile-dual-vmfb":
        script = target_info.get("compile_vmfb_script")
    elif args.action == "run-dual-remote":
        script = target_info.get("run_remote_script")

    if not script:
        utils.eprint(f"Target {args.target} does not define script for {args.action}")
        return 2

    return utils.run_repo_script(script, args.extra_args, args.dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Benchmark Tool")
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
