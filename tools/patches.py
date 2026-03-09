#!/usr/bin/env python3
# tools/patches.py

import argparse
import sys

import utils


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("action", choices=["apply", "verify", "refresh", "drift"], help="Patch workflow action")


def main(args: argparse.Namespace) -> int:
    script_map = {
        "apply": "patches/tools/apply_all.sh",
        "verify": "patches/tools/verify_clean.sh",
        "refresh": "patches/tools/refresh_all.sh",
        "drift": "patches/tools/check_upstream_drift.sh",
    }

    script = script_map.get(args.action)
    if not script:
        utils.eprint(f"Unsupported patches action: {args.action}")
        return 2

    return utils.run_repo_script(script, [], args.dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin Patch Manager")
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
