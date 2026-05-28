"""Read-only chipyard introspection — info, status, firemarshal build."""

from __future__ import annotations

import argparse
import subprocess

import utils

from .config import get_chipyard_root, require_chipyard_root
from .git_ops import _git
from .recipe import _recipe_mode, list_recipes, require_recipe


def cmd_info(args: argparse.Namespace) -> int:
    root = get_chipyard_root(args)
    if root and root.is_dir():
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        sha = _git(root, "rev-parse", "--short", "HEAD")
        print(f"Chipyard root:  {root}")
        print(f"  Branch: {branch}  SHA: {sha}")
    elif root:
        print(f"Chipyard root:  {root} (not found)")
    else:
        print("Chipyard root:  not configured")
        print("  Set with: merlin chipyard set-path /path/to/chipyard")

    print()
    print("Available recipes:")
    for r in list_recipes():
        mode = _recipe_mode(r)
        if mode == "board":
            detail = "physical board"
        elif mode == "firesim":
            fs = r.get("firesim", {})
            br = fs.get("build_recipe", {})
            detail = f"firesim ({br.get('PLATFORM', '?')})"
        elif mode == "bare-metal":
            bm = r.get("bare_metal", {})
            detail = f"{bm.get('simulator', 'vcs')} ({bm.get('config', '?')})"
        else:
            detail = mode
        print(f"  {r['name']:25s}  {detail}")
    return 0


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def cmd_build_firemarshal(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1

    script = SCRIPTS_DIR / "build_firemarshal_base.sh"
    return utils.run(["bash", str(script), str(root)], dry_run=args.dry_run)


def cmd_status(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    mode = _recipe_mode(recipe)

    # Check for running build processes
    print("Active build processes:")
    result = subprocess.run(
        ["bash", "-c", "ps -ef | egrep 'firesim|vivado|vitis|make .*replace-rtl|java .*sbt' | grep -v grep || true"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip())
    else:
        print("  (none)")

    if mode == "firesim":
        fs = recipe.get("firesim", {})
        br = fs.get("build_recipe", {})
        target_config = br.get("TARGET_CONFIG", "")
        recipe_name = br.get("recipe_name", recipe["name"])

        # Check for bitstream
        results_dir = root / "sims" / "firesim" / "deploy" / "results-build"
        if results_dir.is_dir() and target_config:
            result = subprocess.run(
                ["find", str(results_dir), "-path", f"*{target_config}*", "-name", "firesim.tar.gz"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                print("\nBitstream found:")
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
            else:
                print(f"\nNo bitstream found for {target_config}")

        # Check hwdb registration
        hwdb = root / "sims" / "firesim" / "deploy" / "config_hwdb.yaml"
        if hwdb.exists():
            with hwdb.open() as f:
                hwdb_data = yaml.safe_load(f) or {}
            if recipe_name in hwdb_data:
                print(f"\nHWDB entry '{recipe_name}': registered")
            else:
                print(f"\nHWDB entry '{recipe_name}': not registered")
                print(f"  Register with: merlin chipyard register-hwdb {args.recipe}")

    elif mode == "bare-metal":
        bm = recipe.get("bare_metal", {})
        config = bm.get("config", "")
        simulator = bm.get("simulator", "vcs")
        sim_dir = root / "sims" / simulator
        if sim_dir.is_dir():
            result = subprocess.run(
                ["bash", "-c", f"ls -t {sim_dir}/simv-*{config}* 2>/dev/null | head -3"], capture_output=True, text=True
            )
            if result.stdout.strip():
                print("\nSimulator binaries:")
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
            else:
                print(f"\nNo simulator binary for {config}")
                print(f"  Build with: merlin chipyard build-sim {args.recipe}")

    return 0
