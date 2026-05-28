"""FireSim deploy pipeline — configure, bitstream-build, hwdb-register, stage.

Each command is a thin shell wrapper around scripts under
`build_tools/hardware/scripts/` plus FireSim's own infrasetup/runworkload.
"""

from __future__ import annotations

import argparse

import utils

from .config import require_chipyard_root
from .recipe import require_recipe


def cmd_configure_firesim(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    fs = recipe.get("firesim")
    if not fs:
        utils.eprint(f"Recipe '{recipe['name']}' is not a FireSim recipe")
        return 1

    br = fs.get("build_recipe", {})
    rt = fs.get("runtime", {})
    recipe_name = br.get("recipe_name", recipe["name"])
    hw_config = rt.get("default_hw_config", recipe_name)
    workload_name = fs.get("workload", {}).get("name", "merlin-iree") + ".json"

    script = SCRIPTS_DIR / "configure_firesim.sh"
    return utils.run(
        ["bash", str(script), str(root), recipe_name, hw_config, workload_name],
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# build-bitstream
# ---------------------------------------------------------------------------


def cmd_build_bitstream(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    fs = recipe.get("firesim")
    if not fs:
        utils.eprint(f"Recipe '{recipe['name']}' is not a FireSim recipe")
        return 1

    deploy_dir = root / "sims" / "firesim" / "deploy"
    print("Building FireSim bitstream...")
    print("  (This can take hours. Use tmux to avoid losing progress.)")
    return utils.run(["firesim", "buildbitstream"], cwd=deploy_dir, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# register-hwdb — finds built bitstream and registers it
# ---------------------------------------------------------------------------


def cmd_register_hwdb(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    fs = recipe.get("firesim")
    if not fs:
        utils.eprint(f"Recipe '{recipe['name']}' is not a FireSim recipe")
        return 1

    br = fs.get("build_recipe", {})
    recipe_name = br.get("recipe_name", recipe["name"])
    target_config = br.get("TARGET_CONFIG", "")
    platform_config = br.get("PLATFORM_CONFIG", "")

    script = SCRIPTS_DIR / "register_hwdb.sh"
    return utils.run(
        ["bash", str(script), str(root), recipe_name, target_config, platform_config],
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# stage-workload — builds overlay and workload JSON
# ---------------------------------------------------------------------------


def cmd_stage_workload(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    fs = recipe.get("firesim")
    if not fs:
        utils.eprint(f"Recipe '{recipe['name']}' is not a FireSim recipe")
        return 1

    wl = fs.get("workload", {})
    workload_name = wl.get("name", "merlin-iree")
    command = wl.get("command", "/opt/merlin/run.sh && poweroff -f")
    overlay_src = pathlib.Path(args.overlay_dir).resolve() if args.overlay_dir else None

    if not overlay_src:
        # Default: use the firesim build output
        overlay_src = utils.REPO_ROOT / "build" / "firesim-merlin-release" / "install"

    script = SCRIPTS_DIR / "stage_firesim_workload.sh"
    return utils.run(
        ["bash", str(script), str(root), workload_name, str(overlay_src), command],
        dry_run=args.dry_run,
    )
