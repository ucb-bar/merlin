"""Bare-metal RTL simulator commands — cmd_build_sim, cmd_run.

Targets VCS/Verilator builds under chipyard's sim/ dir. Used for
non-FireSim/non-FPGA execution paths.
"""

from __future__ import annotations

import argparse

import utils

from .config import require_chipyard_root
from .recipe import _recipe_mode, require_recipe


def cmd_build_sim(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    bm = recipe.get("bare_metal", {})
    config = bm.get("config", recipe.get("config"))
    simulator = bm.get("simulator", "vcs")
    if not config:
        utils.eprint(f"Recipe '{recipe['name']}' has no config class defined")
        return 1

    sim_dir = root / "sims" / simulator
    if not sim_dir.is_dir():
        utils.eprint(f"Simulator directory not found: {sim_dir}")
        return 1

    print(f"Building {simulator} simulator for {config}...")
    return utils.run(["make", f"CONFIG={config}"], cwd=sim_dir, dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# run (bare-metal)
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    binary = pathlib.Path(args.binary).resolve()
    if not binary.exists() and not args.dry_run:
        utils.eprint(f"Binary not found: {binary}")
        return 1

    mode = _recipe_mode(recipe)

    if mode == "bare-metal":
        bm = recipe.get("bare_metal", {})
        config = bm.get("config")
        simulator = bm.get("simulator", "vcs")
        sim_dir = root / "sims" / simulator
        cmd = ["make", f"CONFIG={config}", f"BINARY={binary}", "LOADMEM=1", "run-binary"]
        print(f"Running {binary.name} on {simulator} ({config})...")
        return utils.run(cmd, cwd=sim_dir, dry_run=args.dry_run)

    elif mode == "firesim":
        print("For FireSim targets, use the full workflow:")
        print(f"  merlin chipyard configure-firesim {recipe['name']}")
        print(f"  merlin chipyard stage-workload {recipe['name']} <overlay_dir>")
        print(f"  cd {root}/sims/firesim/deploy && firesim infrasetup && firesim runworkload")
        return 0

    utils.eprint(f"Recipe '{recipe['name']}' mode '{mode}' does not support direct run")
    return 1
