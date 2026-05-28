"""Radiance Muon kernel runner — bare-metal simulator execution.

Resolves a built `kernel.radiance.elf` and runs it on Radiance's bare-metal
sim. Pre-build the ELF via `./merlin build --profile radiance_muon` first.
"""

from __future__ import annotations

import argparse
import pathlib

import utils

from .config import require_chipyard_root
from .recipe import require_recipe


def cmd_run_radiance_muon(args: argparse.Namespace) -> int:
    """Run a Muon kernel ELF on a Chipyard Radiance bare-metal simulator.

    Resolves the kernel ELF from (in order):
      1. --kernel <path>
      2. $MERLIN_ROOT/build/radiance_muon-vanilla-release/<recipe.merlin.default_kernel_dir basename>.radiance.elf
      3. recipe.bare_metal.reference_binary (relative to RADIANCE_KERNELS_ROOT)

    Resolves the SoC config and simulator from the recipe's `bare_metal`
    block. Wraps build_tools/hardware/scripts/run_radiance_muon.sh.
    """
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    bm = recipe.get("bare_metal", {})
    if not bm:
        utils.eprint(f"Recipe '{recipe['name']}' has no bare_metal block")
        return 1

    config = bm.get("config")
    simulator = bm.get("simulator", "vcs")
    if not config:
        utils.eprint(f"Recipe '{recipe['name']}' bare_metal.config is missing")
        return 1

    # Resolve kernel ELF.
    kernel_path: pathlib.Path | None = None
    if getattr(args, "kernel", None):
        kernel_path = pathlib.Path(args.kernel).resolve()
    else:
        # Try the Merlin radiance_muon build output first.
        merlin_build = utils.REPO_ROOT / "build" / "radiance_muon-vanilla-release"
        if merlin_build.is_dir():
            elfs = sorted(merlin_build.glob("*.radiance.elf"))
            if len(elfs) == 1:
                kernel_path = elfs[0]
            elif len(elfs) > 1:
                utils.eprint("Multiple .radiance.elf candidates in " f"{merlin_build}; pass --kernel <path>:")
                for e in elfs:
                    utils.eprint(f"  {e}")
                return 1
        # Fall back to recipe.bare_metal.reference_binary.
        if not kernel_path:
            ref = bm.get("reference_binary")
            radiance_root = os.environ.get("RADIANCE_KERNELS_ROOT")
            if ref and radiance_root:
                kernel_path = pathlib.Path(radiance_root) / ref

    if not kernel_path or not kernel_path.is_file():
        utils.eprint(
            "run-radiance-muon: cannot resolve kernel ELF.\n"
            "  Pass --kernel <path>, or run `merlin build --profile radiance_muon`,\n"
            "  or set RADIANCE_KERNELS_ROOT and ensure the recipe's "
            "  bare_metal.reference_binary exists."
        )
        return 1

    print(f"Running Muon kernel: {kernel_path}")
    print(f"  config:    {config}")
    print(f"  simulator: {simulator}")

    script = SCRIPTS_DIR / "run_radiance_muon.sh"
    return utils.run(
        ["bash", str(script), str(root), config, str(kernel_path), simulator],
        dry_run=args.dry_run,
    )
