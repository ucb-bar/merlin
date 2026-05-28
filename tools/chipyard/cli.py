#!/usr/bin/env python3
# tools/chipyard.py
#
# Merlin <-> Chipyard interaction tool.
# Reads hardware recipes from build_tools/hardware/*.yaml and automates
# Chipyard validation, simulator builds, FireSim configuration, workload
# staging, and bare-metal execution.
#
# Users should never need to manually edit Chipyard config files.

import argparse
import sys

import utils

HARDWARE_DIR = utils.REPO_ROOT / "build_tools" / "hardware"
SCRIPTS_DIR = HARDWARE_DIR / "scripts"
CONFIG_FILE = utils.REPO_ROOT / ".chipyard_config.json"


# Subcommand handlers extracted to topic modules
from .bare_metal import cmd_build_sim, cmd_run
from .config import cmd_set_path
from .firesim import cmd_build_bitstream, cmd_configure_firesim, cmd_register_hwdb, cmd_stage_workload
from .git_ops import cmd_checkout, cmd_validate
from .radiance import cmd_run_radiance_muon
from .status import cmd_build_firemarshal, cmd_info, cmd_status
from .zephyr import cmd_run_zephyr, cmd_stage_zephyr_workload


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--chipyard-root", default=None, help="Override chipyard root for this invocation")

    sub = parser.add_subparsers(dest="chipyard_action", required=True)

    sub.add_parser("set-path", help="Save chipyard workspace path").add_argument(
        "path", help="Path to chipyard repository root"
    )

    sub.add_parser("info", help="Show chipyard state and available recipes")

    sp = sub.add_parser("validate", help="Validate chipyard checkout matches a recipe")
    sp.add_argument("recipe", help="Recipe name (e.g., gemmini_mx, saturn_opu_u250)")

    sp = sub.add_parser("checkout", help="Switch chipyard branch and submodules to match a recipe")
    sp.add_argument("recipe", help="Recipe name (e.g., gemmini_mx, saturn_opu_u250)")

    # Bare-metal commands
    sp = sub.add_parser("build-sim", help="Build VCS/Verilator RTL simulator")
    sp.add_argument("recipe", help="Recipe name")

    sp = sub.add_parser("run", help="Run bare-metal ELF on simulator")
    sp.add_argument("recipe", help="Recipe name")
    sp.add_argument("binary", help="Path to bare-metal ELF")

    # FireSim commands
    sp = sub.add_parser("configure-firesim", help="Write FireSim deploy configs for a recipe")
    sp.add_argument("recipe", help="Recipe name")

    sp = sub.add_parser("build-bitstream", help="Build FireSim FPGA bitstream")
    sp.add_argument("recipe", help="Recipe name")

    sp = sub.add_parser("register-hwdb", help="Register built bitstream in FireSim HWDB")
    sp.add_argument("recipe", help="Recipe name")

    sp = sub.add_parser("stage-workload", help="Stage Merlin workload for FireSim")
    sp.add_argument("recipe", help="Recipe name")
    sp.add_argument(
        "overlay_dir",
        nargs="?",
        default=None,
        help="Directory to overlay into workload (default: build/firesim-merlin-release/install)",
    )

    sp = sub.add_parser(
        "stage-zephyr-workload",
        help="Stage a Zephyr ELF as a bare-metal FireSim workload",
    )
    sp.add_argument("recipe", help="Recipe name (workload.kind must be bare-metal-zephyr)")
    sp.add_argument(
        "--elf",
        default=None,
        help=(
            "Path to the Zephyr ELF to stage. Falls back to the recipe's "
            "firesim.workload.elf field, then to "
            "$ZEPHYR_BUILD_DIR/zephyr/zephyr.elf."
        ),
    )

    sp = sub.add_parser(
        "run-zephyr",
        help="Stage Zephyr ELF + firesim infrasetup + firesim runworkload",
    )
    sp.add_argument("recipe", help="Recipe name (workload.kind must be bare-metal-zephyr)")
    sp.add_argument("--elf", default=None, help="Override Zephyr ELF path (see stage-zephyr-workload)")

    sp = sub.add_parser(
        "run-radiance-muon",
        help="Run a Muon kernel ELF on a Radiance bare-metal sim (RadianceMuonConfig etc.)",
    )
    sp.add_argument("recipe", help="Recipe name (bare_metal mode, e.g. radiance_muon)")
    sp.add_argument(
        "--kernel",
        default=None,
        help=(
            "Path to a Muon kernel ELF (kernel.radiance.elf). Defaults to the "
            "single .radiance.elf in build/radiance_muon-vanilla-release/, then "
            "to the recipe's bare_metal.reference_binary."
        ),
    )

    sp = sub.add_parser("build-firemarshal", help="Build FireMarshal base Linux image")

    sp = sub.add_parser("status", help="Check build/simulation status")
    sp.add_argument("recipe", help="Recipe name")


def main(args: argparse.Namespace) -> int:
    dispatch = {
        "set-path": cmd_set_path,
        "info": cmd_info,
        "validate": cmd_validate,
        "checkout": cmd_checkout,
        "build-sim": cmd_build_sim,
        "run": cmd_run,
        "configure-firesim": cmd_configure_firesim,
        "build-bitstream": cmd_build_bitstream,
        "register-hwdb": cmd_register_hwdb,
        "stage-workload": cmd_stage_workload,
        "stage-zephyr-workload": cmd_stage_zephyr_workload,
        "run-zephyr": cmd_run_zephyr,
        "run-radiance-muon": cmd_run_radiance_muon,
        "build-firemarshal": cmd_build_firemarshal,
        "status": cmd_status,
    }
    handler = dispatch.get(args.chipyard_action)
    if not handler:
        utils.eprint(f"Unknown chipyard action: {args.chipyard_action}")
        return 2
    return handler(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merlin <-> Chipyard interaction tool")
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
