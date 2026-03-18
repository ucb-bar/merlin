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
import json
import os
import pathlib
import subprocess
import sys

import utils
import yaml

HARDWARE_DIR = utils.REPO_ROOT / "build_tools" / "hardware"
SCRIPTS_DIR = HARDWARE_DIR / "scripts"
CONFIG_FILE = utils.REPO_ROOT / ".chipyard_config.json"


# ---------------------------------------------------------------------------
# Config persistence (chipyard path)
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def get_chipyard_root(args: argparse.Namespace) -> pathlib.Path | None:
    """Resolve chipyard root: --chipyard-root > $CHIPYARD_ROOT > saved config."""
    if getattr(args, "chipyard_root", None):
        return pathlib.Path(args.chipyard_root).resolve()
    env_val = os.environ.get("CHIPYARD_ROOT")
    if env_val:
        return pathlib.Path(env_val).resolve()
    saved = _load_config().get("chipyard_root")
    if saved:
        return pathlib.Path(saved).resolve()
    return None


def require_chipyard_root(args: argparse.Namespace) -> pathlib.Path | None:
    root = get_chipyard_root(args)
    if not root:
        utils.eprint("Chipyard root not configured.")
        utils.eprint("  merlin chipyard set-path /path/to/chipyard")
        utils.eprint("  OR: export CHIPYARD_ROOT=/path/to/chipyard")
        return None
    if not root.is_dir():
        utils.eprint(f"Chipyard root does not exist: {root}")
        return None
    return root


# ---------------------------------------------------------------------------
# Recipe loading
# ---------------------------------------------------------------------------


def list_recipes() -> list[dict]:
    recipes = []
    for f in sorted(HARDWARE_DIR.glob("*.yaml")):
        with f.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            data["_file"] = str(f)
            recipes.append(data)
    return recipes


def load_recipe(name: str) -> dict | None:
    for f in HARDWARE_DIR.glob("*.yaml"):
        with f.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            data["_file"] = str(f)
            if f.stem == name or data.get("name") == name:
                return data
    return None


def require_recipe(name: str) -> dict | None:
    recipe = load_recipe(name)
    if not recipe:
        utils.eprint(f"Recipe not found: {name}")
        utils.eprint(f"Available: {', '.join(r['name'] for r in list_recipes())}")
    return recipe


def _recipe_mode(recipe: dict) -> str:
    return recipe.get("mode", "bare-metal")


def _git(repo: pathlib.Path, *git_args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo)] + list(git_args), capture_output=True, text=True)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# set-path
# ---------------------------------------------------------------------------


def cmd_set_path(args: argparse.Namespace) -> int:
    path = pathlib.Path(args.path).resolve()
    if not path.is_dir():
        utils.eprint(f"Directory does not exist: {path}")
        return 1

    cfg = _load_config()
    cfg["chipyard_root"] = str(path)
    if not args.dry_run:
        _save_config(cfg)
    print(f"Chipyard root saved: {path}")
    os.environ["CHIPYARD_ROOT"] = str(path)
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


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


def cmd_validate(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1
    recipe = require_recipe(args.recipe)
    if not recipe:
        return 1

    if _recipe_mode(recipe) == "board":
        print(f"Recipe '{recipe['name']}' is a physical board — no Chipyard validation needed.")
        return 0

    chipyard_spec = recipe.get("chipyard", {})
    errors = 0

    # Branch
    actual_branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    expected_branch = chipyard_spec.get("branch")
    if expected_branch and actual_branch != expected_branch:
        utils.eprint(f"  MISMATCH branch: expected '{expected_branch}', got '{actual_branch}'")
        errors += 1
    else:
        print(f"  OK  branch: {actual_branch}")

    # SHA
    actual_sha = _git(root, "rev-parse", "HEAD")
    expected_sha = chipyard_spec.get("sha")
    if expected_sha and actual_sha != expected_sha:
        utils.eprint(f"  MISMATCH SHA: expected {expected_sha[:10]}, got {actual_sha[:10]}")
        errors += 1
    else:
        print(f"  OK  SHA: {actual_sha[:10]}")

    # Submodules
    for sub_name, sub_spec in recipe.get("submodules", {}).items():
        expected_sub_sha = sub_spec.get("sha")
        if not expected_sub_sha:
            continue
        for prefix in ["generators/", "sims/", ""]:
            sub_path = root / f"{prefix}{sub_name}"
            if sub_path.is_dir():
                actual_sub_sha = _git(sub_path, "rev-parse", "HEAD")
                if actual_sub_sha != expected_sub_sha:
                    utils.eprint(
                        f"  MISMATCH submodule {sub_name}: expected {expected_sub_sha[:10]}, got {actual_sub_sha[:10]}"
                    )
                    errors += 1
                else:
                    print(f"  OK  submodule {sub_name}: {actual_sub_sha[:10]}")
                break
        else:
            utils.eprint(f"  WARNING submodule {sub_name}: not found")

    # Sysroot
    sysroot = root / ".conda-env" / "riscv-tools" / "riscv64-unknown-elf"
    if sysroot.is_dir():
        print(f"  OK  sysroot: {sysroot}")
    else:
        utils.eprint(f"  MISSING sysroot: {sysroot}")
        errors += 1

    if errors:
        utils.eprint(f"\n{errors} validation error(s).")
        return 1
    print("\nAll checks passed.")
    return 0


# ---------------------------------------------------------------------------
# build-sim (bare-metal VCS/Verilator)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# configure-firesim — writes all FireSim deploy YAML configs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build-firemarshal — builds the base Linux image
# ---------------------------------------------------------------------------


def cmd_build_firemarshal(args: argparse.Namespace) -> int:
    root = require_chipyard_root(args)
    if not root:
        return 1

    script = SCRIPTS_DIR / "build_firemarshal_base.sh"
    return utils.run(["bash", str(script), str(root)], dry_run=args.dry_run)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--chipyard-root", default=None, help="Override chipyard root for this invocation")

    sub = parser.add_subparsers(dest="chipyard_action", required=True)

    sub.add_parser("set-path", help="Save chipyard workspace path").add_argument(
        "path", help="Path to chipyard repository root"
    )

    sub.add_parser("info", help="Show chipyard state and available recipes")

    sp = sub.add_parser("validate", help="Validate chipyard checkout matches a recipe")
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

    sp = sub.add_parser("build-firemarshal", help="Build FireMarshal base Linux image")

    sp = sub.add_parser("status", help="Check build/simulation status")
    sp.add_argument("recipe", help="Recipe name")


def main(args: argparse.Namespace) -> int:
    dispatch = {
        "set-path": cmd_set_path,
        "info": cmd_info,
        "validate": cmd_validate,
        "build-sim": cmd_build_sim,
        "run": cmd_run,
        "configure-firesim": cmd_configure_firesim,
        "build-bitstream": cmd_build_bitstream,
        "register-hwdb": cmd_register_hwdb,
        "stage-workload": cmd_stage_workload,
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
