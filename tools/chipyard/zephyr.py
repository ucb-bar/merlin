"""Zephyr workload pipeline — stage_zephyr_workload + run_zephyr (chained).

Bare-metal Zephyr ELFs are staged into a FireSim deploy overlay and then
run via firesim infrasetup/runworkload.
"""

from __future__ import annotations

import argparse
import pathlib

import utils

from .config import require_chipyard_root
from .recipe import require_recipe


def cmd_stage_zephyr_workload(args: argparse.Namespace) -> int:
    """Stage a Zephyr `.elf` as a FireSim workload of kind `bare-metal-zephyr`.

    Recognises recipes whose `firesim.workload.kind` is `bare-metal-zephyr`
    and resolves the ELF path from either:
      1. --elf <path>  (explicit override)
      2. recipe.firesim.workload.elf (with {merlin_root}/{zephyr_build_dir}
         token substitution)
      3. $ZEPHYR_BUILD_DIR/zephyr/zephyr.elf (env fallback)

    Drops the ELF + workload JSON into
    `$CHIPYARD_ROOT/sims/firesim/deploy/workloads/<name>{,.json}` via the
    companion shell script `stage_firesim_zephyr.sh`.
    """
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
    kind = wl.get("kind", "linux-overlay")
    if kind != "bare-metal-zephyr":
        utils.eprint(
            f"Recipe '{recipe['name']}' workload kind is '{kind}', not "
            "'bare-metal-zephyr'. Use `merlin chipyard stage-workload` instead."
        )
        return 1

    workload_name = wl.get("name", "zephyr-merlin")

    # Resolve ELF path.
    elf_path: pathlib.Path | None = None
    if getattr(args, "elf", None):
        elf_path = pathlib.Path(args.elf).resolve()
    else:
        elf_template = wl.get("elf")
        if elf_template:
            elf_str = elf_template.replace("{merlin_root}", str(utils.REPO_ROOT))
            zbuild = os.environ.get("ZEPHYR_BUILD_DIR")
            if not zbuild:
                # Default to <chipyard>/software/zephyrproject/zephyr/build.
                zbuild = str(root / "software" / "zephyrproject" / "zephyr" / "build")
            elf_str = elf_str.replace("{zephyr_build_dir}", zbuild)
            elf_path = pathlib.Path(elf_str).resolve()
        else:
            zbuild = os.environ.get("ZEPHYR_BUILD_DIR")
            if zbuild:
                elf_path = pathlib.Path(zbuild) / "zephyr" / "zephyr.elf"

    if not elf_path or not elf_path.is_file():
        utils.eprint(
            "stage-zephyr-workload: cannot resolve Zephyr ELF.\n"
            f"  Tried: {elf_path}\n"
            "  Pass --elf <path>, set ZEPHYR_BUILD_DIR, or fix the recipe's"
            " firesim.workload.elf field."
        )
        return 1

    print(f"Staging Zephyr ELF: {elf_path}")
    script = SCRIPTS_DIR / "stage_firesim_zephyr.sh"
    return utils.run(
        ["bash", str(script), str(root), workload_name, str(elf_path)],
        dry_run=args.dry_run,
    )


# ---------------------------------------------------------------------------
# run-zephyr — one-shot stage + infrasetup + runworkload
# ---------------------------------------------------------------------------


def cmd_run_zephyr(args: argparse.Namespace) -> int:
    """End-to-end: stage the Zephyr ELF, then `firesim infrasetup runworkload`.

    Reuses cmd_stage_zephyr_workload then chains the two FireSim commands
    a Zephyr workload needs (it's `runworkload` not `runonly` because we
    want the deploy/results-workload/ tree generated for the uartlog).
    """
    rc = cmd_stage_zephyr_workload(args)
    if rc != 0:
        return rc
    root = require_chipyard_root(args)
    if not root:
        return 1
    deploy_dir = root / "sims" / "firesim" / "deploy"
    if not deploy_dir.is_dir():
        utils.eprint(f"FireSim deploy dir missing: {deploy_dir}")
        return 1
    rc = utils.run(["firesim", "infrasetup"], cwd=deploy_dir, dry_run=args.dry_run)
    if rc != 0:
        return rc
    return utils.run(["firesim", "runworkload"], cwd=deploy_dir, dry_run=args.dry_run)
