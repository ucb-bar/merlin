#!/usr/bin/env python
"""Install merlin's multicore Saturn-vectors SoC configs into a chipyard checkout.

Merlin's multicore RVV path needs an SoC where **every** tile has a vector unit. No stock
chipyard config provides one: every Saturn config is single-core (`WithNShuttleCores(1)` /
`WithNHugeCores(1)`), and the only multi-tile SoC (`GemminiAndOPUShuttleConfig`) builds its
vector unit on tile 1 only. So the configs live here and are installed into the (out-of-repo)
chipyard checkout, the same way `build_tools/SpacemiT/setup_toolchain.sh` handles that
toolchain.

What it installs:
  * `chipyard.DualSaturnV256D128ShuttleConfig` / `MultiSaturnV256D128ShuttleConfig` — 2 and 4
    Shuttle tiles, each with its own Saturn unit at vLen=256/dLen=128 (matching the SpacemiT
    K1, so a schedule tuned on the board transfers without a re-tune).
  * the matching `FireSim*` target configs and Alveo U250 build recipes.

Idempotent: re-running detects what is already present and changes nothing.

    .venv/bin/python build_tools/chipyard/setup_multicore_saturn.py            # install
    .venv/bin/python build_tools/chipyard/setup_multicore_saturn.py --check    # report only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Appended to firechip's TargetConfigs.scala. FireSim rather than Verilator because RTL
# simulation runs ~10^4 cycles/s and a whole-model inference is ~10^10 cycles; on the FPGA the
# same design runs at tens of MHz, which is what makes whole-model multicore RVV measurable.
FIRESIM_TARGETS = """
// ---- merlin multicore Saturn-vectors (build_tools/chipyard/setup_multicore_saturn.py) ----
// Every Shuttle tile carries its own Saturn vector unit, which no stock config provides.
// Build DUAL first: two vLen=256 units are already a large area step up from the single
// vLen=128 OPU that fits today, and a placement failure only surfaces after hours of synthesis.
class FireSimDualSaturnV256D128ShuttleConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.DualSaturnV256D128ShuttleConfig)

class FireSimMultiSaturnV256D128ShuttleConfig extends Config(
  new WithDefaultFireSimBridges ++
  new WithFireSimConfigTweaks ++
  new chipyard.MultiSaturnV256D128ShuttleConfig)
"""

# fpga_frequency is deliberately below the 30 MHz the single-OPU bitstream uses: several
# vLen=256 vector units are a large timing step up, and a frequency that will not close wastes
# the entire synthesis run.
BUILD_RECIPES = """
# ---- merlin multicore Saturn-vectors (build_tools/chipyard/setup_multicore_saturn.py) ----
alveo_u250_firesim_dual_saturn_v256d128:
    PLATFORM: xilinx_alveo_u250
    TARGET_PROJECT: firesim
    TARGET_PROJECT_MAKEFRAG: null
    DESIGN: FireSim
    TARGET_CONFIG: FireSimDualSaturnV256D128ShuttleConfig
    CONFIG_PACKAGE: firechip.chip
    PLATFORM_CONFIG: BaseXilinxAlveoU250Config
    deploy_quintuplet: null
    platform_config_args:
        fpga_frequency: 25
        build_strategy: TIMING
    post_build_hook: null
    metasim_customruntimeconfig: null
    bit_builder_recipe: bit-builder-recipes/xilinx_alveo_u250.yaml

alveo_u250_firesim_multi_saturn_v256d128:
    PLATFORM: xilinx_alveo_u250
    TARGET_PROJECT: firesim
    TARGET_PROJECT_MAKEFRAG: null
    DESIGN: FireSim
    TARGET_CONFIG: FireSimMultiSaturnV256D128ShuttleConfig
    CONFIG_PACKAGE: firechip.chip
    PLATFORM_CONFIG: BaseXilinxAlveoU250Config
    deploy_quintuplet: null
    platform_config_args:
        fpga_frequency: 25
        build_strategy: TIMING
    post_build_hook: null
    metasim_customruntimeconfig: null
    bit_builder_recipe: bit-builder-recipes/xilinx_alveo_u250.yaml
"""

MARKER = "merlin multicore Saturn-vectors"


def chipyard_root() -> Path:
    sys.path.insert(0, str(HERE.parents[1] / "merlin" / "python"))
    from merlin.common.paths import env  # noqa: PLC0415
    root = env("MERLIN_CHIPYARD") or env("MERLIN_EXT_CHIPYARD")
    if not root:
        raise SystemExit("MERLIN_CHIPYARD unset — point it at your chipyard checkout (.env)")
    p = Path(root)
    if not (p / "generators" / "chipyard").is_dir():
        raise SystemExit(f"{p} does not look like a chipyard checkout")
    return p


def _install_scala(cy: Path, check: bool) -> str:
    dest = cy / "generators/chipyard/src/main/scala/config/MerlinSaturnConfigs.scala"
    src = HERE / "MerlinSaturnConfigs.scala"
    if dest.is_file() and dest.read_text() == src.read_text():
        return f"[ok]      {dest.relative_to(cy)} already current"
    if check:
        return f"[MISSING] {dest.relative_to(cy)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return f"[written] {dest.relative_to(cy)}"


def _append_once(path: Path, text: str, check: bool, label: str) -> str:
    if not path.is_file():
        return f"[SKIP]    {label}: {path} not found (older chipyard layout?)"
    body = path.read_text()
    if MARKER in body:
        return f"[ok]      {label} already present"
    if check:
        return f"[MISSING] {label}"
    path.write_text(body.rstrip("\n") + "\n" + text)
    return f"[written] {label}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    a = ap.parse_args()

    cy = chipyard_root()
    print(f"chipyard: {cy}")
    results = [
        _install_scala(cy, a.check),
        _append_once(cy / "generators/firechip/chip/src/main/scala/TargetConfigs.scala",
                     FIRESIM_TARGETS, a.check, "firechip TargetConfigs.scala"),
        _append_once(cy / "sims/firesim/deploy/config_build_recipes.yaml",
                     BUILD_RECIPES, a.check, "firesim config_build_recipes.yaml"),
    ]
    for r in results:
        print("  " + r)
    missing = [r for r in results if r.startswith("[MISSING]")]
    if a.check:
        print("\nall present" if not missing else f"\n{len(missing)} item(s) not installed")
        return 1 if missing else 0

    print("""
next:
  # cycle-accurate RTL simulation (mechanism + small kernels only -- ~10^4 cycles/s)
  make -C $MERLIN_CHIPYARD/sims/verilator CONFIG=MultiSaturnV256D128ShuttleConfig -j16

  # FPGA bitstream (whole-model multicore RVV cycles; HOURS of Vivado synthesis)
  #   add alveo_u250_firesim_dual_saturn_v256d128 to sims/firesim/deploy/config_build.yaml
  #   builds_to_run, then:
  cd $MERLIN_CHIPYARD/sims/firesim && source sourceme-manager.sh
  cd deploy && ./firesim buildbitstream""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
