"""Radiance Muon build path — links a kernel.cpp wrapper + body .o
+ libmuonrt.a + tohost.S into `kernel.radiance.elf`.

Short-circuited from `cli.main()` when `args.target == "radiance_muon"`.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import utils


def build_radiance_muon(args: argparse.Namespace) -> int:
    """Cross-compile a Muon kernel via build_tools/radiance/ (not IREE).

    Short-circuits the IREE build pipeline entirely: the radiance_muon target
    only wraps llvm-muon clang against a single kernel directory. Output is
    `build/radiance_muon-vanilla-release/<name>.radiance.elf`, intended to be
    byte-equivalent to a `radiance-kernels/kernels/<name>/kernel.radiance.elf`
    built via the upstream Make recipe.
    """
    if not os.environ.get("LLVM_MUON"):
        utils.eprint(
            "❌ radiance_muon: LLVM_MUON not set.\n"
            "  export LLVM_MUON=$RADIANCE_KERNELS_ROOT/llvm/llvm-muon")
        return 1
    if not os.environ.get("RADIANCE_KERNELS_ROOT"):
        utils.eprint(
            "❌ radiance_muon: RADIANCE_KERNELS_ROOT not set.\n"
            "  export RADIANCE_KERNELS_ROOT=/path/to/radiance-kernels")
        return 1

    radiance_root = pathlib.Path(os.environ["RADIANCE_KERNELS_ROOT"]).resolve()
    kernel_dir = (
        pathlib.Path(args.kernel_dir).resolve()
        if args.kernel_dir
        else radiance_root / "kernels" / "vecadd"
    )
    if not (kernel_dir / "kernel.cpp").is_file():
        utils.eprint(
            f"❌ radiance_muon: {kernel_dir}/kernel.cpp not found.\n"
            "  Pass --kernel-dir <abs-path-to-radiance-kernel-dir>")
        return 1

    kernel_name = args.kernel_name or kernel_dir.name

    radiance_drv = utils.REPO_ROOT / "build_tools" / "radiance"
    toolchain = radiance_drv / "riscv_muon.toolchain.cmake"
    if not toolchain.is_file():
        utils.eprint(f"❌ radiance_muon: toolchain missing: {toolchain}")
        return 1

    build_name = f"radiance_muon-vanilla-{args.config}"
    build_dir = utils.REPO_ROOT / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_bin = args.cmake_bin or "cmake"
    cmake_args = [
        cmake_bin,
        "-G", "Ninja",
        f"-B{build_dir}",
        f"-S{radiance_drv}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        f"-DMERLIN_RADIANCE_KERNEL_DIR={kernel_dir}",
        f"-DMERLIN_RADIANCE_KERNEL_NAME={kernel_name}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if args.kernel_body_obj:
        body_obj = pathlib.Path(args.kernel_body_obj).resolve()
        if not body_obj.is_file():
            utils.eprint(f"❌ radiance_muon: --kernel-body-obj not found: {body_obj}")
            return 1
        cmake_args.append(f"-DMERLIN_RADIANCE_KERNEL_BODY_OBJ={body_obj}")
    if args.cmake_arg:
        cmake_args.extend(args.cmake_arg)

    print(f"🔧 radiance_muon: kernel_dir={kernel_dir}")
    print(f"🔧 radiance_muon: kernel_name={kernel_name}")
    print(f"📂 Build Dir: {build_dir}")

    env = os.environ.copy()
    if utils.run(cmake_args, dry_run=args.dry_run, env=env) != 0:
        return 1

    target = args.cmake_target or f"{kernel_name}_radiance"
    build_cmd = [cmake_bin, "--build", str(build_dir), "--target", target]
    build_cmd.extend(args.cmake_build_arg or [])
    if args.verbose:
        build_cmd.append("--verbose")
    return utils.run(build_cmd, dry_run=args.dry_run, env=env)
