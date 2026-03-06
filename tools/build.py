#!/usr/bin/env python3
# tools/build.py

import argparse
import sys
import os
import json
import pathlib
import subprocess
import utils

def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--target", 
        choices=["host", "spacemit", "firesim"], 
        default="host",
        help="Target platform (default: host)"
    )
    parser.add_argument(
        "--config", 
        choices=["debug", "release", "asan", "trace", "perf"], 
        default="debug",
        help="Build configuration type"
    )
    parser.add_argument(
        "--cmake-target",
        help="Build specific CMake target (default: install)"
    )
    parser.add_argument(
        "--with-plugin", 
        action="store_true", 
        help="Enable Merlin compiler plugin"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Delete build directory before building"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose build output"
    )

def get_iree_version(iree_src: pathlib.Path) -> str:
    try:
        with (iree_src / "runtime" / "version.json").open() as f:
            return json.load(f).get("package-version", "unknown")
    except Exception:
        return "unknown"

def main(args: argparse.Namespace) -> int:
    # 1. Setup Paths
    iree_src = utils.resolve_repo_path("third_party/iree_bar")
    plugin_src = utils.REPO_ROOT
    
    variant = "merlin" if args.with_plugin else "vanilla"
    version = get_iree_version(iree_src)
    
    # Clean structure: build/spacemit-merlin-perf
    build_name = f"{args.target}-{variant}-{args.config}"
    build_dir = utils.REPO_ROOT / "build" / build_name
    install_dir = build_dir / "install"

    print(f"🔧 Configuration: {args.target} | {args.config} | Plugin: {args.with_plugin}")
    print(f"📂 Build Dir:     {build_dir}")
    print(f"📂 Install Dir:   {install_dir}")

    if args.clean and build_dir.exists():
        print("Cleaning build directory...")
        import shutil
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # 2. Base CMake Flags
    cmake_args = [
        "cmake", "-G", "Ninja",
        f"-B{build_dir}",
        f"-S{iree_src}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DIREE_ENABLE_LLD=ON",
        f"-DPython3_EXECUTABLE={sys.executable}",
        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    ]

    env = os.environ.copy()

    # 3. Config Specific Flags
    if args.config == "debug":
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DIREE_ENABLE_ASSERTIONS=ON",
            "-DIREE_ENABLE_ASAN=OFF",
            "-DCMAKE_CXX_FLAGS=-Wno-error=cpp -Wno-error=maybe-uninitialized -fno-omit-frame-pointer -gz=none",
            "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer -fdebug-types-section",
        ])
    elif args.config == "asan":
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DIREE_ENABLE_ASAN=ON",
            "-DIREE_ENABLE_ASSERTIONS=ON",
        ])
        # Attempt to inject LD_PRELOAD for ASan
        try:
            cc = env.get("CC", "clang")
            if "clang" in cc:
                res = subprocess.run([cc, "-print-resource-dir"], capture_output=True, text=True)
                if res.returncode == 0:
                    resource_dir = pathlib.Path(res.stdout.strip())
                    candidates = list(resource_dir.glob("lib/**/libclang_rt.asan-x86_64.so"))
                    if candidates:
                        env["LD_PRELOAD"] = str(candidates[0])
                        print(f"⚠️  Injecting LD_PRELOAD={candidates[0]}")
        except Exception as e:
            print(f"Warning: Failed ASan LD_PRELOAD detection: {e}")

    elif args.config == "release" or args.config == "perf":
        build_type = "Release" if args.config == "perf" else "RelWithDebInfo"
        cmake_args.extend([
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-DIREE_ENABLE_ASSERTIONS=ON",
            "-DCMAKE_CXX_FLAGS=-Wno-error=cpp -fno-omit-frame-pointer -fdebug-types-section -gz=none",
            "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer -fdebug-types-section -gz=none",
        ])
        if args.config == "perf":
             cmake_args.extend([
                 "-DIREE_ENABLE_RUNTIME_TRACING=OFF",
                 "-DIREE_ENABLE_CPUINFO=OFF"
             ])

    elif args.config == "trace":
        cmake_args.extend([
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DIREE_ENABLE_RUNTIME_TRACING=ON",
            "-DIREE_ENABLE_COMPILER_TRACING=ON",
            "-DIREE_TRACING_MODE=4", # Tracy
            "-DIREE_ENABLE_ASSERTIONS=ON",
            "-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer",
            "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer",
        ])
        if args.target in ["spacemit", "firesim"]:
             cmake_args.append("-DTRACY_NO_POINTER_COMPRESSION=ON")

    # 4. Target Specific Logic

    # For cross-compilation targets, we must provide the path to the native host tools.
    # We ALWAYS use the 'release' build of the host tools for maximum compilation speed.
    if args.target != "host":
        # UPDATED: Reconstruct the name to match the flat build/target-variant-config structure
        # We always check for 'release' config for host tools.
        primary_host_name = f"host-{variant}-release"
        fallback_host_name = f"host-vanilla-release"
        
        host_variant_bin_dir = utils.REPO_ROOT / "build" / primary_host_name / "install" / "bin"
        host_vanilla_bin_dir = utils.REPO_ROOT / "build" / fallback_host_name / "install" / "bin"

        host_bin_dir = None

        if host_variant_bin_dir.exists():
            host_bin_dir = host_variant_bin_dir
        elif host_vanilla_bin_dir.exists():
            print(f"ℹ️  Note: Using vanilla host tools from {host_vanilla_bin_dir}")
            host_bin_dir = host_vanilla_bin_dir
        
        if host_bin_dir:
            cmake_args.append(f"-DIREE_HOST_BIN_DIR={host_bin_dir}")
        else:
            # This error message now correctly reflects the paths actually being checked
            print(f"❌ Error: No host tools found at {host_variant_bin_dir} or {host_vanilla_bin_dir}")
            print("   Please build host tools first: python3 tools/build.py --target host --config release")
            return 1
    
    if args.target == "host":
        cmake_args.extend([
            "-DIREE_TARGET_BACKEND_DEFAULTS=OFF",
            "-DIREE_TARGET_BACKEND_LLVM_CPU=ON",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
            "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
            "-DIREE_BUILD_PYTHON_BINDINGS=ON",
            "-DIREE_BUILD_SAMPLES=OFF",
            "-DIREE_BUILD_TESTS=ON",
        ])

    elif args.target == "spacemit":
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
            default_tc = utils.REPO_ROOT / "build_tools" / "riscv-tools-spacemit" / "spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
            if default_tc.exists():
                tc_root = str(default_tc)
        
        if not tc_root:
            utils.eprint("❌ Error: SpacemiT toolchain not found. Set RISCV_TOOLCHAIN_ROOT.")
            return 1

        cmake_args.extend([
            "-DMERLIN_BUILD_SPACEMITX60=ON",
            f"-DCMAKE_TOOLCHAIN_FILE={iree_src}/build_tools/cmake/riscv.toolchain.cmake",
            "-DRISCV_CPU=linux-riscv_64",
            "-DIREE_BUILD_COMPILER=OFF",
            f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
            "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
            "-DIREE_BUILD_TESTS=OFF",
            "-DIREE_BUILD_SAMPLES=ON",
            "-DCMAKE_C_FLAGS=-march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
            "-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer -march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
            "-DIREE_ENABLE_CPUINFO=ON",
        ])

    elif args.target == "firesim":
        tc_file = utils.REPO_ROOT / "build_tools" / "firesim" / "riscv_firesim.toolchain.cmake"
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
             tc_root = str(utils.REPO_ROOT / "build_tools" / "riscv-tools-iree" / "toolchain" / "clang" / "linux" / "RISCV")

        cmake_args.extend([
            "-DMERLIN_BUILD_SATURN_OPU=ON",
            f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
            "-DIREE_BUILD_COMPILER=OFF",
            f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
            "-DIREE_ARCH=riscv_64",
            "-DIREE_ENABLE_THREADING=OFF",
            "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
            "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
            "-DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF",
            "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON",
            "-DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF",
            "-DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON",
            "-DIREE_UK_BUILD_RISCV_64_ZVFH=OFF",
            "-DIREE_UK_BUILD_RISCV_64_ZVFHMIN=OFF",
            "-DIREE_BUILD_SAMPLES=ON",
            "-DIREE_BUILD_TESTS=OFF",
        ])

    # 5. Plugin Activation
    if args.with_plugin:
        cmake_args.extend([
            f"-DIREE_CMAKE_PLUGIN_PATHS={plugin_src}",
            "-DMERLIN_ENABLE_CORE=ON",
            "-DMERLIN_ENABLE_TARGET_GEMMINI=ON",
            "-DMERLIN_ENABLE_TARGET_SATURN=ON",
            "-DMERLIN_ENABLE_TARGET_SPACEMIT=ON",
            # Disable the in-tree IREE fork's gemmini plugin so that
            # the Merlin-provided merlin_gemmini plugin is used instead.
            "-DIREE_GEMMINI_EXTERNAL_PLUGIN=ON",
        ])

    # 6. Configure & Build Main Target
    if utils.run(cmake_args, dry_run=args.dry_run, env=env) != 0:
        return 1
    
    target_arg = args.cmake_target if args.cmake_target else "install"
    build_cmd = ["cmake", "--build", str(build_dir), "--target", target_arg]
    if args.verbose:
        build_cmd.append("--verbose")
        
    if utils.run(build_cmd, dry_run=args.dry_run, env=env) != 0:
        return 1

    # 7. Build Extra Tools (Host Only)
    # This replicates the logic from `build_debug_asan.sh`
    if args.target == "host" and not args.cmake_target:
        print(">> Building extra LLVM tools (llvm-mca, llvm-objdump)...")
        extra_tools_cmd = ["cmake", "--build", str(build_dir), "--target", "llvm-mca", "llvm-objdump"]
        if args.verbose:
            extra_tools_cmd.append("--verbose")
        if utils.run(extra_tools_cmd, dry_run=args.dry_run, env=env) != 0:
            return 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))