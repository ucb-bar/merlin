#!/usr/bin/env python3
# tools/build.py

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import utils

PROFILE_PRESETS: dict[str, dict[str, object]] = {
    "vanilla": {
        "target": "host",
        "config": "debug",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": False,
        "build_compiler": True,
        "build_python_bindings": True,
        "build_samples": False,
        "build_tests": True,
        "enable_libbacktrace": True,
    },
    "full-plugin": {
        "target": "host",
        "config": "debug",
        "with_plugin": True,
        "plugin_compiler": True,
        "plugin_runtime": True,
        "plugin_runtime_radiance": True,
        "plugin_runtime_samples": True,
        "plugin_runtime_benchmarks": False,
        "plugin_runtime_radiance_tests": True,
        "build_compiler": True,
        "build_python_bindings": True,
        "build_samples": False,
        "build_tests": True,
        "enable_libbacktrace": True,
        "compiler_scope": "all",
    },
    "radiance": {
        "target": "host",
        "config": "debug",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": True,
        "plugin_runtime_radiance": True,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "plugin_runtime_radiance_tests": True,
        "plugin_runtime_radiance_rpc": True,
        "plugin_runtime_radiance_direct": True,
        "plugin_runtime_radiance_kmod": True,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": True,
        "enable_libbacktrace": False,
    },
    "gemmini": {
        "target": "host",
        "config": "debug",
        "with_plugin": False,
        "plugin_compiler": True,
        "plugin_runtime": False,
        "build_compiler": True,
        "build_python_bindings": True,
        "build_samples": False,
        "build_tests": True,
        "enable_libbacktrace": True,
        "compiler_scope": "gemmini",
    },
    "npu": {
        "target": "host",
        "config": "debug",
        "with_plugin": False,
        "plugin_compiler": True,
        "plugin_runtime": False,
        "build_compiler": True,
        "build_python_bindings": True,
        "build_samples": False,
        "build_tests": True,
        "enable_libbacktrace": True,
        "compiler_scope": "npu",
    },
    "spacemit": {
        "target": "spacemit",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": True,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": True,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": True,
        "build_tests": False,
        "enable_libbacktrace": False,
    },
    "firesim": {
        "target": "firesim",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": True,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": True,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": True,
        "build_tests": False,
        "enable_libbacktrace": False,
    },
}


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        help=(
            "High-level user profile preset. "
            "Use this for normal workflows; advanced flags may still override details."
        ),
    )
    parser.add_argument("--target", choices=["host", "spacemit", "firesim"], default=None, help="Target platform.")
    parser.add_argument(
        "--config",
        choices=["debug", "release", "asan", "trace", "perf"],
        default=None,
        help="Build configuration type",
    )
    parser.add_argument("--cmake-target", help="Build specific CMake target (default: install)")
    parser.add_argument(
        "--with-plugin",
        action="store_true",
        help="Enable Merlin compiler+runtime plugins (legacy umbrella switch).",
    )
    parser.add_argument(
        "--plugin-compiler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Merlin compiler plugin targets (default follows --with-plugin).",
    )
    parser.add_argument(
        "--plugin-runtime",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Merlin runtime plugin integration (default follows --with-plugin).",
    )
    parser.add_argument(
        "--plugin-runtime-radiance",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Radiance HAL runtime plugin path (default: host+plugin only).",
    )
    parser.add_argument(
        "--plugin-runtime-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable runtime plugin samples subdir.",
    )
    parser.add_argument(
        "--plugin-runtime-benchmarks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable runtime plugin benchmarks subdir.",
    )
    parser.add_argument(
        "--plugin-runtime-radiance-tests",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Radiance runtime plugin tests.",
    )
    parser.add_argument(
        "--plugin-runtime-radiance-rpc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Radiance RPC-compat transport backend.",
    )
    parser.add_argument(
        "--plugin-runtime-radiance-direct",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Radiance direct-submit transport backend.",
    )
    parser.add_argument(
        "--plugin-runtime-radiance-kmod",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Radiance kmod transport backend.",
    )
    parser.add_argument(
        "--compiler-scope",
        choices=["all", "gemmini", "npu", "saturn", "spacemit", "none"],
        default=None,
        help=(
            "Limit compiler-plugin target registration scope. "
            "Only used when compiler plugin + compiler build are enabled."
        ),
    )
    parser.add_argument(
        "--build-compiler",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override IREE_BUILD_COMPILER for this build.",
    )
    parser.add_argument(
        "--build-python-bindings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override IREE_BUILD_PYTHON_BINDINGS for this build.",
    )
    parser.add_argument(
        "--build-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override IREE_BUILD_SAMPLES for this build.",
    )
    parser.add_argument(
        "--build-tests",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override IREE_BUILD_TESTS for this build.",
    )
    parser.add_argument(
        "--enable-libbacktrace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override IREE_ENABLE_LIBBACKTRACE for this build.",
    )
    parser.add_argument(
        "--offline-friendly",
        action="store_true",
        help=(
            "Prefer settings that avoid network fetches in CMake "
            "(equivalent to --no-build-compiler --no-build-python-bindings "
            "--no-enable-libbacktrace unless explicitly overridden)."
        ),
    )
    parser.add_argument(
        "--cmake-bin",
        default="cmake",
        help="CMake executable to use (default: cmake).",
    )
    parser.add_argument(
        "--use-system-cmake",
        action="store_true",
        help="Use /usr/bin/cmake instead of cmake from PATH.",
    )
    parser.add_argument(
        "--use-ccache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable ccache compiler launchers (default: enabled).",
    )
    parser.add_argument(
        "--cmake-arg",
        "--configure-custom-arg",
        action="append",
        dest="cmake_arg",
        default=[],
        help="Extra argument forwarded to CMake configure (repeatable).",
    )
    parser.add_argument(
        "--cmake-build-arg",
        "--build-custom-arg",
        action="append",
        dest="cmake_build_arg",
        default=[],
        help="Extra argument forwarded to CMake build command (repeatable).",
    )
    parser.add_argument(
        "--native-build-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the native build tool after '--' (repeatable).",
    )
    parser.add_argument("--clean", action="store_true", help="Delete build directory before building")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose build output")


def get_iree_version(iree_src: pathlib.Path) -> str:
    try:
        with (iree_src / "runtime" / "version.json").open() as f:
            return json.load(f).get("package-version", "unknown")
    except Exception:
        return "unknown"


def cmake_bool(value: bool) -> str:
    return "ON" if value else "OFF"


def resolve_bool(default_value: bool, override: bool | None) -> bool:
    return default_value if override is None else override


def apply_profile(args: argparse.Namespace) -> None:
    if not args.profile:
        return
    preset = PROFILE_PRESETS[args.profile]
    for key, value in preset.items():
        current_value = getattr(args, key)
        if current_value is None:
            setattr(args, key, value)
            continue
        # `with_plugin` uses store_true and defaults to False.
        if key == "with_plugin" and current_value is False and value is True:
            setattr(args, key, value)


def is_cmake_usable(cmake_path: str) -> bool:
    try:
        result = subprocess.run(
            [cmake_path, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def main(args: argparse.Namespace) -> int:
    apply_profile(args)

    if args.target is None:
        args.target = "host"
    if args.config is None:
        args.config = "debug"

    # 1. Setup Paths
    iree_src = utils.resolve_repo_path("third_party/iree_bar")
    plugin_src = utils.REPO_ROOT

    plugin_compiler_enabled = args.with_plugin
    plugin_runtime_enabled = args.with_plugin
    if args.plugin_compiler is not None:
        plugin_compiler_enabled = args.plugin_compiler
    if args.plugin_runtime is not None:
        plugin_runtime_enabled = args.plugin_runtime
    if args.plugin_runtime_radiance is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_samples is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_benchmarks is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_radiance_tests is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_radiance_rpc is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_radiance_direct is True:
        plugin_runtime_enabled = True
    if args.plugin_runtime_radiance_kmod is True:
        plugin_runtime_enabled = True

    with_any_plugin = plugin_compiler_enabled or plugin_runtime_enabled
    variant = "merlin" if with_any_plugin else "vanilla"

    runtime_radiance_enabled = resolve_bool(
        plugin_runtime_enabled and args.target == "host", args.plugin_runtime_radiance
    )
    runtime_samples_enabled = resolve_bool(plugin_runtime_enabled, args.plugin_runtime_samples)
    runtime_benchmarks_enabled = resolve_bool(False, args.plugin_runtime_benchmarks)
    runtime_radiance_tests_enabled = resolve_bool(
        runtime_radiance_enabled and args.target == "host", args.plugin_runtime_radiance_tests
    )
    runtime_radiance_backend_rpc = resolve_bool(True, args.plugin_runtime_radiance_rpc)
    runtime_radiance_backend_direct = resolve_bool(True, args.plugin_runtime_radiance_direct)
    runtime_radiance_backend_kmod = resolve_bool(True, args.plugin_runtime_radiance_kmod)

    if args.offline_friendly:
        if args.build_compiler is None:
            args.build_compiler = False
        if args.build_python_bindings is None:
            args.build_python_bindings = False
        if args.enable_libbacktrace is None:
            args.enable_libbacktrace = False

    default_build_compiler = args.target == "host"
    default_build_python_bindings = args.target == "host"
    default_build_samples = args.target in ["spacemit", "firesim"]
    default_build_tests = args.target == "host"
    default_enable_libbacktrace = args.target == "host"

    build_compiler = resolve_bool(default_build_compiler, args.build_compiler)
    build_python_bindings = resolve_bool(default_build_python_bindings, args.build_python_bindings)
    build_samples = resolve_bool(default_build_samples, args.build_samples)
    build_tests = resolve_bool(default_build_tests, args.build_tests)
    enable_libbacktrace = resolve_bool(default_enable_libbacktrace, args.enable_libbacktrace)

    cmake_bin = "/usr/bin/cmake" if args.use_system_cmake else args.cmake_bin
    if os.path.sep in cmake_bin:
        if not pathlib.Path(cmake_bin).exists():
            utils.eprint(f"❌ Error: CMake binary not found: {cmake_bin}")
            return 1
    else:
        resolved_cmake = shutil.which(cmake_bin)
        if not resolved_cmake:
            utils.eprint(f"❌ Error: CMake executable '{cmake_bin}' not found in PATH.")
            return 1
        cmake_bin = resolved_cmake
    if not is_cmake_usable(cmake_bin):
        fallback_cmake = "/usr/bin/cmake"
        if (
            not args.use_system_cmake
            and args.cmake_bin == "cmake"
            and pathlib.Path(fallback_cmake).exists()
            and is_cmake_usable(fallback_cmake)
        ):
            print("⚠️  Resolved cmake from PATH is not runnable " f"({cmake_bin}); falling back to {fallback_cmake}.")
            cmake_bin = fallback_cmake
        else:
            utils.eprint(
                "❌ Error: Selected CMake binary is not runnable: "
                f"{cmake_bin}. Try --use-system-cmake or --cmake-bin."
            )
            return 1

    get_iree_version(iree_src)

    # Clean structure: build/spacemit-merlin-perf
    build_name = f"{args.target}-{variant}-{args.config}"
    build_dir = utils.REPO_ROOT / "build" / build_name
    install_dir = build_dir / "install"

    print(f"🔧 Configuration: {args.target} | {args.config} | Plugin: {with_any_plugin}")
    if args.profile:
        print(f"🧭 Profile:      {args.profile}")
    print(
        "🧩 Plugin Split: "
        f"compiler={plugin_compiler_enabled} runtime={plugin_runtime_enabled} "
        f"runtime_radiance={runtime_radiance_enabled}"
    )
    if plugin_compiler_enabled:
        print(f"🎯 Compiler Scope: {args.compiler_scope or 'all'}")
    print(
        "📦 IREE Build: "
        f"compiler={build_compiler} python_bindings={build_python_bindings} "
        f"samples={build_samples} tests={build_tests} libbacktrace={enable_libbacktrace}"
    )
    if args.cmake_arg:
        print(f"🧱 Extra CMake Configure Args: {args.cmake_arg}")
    if args.cmake_build_arg or args.native_build_arg:
        print("🏗️  Extra CMake Build Args: " f"cmake={args.cmake_build_arg or []} native={args.native_build_arg or []}")
    print(f"🛠️  CMake:         {cmake_bin}")
    print(f"📂 Build Dir:     {build_dir}")
    print(f"📂 Install Dir:   {install_dir}")

    if args.clean and build_dir.exists():
        print("Cleaning build directory...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    ccache_path = shutil.which("ccache")
    use_ccache = bool(args.use_ccache and ccache_path)
    if args.use_ccache and not ccache_path:
        print("⚠️  ccache requested but not found in PATH; continuing without compiler launcher.")
    if use_ccache:
        ccache_dir = build_dir / ".ccache"
        ccache_tmp_dir = ccache_dir / "tmp"
        ccache_tmp_dir.mkdir(parents=True, exist_ok=True)
        env["CCACHE_DIR"] = str(ccache_dir)
        env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)

    # 2. Base CMake Flags
    cmake_args = [
        cmake_bin,
        "-G",
        "Ninja",
        f"-B{build_dir}",
        f"-S{iree_src}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DIREE_ENABLE_LLD=ON",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    if use_ccache:
        cmake_args.extend(
            [
                f"-DCMAKE_C_COMPILER_LAUNCHER={ccache_path}",
                f"-DCMAKE_CXX_COMPILER_LAUNCHER={ccache_path}",
            ]
        )
    else:
        cmake_args.extend(["-DCMAKE_C_COMPILER_LAUNCHER=", "-DCMAKE_CXX_COMPILER_LAUNCHER="])

    # 3. Config Specific Flags
    if args.config == "debug":
        cmake_args.extend(
            [
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DIREE_ENABLE_ASSERTIONS=ON",
                "-DIREE_ENABLE_ASAN=OFF",
                "-DCMAKE_CXX_FLAGS=-Wno-error=cpp -Wno-error=maybe-uninitialized -fno-omit-frame-pointer -fdebug-types-section -gz=none",
                "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer -fdebug-types-section -gz=none",
            ]
        )
    elif args.config == "asan":
        cmake_args.extend(
            [
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DIREE_ENABLE_ASAN=ON",
                "-DIREE_ENABLE_ASSERTIONS=ON",
            ]
        )
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
        cmake_args.extend(
            [
                f"-DCMAKE_BUILD_TYPE={build_type}",
                "-DIREE_ENABLE_ASSERTIONS=ON",
                "-DCMAKE_CXX_FLAGS=-Wno-error=cpp -fno-omit-frame-pointer -fdebug-types-section -gz=none",
                "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer -fdebug-types-section -gz=none",
            ]
        )
        if args.config == "perf":
            cmake_args.extend(["-DIREE_ENABLE_RUNTIME_TRACING=OFF", "-DIREE_ENABLE_CPUINFO=OFF"])

    elif args.config == "trace":
        cmake_args.extend(
            [
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DIREE_ENABLE_RUNTIME_TRACING=ON",
                "-DIREE_ENABLE_COMPILER_TRACING=ON",
                "-DIREE_TRACING_MODE=4",  # Tracy
                "-DIREE_ENABLE_ASSERTIONS=ON",
                "-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer",
                "-DCMAKE_C_FLAGS=-fno-omit-frame-pointer",
            ]
        )
        if args.target in ["spacemit", "firesim"]:
            cmake_args.append("-DTRACY_NO_POINTER_COMPRESSION=ON")

    # 4. Target Specific Logic

    # For cross-compilation targets, we must provide the path to the native host tools.
    # We ALWAYS use the 'release' build of the host tools for maximum compilation speed.
    if args.target != "host":
        # UPDATED: Reconstruct the name to match the flat build/target-variant-config structure
        # We always check for 'release' config for host tools.
        host_tools_variant = "merlin" if plugin_compiler_enabled else "vanilla"
        primary_host_name = f"host-{host_tools_variant}-release"
        fallback_host_name = "host-vanilla-release"

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
        cmake_args.extend(
            [
                "-DIREE_TARGET_BACKEND_DEFAULTS=OFF",
                "-DIREE_TARGET_BACKEND_LLVM_CPU=ON",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
            ]
        )

    elif args.target == "spacemit":
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
            default_tc = (
                utils.REPO_ROOT
                / "build_tools"
                / "riscv-tools-spacemit"
                / "spacemit-toolchain-linux-glibc-x86_64-v1.1.2"
            )
            if default_tc.exists():
                tc_root = str(default_tc)

        if not tc_root:
            utils.eprint("❌ Error: SpacemiT toolchain not found. Set RISCV_TOOLCHAIN_ROOT.")
            return 1

        cmake_args.extend(
            [
                "-DMERLIN_BUILD_SPACEMITX60=ON",
                f"-DCMAKE_TOOLCHAIN_FILE={iree_src}/build_tools/cmake/riscv.toolchain.cmake",
                "-DRISCV_CPU=linux-riscv_64",
                f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
                "-DCMAKE_C_FLAGS=-march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
                "-DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer -march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
                "-DIREE_ENABLE_CPUINFO=ON",
            ]
        )

    elif args.target == "firesim":
        tc_file = utils.REPO_ROOT / "build_tools" / "firesim" / "riscv_firesim.toolchain.cmake"
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
            tc_root = str(
                utils.REPO_ROOT / "build_tools" / "riscv-tools-iree" / "toolchain" / "clang" / "linux" / "RISCV"
            )

        cmake_args.extend(
            [
                "-DMERLIN_BUILD_SATURN_OPU=ON",
                f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
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
            ]
        )

    # 5. Plugin Activation
    if with_any_plugin:
        cmake_args.append(f"-DIREE_CMAKE_PLUGIN_PATHS={plugin_src}")

    compiler_scope = args.compiler_scope
    if compiler_scope is None:
        compiler_scope = "all"

    compiler_target_gemmini = compiler_scope in ["all", "gemmini"]
    compiler_target_npu = compiler_scope in ["all", "npu"]
    compiler_target_saturn = compiler_scope in ["all", "saturn"]
    compiler_target_spacemit = compiler_scope in ["all", "spacemit"]

    if plugin_compiler_enabled and build_compiler:
        cmake_args.extend(
            [
                "-DMERLIN_ENABLE_CORE=ON",
                f"-DMERLIN_ENABLE_TARGET_GEMMINI={cmake_bool(compiler_target_gemmini)}",
                f"-DMERLIN_ENABLE_TARGET_NPU={cmake_bool(compiler_target_npu)}",
                f"-DMERLIN_ENABLE_TARGET_SATURN={cmake_bool(compiler_target_saturn)}",
                f"-DMERLIN_ENABLE_TARGET_SPACEMIT={cmake_bool(compiler_target_spacemit)}",
            ]
        )
    elif plugin_compiler_enabled and not build_compiler:
        print("ℹ️  Compiler plugin requested but IREE_BUILD_COMPILER=OFF; skipping compiler plugin target toggles.")
    elif with_any_plugin and build_compiler:
        cmake_args.extend(
            [
                "-DMERLIN_ENABLE_CORE=OFF",
                "-DMERLIN_ENABLE_TARGET_GEMMINI=OFF",
                "-DMERLIN_ENABLE_TARGET_NPU=OFF",
                "-DMERLIN_ENABLE_TARGET_SATURN=OFF",
                "-DMERLIN_ENABLE_TARGET_SPACEMIT=OFF",
            ]
        )

    if plugin_runtime_enabled:
        cmake_args.extend(
            [
                f"-DMERLIN_RUNTIME_ENABLE_SAMPLES={cmake_bool(runtime_samples_enabled)}",
                f"-DMERLIN_RUNTIME_ENABLE_BENCHMARKS={cmake_bool(runtime_benchmarks_enabled)}",
                f"-DMERLIN_RUNTIME_ENABLE_HAL_RADIANCE={cmake_bool(runtime_radiance_enabled)}",
                f"-DMERLIN_ENABLE_HAL_RADIANCE={cmake_bool(runtime_radiance_enabled)}",
            ]
        )
        if runtime_radiance_enabled:
            cmake_args.extend(
                [
                    f"-DMERLIN_HAL_RADIANCE_BUILD_TESTS={cmake_bool(runtime_radiance_tests_enabled)}",
                    f"-DMERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT={cmake_bool(runtime_radiance_backend_rpc)}",
                    f"-DMERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT={cmake_bool(runtime_radiance_backend_direct)}",
                    f"-DMERLIN_HAL_RADIANCE_ENABLE_KMOD={cmake_bool(runtime_radiance_backend_kmod)}",
                ]
            )
        else:
            cmake_args.append("-DMERLIN_HAL_RADIANCE_BUILD_TESTS=OFF")
    elif with_any_plugin:
        cmake_args.extend(
            [
                "-DMERLIN_RUNTIME_ENABLE_SAMPLES=OFF",
                "-DMERLIN_RUNTIME_ENABLE_BENCHMARKS=OFF",
                "-DMERLIN_RUNTIME_ENABLE_HAL_RADIANCE=OFF",
                "-DMERLIN_ENABLE_HAL_RADIANCE=OFF",
                "-DMERLIN_HAL_RADIANCE_BUILD_TESTS=OFF",
            ]
        )

    # 6. Generic build toggles (last assignment wins over target defaults)
    cmake_args.extend(
        [
            f"-DIREE_BUILD_COMPILER={cmake_bool(build_compiler)}",
            f"-DIREE_BUILD_PYTHON_BINDINGS={cmake_bool(build_python_bindings)}",
            f"-DIREE_BUILD_SAMPLES={cmake_bool(build_samples)}",
            f"-DIREE_BUILD_TESTS={cmake_bool(build_tests)}",
            f"-DIREE_ENABLE_LIBBACKTRACE={cmake_bool(enable_libbacktrace)}",
        ]
    )
    cmake_args.extend(args.cmake_arg)

    # 7. Configure & Build Main Target
    if utils.run(cmake_args, dry_run=args.dry_run, env=env) != 0:
        return 1

    target_arg = args.cmake_target if args.cmake_target else "install"
    build_cmd = [cmake_bin, "--build", str(build_dir), "--target", target_arg]
    build_cmd.extend(args.cmake_build_arg)
    if args.verbose:
        build_cmd.append("--verbose")
    if args.native_build_arg:
        build_cmd.append("--")
        build_cmd.extend(args.native_build_arg)

    if utils.run(build_cmd, dry_run=args.dry_run, env=env) != 0:
        return 1

    # 8. Build Extra Tools (Host Only)
    # This replicates the logic from `build_debug_asan.sh`
    if args.target == "host" and not args.cmake_target:
        print(">> Building extra LLVM tools (llvm-mca, llvm-objdump)...")
        extra_tools_cmd = [cmake_bin, "--build", str(build_dir), "--target", "llvm-mca", "llvm-objdump"]
        extra_tools_cmd.extend(args.cmake_build_arg)
        if args.verbose:
            extra_tools_cmd.append("--verbose")
        if args.native_build_arg:
            extra_tools_cmd.append("--")
            extra_tools_cmd.extend(args.native_build_arg)
        if utils.run(extra_tools_cmd, dry_run=args.dry_run, env=env) != 0:
            return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
