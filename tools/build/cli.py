#!/usr/bin/env python3
# tools/build.py
"""Backs `./merlin build`: configures and builds Merlin host tools and target
runtimes via cmake/ninja, with curated `--profile` presets (vanilla, full-plugin,
spacemit, firesim, gemmini, etc.).

See docs/how_to/use_build_py.md for profile reference and examples.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

import utils


# Split-out modules (extracted from this file)
from build.presets import PROFILE_PRESETS, apply_profile
from build.cmake import (cmake_bool, resolve_bool, is_cmake_usable,
                         is_darwin_host, make_common_cmake_flags,
                         get_iree_version)
from build.packaging import package_dist, maybe_install
from build.radiance import build_radiance_muon


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        help=(
            "High-level user profile preset. "
            "Use this for normal workflows; advanced flags may still override details."
        ),
    )
    parser.add_argument(
        "--target",
        choices=["host", "spacemit", "qrb5165", "firesim", "zephyr", "radiance_muon"],
        default=None,
        help="Target platform.",
    )
    parser.add_argument(
        "--kernel-dir",
        default=None,
        help=(
            "For --target radiance_muon: absolute path to a directory "
            "containing kernel.cpp (and optionally host.cpp). Defaults to "
            "$RADIANCE_KERNELS_ROOT/kernels/vecadd."
        ),
    )
    parser.add_argument(
        "--kernel-name",
        default=None,
        help=(
            "For --target radiance_muon: basename of the produced ELF "
            "(<name>.radiance.elf). Default: derived from --kernel-dir."
        ),
    )
    parser.add_argument(
        "--kernel-body-obj",
        default=None,
        help=(
            "For --target radiance_muon (manifest mode): path to a "
            "precompiled Muon kernel-body .o file (typically produced by "
            "kernels/core/precompile.py from the Radiance manifest). When "
            "set, the wrapper template declares the kernel as `extern \"C\"` "
            "and the body .o is linked into kernel.radiance.elf at link time."
        ),
    )
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
        "--plugin-runtime-qnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable QNN HAL runtime plugin path for QRB5165 profiling.",
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
        choices=["all", "gemmini", "npu", "saturn", "spacemit", "radiance", "none"],
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
        "--enable-tracy",
        action="store_true",
        default=False,
        help=(
            "Enable Tracy runtime tracing (IREE_ENABLE_RUNTIME_TRACING=ON, "
            "IREE_TRACING_MODE=4). Compatible with any --config."
        ),
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



def main(args: argparse.Namespace) -> int:
    apply_profile(args)

    if args.target is None:
        args.target = "host"
    if args.config is None:
        # 2026-05-08: default to release. Debug builds balloon to 150GB+
        # of DWARF and have hit disk-full failures (host-merlin-debug
        # alone hit 165GB during the mxGemmini Phase B work). Release is
        # ~5GB. Pass --config=debug explicitly if you genuinely need
        # symbolicated stack traces.
        args.config = "release"

    # radiance_muon is a thin wrap around the radiance-kernels Makefile flow,
    # not an IREE build. Handle it before any of the IREE-specific machinery.
    if args.target == "radiance_muon":
        return build_radiance_muon(args)

    package_profile = args.profile in {"package-host", "package-spacemit", "package-firesim"}
    package_runtime_samples = args.profile in {"package-spacemit", "package-firesim"}

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
    if args.plugin_runtime_qnn is True:
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
    runtime_qnn_enabled = resolve_bool(
        plugin_runtime_enabled and args.target == "qrb5165",
        args.plugin_runtime_qnn,
    )

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

    # Clean structure: build/spacemit-merlin-perf. QRB5165 is runtime-only
    # today; keep the established build/qrb5165-runtime-release directory so
    # board staging scripts do not need to chase a one-off hand config.
    if args.profile == "qnn-compiler":
        build_name = "host-merlin-release-qrb"
    elif args.target == "qrb5165" and plugin_runtime_enabled and not plugin_compiler_enabled:
        build_name = f"{args.target}-runtime-{args.config}"
    elif args.profile == "zephyr-task":
        # Separate output dir from the threading-off "zephyr" profile so
        # both archives can coexist. Consumer apps pick which one to link
        # via MERLIN_BUILD_DIR at west-build time.
        build_name = f"{args.target}-task-{args.config}"
    else:
        build_name = f"{args.target}-{variant}-{args.config}"
    build_dir = utils.REPO_ROOT / "build" / build_name
    install_dir = build_dir / "install"

    dist_dir = utils.REPO_ROOT / "dist"
    dist_name = build_name

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
    if package_profile:
        print(f"📦 Dist Dir:      {dist_dir}")
        print(f"📦 Dist Name:     {dist_name}")

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
    iree_enable_lld = "OFF" if is_darwin_host() or args.target == "qrb5165" else "ON"

    cmake_args = [
        cmake_bin,
        "-G",
        "Ninja",
        f"-B{build_dir}",
        f"-S{iree_src}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        f"-DIREE_ENABLE_LLD={iree_enable_lld}",
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
        common_c_flags, common_cxx_flags = make_common_cmake_flags(
            cxx_warn_cpp=True,
            cxx_warn_maybe_uninitialized=True,
        )
        cmake_args.extend(
            [
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DIREE_ENABLE_ASSERTIONS=ON",
                "-DIREE_ENABLE_ASAN=OFF",
                f"-DCMAKE_CXX_FLAGS={common_cxx_flags}",
                f"-DCMAKE_C_FLAGS={common_c_flags}",
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
        # Attempt to inject LD_PRELOAD for Linux ASan only.
        if not is_darwin_host():
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
        build_type = "Release"
        common_c_flags, common_cxx_flags = make_common_cmake_flags(cxx_warn_cpp=True)
        cmake_args.extend([f"-DCMAKE_BUILD_TYPE={build_type}", "-DIREE_ENABLE_ASSERTIONS=ON"])
        if args.target != "qrb5165":
            cmake_args.extend(
                [
                    f"-DCMAKE_CXX_FLAGS={common_cxx_flags}",
                    f"-DCMAKE_C_FLAGS={common_c_flags}",
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
        if args.target in ["spacemit", "firesim", "zephyr"]:
            cmake_args.append("-DTRACY_NO_POINTER_COMPRESSION=ON")

    # --enable-tracy: overlay runtime tracing onto any config.
    if args.enable_tracy:
        # Mode 1 = instrumentation zones + log messages (no alloc tracking,
        # no callstacks). Higher modes crash on RISC-V due to pointer
        # compression and callstack issues in Tracy's server code.
        print("  🔬 Enabling Tracy runtime tracing (IREE_TRACING_MODE=1)")
        cmake_args.extend(
            [
                "-DIREE_ENABLE_RUNTIME_TRACING=ON",
                "-DIREE_TRACING_MODE=1",
            ]
        )
        if args.target in ["spacemit", "firesim", "zephyr"]:
            # RISC-V 64-bit uses address ranges incompatible with Tracy's
            # default 48-bit pointer compression (PackPointer assert).
            cmake_args.append("-DTRACY_NO_POINTER_COMPRESSION=ON")

    # 4. Target Specific Logic

    # For cross-compilation targets, we must provide the path to native host tools.
    # Prefer Merlin host tools when this build enables any Merlin plugin path,
    # then fall back to vanilla host tools.
    if args.target != "host":
        preferred_host_names: list[str] = []

        if with_any_plugin:
            preferred_host_names.extend(
                [
                    f"host-merlin-{args.config}",
                    "host-merlin-release",
                    "host-merlin-debug",
                ]
            )

        preferred_host_names.extend(
            [
                f"host-vanilla-{args.config}",
                "host-vanilla-release",
                "host-vanilla-debug",
            ]
        )

        # Also fall back to host-merlin-* if no vanilla host build is present.
        # The cross-build only needs `iree-compile` (and friends) for ELF
        # embedding — host-merlin-* has the same binaries as host-vanilla-*
        # plus the plugin compiler stuff that the cross-build ignores.
        if "host-merlin-release" not in preferred_host_names:
            preferred_host_names.extend(
                [
                    f"host-merlin-{args.config}",
                    "host-merlin-release",
                    "host-merlin-debug",
                ]
            )

        host_bin_dir = None
        selected_candidate = None

        for candidate_name in preferred_host_names:
            candidate_dirs = [
                utils.REPO_ROOT / "build" / candidate_name / "install" / "bin",
                utils.REPO_ROOT / "build" / candidate_name / "tools",
            ]
            for candidate_dir in candidate_dirs:
                if not (candidate_dir / "iree-compile").exists():
                    continue
                host_bin_dir = candidate_dir
                selected_candidate = candidate_name
                break
            if host_bin_dir:
                break

        if host_bin_dir:
            if selected_candidate != preferred_host_names[0]:
                print(f"ℹ️  Note: Using fallback host tools from {host_bin_dir}")
            cmake_args.append(f"-DIREE_HOST_BIN_DIR={host_bin_dir}")
        else:
            print("❌ Error: No host tools found for cross compilation.")
            print("   Checked:")
            for name in preferred_host_names:
                print(f"     build/{name}/install/bin")
                print(f"     build/{name}/tools")
            print("   Please build host tools first, for example:")
            print("   python3 tools/build.py --profile package-host")
            return 1

    if args.target == "host":
        cmake_args.extend(
            [
                "-DIREE_TARGET_BACKEND_DEFAULTS=OFF",
                "-DIREE_TARGET_BACKEND_LLVM_CPU=ON",
                "-DIREE_TARGET_BACKEND_VMVX=OFF",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
                # QRB5165 data-flow wrappers are cross-compiled from the host
                # compiler to AArch64 embedded ELF. Keep AArch64 enabled in
                # Merlin host compiler builds so the QRB runtime and wrapper
                # VMFBs can be built from the same IREE revision.
                "-DIREE_DEFAULT_CPU_LLVM_TARGETS=X86;RISCV;AArch64",
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

        toolchain_file = iree_src / "build_tools" / "cmake" / "riscv.toolchain.cmake"
        if not toolchain_file.exists():
            toolchain_file = iree_src / "build_tools" / "cmake" / "linux_riscv64.cmake"
        if not toolchain_file.exists():
            utils.eprint("❌ Error: RISC-V CMake toolchain file not found in third_party/iree_bar/build_tools/cmake.")
            return 1

        cmake_args.extend(
            [
                "-DMERLIN_BUILD_SPACEMITX60=ON",
                f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
                "-DRISCV_CPU=linux-riscv_64",
                f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
                "-DCMAKE_C_FLAGS=" "-march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
                "-DCMAKE_CXX_FLAGS="
                "-fno-omit-frame-pointer"
                " -march=rv64gc_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause -mabi=lp64d",
                "-DIREE_ENABLE_CPUINFO=ON",
            ]
        )

    elif args.target == "qrb5165":
        tc_file = utils.REPO_ROOT / "build_tools" / "qualcomm_qrb5165" / "aarch64_qrb5165.toolchain.cmake"
        qnn_sdk_root = pathlib.Path(
            os.environ.get(
                "QNN_SDK_ROOT",
                "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326",
            )
        )
        if not tc_file.is_file():
            utils.eprint(f"❌ Error: QRB5165 toolchain file not found: {tc_file}")
            return 1
        if runtime_qnn_enabled and not qnn_sdk_root.is_dir():
            utils.eprint(f"❌ Error: QNN_SDK_ROOT not found: {qnn_sdk_root}")
            return 1
        cmake_args.extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
                f"-DQNN_SDK_ROOT={qnn_sdk_root}",
                "-DMERLIN_BUILD_XPU_RT_RUNNER=ON",
                "-DMERLIN_BUILD_QRB5165=ON",
                "-DIREE_ENABLE_WERROR_FLAG=OFF",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
            ]
        )

    elif args.target == "firesim":
        tc_file = utils.REPO_ROOT / "build_tools" / "firesim" / "riscv_firesim.toolchain.cmake"
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
            tc_root = str(
                utils.REPO_ROOT / "build_tools" / "riscv-tools-iree" / "toolchain" / "clang" / "linux" / "RISCV"
            )

        # The FireSim toolchain file expects these as environment variables,
        # not only as CMake cache entries.
        env["RISCV_TOOLCHAIN_ROOT"] = tc_root
        env.setdefault("RISCV", tc_root)

        # Bare-metal CPU feature bitmask for ukernel dispatch.
        # V=0x01, ZVFHMIN=0x02, ZVFH=0x04, XSMTVDOT=0x08, XOPU=0x10
        bare_metal_cpu_features = "0x11"  # V + XOPU (Saturn OPU)

        cmake_args.extend(
            [
                "-DMERLIN_BUILD_SATURN_OPU=ON",
                f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
                f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
                "-DMERLIN_TOOLCHAIN_PROFILE=bare-metal",
                f"-DIREE_RISCV_BARE_METAL_FEATURES={bare_metal_cpu_features}",
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

    elif args.target == "zephyr":
        # Cross-compile the IREE runtime as static libraries for a Zephyr
        # application on chipyard_riscv64. We deliberately do NOT fork
        # IREE's platform layer; instead we reuse IREE_PLATFORM_GENERIC
        # (single-threaded, no synchronization) and let the consuming
        # Zephyr app drive multi-hart parallelism via k_thread. The
        # toolchain file's `zephyr` profile differs from `bare-metal`
        # only in skipping htif.ld + htif-nano.spec (Zephyr supplies its
        # own linker script).
        tc_file = utils.REPO_ROOT / "build_tools" / "firesim" / "riscv_firesim.toolchain.cmake"
        tc_root = os.environ.get("RISCV_TOOLCHAIN_ROOT")
        if not tc_root:
            tc_root = str(
                utils.REPO_ROOT / "build_tools" / "riscv-tools-iree" / "toolchain" / "clang" / "linux" / "RISCV"
            )
        env["RISCV_TOOLCHAIN_ROOT"] = tc_root
        env.setdefault("RISCV", tc_root)

        # Match the Rocket Vector pipe (zvl128b). No XOPU/Gemmini bits.
        bare_metal_cpu_features = "0x01"  # V only

        cmake_args.extend(
            [
                f"-DCMAKE_TOOLCHAIN_FILE={tc_file}",
                f"-DRISCV_TOOLCHAIN_ROOT={tc_root}",
                "-DMERLIN_TOOLCHAIN_PROFILE=zephyr",
                f"-DIREE_RISCV_BARE_METAL_FEATURES={bare_metal_cpu_features}",
                "-DIREE_ARCH=riscv_64",
                # IREE_ENABLE_THREADING is profile-controlled (see
                # PROFILE_PRESETS["zephyr"] vs ["zephyr-task"]):
                #   zephyr      → OFF  (lean: iree_slim_mutex no-op, no
                #                        proactor pool runner thread,
                #                        iree_thread_*/iree_futex_* unused)
                #   zephyr-task → ON   (supports iree_hal_local_task; the
                #                        Zephyr app supplies the k_thread /
                #                        k_mutex+k_condvar bridges at link)
                f"-DIREE_ENABLE_THREADING={'ON' if getattr(args, 'iree_threading', False) else 'OFF'}",
                "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
                "-DIREE_HAL_DRIVER_LOCAL_SYNC=ON",
                # iree_hal_drivers_local_task is always BUILT (because
                # runtime/src/iree/runtime/CMakeLists.txt unconditionally
                # references iree::task::api in its dependency closure),
                # but at runtime it can only be REGISTERED when
                # IREE_ENABLE_THREADING=ON. With threading off the local-task
                # archives are present but contain empty TUs (gated on
                # IREE_ENABLE_THREADING in IREE's own CMake).
                "-DIREE_HAL_DRIVER_LOCAL_TASK=ON",
                "-DIREE_HAL_EXECUTABLE_LOADER_DEFAULTS=OFF",
                "-DIREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF=ON",
                "-DIREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS=OFF",
                "-DIREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF=ON",
                # Disable file I/O surface: Zephyr apps embed bytecode via
                # generate_inc_file_for_target rather than reading a path.
                "-DIREE_FILE_IO_ENABLE=OFF",
                # Saturn-only ukernels disabled on this generic-RVV target.
                "-DIREE_UK_BUILD_RISCV_64_ZVFH=OFF",
                "-DIREE_UK_BUILD_RISCV_64_ZVFHMIN=OFF",
                "-DCMAKE_SKIP_RPATH=ON",
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
    # `radiance` is opt-in via --compiler-scope=radiance OR the umbrella `all`.
    compiler_target_radiance = compiler_scope in ["all", "radiance"]

    if plugin_compiler_enabled and build_compiler:
        cmake_args.extend(
            [
                "-DMERLIN_ENABLE_CORE=ON",
                f"-DMERLIN_ENABLE_TARGET_GEMMINI={cmake_bool(compiler_target_gemmini)}",
                f"-DMERLIN_ENABLE_TARGET_NPU={cmake_bool(compiler_target_npu)}",
                f"-DMERLIN_ENABLE_TARGET_SATURN={cmake_bool(compiler_target_saturn)}",
                f"-DMERLIN_ENABLE_TARGET_SPACEMIT={cmake_bool(compiler_target_spacemit)}",
                f"-DMERLIN_ENABLE_TARGET_RADIANCE={cmake_bool(compiler_target_radiance)}",
            ]
        )
        if args.profile == "qnn-compiler":
            cmake_args.extend(
                [
                    "-DMERLIN_BUILD_QNN_TARGET=ON",
                    "-DMERLIN_RUNTIME_ENABLE_HAL_QNN=OFF",
                    "-DIREE_EXTERNAL_HAL_DRIVERS=",
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
                "-DMERLIN_ENABLE_TARGET_RADIANCE=OFF",
            ]
        )

    if plugin_runtime_enabled:
        cmake_args.extend(
            [
                f"-DMERLIN_RUNTIME_ENABLE_SAMPLES={cmake_bool(runtime_samples_enabled)}",
                f"-DMERLIN_RUNTIME_ENABLE_BENCHMARKS={cmake_bool(runtime_benchmarks_enabled)}",
                f"-DMERLIN_RUNTIME_ENABLE_HAL_RADIANCE={cmake_bool(runtime_radiance_enabled)}",
                f"-DMERLIN_ENABLE_HAL_RADIANCE={cmake_bool(runtime_radiance_enabled)}",
                f"-DMERLIN_RUNTIME_ENABLE_HAL_QNN={cmake_bool(runtime_qnn_enabled)}",
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
            "-DIREE_BUILD_BINDINGS_TFLITE=OFF",
            "-DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF",
            "-DIREE_BUILD_ALL_CHECK_TEST_MODULES=OFF",
            # Disable LLVM valgrind support unconditionally.  CMake's
            # check_include_file detects the header via conda's CFLAGS
            # (-I${CONDA_PREFIX}/include) but the LLVM build itself does not
            # add that include path, causing a compile error in Valgrind.cpp.
            "-DHAVE_VALGRIND_VALGRIND_H=OFF",
        ]
    )
    cmake_args.extend(args.cmake_arg)

    # 7. Configure & Build Main Target
    if utils.run(cmake_args, dry_run=args.dry_run, env=env) != 0:
        return 1

    # The `zephyr` profile builds a curated set of leaf targets rather than
    # the full `install` closure: IREE's `install` includes targets
    # (iree_base_internal_csprng, iree_async_util_signal, etc.) that have
    # hard-coded Linux/Apple/BSD platform assumptions which fail to compile
    # under IREE_PLATFORM_GENERIC + newlib bare-metal. The Zephyr module
    # only links the subset listed below, so we just build that closure.
    #
    # Symbols and the targets that supply them (validated by linking the
    # chipyard merlin/model_benchmark Zephyr application):
    #   iree_async_proactor_pool_*  -> iree_async_util_proactor_pool
    #   iree_async_semaphore_*      -> iree_async_async (semaphore.c)
    #   iree_async_frontier_tracker -> iree_async_async (frontier_tracker.c)
    #   iree_hal_deferred_command_buffer_*, iree_hal_device_queue_emulated_*,
    #   iree_hal_file_*, iree_hal_platform_query_numa_distance ->
    #     iree_hal_utils_{deferred_command_buffer,queue_emulation,
    #     file_cache,file_registry,platform_topology}
    #   iree_io_file_handle_*       -> iree_io_file_handle
    #   iree_slim_mutex_*           -> iree_base_threading_threading
    #   iree_hal_executable_plugin_manager_* ->
    #     iree_hal_local_plugins_registration_registration
    #   iree_hal_executable_infer_elf_format -> iree_hal_local_executable_format
    #   flatcc_verify_*             -> flatcc_parsing (third_party)
    # Local-task targets are only buildable when IREE_ENABLE_THREADING=ON.
    # With threading=OFF, iree/task/CMakeLists.txt exposes stub interface
    # libraries (no ninja target), so we drop them from the explicit
    # target list and rely on iree_hal_drivers_local_sync_sync_driver
    # alone for the runtime HAL closure.
    _iree_threading = getattr(args, "iree_threading", False)
    _maybe_task_targets = [
        "iree_hal_drivers_local_task_task_driver",
        "iree_hal_drivers_local_task_registration_registration",
        "iree_task_task",
        "iree_task_api",
    ] if _iree_threading else []
    ZEPHYR_TARGETS = [
        # Public API (transitive over local-sync HAL closure)
        "iree_modules_hal_hal",
        "iree_hal_drivers_local_sync_sync_driver",
        # Local-task HAL driver and its dependencies (multi-hart executor;
        # Zephyr app supplies iree_thread_* + iree_futex_* impls).
        # Only included when --profile zephyr-task (threading on).
        *_maybe_task_targets,
        "iree_hal_local_loaders_embedded_elf_loader",
        "iree_hal_local_executable_format",
        "iree_hal_local_plugins_registration_registration",
        "iree_hal_local_executable_plugin_manager",
        "iree_hal_utils_resource_set",
        "iree_vm_bytecode_module",
        # iree_async semaphore + frontier tracker + util/proactor_pool +
        # platform abstraction + thread runner factory
        "iree_async_async",
        "iree_async_platform",
        "iree_async_util_proactor_pool",
        "iree_async_util_proactor_thread",
        "iree_async_util_proactor_thread_runner",
        # iree_io for file_handle wrapping (referenced by VM bytecode loader)
        "iree_io_file_handle",
        # threading: iree_slim_mutex_* lives in base/threading
        "iree_base_threading_threading",
        # hal/utils: deferred command buffer, queue emulation, file IO,
        # platform topology (numa distance)
        "iree_hal_utils_deferred_command_buffer",
        "iree_hal_utils_queue_emulation",
        "iree_hal_utils_file_cache",
        "iree_hal_utils_file_transfer",
        "iree_hal_utils_files",
        "iree_hal_utils_platform_topology",
        # flatcc verifier (used by VM bytecode module verification)
        "flatcc_parsing",
    ]

    if args.cmake_target:
        targets = [args.cmake_target]
    elif args.target == "zephyr":
        targets = ZEPHYR_TARGETS
    elif package_profile:
        targets = ["all"]
    else:
        targets = ["install"]

    build_cmd = [cmake_bin, "--build", str(build_dir)]
    for t in targets:
        build_cmd.extend(["--target", t])
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
    if args.target == "host" and not args.cmake_target and not package_profile:
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

    if package_profile:
        print(">> Stripping installed binaries...")
        if (
            maybe_install(
                cmake_bin=cmake_bin,
                build_dir=build_dir,
                strip_install=True,
                dry_run=args.dry_run,
                env=env,
            )
            != 0
        ):
            return 1

        try:
            artifact_path = package_dist(
                build_dir=build_dir,
                install_dir=install_dir,
                dist_dir=dist_dir,
                dist_name=dist_name,
                include_runtime_samples=package_runtime_samples,
            )
        except FileNotFoundError as e:
            utils.eprint(f"❌ Error: {e}")
            return 1

        print(f"✅ Packaged artifact: {artifact_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    setup_parser(parser)
    sys.exit(main(parser.parse_args()))
