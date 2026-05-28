"""Build-profile presets — the canonical `PROFILE_PRESETS` dict + `apply_profile`.

Each preset is a dict of cmake-variable defaults that `build/cli.py:main()`
composes into the final cmake invocation. Profile names are the public API
(`./merlin build --profile <name>`); don't rename casually.
"""

from __future__ import annotations

import argparse


PROFILE_PRESETS: dict[str, dict[str, object]] = {
    "vanilla": {
        "target": "host",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": False,
        "build_compiler": True,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": True,
    },
    "full-plugin": {
        "target": "host",
        "config": "release",
        "with_plugin": True,
        "plugin_compiler": True,
        "plugin_runtime": True,
        "plugin_runtime_radiance": True,
        "plugin_runtime_samples": True,
        "plugin_runtime_benchmarks": False,
        "plugin_runtime_radiance_tests": True,
        "build_compiler": True,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": True,
        "compiler_scope": "all",
    },
    "qnn-compiler": {
        "target": "host",
        "config": "release",
        "with_plugin": True,
        "plugin_compiler": True,
        "plugin_runtime": False,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "plugin_runtime_radiance_tests": False,
        "build_compiler": True,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": True,
        "compiler_scope": "none",
    },
    "radiance": {
        "target": "host",
        "config": "release",
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
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": True,
        "plugin_runtime": False,
        "build_compiler": True,
        "build_python_bindings": True,
        "build_samples": False,
        # build_tests=False: this profile is for Gemmini compiler plugin
        # development. Upstream ucb-bar/main hard-errors when CTS
        # testdata targets request an unregistered backend (e.g. vmvx
        # under runtime/src/iree/hal/drivers/local_sync/cts/testdata_vmvx);
        # the gemmini profile doesn't enable vmvx so those steps would
        # always fail. Runtime CTS tests are independent of gemmini
        # plugin work — use --profile vanilla or --profile full-plugin
        # if you need them.
        "build_tests": False,
        "enable_libbacktrace": True,
        "compiler_scope": "gemmini",
    },
    "npu": {
        "target": "host",
        "config": "release",
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
    "package-host": {
        "target": "host",
        "config": "perf",
        "with_plugin": True,
        "plugin_compiler": True,
        "plugin_runtime": True,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "plugin_runtime_radiance_tests": False,
        "build_compiler": True,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": False,
        "compiler_scope": "all",
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
    "qrb5165": {
        "target": "qrb5165",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": True,
        "plugin_runtime_qnn": True,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": True,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": False,
    },
    "package-spacemit": {
        "target": "spacemit",
        "config": "perf",
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
    "zephyr": {
        # Lean Zephyr runtime: IREE_ENABLE_THREADING=OFF.
        #
        # Use this for the production samples (merlin_model_runner,
        # merlin_hetero_runner) where each hart owns its own iree_vm_context
        # backed by iree_hal_local_sync. With no shared IREE state across
        # harts, iree_slim_mutex becomes a no-op stub, the proactor pool
        # runner thread is not spawned, and iree_thread_create/iree_futex_*
        # are unused. ~few % lower per-dispatch overhead than the
        # threading-on build.
        #
        # Output: build/zephyr-vanilla-release/
        "target": "zephyr",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": False,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": False,
        # Profile-controlled IREE flags. None = use the build.py target
        # default; True/False = override.
        "iree_threading": False,
    },
    "zephyr-task": {
        # Heavier Zephyr runtime: IREE_ENABLE_THREADING=ON.
        #
        # Reserved for samples that want iree_hal_local_task (work-stealing
        # task scheduler across harts in a SINGLE inference). Pulls in
        # iree_task_*, iree_futex_*, iree_thread_* + the async proactor
        # pool runner thread, and requires that the consuming Zephyr app
        # provide the iree_thread_zephyr.c / iree_futex_zephyr.c bridge.
        #
        # Output: build/zephyr-task-release/  (different dir from the
        # lean variant so both archives can coexist).
        "target": "zephyr",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": False,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": False,
        "iree_threading": True,
    },
    "radiance_muon": {
        # Cross-compile a Radiance/Muon GPU kernel via the vendored llvm-muon
        # toolchain. *Not* an IREE build -- the source dir is
        # `build_tools/radiance/` (driver CMakeLists), and the only inputs
        # are the kernel.cpp from $RADIANCE_KERNELS_ROOT or a user-supplied
        # --kernel-dir. Output: build/radiance_muon-vanilla-release/<name>.radiance.elf.
        "target": "radiance_muon",
        "config": "release",
        "with_plugin": False,
        "plugin_compiler": False,
        "plugin_runtime": False,
        "plugin_runtime_radiance": False,
        "plugin_runtime_samples": False,
        "plugin_runtime_benchmarks": False,
        "build_compiler": False,
        "build_python_bindings": False,
        "build_samples": False,
        "build_tests": False,
        "enable_libbacktrace": False,
    },
    "package-firesim": {
        "target": "firesim",
        "config": "perf",
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


def apply_profile(args: argparse.Namespace) -> None:
    if not args.profile:
        return
    preset = PROFILE_PRESETS[args.profile]
    for key, value in preset.items():
        # Profile keys that don't have a corresponding argparse arg are
        # attached to args as plain attributes (no default in argparse).
        if not hasattr(args, key):
            setattr(args, key, value)
            continue
        current_value = getattr(args, key)
        if current_value is None:
            setattr(args, key, value)
            continue
        # `with_plugin` uses store_true and defaults to False.
        if key == "with_plugin" and current_value is False and value is True:
            setattr(args, key, value)
