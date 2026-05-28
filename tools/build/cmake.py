"""CMake invocation helpers — small primitives used across the build flow.

Includes `cmake_bool`, `resolve_bool`, `is_cmake_usable`, `is_darwin_host`,
`make_common_cmake_flags`, and `get_iree_version`.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def get_iree_version(iree_src: pathlib.Path) -> str:
    try:
        with (iree_src / "runtime" / "version.json").open() as f:
            return json.load(f).get("package-version", "unknown")
    except FileNotFoundError:
        return "unknown"


def cmake_bool(value: bool) -> str:
    return "ON" if value else "OFF"


def resolve_bool(default_value: bool, override: bool | None) -> bool:
    return default_value if override is None else override


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


def is_darwin_host() -> bool:
    return sys.platform == "darwin"


def make_common_cmake_flags(
    *,
    cxx_warn_cpp: bool,
    cxx_warn_maybe_uninitialized: bool = False,
) -> tuple[str, str]:
    c_flags = ["-fno-omit-frame-pointer"]
    cxx_flags = []

    if cxx_warn_cpp:
        cxx_flags.append("-Wno-error=cpp")
    if cxx_warn_maybe_uninitialized and not is_darwin_host():
        cxx_flags.append("-Wno-error=maybe-uninitialized")

    # Apple clang is hitting upstream IREE warnings in some files.
    if is_darwin_host():
        cxx_flags.append("-Wno-error=unused-but-set-variable")

    cxx_flags.append("-fno-omit-frame-pointer")

    # Linux/ELF-oriented flags: do not use them on macOS.
    if not is_darwin_host():
        c_flags.extend(["-fdebug-types-section", "-gz=none"])
        cxx_flags.extend(["-fdebug-types-section", "-gz=none"])

    return " ".join(c_flags), " ".join(cxx_flags)
