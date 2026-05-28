"""Build MCP tools — typed wrappers around `./merlin build`.

Depends on:
- tools/build/cli.py and tools/build/presets.py (PROFILE_PRESETS dict; the build_dir naming logic
  at line ~840-854 which this module mirrors)
- compiler/, runtime/, build_tools/ source roots (freshness check walks these)
- third_party/iree_bar/CMakeLists.txt (sentinel files)
- root CMakeLists.txt, iree_compiler_plugin.cmake, iree_runtime_plugin.cmake

If any of those change shape, update `_build_dir_for_profile`,
`_SOURCE_ROOTS`, `_IREE_BAR_SENTINELS`, or `_ROOT_CMAKE_SENTINELS` to match.

Killer feature: `build_check_freshness` — a deterministic answer to "do I
need to rebuild before running compile/run/verify?". Without it, an agent
either over-rebuilds (slow) or under-rebuilds (stale binaries that lie
about what the current code does).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOLS_DIR))
import utils  # noqa: E402

REPO_ROOT = utils.REPO_ROOT
MERLIN = REPO_ROOT / "merlin"

# Source roots whose mtimes invalidate a build tree. The list is conservative:
# touching anything under these dirs means *some* profile needs a rebuild. The
# per-profile check is approximated (we don't track every profile's exact
# dependency graph) — under-rebuild is worse than over-rebuild, so we err
# toward suggesting a rebuild.
_SOURCE_ROOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("compiler", (".cpp", ".cc", ".h", ".hpp", ".td", ".mlir", ".inc", ".def")),
    ("runtime", (".cpp", ".cc", ".h", ".hpp", ".c", ".inc")),
    ("build_tools", (".cmake", ".cmake.in", ".txt")),  # toolchain files
    # third_party/iree_bar is huge; we sample only the top-level CMake files
    # below via _IREE_BAR_SENTINELS rather than walking the whole subtree.
)

_IREE_BAR_SENTINELS = (
    "third_party/iree_bar/CMakeLists.txt",
    "third_party/iree_bar/compiler/CMakeLists.txt",
    "third_party/iree_bar/runtime/CMakeLists.txt",
)

_ROOT_CMAKE_SENTINELS = (
    "CMakeLists.txt",
    "iree_compiler_plugin.cmake",
    "iree_runtime_plugin.cmake",
)


class ToolError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# ---- helpers -------------------------------------------------------------


def _profile_presets() -> dict[str, dict[str, object]]:
    """Read PROFILE_PRESETS from tools/build/presets.py.

    Imports the lightweight presets module directly (avoids
    pulling in build.cli, which has cmake-finder side effects).
    """
    import importlib

    mod = importlib.import_module("build.presets")
    return getattr(mod, "PROFILE_PRESETS", {})


def _candidate_build_dirs() -> list[pathlib.Path]:
    """All directories under build/ that look like a configured build tree."""
    build_root = REPO_ROOT / "build"
    if not build_root.exists():
        return []
    out = []
    for d in sorted(build_root.iterdir()):
        if d.is_dir() and (d / "CMakeCache.txt").exists():
            out.append(d)
    return out


def _build_dir_for_profile(profile: str, config: str = "release") -> pathlib.Path | None:
    """Best-effort: derive the build directory name from PROFILE_PRESETS.

    Mirrors the logic in build/cli.py around line 840-854. Returns None
    if the profile isn't known.
    """
    presets = _profile_presets()
    if profile not in presets:
        return None
    preset = presets[profile]
    target = str(preset.get("target", "host"))
    # Variant: matches build/cli.py's variant derivation.
    if preset.get("with_plugin"):
        variant = "merlin"
    else:
        variant = "vanilla"
    if profile == "qnn-compiler":
        build_name = "host-merlin-release-qrb"
    elif target == "qrb5165" and preset.get("plugin_runtime") and not preset.get("plugin_compiler"):
        build_name = f"{target}-runtime-{config}"
    elif profile == "zephyr-task":
        build_name = f"{target}-task-{config}"
    else:
        build_name = f"{target}-{variant}-{config}"
    return REPO_ROOT / "build" / build_name


def _latest_source_mtime(roots: list[pathlib.Path], extensions: tuple[str, ...]) -> tuple[float, str | None]:
    """Walk roots and find the most recent (mtime, path) for files matching
    extensions. Returns (0.0, None) if nothing is found.
    """
    best_mtime = 0.0
    best_path: str | None = None
    for root in roots:
        if not root.exists():
            continue
        for src in root.rglob("*"):
            if not src.is_file():
                continue
            if extensions and src.suffix not in extensions:
                continue
            try:
                m = src.stat().st_mtime
            except OSError:
                continue
            if m > best_mtime:
                best_mtime = m
                best_path = str(src.relative_to(REPO_ROOT))
    return best_mtime, best_path


# ---- tools ---------------------------------------------------------------


def _list_profiles(args: dict[str, Any]) -> dict[str, Any]:
    """List every entry in build/cli.py:PROFILE_PRESETS with the salient
    flags Claude needs to pick the right one."""
    presets = _profile_presets()
    out = []
    for name, preset in presets.items():
        out.append(
            {
                "name": name,
                "target": preset.get("target"),
                "config": preset.get("config", "release"),
                "with_plugin": bool(preset.get("with_plugin")),
                "plugin_compiler": bool(preset.get("plugin_compiler")),
                "plugin_runtime": bool(preset.get("plugin_runtime")),
                "build_compiler": bool(preset.get("build_compiler")),
                "expected_build_dir": str(_build_dir_for_profile(name).relative_to(REPO_ROOT))
                if _build_dir_for_profile(name)
                else None,
            }
        )
    return {"profiles": out, "count": len(out)}


def _status(args: dict[str, Any]) -> dict[str, Any]:
    """List all configured build trees with their state."""
    trees = []
    now = time.time()
    for d in _candidate_build_dirs():
        cmake_cache = d / "CMakeCache.txt"
        cache_mtime = cmake_cache.stat().st_mtime
        # Size: cheap estimate via du-style walk. Cap walking depth.
        size_bytes = 0
        for f in d.rglob("*"):
            try:
                if f.is_file():
                    size_bytes += f.stat().st_size
            except OSError:
                pass
        # Look for a representative binary to confirm there's a real build.
        likely_binaries = [
            p
            for p in [
                d / "tools" / "iree-compile",
                d / "tools" / "iree-run-module",
                d / "iree_compiler" / "tools" / "iree-compile",
            ]
            if p.exists()
        ]
        trees.append(
            {
                "build_dir": str(d.relative_to(REPO_ROOT)),
                "size_mb": round(size_bytes / 1024 / 1024, 1),
                "cmake_cache_mtime": cache_mtime,
                "age_hours": round((now - cache_mtime) / 3600, 1),
                "has_artifacts": bool(likely_binaries),
                "representative_binaries": [str(p.relative_to(REPO_ROOT)) for p in likely_binaries],
            }
        )
    return {"trees": trees, "count": len(trees)}


def _check_freshness(args: dict[str, Any]) -> dict[str, Any]:
    """Deterministically decide whether a rebuild is needed.

    Compares CMakeCache.txt's mtime against the latest mtime of source
    files under compiler/, runtime/, build_tools/, plus a handful of
    sentinel files (root CMakeLists.txt, third_party/iree_bar/...).

    Return shape:
        {
          "needs_rebuild": bool,
          "reason": "no_build_tree" | "sources_newer" | "config_changed" | "up_to_date",
          "build_dir": str | null,
          "cmake_cache_mtime": float | null,
          "newest_source": {"path": str, "mtime": float} | null,
          "newer_source_count": int,
          "newer_source_sample": [str, ...]   # up to 10 examples
        }
    """
    profile = args.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ToolError("missing required arg 'profile' (one of build_list_profiles names).")
    config = args.get("config", "release")

    build_dir = _build_dir_for_profile(profile, config=config)
    if build_dir is None:
        raise ToolError(f"unknown profile {profile!r}; call build_list_profiles to enumerate.")

    cmake_cache = build_dir / "CMakeCache.txt"
    if not cmake_cache.exists():
        return {
            "needs_rebuild": True,
            "reason": "no_build_tree",
            "build_dir": str(build_dir.relative_to(REPO_ROOT)),
            "cmake_cache_mtime": None,
            "newest_source": None,
            "newer_source_count": 0,
            "newer_source_sample": [],
        }

    cache_mtime = cmake_cache.stat().st_mtime

    # Sentinel files first — these are cheap to check.
    config_changed_files: list[str] = []
    for rel in (*_ROOT_CMAKE_SENTINELS, *_IREE_BAR_SENTINELS):
        p = REPO_ROOT / rel
        if p.exists() and p.stat().st_mtime > cache_mtime:
            config_changed_files.append(rel)

    # Source roots: find newest mtime per root.
    newer_sources: list[tuple[float, str]] = []
    overall_newest_mtime = 0.0
    overall_newest_path: str | None = None
    for root_name, extensions in _SOURCE_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for src in root.rglob("*"):
            if not src.is_file():
                continue
            if extensions and src.suffix not in extensions:
                continue
            try:
                m = src.stat().st_mtime
            except OSError:
                continue
            if m > cache_mtime:
                rel = str(src.relative_to(REPO_ROOT))
                newer_sources.append((m, rel))
            if m > overall_newest_mtime:
                overall_newest_mtime = m
                overall_newest_path = str(src.relative_to(REPO_ROOT))

    needs_rebuild = bool(config_changed_files or newer_sources)
    if config_changed_files:
        reason = "config_changed"
    elif newer_sources:
        reason = "sources_newer"
    else:
        reason = "up_to_date"

    newer_sources.sort(reverse=True)  # newest first
    return {
        "needs_rebuild": needs_rebuild,
        "reason": reason,
        "build_dir": str(build_dir.relative_to(REPO_ROOT)),
        "cmake_cache_mtime": cache_mtime,
        "newest_source": (
            {"path": overall_newest_path, "mtime": overall_newest_mtime} if overall_newest_path else None
        ),
        "newer_source_count": len(newer_sources),
        "newer_source_sample": [p for _, p in newer_sources[:10]],
        "config_files_changed": config_changed_files,
    }


def _build_profile(args: dict[str, Any]) -> dict[str, Any]:
    """Invoke `./merlin build --profile <name> [--cmake-target <t>]` and
    return parsed structure.

    Returns:
        {
          "passed": bool,
          "exit_code": int,
          "profile": str,
          "build_dir": str | null,
          "duration_s": float,
          "warning_count": int,
          "error_count": int,
          "command": [str, ...],
          "stdout_tail": [str, ...],
          "stderr_tail": [str, ...]
        }
    """
    profile = args.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ToolError("missing required arg 'profile'.")
    cmake_target = args.get("cmake_target")
    config = args.get("config")
    extra_flags = args.get("extra_flags") or []
    if not isinstance(extra_flags, list):
        raise ToolError("extra_flags must be a list of strings")

    cmd = [str(MERLIN), "build", "--profile", profile]
    if cmake_target:
        cmd.extend(["--cmake-target", str(cmake_target)])
    if config:
        cmd.extend(["--config", str(config)])
    cmd.extend(str(f) for f in extra_flags)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    duration_s = time.time() - start

    combined = result.stdout + "\n" + result.stderr
    warning_count = sum(
        1 for L in combined.splitlines() if " warning:" in L.lower() or "warning generated" in L.lower()
    )
    error_count = sum(1 for L in combined.splitlines() if " error:" in L.lower() or "fatal error" in L.lower())

    build_dir = _build_dir_for_profile(profile, config=str(config or "release"))
    build_dir_str = str(build_dir.relative_to(REPO_ROOT)) if build_dir and build_dir.exists() else None

    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "profile": profile,
        "build_dir": build_dir_str,
        "duration_s": round(duration_s, 1),
        "warning_count": warning_count,
        "error_count": error_count,
        "command": cmd,
        "stdout_tail": result.stdout.splitlines()[-30:],
        "stderr_tail": result.stderr.splitlines()[-30:],
    }


def _build_help(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(MERLIN), "build", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"exit_code": result.returncode, "help": result.stdout}


# ---- registry ------------------------------------------------------------


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="build_list_profiles",
        description=(
            "List every entry in build/cli.py:PROFILE_PRESETS with the "
            "salient flags (target, config, with_plugin, plugin_compiler, "
            "plugin_runtime, build_compiler) and the expected build_dir for "
            "each. Use to discover available profiles before invoking a build."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_profiles,
    ),
    ToolDefinition(
        name="build_status",
        description=(
            "List every currently-configured build tree under build/ with "
            "metadata (size_mb, cmake_cache_mtime, age_hours, has_artifacts, "
            "representative binaries). Use to see what's already built "
            "without re-running ls."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_status,
    ),
    ToolDefinition(
        name="build_check_freshness",
        description=(
            "Deterministically decide whether a rebuild is needed for a "
            "given profile. Compares CMakeCache.txt mtime against the "
            "latest mtime of source files under compiler/, runtime/, "
            "build_tools/, plus root CMakeLists and iree_bar sentinels. "
            "Returns needs_rebuild (bool), reason "
            "(no_build_tree/sources_newer/config_changed/up_to_date), and "
            "a sample of newer files. Use this BEFORE invoking compile/run/verify "
            "if the underlying compiler binary might be stale — it prevents "
            "both over-rebuilds and stale-binary bugs."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "profile": {"type": "string", "description": "Profile name (one of build_list_profiles entries)."},
                "config": {"type": "string", "description": "Build config (default 'release')."},
            },
            "required": ["profile"],
            "additionalProperties": False,
        },
        handler=_check_freshness,
    ),
    ToolDefinition(
        name="build_profile",
        description=(
            "Invoke `./merlin build --profile <name>` and return parsed "
            "structure: passed, exit_code, build_dir, duration_s, warning_count, "
            "error_count, plus stdout/stderr tails. Use after build_check_freshness "
            "indicates a rebuild is needed, or when explicitly asked to build."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "profile": {"type": "string", "description": "Profile name (one of build_list_profiles entries)."},
                "cmake_target": {"type": "string", "description": "Optional specific cmake target to build."},
                "config": {"type": "string", "description": "Build config override (release/debug/trace)."},
                "extra_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra raw flags forwarded to ./merlin build.",
                },
            },
            "required": ["profile"],
            "additionalProperties": False,
        },
        handler=_build_profile,
    ),
    ToolDefinition(
        name="build_help",
        description="Verbatim `./merlin build --help`.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_build_help,
    ),
]


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"unknown tool: {name!r}; known: {[t.name for t in TOOL_REGISTRY]}")
