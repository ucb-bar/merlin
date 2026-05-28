"""Cross-check TargetGen-predicted plugin paths against IREE plugin cmake.

The planner predicts edits to ``iree_compiler_plugin.cmake`` and
``iree_runtime_plugin.cmake``. If the macro signatures or per-target gating
in those files drifts (e.g., after an IREE submodule update), the predicted
edits become stale. This test parses the live cmake files and asserts that
the registration patterns referenced by ``stage_map.py`` and
``target_routes.py`` actually exist.
"""

from __future__ import annotations

import re
from pathlib import Path

from conftest import REPO_ROOT

COMPILER_CMAKE = REPO_ROOT / "iree_compiler_plugin.cmake"
RUNTIME_CMAKE = REPO_ROOT / "iree_runtime_plugin.cmake"


def _read(path: Path) -> str:
    assert path.exists(), f"missing: {path}"
    return path.read_text(encoding="utf-8")


def test_compiler_plugin_cmake_uses_merlin_build_gating() -> None:
    """The compiler plugin cmake must use MERLIN_BUILD_<TARGET> gating per target."""
    text = _read(COMPILER_CMAKE)
    # Every compiler plugin block is gated by MERLIN_BUILD_*; assert at least
    # one such block exists (otherwise the plugin model has changed and the
    # planner needs updating).
    matches = re.findall(r"if\s*\(\s*MERLIN_BUILD_[A-Z0-9_]+\s*\)", text)
    assert matches, "iree_compiler_plugin.cmake no longer gates plugins by MERLIN_BUILD_<TARGET>"


def test_compiler_plugin_cmake_requires_merlin_enable_core() -> None:
    """Each per-target block must guard against MERLIN_ENABLE_CORE=OFF."""
    text = _read(COMPILER_CMAKE)
    assert "MERLIN_ENABLE_CORE" in text, "MERLIN_ENABLE_CORE flag missing — the planner assumes core/plugin split"


def test_runtime_plugin_cmake_uses_iree_register_external_hal_driver() -> None:
    """The runtime plugin cmake must use the IREE external HAL registration macro."""
    text = _read(RUNTIME_CMAKE)
    assert "iree_register_external_hal_driver" in text, (
        "iree_runtime_plugin.cmake no longer uses iree_register_external_hal_driver — "
        "stage_map.py predictions for runtime_hal stages are now stale"
    )


def test_runtime_plugin_cmake_exposes_per_driver_toggle() -> None:
    """At least one MERLIN_RUNTIME_ENABLE_HAL_<DRIVER> option must exist."""
    text = _read(RUNTIME_CMAKE)
    matches = re.findall(r"MERLIN_RUNTIME_ENABLE_HAL_[A-Z0-9_]+", text)
    assert matches, "iree_runtime_plugin.cmake exposes no per-driver toggles"


def test_existing_compiler_plugin_targets_have_cmake_blocks() -> None:
    """Each existing compiler plugin under compiler/plugins/target/<X>/ has a cmake gate."""
    text = _read(COMPILER_CMAKE)
    plugin_dirs = sorted(p.name for p in (REPO_ROOT / "compiler" / "plugins" / "target").iterdir() if p.is_dir())
    if not plugin_dirs:
        return  # no plugins yet; nothing to cross-check
    pattern = r'add_subdirectory\(\s*"\$\{MERLIN_COMPILER_SOURCE_DIR\}' r'/compiler/plugins/target/([A-Za-z0-9_]+)"'
    found = re.findall(pattern, text)
    found_set = {f.lower() for f in found}
    for plugin in plugin_dirs:
        assert plugin.lower() in found_set, (
            f"compiler plugin '{plugin}' exists on disk but is not registered " f"in iree_compiler_plugin.cmake"
        )


def test_existing_runtime_drivers_have_cmake_blocks() -> None:
    """Each runtime/src/iree/hal/drivers/<X>/ subdir has a registration in the runtime cmake."""
    text = _read(RUNTIME_CMAKE)
    drivers_dir = REPO_ROOT / "runtime" / "src" / "iree" / "hal" / "drivers"
    if not drivers_dir.exists():
        return
    drivers = sorted(p.name for p in drivers_dir.iterdir() if p.is_dir())
    if not drivers:
        return
    register_blocks = re.findall(
        r"iree_register_external_hal_driver\s*\(\s*NAME\s+(\w+)",
        text,
        re.IGNORECASE,
    )
    registered = {r.lower() for r in register_blocks}
    for driver in drivers:
        # Skip support directories that aren't drivers.
        if driver in {"common", "registration"}:
            continue
        assert driver.lower() in registered, (
            f"runtime driver '{driver}' exists on disk but is not registered " f"in iree_runtime_plugin.cmake"
        )
