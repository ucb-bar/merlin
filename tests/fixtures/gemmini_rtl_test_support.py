"""Explicit companion-owned RTL checks for Gemmini protocol tests only."""

import pytest

from merlin.targetgen import rtl_checks, target_registry
from merlin.targetgen.plugins import resolve_support

if target_registry.explicit_targets().get("gemmini") is None:
    pytest.skip("requires explicit Gemmini support on MERLIN_TARGET_PATH", allow_module_level=True)

resolve_support("gemmini")
checks = rtl_checks.selected_checks("gemmini")
