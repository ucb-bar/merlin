"""Explicit companion-owned RTL checks for Gemmini protocol tests only."""

import pytest

from merlin.target.isa.rocc import rtl_checks
from merlin.target.registry import identity as target_registry
from merlin.target.registry.plugins import resolve_support

if target_registry.explicit_targets().get("gemmini") is None:
    pytest.skip("requires explicit Gemmini support on MERLIN_TARGET_PATH", allow_module_level=True)

resolve_support("gemmini")
checks = rtl_checks.selected_checks("gemmini")
