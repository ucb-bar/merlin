"""Synthetic out-of-tree runtime backend for the OOT discovery regression test.

This module is NEVER imported by name; base._load_oot_backend imports it by FILE PATH under
the synthetic name ``merlin._oot_backends.fixture_npu`` (because its package's contract names
it via ``plugin.backend``). At import time it self-registers with the runtime backend registry,
exactly as an evicted accelerator backend in a published ``<target>-mlir`` package would. It
carries no real toolchain — the test only proves the discover -> import -> register plumbing.
"""
from __future__ import annotations

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass

# Module-level self-registration: BackendInfo.module is this module's own synthetic __name__,
# so base.get_backend("fixture_npu") re-resolves it from sys.modules without core knowing the
# package layout.
base.register(BackendInfo("fixture_npu", TargetClass.NPU, BackendKind.KERNEL, __name__))
