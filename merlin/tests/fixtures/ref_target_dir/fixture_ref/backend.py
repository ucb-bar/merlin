"""Synthetic curated-REFERENCE runtime backend for the reference-plugin discovery test.

Imported BY FILE PATH under ``merlin._oot_backends.fixture_ref`` (because its reference-target contract
names it via ``plugin.backend``, and ``base._oot_backend_modules`` now walks reference targets too, using
the reference package root as the backend path). Self-registers at import — exactly as the gemmini
reference backend will once evicted into ``merlin/targets/gemmini/backend/``.
"""
from __future__ import annotations

from merlin.runtime.backends import base
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass

base.register(BackendInfo("fixture_ref", TargetClass.NPU, BackendKind.KERNEL, __name__))
