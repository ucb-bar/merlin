"""Muon's thin binding to the generic fork-free boot/BSP builder.

The builder itself is fully target-agnostic and lives in its generic home
:mod:`merlin.targetgen.fixed_format.boot` (every ISA fact is derived from the target's RTL model, and
the launch-width shim is supplied through the ``occupancy=(symbol, value)`` seam rather than a baked
vendor symbol). This module only re-exports it under the muon backend's own namespace so the relocated
SIMT backend can reach it as a sibling (``get_backend("muon").muon_bsp`` / ``from . import muon_bsp``)
without re-homing the generic code under a target-named directory. The muon-owned occupancy SYMBOL is
``merlin...muon.OCCUPANCY_SYMBOL`` and is passed by the backend at call time.
"""
from __future__ import annotations

from merlin.targetgen.fixed_format.boot import *  # noqa: F401,F403
from merlin.targetgen.fixed_format.boot import (  # noqa: F401  (explicit: names the backend calls)
    BootBuildError,
    build_boot_object,
    build_bsp,
    occupancy_shim,
    transcode_boot_object,
)
