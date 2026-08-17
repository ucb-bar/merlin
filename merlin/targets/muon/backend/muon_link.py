"""Muon's thin binding to the generic fork-free linker.

The linker is fully target-agnostic (a stock ``ld.lld`` lays out the image; relocations are re-applied
at the target's RTL-derived field positions) and lives in its generic home
:mod:`merlin.targetgen.fixed_format.link`. This module re-exports it under the muon backend's own
namespace so the relocated SIMT backend can reach it as a sibling
(``get_backend("muon").muon_link`` / ``from . import muon_link``) without re-homing the generic code
under a target-named directory.
"""
from __future__ import annotations

from merlin.targetgen.fixed_format.link import *  # noqa: F401,F403
from merlin.targetgen.fixed_format.link import (  # noqa: F401  (explicit: names the backend calls)
    FixedFormatLinkError,
    link_fork_free,
    patch_relocations,
    resolve_stock_linker,
    stock_layout_link,
)
