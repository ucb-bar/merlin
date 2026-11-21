"""Shared visibility vocabulary for the core compiler dialects."""

from __future__ import annotations

try:
    from xdsl.utils.str_enum import StrEnum

    HAS_XDSL = True
except ImportError:
    HAS_XDSL = False

if HAS_XDSL:

    class Visibility(StrEnum):
        """DSE variant tag carried by contract, schedule and Interface IR."""

        BASELINE = "baseline"
        SOFTWARE_VISIBLE = "software_visible"
        HARDWARE_MANAGED = "hardware_managed"
        ORACLE = "oracle"

else:
    Visibility = None
