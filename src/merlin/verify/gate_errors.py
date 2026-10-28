"""Outcomes of a gate check other than acceptance; the gate worker turns each into a verdict."""

from __future__ import annotations


class GateError(RuntimeError):
    """A check did not accept; never a pass."""


class GateUnsupported(GateError):
    """The input lies outside what the check models, so nothing was proved."""


class GateRejected(GateError):
    """The check ran and did not accept: the pass changed what the program computes."""


class GateUnavailable(GateError):
    """The checker or its toolchain could not be used."""
