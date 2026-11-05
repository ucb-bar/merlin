"""Merlin's generic RoCC rule engine bound to the in-repo Gemmini contract, for protocol tests.

``checks`` is :mod:`merlin.targetgen.rtl_checks_generic`; ``protocol`` is the ``rtl_checks`` protocol
the Gemmini contract declares (for pure helpers driven with hand-built facts). Needs only that
contract and the target's RTL facts (``MERLIN_RTL_FACTS`` or a regenerable cache) — no support
backend selection. When Gemmini support IS explicitly selected and resolves to the generic engine,
``selected`` is that capability (else ``None``).
"""

import pytest

from merlin.targetgen import rtl_checks
from merlin.targetgen import rtl_checks_generic as checks

try:
    protocol = checks.protocol_for("gemmini")
    from merlin.targetgen.rtl.facts import load_facts

    load_facts("gemmini")
except Exception as exc:  # noqa: BLE001 — no contract protocol / no RTL facts on this host
    pytest.skip(f"Gemmini rtl_checks protocol or RTL facts unavailable: {exc}", allow_module_level=True)

try:
    from merlin.targetgen import target_registry

    selected = rtl_checks.selected_checks("gemmini") if target_registry.explicit_targets().get("gemmini") else None
except Exception:  # noqa: BLE001 — selection is optional for these tests
    selected = None
