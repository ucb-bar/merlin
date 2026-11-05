"""Tests that need a real pinned support package: use the fetched one, or skip with the setup command.

Chip packages are not tracked in this repository (``merlin.targetgen.support_pins``); a host that has
fetched them (``merlin-target-fetch <target>`` or the chip example's ``run.sh setup``) runs these tests
against exactly the pinned revision, and any other host skips them, naming the command.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _setup_command(targets: list[str]) -> str:
    from merlin.targetgen import support_pins

    pins = support_pins.load_pins()
    hints = [pins.get(t).setup_hint() for t in targets if pins.get(t) is not None]
    return "; ".join(hints) or "merlin-target-fetch <target>"


def require_support(target: str) -> Path:
    """The fetched pinned package for ``target``, or skip naming how to fetch it."""
    from merlin.targetgen import support_pins

    pin = support_pins.load_pins().get(target)
    if pin is None:
        pytest.skip(f"{target!r} is not a pinned support package")
    root = support_pins.available_root(pin)
    if root is None:
        pytest.skip(f"{pin.repo_name}@{pin.short} is not fetched here; run {pin.setup_hint()}")
    return root


def require_board(name: str, **overrides):
    """``boards.board(name)``, or skip when the board ships with a pinned package that is not fetched."""
    from merlin.runtime import boards
    from merlin.targetgen import support_pins

    if name in boards.BOARDS:
        return boards.board(name, **overrides)
    pins = support_pins.load_pins()
    missing = [pin.target for pin in support_pins.default_selection(pins).missing.values() if pin.extends]
    if not missing:
        return boards.board(name, **overrides)  # absent for another reason: fail as the test would
    pytest.skip(
        f"board {name!r} is not selected; chip boards ship with their support packages, so run the "
        f"setup of the chip that owns it: {_setup_command(missing)}"
    )
