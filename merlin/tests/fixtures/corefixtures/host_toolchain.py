from __future__ import annotations

import pytest


def _refuse_selected_missing(absent, selection_keys, *, reason):
    """Missing optional tools may skip; an explicit unavailable selection fails."""
    from merlin.llvmlower import toolchain

    if absent and any((toolchain._env(key) for key in selection_keys)):
        pytest.fail("selected " + reason, pytrace=False)
