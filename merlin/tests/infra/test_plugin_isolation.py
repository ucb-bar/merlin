"""A test that selects its own support provider must leave its worker as it found it.

Plugin ownership is process-immutable: once a backend has loaded from the provider selected on
``MERLIN_TARGET_PATH``, dropping that selection makes every later registry query refuse. Tests that
select a provider in-process used to leave exactly that behind, so a whole xdist worker's later tests
failed with ``PluginOwnershipError`` naming a synthetic target they never touched.
``plugin_isolation.fresh_plugin_state`` is the fresh process those tests owe the rest of the suite;
these cases pin both halves of it -- the refusal still fires inside the block, and nothing the block
loaded survives it.
"""

from __future__ import annotations

import os
import sys

import plugin_isolation
import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.backends import base

FIXTURE = merlin_dir() / "tests" / "fixtures" / "oot_backend_pkg" / "fixture_npu"
MODULE = "merlin._oot_backends.fixture_npu"


def _select_fixture(patch: pytest.MonkeyPatch) -> None:
    """Prepend the fixture provider so a provider the session already selected stays selected."""
    patch.setenv(
        "MERLIN_TARGET_PATH", os.pathsep.join(filter(None, (str(FIXTURE), os.environ.get("MERLIN_TARGET_PATH"))))
    )


def test_a_provider_loaded_inside_the_block_does_not_outlive_it():
    with plugin_isolation.fresh_plugin_state():
        with pytest.MonkeyPatch.context() as patch:
            _select_fixture(patch)
            assert base.get_backend("fixture_npu").__name__ == MODULE
        # The selection is gone while the backend is still owned: production refuses here, and so
        # must the block -- isolation is a fresh process AFTER it, not a relaxed check inside it.
        with pytest.raises(base.PluginOwnershipError, match="fixture_npu"):
            base.list_backends()
    assert "fixture_npu" not in base.list_backends(), "the block's backend leaked into the worker"
    assert MODULE not in sys.modules
    assert "fixture_npu" not in base.load_failures()


def test_the_same_provider_loads_again_in_a_later_block():
    """A later test selecting the same provider gets a fresh load, not the earlier block's stale owner."""
    for _ in range(2):
        with plugin_isolation.fresh_plugin_state(), pytest.MonkeyPatch.context() as patch:
            _select_fixture(patch)
            assert base.get_backend("fixture_npu").__name__ == MODULE
    assert "fixture_npu" not in base.list_backends()
