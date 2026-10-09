"""Replay exact separately selected original neutral queue lifecycle tests."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import unittest
from pathlib import Path

import pytest

OWNER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_queue_patched_source_fixture", OWNER / "test_committed_inputs.py")
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)
queue = fixture_module.queue
ORIGINAL_TEST_SHA = "97abd06301b9f4c7fb8e9485e1155e10aa5a5d78991b772687a6b4fc7e10964e"


@pytest.mark.parametrize(
    "name",
    [
        "test_every_lifecycle_command_uses_exact_readonly_snapshot",
        "test_source_tamper_after_submit_fails_before_any_firesim_call",
        "test_untrusted_owner_cannot_authorize_parent_by_mode",
        "test_private_snapshot_tamper_between_commands_stops_lifecycle",
    ],
)
def test_original_neutral_hwdb_lifecycle(queue, name):
    selected = os.environ.get("MERLIN_TEST_QUEUE_ORIGINAL_TESTS")
    if not selected:
        pytest.skip("explicit original neutral test source selection required")
    path = Path(selected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_TEST_SHA
    spec = importlib.util.spec_from_file_location("_original_queue_hwdb_controls", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    # Select the actual patched source; leave all original test code and gates
    # byte-identical. These controls use only their own disposable queues.
    original.QUEUE_MODULE = Path(queue.__file__)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(unittest.TestSuite([original.HwdbSnapshotTest(name)]))
    assert result.testsRun == 1 and not result.skipped
    assert result.wasSuccessful(), stream.getvalue()
