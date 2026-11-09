"""Explicit prerequisites for tests that invoke the external host compiler."""

import pytest


@pytest.fixture
def upstream_host_tools():
    """Only unselected absence skips; selected or present broken tools still fail."""
    from logical_device_target import require_native_tools

    require_native_tools(host=True)
