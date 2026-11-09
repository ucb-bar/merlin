"""The refactored source closure preserves reviewed public authority."""

import copy

import pytest
from test_compiler_tool_profile_v6 import bundle, prepare, reseal

from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V19, tool_profile_prompt, verify_tool_profile


def test_v25_profile_has_closed_sources_without_new_public_grants(tmp_path):
    root = prepare(tmp_path / "release", version=26)
    selected = bundle(root, version=COMMON_SUBSTRATE_V19)
    profile = verify_tool_profile(selected)
    assert profile["shared_scheduling_library"]["inventory_version"] == 26
    assert "merlin.compiler.passes.xdsl" not in profile["common_public_modules"]
    assert "V26 uses canonical" in tool_profile_prompt(selected)
    assert not (root / "merlin/compile").exists()
    assert all(not name.startswith("merlin.compile.") and name != "merlin.compile"
               for name in profile["common_public_modules"])
    foreign = copy.deepcopy(selected)
    foreign["compiler_tool_profile"]["common_substrate"] = "merlin.common_compiler_substrate.v17"
    reseal(foreign)
    with pytest.raises(ValueError):
        verify_tool_profile(foreign)
