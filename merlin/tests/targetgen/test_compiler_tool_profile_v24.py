"""The prospective V24 SDK has an explicit common profile; V23 keeps its grants."""

import copy

import pytest
from test_compiler_tool_profile_v6 import bundle, prepare, reseal

from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V17, tool_profile_prompt, verify_tool_profile


def test_v24_profile_transports_its_exact_closed_inventory(tmp_path):
    selected = bundle(prepare(tmp_path / "release", version=24), version=COMMON_SUBSTRATE_V17)
    profile = verify_tool_profile(selected)
    assert profile["shared_scheduling_library"]["inventory_version"] == 24
    assert "merlin.compile.scheduling.dense_shared_rhs" in profile["common_public_modules"]
    assert "scheduling_capabilities(24)" in tool_profile_prompt(selected)
    assert "V24 additionally exposes source-checked shared-RHS" in tool_profile_prompt(selected)
    foreign = copy.deepcopy(selected)
    foreign["compiler_tool_profile"]["common_substrate"] = "merlin.common_compiler_substrate.v16"
    reseal(foreign)
    with pytest.raises(ValueError):
        verify_tool_profile(foreign)
