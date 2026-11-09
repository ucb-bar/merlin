"""The refactored source closure preserves reviewed public authority."""

import copy

import pytest
import yaml
from test_compiler_tool_profile_v6 import bundle, prepare, reseal

from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V19, tool_profile_prompt, verify_tool_profile


def test_v25_profile_has_closed_sources_without_new_public_grants(tmp_path):
    root = prepare(tmp_path / "release", version=26)
    selected = bundle(root, version=COMMON_SUBSTRATE_V19)
    profile = verify_tool_profile(selected)
    assert profile["shared_scheduling_library"]["inventory_version"] == 26
    # Preparation must preserve real namespace packages, not invent scalar.py.
    from merlin.compiler.schedule.export import namespace_replacements

    scalar = root / "merlin/compiler/lower/scalar"
    assert (scalar / "__init__.py").read_bytes() == namespace_replacements(26)["merlin.compiler.lower.scalar"].encode()
    assert (scalar / "casts.py").is_file()
    assert not scalar.with_suffix(".py").exists()
    for lemma in ("Foundation", "ResourceGeometry", "StateEffects"):
        assert (root / "merlin/verification/lean/Merlin" / (lemma + ".lean")).is_file()
    assert "merlin.compiler.passes.xdsl" not in profile["common_public_modules"]
    assert "V26 uses canonical" in tool_profile_prompt(selected)
    assert not (root / "merlin/compile").exists()
    assert all(
        not name.startswith("merlin.compile.") and name != "merlin.compile" for name in profile["common_public_modules"]
    )
    from merlin.common.paths import repo_root
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.sandbox import bwrap
    from merlin_experiments.phase2.common_transforms import supports_check

    assert supports_check(profile)
    assert profile["lean_toolchain"] == {"selection": "not_selected", "readiness": "unavailable"}
    archived = tmp_path / "original.yaml"
    archived.write_text(yaml.safe_dump(selected))
    workspace = tmp_path / "original_workspace"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, selected, repo=repo_root())
    owner = CommonTransformService(workspace=workspace, bundle_manifest=archived, repo=repo_root())
    request = {
        "schema": "merlin.common_check_request.v1",
        "before_ir": """builtin.module {
          func.func @forward(%x: i8) -> i8 {
            %zero = arith.constant 0 : i8
            %y = arith.addi %x, %zero : i8
            func.return %y : i8
          }
        }""",
        "after_ir": "builtin.module { func.func @forward(%x: i8) -> i8 { func.return %x : i8 } }",
        "entry": "forward",
    }
    # Current dispatch reaches the original toolchain requirement. No proof,
    # numerical execution or hardware qualification is manufactured by this smoke.
    with pytest.raises(FileNotFoundError, match="no explicitly selected common formal toolchain"):
        owner.check(request, work_root=tmp_path / "host_work", timeout_s=5)
    foreign = copy.deepcopy(selected)
    foreign["compiler_tool_profile"]["common_substrate"] = "merlin.common_compiler_substrate.v17"
    reseal(foreign)
    with pytest.raises(ValueError):
        verify_tool_profile(foreign)
