"""Current SDK byte custody and independently frozen historical API metadata."""

import subprocess
import sys
from pathlib import Path

import pytest
from compiler_profile import bundle, prepare, reseal

from merlin.common import frozen_imports
from merlin.compiler.schedule.export import namespace_replacements
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import tool_profile_prompt, verify_tool_profile


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    return prepare(tmp_path_factory.mktemp("compiler-tool-profile") / "release", tools=("lean_proof_authoring",))


def test_current_release_preserves_real_packages_and_independent_check_inputs(release, tmp_path):
    selected = bundle(release)
    profile = verify_tool_profile(selected)
    assert profile["shared_scheduling_library"]["inventory_version"] == 27
    scalar = release / "merlin/compiler/lower/scalar"
    assert (scalar / "__init__.py").read_bytes() == namespace_replacements(27)["merlin.compiler.lower.scalar"].encode()
    assert (scalar / "casts.py").is_file() and (not scalar.with_suffix(".py").exists())
    for lemma in ("Foundation", "ResourceGeometry", "StateEffects"):
        assert (release / "merlin/verification/lean/Merlin" / (lemma + ".lean")).is_file()
    assert not (release / "merlin/compile").exists()
    assert not any(
        name == "merlin.compile" or name.startswith("merlin.compile.") for name in profile["common_public_modules"]
    )
    assert "merlin.compiler.passes.xdsl" not in profile["common_public_modules"]
    assert profile["proof_policy"]["mode"] == "strict_within_coverage"
    assert profile["proof_policy"]["unknown_within_coverage"] == "reject"
    assert profile["lean_toolchain"] == {"selection": "not_selected", "readiness": "unavailable"}
    hosts = [(row["path"], Path(row["path"])) for row in selected["host_inputs"]]
    assert hosts and all((path.is_absolute() and path.resolve() == path for _, path in hosts))
    bwrap._validate_host_sources([], hosts)
    alias = tmp_path / "private-checker-alias.py"
    alias.symlink_to(hosts[0][1])
    with pytest.raises(RuntimeError, match="symlink aliases"):
        bwrap._validate_host_sources([], [(str(alias), alias)])

    # A renamed workflow must not reopen its old namespace through dependency paths.
    snapshot = tmp_path / "frozen"
    snapshot.mkdir()
    ambient = tmp_path / "ambient"
    package = ambient / "merlin_experiments"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise AssertionError('unverified workflow imported')\n")
    probe = """
import importlib, importlib.util, sys
spec = importlib.util.spec_from_file_location("frozen_guard", sys.argv[1])
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
guard.activate(snapshot_root=sys.argv[2], import_roots=[sys.argv[2]], sources={})
sys.path.insert(0, sys.argv[3])
try:
    importlib.import_module("merlin_experiments")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("historical workflow fell through its frozen receipt")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, frozen_imports.__file__, str(snapshot), str(ambient)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_treatments_share_algorithms_but_only_declared_optional_tools_are_exposed(release):
    assert verify_tool_profile({"arm": "raw_baseline", "tools": []}) is None
    for arm in ("merlin_isa", "merlin_lean"):
        with pytest.raises(ValueError, match="versioned common compiler substrate"):
            verify_tool_profile({"arm": arm, "tools": list(TR.ARM_TOOLS[arm])})
    profiles = []
    for arm in ("raw_baseline", "merlin_isa", "merlin_lean"):
        selected = bundle(release, arm=arm)
        profile = verify_tool_profile(selected)
        profiles.append(profile)
        prompt = tool_profile_prompt(selected)
        assert "V27 uses canonical" in prompt
        for tool in ("isa_dialect_generation", "lean_proof_authoring"):
            assert ("`" + tool + "`" in prompt) == (tool in selected["tools"])
        allowed = {row["path"] for row in selected["allowed"]}
        denied = {row["path"] for row in selected["denied"]}
        cli = str(release / "merlin/verification/lean_cli.py")
        assert (cli in allowed) == (arm == "merlin_lean")
        assert (cli in denied) != (arm == "merlin_lean")
        hosts = {row["path"] for row in selected["host_inputs"]}
        for name in ("Foundation", "ResourceGeometry", "StateEffects"):
            source = str(release / "merlin/verification/lean/Merlin" / (name + ".lean"))
            assert (source in allowed) == (arm == "merlin_lean")
            assert (source in hosts) != (arm == "merlin_lean")
        with pytest.raises(ValueError, match="immutable compiler tool profile"):
            verify_tool_profile(selected, tools=("undeclared",))
    for profile in profiles[1:]:
        for field in ("common_sources", "automatic_checker_sources", "proof_policy", "shared_scheduling_library"):
            assert profile[field] == profiles[0][field]
    with pytest.raises(ValueError, match="explicit EL5/EL6"):
        TR.arm_tools("merlin_rtlchecks", add=("lean_proof_authoring",))


def test_changed_shared_algorithm_refuses_before_profile_admission(release):
    path = release / "merlin/compiler/schedule/configuration.py"
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\n# changed source\n")
        with pytest.raises(ValueError):
            bundle(release)
    finally:
        path.write_bytes(before)


def test_resealed_profile_cannot_publish_reference_implementation(release):
    selected = bundle(release)
    selected["compiler_tool_profile"]["common_public_modules"].append("merlin.runtime.reference")
    reseal(selected)
    with pytest.raises(ValueError):
        verify_tool_profile(selected)
