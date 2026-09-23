"""Generic sandbox imports must not require an unrelated target toolchain."""

import os
import subprocess
import sys

from merlin.common.paths import python_import_roots


def test_generic_sandbox_import_without_chipyard_and_selected_family_fails_closed(tmp_path):
    environment = dict(os.environ, MERLIN_REPO_ROOT=str(tmp_path), MERLIN_OUT_ROOT=str(tmp_path / "out"))
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in python_import_roots())
    environment.pop("MERLIN_EXT_CHIPYARD", None)
    source = """
from types import SimpleNamespace
import merlin.targetgen.sandbox.bwrap
from merlin.targetgen.sandbox import toolchain
assert toolchain._sim(SimpleNamespace(sim_via="unregistered")).bind_paths == ()
assert toolchain._sim(SimpleNamespace(sim_via="")).bind_paths == ()
assert "chipyard" in toolchain.SIM_TOOLCHAINS
for lookup in (
    lambda: toolchain._sim(SimpleNamespace(sim_via="chipyard")),
    lambda: toolchain.SIM_TOOLCHAINS["chipyard"],
    lambda: toolchain.SIM_TOOLCHAINS.get("chipyard"),
):
    try:
        lookup()
    except KeyError as exc:
        assert "MERLIN_EXT_CHIPYARD" in str(exc)
    else:
        raise AssertionError("selected missing toolchain silently became empty")
"""
    subprocess.run([sys.executable, "-c", source], env=environment, capture_output=True, text=True, check=True)


def test_registry_retains_direct_mapping_access_and_plugin_registration():
    from merlin.targetgen.sandbox.toolchain import SimToolchain, _SimToolchainRegistry

    calls = []
    expected = SimToolchain(bind_paths=("/configured/toolchain",))
    registry = _SimToolchainRegistry({"fixture": lambda: calls.append(True) or expected})
    assert list(registry) == ["fixture"] and calls == []
    assert registry["fixture"] is expected
    assert registry.get("fixture") is expected
    assert list(registry.values()) == [expected] and calls == [True]
    registry["plugin"] = expected
    assert registry["plugin"] is expected
