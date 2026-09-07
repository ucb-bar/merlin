"""Environment resolution creates obligations, not hidden executable grants."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import get_backend


def fixture(tmp_path, monkeypatch):
    backend = get_backend("gemmini")
    module = importlib.import_module(backend.__package__+".gemmini_program_environment")
    implementation = importlib.import_module(backend.__package__+".gemmini")
    root = tmp_path/"frozen"
    python = root/"merlin/python"
    python.mkdir(parents=True)
    interpreter = root/".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("trusted existing interpreter")
    harness, alias = tmp_path/"selected_harness", tmp_path/"existing_alias"
    harness.mkdir()
    alias.mkdir()
    for directory in (harness, alias):
        (directory/"support.c").write_text("support")
        (directory/"link.ld").write_text("link")
    monkeypatch.setattr(implementation, "rocc_tests_dir", lambda: harness)
    monkeypatch.setattr(implementation, "harness_build_recipe", lambda: SimpleNamespace(
        support_sources=(harness/"support.c",), link_script=harness/"link.ld"))
    from merlin.targetgen import gsim_emulator
    monkeypatch.setattr(gsim_emulator, "citation", lambda *a, **k: {
        "available": True, "refused": False, "receipt_status": "bound", "flavour": "binary"})
    prefix = ["bwrap", "--clearenv", "--ro-bind", str(root), str(root),
        "--ro-bind", str(alias), str(alias), "bash", "-c",
        f'export PYTHONPATH={python}${{PYTHONPATH:+:$PYTHONPATH}}; export MERLIN_GEMMINI_HARNESS_DIR={alias}; exec "$@"', "host-tool"]
    return module, {"command_prefix": prefix}, root, alias


def test_exact_existing_namespace_and_recipe_alias_no_grants(tmp_path, monkeypatch):
    module, policy, root, alias = fixture(tmp_path, monkeypatch)
    original = list(policy["command_prefix"])
    result = module.short_program_environment(policy)
    assert result["build_namespace_root"] == str(root)
    assert result["python_executable"] == str(root/".venv/bin/python")
    assert result["build_path_bindings"][0][1] == str(alias)
    assert not result["mount_grants"] and policy["command_prefix"] == original
    assert len(result["adapter_source_pins"]) == 2


@pytest.mark.parametrize("mutation", ["shadow", "masked", "namespace", "engine"])
def test_untrusted_or_missing_existing_obligations_refused(tmp_path, monkeypatch, mutation):
    module, policy, root, alias = fixture(tmp_path, monkeypatch)
    if mutation == "shadow":
        (alias/"support.c").write_text("different bytes")
    elif mutation == "masked":
        policy["command_prefix"][8:8] = ["--tmpfs", str(alias)]
    elif mutation == "namespace":
        policy["command_prefix"][-2] = 'export PYTHONPATH=$USER/merlin/python; exec "$@"'
    else:
        from merlin.targetgen import gsim_emulator
        monkeypatch.setattr(gsim_emulator, "citation", lambda *a, **k: {"receipt_status": "absent"})
    with pytest.raises(ValueError):
        module.short_program_environment(policy)
