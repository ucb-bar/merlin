"""Bundled legacy bindings remain closed, explicit, and contained."""

from importlib.resources import files

import pytest
from merlin_experiments import adapters
from merlin_experiments.spec import SpecError


def test_all_production_adapters_resolve_the_bundled_table():
    assert files("merlin_experiments").joinpath("resources", "legacy_entrypoints.json").is_file()
    table = adapters._legacy_entrypoints()
    assert set(table) == set(adapters.ADAPTERS)
    for name, adapter in adapters.ADAPTERS.items():
        assert adapter.script == table[name]["default"]
    assert table["capsule_bench"]["rtlchecks"].endswith("run_rtlchecks_qa_loop.py")


@pytest.mark.parametrize("path", ["", "/absolute.py", "../escape.py", "folder/../../escape.py", "command.sh", None])
def test_binding_rejects_missing_or_escaping_source_paths(tmp_path, path):
    with pytest.raises(SpecError):
        adapters._entrypoint_path(tmp_path, path)


def test_binding_rejects_missing_file_and_symlink_escape(tmp_path):
    with pytest.raises(SpecError, match="entrypoint missing"):
        adapters._entrypoint_path(tmp_path, "missing.py")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("# not an engine in the selected checkout\n")
    (checkout / "engine.py").symlink_to(outside)
    with pytest.raises(SpecError, match="escapes"):
        adapters._entrypoint_path(checkout, "engine.py")


def test_unknown_binding_variant_is_not_a_command_escape():
    with pytest.raises(SpecError, match="binding missing"):
        adapters._legacy_script("capsule_bench", "../../another.py")
