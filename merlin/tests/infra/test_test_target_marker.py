"""check_structure "test target marker": a target-heavy test in a subsystem bucket must declare its target."""
from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_structure.py"


def _gate(root):
    spec = importlib.util.spec_from_file_location("check_structure_marker_test", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = str(root)
    mod.TARGET_MARKER_RATCHET = str(root / "build_tools" / "scripts" / "test_target_marker_ratchet.txt")
    return mod


def _tree(tmp_path):
    (tmp_path / "merlin/targets/foo_hw/contracts").mkdir(parents=True)
    (tmp_path / "merlin/targets/foo_hw/contracts/target_contract.yaml").write_text("name: foo_hw\n")
    (tmp_path / "build_tools/scripts").mkdir(parents=True)
    import shutil
    shutil.copy(repo_root() / "build_tools/scripts/_target_roster.py", tmp_path / "build_tools/scripts/")
    (tmp_path / "merlin/tests/infra").mkdir(parents=True)
    return tmp_path


def _errors(root):
    errors: list[str] = []
    _gate(root).check_test_target_marker(errors)
    return errors


def test_a_target_heavy_unmarked_test_is_reported(tmp_path):
    root = _tree(tmp_path)
    (root / "merlin/tests/infra/test_x.py").write_text('T = ["foo_hw"] * 1\n' + '"foo_hw"\n' * 5)
    assert any("test_x.py" in e for e in _errors(root))


def test_the_marker_or_the_ledger_clears_it(tmp_path):
    root = _tree(tmp_path)
    body = '"foo_hw"\n' * 6
    (root / "merlin/tests/infra/test_marked.py").write_text("pytestmark = pytest.mark.target('foo_hw')\n" + body)
    (root / "merlin/tests/infra/test_ledgered.py").write_text(body)
    (root / "build_tools/scripts/test_target_marker_ratchet.txt").write_text("merlin/tests/infra/test_ledgered.py\n")
    assert _errors(root) == []


def test_a_light_mention_is_not_target_heavy(tmp_path):
    root = _tree(tmp_path)
    (root / "merlin/tests/infra/test_light.py").write_text('"foo_hw"\n' * 4)
    assert _errors(root) == []


def test_the_real_tree_passes():
    errors: list[str] = []
    _gate(repo_root()).check_test_target_marker(errors)
    assert errors == []
