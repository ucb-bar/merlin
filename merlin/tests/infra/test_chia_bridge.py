"""The CHIA bridge must import cleanly where CHIA is absent.

CHIA lives in its own venv (``build/chia-venv``) because it hard-pins pydantic/ray[default].
The whole point of ``chia_bridge`` is that ``merlin`` keeps importing under the main ``.venv``,
where Ray is not installed. These tests run under that main ``.venv``, so they are the guarantee.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.benchharness import chia_bridge


def test_chia_run_exposes_the_aet_handle_run_id(tmp_path):
    """The CPU-host runner uses the canonical AET identity to name all subordinate evidence."""
    run = chia_bridge.ChiaRun(
        handle=SimpleNamespace(run_id="campaign__arm1__r00__seed001", run_dir=tmp_path),
        metrics=object(), profile_path=tmp_path / "profile.jsonl")
    assert run.run_id == "campaign__arm1__r00__seed001"


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import names in a module, ignoring anything nested in a def/class."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:  # body only == module level
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_module_level_chia_or_ray_import():
    """The lazy-import contract, asserted structurally so it survives a chia-venv run.

    ``chia_available()`` cannot catch a regression here once CHIA *is* installed, but this can.
    """
    src = Path(inspect.getsourcefile(chia_bridge))
    top = _module_level_imports(src)
    assert "chia" not in top, f"chia imported at module level: {sorted(top)}"
    assert "ray" not in top, f"ray imported at module level: {sorted(top)}"


def test_chia_available_is_a_bool():
    assert isinstance(chia_bridge.chia_available(), bool)


def test_require_chia_explains_the_venv_when_absent():
    if chia_bridge.chia_available():
        pytest.skip("CHIA importable here — nothing to explain")
    with pytest.raises(RuntimeError, match="chia-venv"):
        chia_bridge.require_chia()


def test_driver_python_prefers_the_main_venv():
    """Repeats must shell out under the main .venv so no ray/mcp reaches the agent's process tree."""
    from merlin.common.paths import repo_root

    got = Path(chia_bridge.driver_python())
    expected = repo_root() / ".venv" / "bin" / "python"
    if expected.is_file():
        assert got == expected
    else:
        assert got.is_file()  # fell back to the current interpreter


@pytest.mark.skipif(not chia_bridge.chia_available(), reason="needs the chia venv")
def test_aet_metrics_backend_writes_jsonl(tmp_path):
    import json

    backend = chia_bridge._aet_backend_cls()(run_dir=tmp_path)
    backend.log_scalar("repeat/wall_s", 1.5, 0)
    backend.flush()
    backend.close()

    lines = (tmp_path / "chia" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == [{"tag": "repeat/wall_s", "value": 1.5, "step": 0}]
