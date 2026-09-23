"""Dependency direction and stable identity at the optional DSE packaging seam."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root


def test_shared_apis_keep_one_identity():
    from merlin.capture import roles, shape_taxonomy
    from merlin.common.grid_search import grid_search
    from merlin.dse.search.grid import grid_search as legacy_grid
    from merlin.dse_guidance import attribution
    from merlin.dse_guidance import shape_taxonomy as legacy_shape

    assert legacy_grid is grid_search
    assert legacy_shape is shape_taxonomy
    assert attribution.role_from_fqn is roles.role_from_fqn
    assert attribution._FQN_ROLE_KEYWORDS is roles._FQN_ROLE_KEYWORDS
    assert grid_search({"n": [1, 3, 2]}, lambda point: point["n"]) == [
        {"n": 3, "score": 3},
        {"n": 2, "score": 2},
        {"n": 1, "score": 1},
    ]


def test_research_implementation_not_shipped_in_core_tree():
    for name in ("dse", "design_pressure", "dse_guidance"):
        assert not (repo_root() / "src/merlin" / name).exists()
        module = importlib.import_module("merlin." + name)
        assert Path(module.__file__).is_relative_to(repo_root() / "packages/merlin-dse/src")


def test_core_shared_apis_work_without_optional_distribution_or_site_packages():
    code = """
import importlib.util, json, sys
sys.path.insert(0, sys.argv[1])
from merlin.capture.roles import role_from_fqn
from merlin.capture.shape_taxonomy import classify_geometry
from merlin.common.grid_search import grid_search
assert importlib.util.find_spec("merlin.dse") is None
assert importlib.util.find_spec("merlin.dse_guidance") is None
assert role_from_fqn("model.action_expert.layers.0") == "repeated_head"
assert classify_geometry(1, 32, 8) == "gemv_like"
assert grid_search({"n": [1]}, lambda point: point["n"])[0]["score"] == 1
print(json.dumps({"core_only": True}))
"""
    env = dict(os.environ, PYTHONPATH="")
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(repo_root() / "src")],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"core_only": True}


def test_core_does_not_import_optional_research_packages():
    forbidden = ("merlin.dse", "merlin.dse_guidance", "merlin.design_pressure")
    found = []
    for source in (repo_root() / "src/merlin").rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                if node.module == "merlin":
                    names.extend("merlin." + alias.name for alias in node.names)
            if any(name == prefix or name.startswith(prefix + ".") for name in names for prefix in forbidden):
                found.append(f"{source.relative_to(repo_root())}:{node.lineno}")
    assert found == []


def test_legacy_cli_help_is_available_without_execution():
    for name in ("search_main", "pressure_main", "guidance_main"):
        result = subprocess.run(
            [sys.executable, "-c", f"from merlin_dse.cli import {name}; {name}()", "--help"],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
