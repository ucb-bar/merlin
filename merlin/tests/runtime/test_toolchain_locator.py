"""A cross-toolchain prefix is located from what a target contract DECLARES, never a vendor literal.

``merlin.baselines.toolchain_locator`` replaced three copies of one resolver, each of which spelled the
vendor's extracted-release directory name. These tests pin its search order, its refusal to pick a
directory the declaration does not name, its behaviour with no declaration, and the one real
declaration the board adapter depends on at import time.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from merlin.baselines import toolchain_locator as TL
from merlin.common.paths import repo_root

BOARD_ENV = "MERLIN_K1_TOOLCHAIN"


def _prefix(root: Path, *tools: str) -> Path:
    for tool in tools:
        path = root / "bin" / tool
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return root


def _declare(monkeypatch, env_var: str, globs: tuple[str, ...]) -> None:
    decl = TL.ToolchainDeclaration(target="t", env=env_var, release_dirs=globs)
    monkeypatch.setattr(TL, "declared", lambda var: decl if var == env_var else None)


def test_an_install_that_is_itself_a_prefix_wins(tmp_path, monkeypatch):
    _declare(monkeypatch, "X_TC", ("vendor-*",))
    _prefix(tmp_path, "clang")
    _prefix(tmp_path / "vendor-1", "clang")
    assert TL.find_prefix(tmp_path, "X_TC") == tmp_path


def test_declared_release_dirs_are_searched_one_level_then_two(tmp_path, monkeypatch):
    _declare(monkeypatch, "X_TC", ("vendor-*",))
    both = ("clang", "clang++")
    deep = _prefix(tmp_path / "a" / "vendor-2", *both)
    assert TL.find_prefix(tmp_path, "X_TC", tools=both) == deep
    shallow = _prefix(tmp_path / "vendor-1", *both)
    assert TL.find_prefix(tmp_path, "X_TC", tools=both) == shallow


def test_a_directory_outside_the_declared_naming_is_never_picked(tmp_path, monkeypatch):
    _declare(monkeypatch, "X_TC", ("vendor-*",))
    _prefix(tmp_path / "some-other-llvm", "clang")
    assert TL.find_prefix(tmp_path, "X_TC") is None


def test_a_prefix_missing_a_requested_tool_is_not_a_prefix(tmp_path, monkeypatch):
    _declare(monkeypatch, "X_TC", ("vendor-*",))
    _prefix(tmp_path / "vendor-1", "clang")
    assert TL.find_prefix(tmp_path, "X_TC", tools=("clang", "clang++")) is None


def test_without_a_declaration_only_an_explicit_prefix_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(TL, "declared", lambda _var: None)
    nested = _prefix(tmp_path / "vendor-1", "clang")
    assert TL.find_prefix(tmp_path, "X_TC") is None
    assert TL.find_prefix(nested, "X_TC") == nested
    assert TL.default_install("X_TC") is None


def test_the_board_toolchain_is_declared_by_exactly_one_contract():
    decl = TL.declared(BOARD_ENV)
    assert decl is not None, "no (or more than one) target contract declares the board toolchain"
    assert decl.release_dirs and decl.default_install
    assert TL.default_install(BOARD_ENV) == repo_root() / decl.default_install


def test_board_adapter_imports_with_the_variable_unset():
    """Unset, the adapter reads its default install from the contract while it is being imported. That
    lookup must not re-enter the adapter, and must land on the declared default."""
    code = textwrap.dedent(f"""
        import os
        os.environ.pop({BOARD_ENV!r}, None)
        import merlin.common.paths as paths
        paths._dotenv = lambda: {{}}
        from merlin.baselines import toolchain_locator as TL
        import merlin.mining.k1 as board
        assert board.K1_TOOLCHAIN == TL.default_install({BOARD_ENV!r}), board.K1_TOOLCHAIN
        print("ok")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]
