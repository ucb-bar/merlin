"""The CHIA bridge must import cleanly where CHIA is absent.

CHIA is an optional dependency qualified in an isolated ``chia-venv``.
The whole point of ``chia_bridge`` is that ``merlin`` keeps importing under the main ``.venv``,
where Ray is not installed. These tests run under that main ``.venv``, so they are the guarantee.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.benchharness import chia_bridge


def test_chia_run_exposes_the_aet_handle_run_id(tmp_path):
    """The CPU-host runner uses the canonical AET identity to name all subordinate evidence."""
    run = chia_bridge.ChiaRun(
        handle=SimpleNamespace(run_id="campaign__arm1__r00__seed001", run_dir=tmp_path),
        metrics=object(),
        profile_path=tmp_path / "profile.jsonl",
    )
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
    from merlin.common.paths import checkout_root

    got = Path(chia_bridge.driver_python())
    checkout = checkout_root()
    expected = checkout / ".venv" / "bin" / "python" if checkout is not None else None
    if expected is not None and expected.is_file():
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


def test_driver_uses_physical_checkout_not_workspace_override(tmp_path, monkeypatch):
    physical = tmp_path / "physical"
    python = physical / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setattr(chia_bridge, "checkout_root", lambda: physical)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "workspace"))
    monkeypatch.delenv("MERLIN_EXPERIMENT_PYTHON", raising=False)
    assert chia_bridge.driver_python() == str(python)


def test_installed_driver_defaults_to_current_python(monkeypatch):
    monkeypatch.setattr(chia_bridge, "checkout_root", lambda: None)
    monkeypatch.delenv("MERLIN_EXPERIMENT_PYTHON", raising=False)
    assert chia_bridge.driver_python() == sys.executable


@pytest.mark.parametrize(
    "variable,resolver",
    [
        ("MERLIN_CHIA_PYTHON", chia_bridge.chia_python),
        ("MERLIN_EXPERIMENT_PYTHON", chia_bridge.driver_python),
    ],
)
def test_explicit_interpreter_preserves_venv_symlink_and_rejects_invalid(
    tmp_path,
    monkeypatch,
    variable,
    resolver,
):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv(variable, str(python))
    assert resolver() == str(python)
    for invalid in ("", str(tmp_path / "missing"), str(tmp_path)):
        monkeypatch.setenv(variable, invalid)
        with pytest.raises(RuntimeError, match=variable):
            resolver()


def test_chia_interpreter_uses_canonical_build_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_CHIA_PYTHON", raising=False)
    monkeypatch.setattr(chia_bridge, "build_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="MERLIN_CHIA_PYTHON"):
        chia_bridge.chia_python()
    python = tmp_path / "chia-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert chia_bridge.chia_python() == str(python)


def test_chia_cli_interpreter_precedes_environment_and_invalid_is_not_ignored(tmp_path, monkeypatch):
    python = tmp_path / "explicit-python"
    python.symlink_to(sys.executable)
    monkeypatch.setenv("MERLIN_CHIA_PYTHON", str(tmp_path / "invalid-env"))
    assert chia_bridge.chia_python(python) == str(python)
    assert chia_bridge.chia_python(str(python)) == str(python)
    monkeypatch.setenv("MERLIN_CHIA_PYTHON", sys.executable)
    for invalid in ("", tmp_path / "missing", tmp_path):
        with pytest.raises(RuntimeError, match="explicit interpreter"):
            chia_bridge.chia_python(invalid)


def test_chia_get_delegates_public_scalar_api_without_private_profiler_types(monkeypatch):
    public = types.ModuleType("chia.base.ChiaFunction")
    seen = []
    sentinel = object()
    public.get = lambda refs, **kwargs: seen.append((refs, kwargs)) or sentinel
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", public)
    assert chia_bridge.chia_get("first", timeout=3) is sentinel
    assert seen == [("first", {"timeout": 3})]


def test_main_batch_uses_scalar_get_in_order_with_one_timeout_budget(monkeypatch):
    public = types.ModuleType("chia.base.ChiaFunction")
    seen = []
    public.get = lambda ref, **kwargs: seen.append((ref, kwargs)) or ref.upper()
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", public)
    import time

    ticks = iter([10.0, 10.0, 11.0, 13.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    assert chia_bridge.chia_get(["a", "b", "a"], timeout=5, callback=tuple) == ("A", "B", "A")
    assert seen == [("a", {"timeout": 5.0}), ("b", {"timeout": 4.0}), ("a", {"timeout": 2.0})]


def test_official_main_contract_needs_no_fork_only_extensions(monkeypatch):
    import importlib

    class PublicModule:
        def __getattr__(self, member):
            if member == "register_backend":
                raise AttributeError(member)
            return lambda *args, **kwargs: None

    def load(name):
        if name == "chia.trace.aet_sink":
            raise ModuleNotFoundError(name)
        return PublicModule()

    monkeypatch.setattr(importlib, "import_module", load)
    chia_bridge.require_chia()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), -1, True])
@pytest.mark.parametrize("refs", ["scalar", ["batch"]])
def test_invalid_timeout_refused_without_resolution(monkeypatch, timeout, refs):
    public = types.ModuleType("chia.base.ChiaFunction")
    public.get = lambda *args, **kwargs: pytest.fail("invalid timeout must not resolve")
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", public)
    with pytest.raises(ValueError, match="timeout"):
        chia_bridge.chia_get(refs, timeout=timeout)


def test_public_capability_failure_has_actionable_extra_without_private_fallback(monkeypatch):
    import importlib

    def module(name):
        # Everything is callable except the required public registration seam.
        return SimpleNamespace(
            **{
                key: lambda: None
                for key in (
                    "init",
                    "shutdown",
                    "is_initialized",
                    "cluster_resources",
                    "kill",
                    "ObjectRef",
                    "wait",
                    "get",
                    "cancel",
                    "MetricsLogger",
                    "start_collector",
                    "stop_collector",
                    "get_collector",
                    "MetricsBackend",
                )
            }
        )

    monkeypatch.setattr(importlib, "import_module", module)
    assert chia_bridge.chia_available() is False
    with pytest.raises(
        RuntimeError, match=r"missing public API chia.trace.profiler.reset_profiler.*merlin-experiments\[chia\]"
    ):
        chia_bridge.require_chia()


@pytest.mark.parametrize("missing", ["ObjectRef", "wait", "get", "cancel"])
def test_task_ownership_requires_public_ray_api(monkeypatch, missing):
    import importlib

    class PublicModule:
        def __init__(self, name):
            self.name = name

        def __getattr__(self, member):
            if self.name == "ray" and member == missing:
                raise AttributeError(member)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(importlib, "import_module", PublicModule)
    assert chia_bridge.chia_available() is False
    with pytest.raises(RuntimeError, match=rf"missing public API ray\.{missing}.*merlin-experiments\[chia\]"):
        chia_bridge.require_chia()
