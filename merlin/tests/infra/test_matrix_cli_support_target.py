"""Execute CLI parsing without imports, builds, hardware or provider discovery."""

import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def _main(name, **globals_):
    path = repo_root() / "build_tools/scripts" / name
    tree = ast.parse(path.read_text())
    main = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][-1]
    namespace = {"argparse": argparse, "Path": Path, "__doc__": "test", **globals_}
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["main"]


@pytest.mark.parametrize(
    "script,argv",
    [
        ("firesim_sweep.py", ["--matrix-unit", "unit", "--matrix-config", "config"]),
        (
            "zephyr_firesim_leg.py",
            [
                "bundle",
                "device",
                "out",
                "--unit",
                "unit",
                "--config",
                "config",
                "--board",
                "board",
                "--package",
                "package",
            ],
        ),
        (
            "make_delivery.py",
            ["--board", "board", "--matrix-harts", "1", "--matrix-unit", "unit", "--matrix-config", "config"],
        ),
    ],
)
def test_missing_support_target_refuses_before_build_dependencies(script, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [script, *argv])
    with pytest.raises(SystemExit) as error:
        _main(script)()
    assert error.value.code == 2
    assert "--matrix-support-target" in capsys.readouterr().err


def test_sweep_passes_explicit_provider_distinct_from_unit(monkeypatch):
    seen = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep", "--matrix-unit", "unit", "--matrix-config", "config", "--matrix-support-target", "provider"],
    )
    # No bundles: parse and construct routing, never call a build.
    _main("firesim_sweep.py", zm=SimpleNamespace(MatrixRouting=lambda **kw: seen.append(kw)))()
    assert seen == [{"unit": "unit", "config": "config", "support_target": "provider"}]


@pytest.mark.parametrize("script", ["firesim_sweep.py", "zephyr_firesim_leg.py", "make_delivery.py"])
def test_every_matrix_constructor_and_delivery_dispatch_forwards_support(script):
    tree = ast.parse((repo_root() / "build_tools/scripts" / script).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            isinstance(n.func, ast.Attribute)
            and n.func.attr == "MatrixRouting"
            or isinstance(n.func, ast.Name)
            and n.func.id == "build_matrix"
        )
    ]
    assert len(calls) == (3 if script == "make_delivery.py" else 1)
    assert all("support_target" in {k.arg for k in call.keywords} for call in calls)


def test_control_leg_also_requires_explicit_audit_provider(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leg",
            "bundle",
            "control",
            "out",
            "--unit",
            "unit",
            "--config",
            "config",
            "--board",
            "board",
            "--package",
            "package",
        ],
    )
    with pytest.raises(SystemExit) as error:
        _main("zephyr_firesim_leg.py")()
    assert error.value.code == 2
    assert "--matrix-support-target" in capsys.readouterr().err


def test_delivery_routing_facts_validate_handoff_before_selected_sidecar(tmp_path):
    path = repo_root() / "build_tools/scripts/make_delivery.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_routing_facts")
    seen = []
    routing = SimpleNamespace(provider=lambda: SimpleNamespace(SIDECAR_NAME="selected.json"))

    def verify(work, selected):
        seen.append((work, selected))

    namespace = {"Path": Path, "json": json, "zm": SimpleNamespace(load_matrix_signatures=verify)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    (tmp_path / "selected.json").write_text(
        json.dumps({"count": 1, "signatures": {"s": [1]}, "routed": [{"m": 2, "n": 3, "k": 4}]})
    )
    assert namespace["_routing_facts"](tmp_path, routing)["macs_routed"] == 24
    assert seen == [(tmp_path, routing)]

    def refuse(*args):
        raise ValueError("mismatched provider")

    namespace["zm"].load_matrix_signatures = refuse
    with pytest.raises(ValueError, match="mismatched provider"):
        namespace["_routing_facts"](tmp_path, routing)
