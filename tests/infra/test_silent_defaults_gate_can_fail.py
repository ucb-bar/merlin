from __future__ import annotations

import importlib.util
import sys

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "checks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SD = _load("check_silent_defaults")


def _shapes(source: str) -> list[tuple[str, str]]:
    return [(f.shape, f.subject) for f in SD.findings_for_source(source, "probe.py")]


FLAGGED = {"get-default (a number)": ("def grade(r):\n    return r.get('explicit_cycles', 0)\n", "get-default")}


def test_numeric_default_reaches_the_gate_finding():
    name = "get-default (a number)"
    source, shape = FLAGGED[name]
    found = _shapes(source)
    assert shape in [s for s, _ in found], f"{name} produced no {shape} finding; got {found}"
