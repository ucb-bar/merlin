"""The declared Python floor must be one the code can actually be parsed by.

`requires-python` said `>=3.11` while five modules used PEP 701 f-strings — reused outer quotes,
escape sequences, and line breaks inside replacement fields — which are a SyntaxError before 3.12.
An install on 3.11 would have resolved, installed, and then failed at import. Nothing compared the
declaration to the syntax actually written, and ruff could not say so either: its own
`target-version` was pinned two releases lower still, so it reported those seven working f-strings
as invalid syntax rather than reporting the floor as wrong.
"""

from __future__ import annotations

import tomllib

from merlin.common.paths import repo_root

PYPROJECT = repo_root() / "pyproject.toml"


def _config() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _floor(spec: str) -> tuple[int, int]:
    """(major, minor) from a `>=X.Y` specifier — the only form this project uses."""
    assert spec.startswith(">="), f"unexpected requires-python spec {spec!r}"
    major, _, minor = spec[2:].strip().partition(".")
    return int(major), int(minor)


def test_ruff_targets_the_version_the_project_requires():
    """A linter aimed below the floor reports live code as broken; aimed above it, it misses real
    incompatibilities. Either way its output stops being believed."""
    cfg = _config()
    major, minor = _floor(cfg["project"]["requires-python"])
    assert cfg["tool"]["ruff"]["target-version"] == f"py{major}{minor}"


def test_the_floor_admits_the_syntax_the_code_is_written_in():
    """PEP 701 arrived in 3.12 and this tree uses it, so the floor cannot be lower."""
    major, minor = _floor(_config()["project"]["requires-python"])
    assert (major, minor) >= (3, 12), (
        "merlin/python/merlin/kernels/opu_cert.py and four others use PEP 701 f-strings; a floor "
        "below 3.12 promises an install that cannot import"
    )


def test_the_interpreter_running_the_suite_satisfies_the_floor():
    import sys

    assert sys.version_info[:2] >= _floor(_config()["project"]["requires-python"])
