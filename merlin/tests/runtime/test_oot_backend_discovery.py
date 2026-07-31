"""Regression test for out-of-tree (OOT) runtime-backend discovery via ``MERLIN_TARGET_PATH``.

``merlin.runtime.backends.base`` lets a target package ship its own runtime backend: on the first
registry query, ``_ensure_oot_discovered()`` walks the OOT target packages reachable via
``MERLIN_TARGET_PATH`` (``target_registry.external_targets()``), reads each contract's
``plugin.backend`` module path (``target_registry.resolve(name).plugin()`` -> ``{path, backend}``),
and imports that module BY FILE PATH under the synthetic name ``merlin._oot_backends.<name>`` so its
module-level ``base.register(...)`` runs. It is additive + target-agnostic, and never runs at
``import base`` — only on the first registry query, re-scanning only when the env value changes.

We prove that plumbing hermetically, against a hand-authored fixture target package under
``merlin/tests/fixtures/oot_backend_pkg/fixture_npu/`` (``fixture_npu`` is a synthetic name — not a
shipped target). Discovery mutates process-global state (``base._REGISTRY`` + ``sys.modules``), so
each assertion runs in a FRESH interpreter via ``subprocess`` with a tailored ``MERLIN_TARGET_PATH``;
running it in-process would leak the fixture backend into every other test in the session.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir

_FIXTURE_PKG = merlin_dir() / "tests" / "fixtures" / "oot_backend_pkg" / "fixture_npu"


def _run(code: str, *, target_path: str | None) -> dict:
    """Run ``code`` in a clean interpreter and return the JSON dict it prints on the last line.

    ``target_path`` is set as ``MERLIN_TARGET_PATH`` (or removed when ``None``) so each case starts
    from a known env; nothing else about the parent env is disturbed."""
    env = dict(os.environ)
    env.pop("MERLIN_TARGET_PATH", None)
    if target_path is not None:
        env["MERLIN_TARGET_PATH"] = target_path
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    # The probe prints a single JSON line last; be robust to any preceding import chatter.
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(last)


_PROBE = """
import json, sys
import merlin.runtime.backends.base as base
result = {
    "backends": base.list_backends(),
    "oot_in_sys_modules": "merlin._oot_backends.fixture_npu" in sys.modules,
}
if "fixture_npu" in base.list_backends():
    mod = base.get_backend("fixture_npu")
    result["fixture_module_name"] = getattr(mod, "__name__", None)
    result["fixture_class"] = base.class_of("fixture_npu").value
print(json.dumps(result))
"""


def _require_fixture() -> None:
    if not (_FIXTURE_PKG / "contracts" / "target_contract.yaml").is_file():
        pytest.skip(f"fixture OOT target package missing at {_FIXTURE_PKG}")


def test_core_clean_without_target_path():
    """With ``MERLIN_TARGET_PATH`` unset, the fixture backend is absent and never imported: importing
    ``base`` must not eagerly pull in any OOT backend."""
    _require_fixture()
    res = _run(_PROBE, target_path=None)
    assert "fixture_npu" not in res["backends"], res["backends"]
    assert res["oot_in_sys_modules"] is False
    # the seeded generic-ISA backends are still there (sanity that the probe really queried the registry)
    assert "spike" in res["backends"]


def test_fixture_resolves_via_target_path():
    """With the fixture package on ``MERLIN_TARGET_PATH``, its backend self-registers: it appears in
    the registry under the right target CLASS, ``get_backend`` resolves it to the file-path-imported
    synthetic module, and the built-in backends are still present (discovery is additive)."""
    _require_fixture()
    res = _run(_PROBE, target_path=str(_FIXTURE_PKG))
    assert "fixture_npu" in res["backends"], res["backends"]
    # get_backend imported the module by file path under the stable synthetic name
    assert res["fixture_module_name"] == "merlin._oot_backends.fixture_npu"
    assert res["oot_in_sys_modules"] is True
    # addressed by target CLASS, not instance: the fixture declares itself NPU
    assert res["fixture_class"] == "npu"
    # additive — the in-tree built-ins are untouched
    assert "spike" in res["backends"]
    assert "gemmini" in res["backends"]
