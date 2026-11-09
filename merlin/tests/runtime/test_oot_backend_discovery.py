from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir

_FIXTURE_ROOT = merlin_dir() / "tests" / "fixtures" / "oot_backend_pkg"
_FIXTURE_PKG = _FIXTURE_ROOT / "fixture_npu"


def _run(code: str, *, target_path: str | None, targets_dir: str | None = None) -> dict:
    """Run ``code`` in a clean interpreter and return the JSON dict it prints on the last line.

    ``target_path`` is set as ``MERLIN_TARGET_PATH`` (empty, i.e. nothing selected, when ``None``) so each case starts
    from a known env; nothing else about the parent env is disturbed."""
    env = dict(os.environ)
    env["MERLIN_TARGET_PATH"] = ""
    env.pop("MERLIN_TARGETS_DIR", None)
    if targets_dir is not None:
        env["MERLIN_TARGETS_DIR"] = targets_dir
    if target_path is not None:
        env["MERLIN_TARGET_PATH"] = target_path
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 0, f"subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    last = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
    return json.loads(last)


_PROBE = '\nimport json, sys\nimport merlin.runtime.backends.base as base\nresult = {\n    "backends": base.list_backends(),\n    "oot_in_sys_modules": "merlin._oot_backends.fixture_npu" in sys.modules,\n}\nif "fixture_npu" in base.list_backends():\n    mod = base.get_backend("fixture_npu")\n    result["fixture_module_name"] = getattr(mod, "__name__", None)\n    result["fixture_class"] = base.class_of("fixture_npu").value\nprint(json.dumps(result))\n'


def _require_fixture() -> None:
    if not (_FIXTURE_PKG / "contracts" / "target_contract.yaml").is_file():
        pytest.fail(f"fixture OOT target package missing at {_FIXTURE_PKG}")


def test_fixture_resolves_via_target_path():
    """With the fixture package on ``MERLIN_TARGET_PATH``, its backend self-registers: it appears in
    the registry under the right target CLASS, ``get_backend`` resolves it to the file-path-imported
    synthetic module, and the built-in backends are still present (discovery is additive)."""
    _require_fixture()
    res = _run(_PROBE, target_path=str(_FIXTURE_PKG))
    assert "fixture_npu" in res["backends"], res["backends"]
    assert res["fixture_module_name"] == "merlin._oot_backends.fixture_npu"
    assert res["oot_in_sys_modules"] is True
    assert res["fixture_class"] == "npu"
    assert "spike" in res["backends"]


@pytest.mark.parametrize("key", ["backend"])
def test_reference_metadata_never_contributes_executable_plugins(tmp_path, key):
    package = tmp_path / "references/fixture_ref"
    (package / "contracts").mkdir(parents=True)
    (package / "contracts/target_contract.yaml").write_text(f"name: fixture_ref\nplugin:\n  {key}: forbidden.py\n")
    (package / "forbidden.py").write_text("raise AssertionError('reference code executed')\n")
    code = f'\nimport json\nfrom merlin.runtime.backends import base\nprint(json.dumps({{"plugins": [(name, str(path)) for name, path in base._oot_plugin_modules({key!r})]}}))\n'
    res = _run(code, target_path=None, targets_dir=str(package.parent))
    assert res["plugins"] == []
