from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common import paths

CHECKER = paths.repo_root() / "build_tools" / "scripts" / "check_inert_capabilities.py"


def _load_checker():
    """Import the gate as a module (it lives in ``build_tools/scripts``, which is not a package)."""
    spec = importlib.util.spec_from_file_location("check_inert_capabilities", CHECKER)
    assert spec and spec.loader, f"cannot load {CHECKER}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


CONV_REPORT_INERT = {
    "lower_conv_int8": {
        "generics_scanned": 280,
        "compound_map_generics": 4,
        "windowed_map_generics": 0,
        "conv_prov_ops": 190,
        "lowered": 0,
    }
}


def test_runtime_inert_flags_a_pass_that_saw_candidates_and_transformed_none(checker, tmp_path):
    rp = tmp_path / "quant_report.json"
    rp.write_text(json.dumps(CONV_REPORT_INERT), encoding="utf-8")
    found, notes = checker.detect_runtime_inert([rp])
    assert [f.ident for f in found] == ["lower_conv_int8"], (found, notes)
    assert "190" in found[0].blocked_by or "280" in found[0].blocked_by
