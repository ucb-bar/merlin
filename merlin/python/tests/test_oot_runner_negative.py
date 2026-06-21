"""K10: broken packages must fail closed, with the expected plane + FailureCategory."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.targetgen.oot_runner import certify

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "broken_packages"
G0 = REPO / "bench_contract" / "examples" / "g0_matmul.interface.mlir"
EXPECTED = yaml.safe_load((FIXTURES / "EXPECTED.yaml").read_text())


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_broken_package_fails_closed(name, tmp_path):
    exp = EXPECTED[name]
    res = certify(FIXTURES / name, G0, runs_root=tmp_path, run_id=f"neg_{name}", simulator="spike")
    assert res["status"] == "fail", f"{name} did not fail closed: {res}"
    assert res["failure"] is not None
    assert res["failure"]["plane"] == exp["plane"]
    assert res["failure"]["category"] == exp["category"]
