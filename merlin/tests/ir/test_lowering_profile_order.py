"""Profile products and stage boundaries belong to the current lowering call."""

import hashlib
import json
from pathlib import Path

import pytest

from merlin.llvmlower import op_profile, toolchain
from merlin.llvmlower.lower import lower_model

SOURCE = """module { func.func @forward(%x: f32) -> f32 {
  %one = arith.constant 1.0 : f32
  %sum = arith.addf %x, %one : f32
  return %sum : f32
} }"""


@pytest.mark.parametrize("selection", [None, 1, "yes"])
def test_profile_selection_requires_a_boolean_before_creating_work(tmp_path, selection):
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="boolean"):
        lower_model(SOURCE, work, targets=(), op_profile=selection)
    assert not work.exists()


def test_profile_products_are_removed_for_a_new_unselected_or_failed_call(tmp_path, monkeypatch):
    if not toolchain.available():
        pytest.skip("selected upstream toolchain unavailable")
    profiled = lower_model(SOURCE, tmp_path, targets=(), op_profile=True)
    report = profiled.stats["operation_profile"]
    assert report == json.loads((tmp_path / "operation-profile.json").read_text())
    assert report["source_file_sha256"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert "@merlin_prof_mark" in profiled.ll_path.read_text()
    for product in ("table", "instrumented_source"):
        assert hashlib.sha256(Path(report[product]["path"]).read_bytes()).hexdigest() == report[product]["sha256"]
    plain = lower_model(SOURCE, tmp_path, targets=())
    assert "operation_profile" not in plain.stats and "@merlin_prof_mark" not in plain.ll_path.read_text()
    names = ("operation-profile.json", "model.profiled.mlir", "op_profile_table.json")
    assert all(not (tmp_path / name).exists() for name in names)
    for name in names:
        (tmp_path / name).write_text("previous compilation")

    def refused(*args, **kwargs):
        raise op_profile.OpProfileError("selected instrumentation refused")

    monkeypatch.setattr(op_profile, "instrument", refused)
    with pytest.raises(op_profile.OpProfileError, match="refused"):
        lower_model(SOURCE, tmp_path, targets=(), op_profile=True)
    assert all(not (tmp_path / name).exists() for name in names)
