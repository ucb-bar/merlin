"""The optional observer receives the store reset without inventing execution state."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
from merlin_timing.capacities import apply

MODULE = Path(__file__).resolve().parents[3] / "examples/gemmini/phase2/el4_20261018/reset_facts.py"
SPEC = importlib.util.spec_from_file_location("selected_reset_facts", MODULE)
RESET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESET)


def derive(tmp_path, text):
    source = tmp_path / "target.fir"
    source.write_text(text)
    return RESET.accumulator_activation_reset(source, hashlib.sha256(source.read_bytes()).hexdigest(), "a" * 64)


def test_only_store_reset_fills_declared_hardware_slot(tmp_path):
    facts = derive(
        tmp_path,
        """circuit Target :
  module StoreController :
    regreset activation : UInt<3>, clock, reset, UInt<3>(0h0)
  module ExecuteController :
    reg activation : UInt<3>, clock
""",
    )
    profile = {"resources": {}, "target": {"hardware": {"reset_accumulator_activation": None}}}
    updated = apply(profile, facts, profile_sha256="a" * 64)
    assert updated["target"]["hardware"]["reset_accumulator_activation"] == 0
    assert profile["target"]["hardware"]["reset_accumulator_activation"] is None


@pytest.mark.parametrize(
    "declaration",
    [
        "reg activation : UInt<3>, clock",
        "regreset activation : UInt<3>, clock, reset, UInt<3>(0h1)",
        "regreset activation : UInt<2>, clock, reset, UInt<2>(0h0)",
    ],
)
def test_unreviewed_or_uninitialized_store_reset_refuses(tmp_path, declaration):
    with pytest.raises(ValueError, match="reviewed"):
        derive(tmp_path, "  module StoreController :\n    " + declaration + "\n")


def test_source_change_refuses(tmp_path):
    source = tmp_path / "target.fir"
    source.write_text("  module StoreController :\n")
    with pytest.raises(ValueError, match="FIRRTL changed"):
        RESET.accumulator_activation_reset(source, "b" * 64, "a" * 64)
