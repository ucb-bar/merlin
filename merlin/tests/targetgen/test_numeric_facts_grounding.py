"""The generated numeric checker must ground its accumulator width in the target's OWN RTL facts.

Two failure modes bracket this. A baked ``or 32`` default silently hands every target gemmini's
accumulator width — the derive-vs-overfit violation the generator exists to avoid. Removing the default
without reading the fact that IS present leaves ACC_WIDTH_BITS None, and the generated checker then
fail-closed SKIPS the narrow-accumulator rule: an arm-4 lever that looks shipped and never fires.

Gemmini's facts carry the width on the accumulator DATAPATH (``i32``, evidence ``AccumulatorMem
SInt<32>``) but not as ``memories[].lane_bits``, so both sources have to be read.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.targetgen.rtl.gen_numeric_facts import _dtype_bits, generate


def _load(src: str, tmp_path):
    p = tmp_path / "numeric_facts.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("numeric_facts_under_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("numeric_facts_under_test", None)
    return mod


@pytest.mark.parametrize("dtype,bits", [("i8", 8), ("i32", 32), ("bf16", 16), ("f8E4M3FN", 8),
                                        ("f32", 32), (None, None), ("", None), ("int", None)])
def test_dtype_bits_is_structural(dtype, bits):
    assert _dtype_bits(dtype) == bits


def _facts(*, acc_dtype=None, lane_bits=None):
    mem = {"name": "accumulator", "bytes": 65536, "depth": 512}
    if lane_bits is not None:
        mem["lane_bits"] = lane_bits
    dps = [{"name": "input", "dtype": "i8"}]
    if acc_dtype:
        dps.append({"name": "accumulator", "dtype": acc_dtype})
    return {"target": "t", "facts": {"datapaths": dps, "memories": [mem]}}


def test_width_comes_from_the_datapath_when_the_memory_fact_lacks_lane_bits(tmp_path):
    mod = _load(generate(_facts(acc_dtype="i32")), tmp_path)
    assert mod.ACC_WIDTH_BITS == 32
    findings = mod.check_numeric_shapes(
        {"tensors": {"acc": {"dtype": "i8"}},
         "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]})
    assert findings, "a narrow accumulator must be flagged once the width is grounded"


def test_an_explicit_lane_bits_fact_still_wins(tmp_path):
    mod = _load(generate(_facts(acc_dtype="i32", lane_bits=24)), tmp_path)
    assert mod.ACC_WIDTH_BITS == 24


def test_a_target_with_neither_fact_fails_closed_rather_than_defaulting(tmp_path):
    mod = _load(generate(_facts()), tmp_path)
    assert mod.ACC_WIDTH_BITS is None
    assert mod.check_numeric_shapes(
        {"tensors": {"acc": {"dtype": "i8"}},
         "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]}) == [], \
        "an ungrounded width must SKIP the rule, never assume a width"


def test_a_correctly_typed_accumulator_is_clean(tmp_path):
    mod = _load(generate(_facts(acc_dtype="i32")), tmp_path)
    assert mod.check_numeric_shapes(
        {"tensors": {"acc": {"dtype": "i32"}},
         "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc"}}]}) == []
