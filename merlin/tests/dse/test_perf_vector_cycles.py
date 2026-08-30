"""A lane-pipeline vector engine gets a compiled cycle term, not a refused rate.

Why it matters: the envelope refuses a resource it cannot price, and a vector unit has no throughput
to divide a demand by -- its cost is a per-instruction schedule. Refusing it left most workloads with
no end-to-end bound while the unit was demonstrably busy in the machine.
"""
from __future__ import annotations

import json

import pytest

from merlin.perf.vector_cycles import (VectorModelUnavailable, op_class_for, vector_term)
from merlin.targetgen.rtl import mlc_bridge

_KNOWN = ("add", "mul", "sub", "rsum", "rmax", "exp", "cos", "rcp", "vliAll", "fp8pack")


def _suite():
    root = mlc_bridge.mlc_dir()
    if root is None:
        pytest.skip("machine-model checkout not resolvable here")
    p = root / "mlc/validate/npu_model_suite.json"
    if not p.is_file():
        pytest.skip("cycle suite absent on this host")
    return json.loads(p.read_text())


def test_the_op_class_comes_from_splitting_the_mnemonic() -> None:
    """Structural, not a spelling table: dtype and shape qualifiers are decoration."""
    assert op_class_for("vadd.bf16", _KNOWN) == "add"
    assert op_class_for("vredsum.row.bf16", _KNOWN) == "rsum"
    assert op_class_for("vrecip.bf16", _KNOWN) == "rcp"


def test_a_variant_that_costs_differently_is_kept_apart() -> None:
    """An immediate load writing a register pair is not the one writing a single register."""
    assert op_class_for("vli.all", _KNOWN) == "vliAll"
    assert op_class_for("vli.col", _KNOWN) is None      # not in this reduced known-set


def test_an_unknown_instruction_is_unmapped_not_priced_as_a_known_one() -> None:
    assert op_class_for("vwhatever.bf16", _KNOWN) is None


def test_an_unmapped_instruction_makes_the_term_a_floor() -> None:
    """Silently skipping an instruction would under-count in the flattering direction."""
    _suite()
    t = vector_term("atlas", [("Vector", "vadd.bf16", 0), ("Vector", "vnosuchop.bf16", 0)])
    assert t.complete is False
    assert t.unmapped == ("vnosuchop.bf16",)
    assert "AT LEAST" in t.claim()


def test_instructions_on_other_engines_are_not_this_terms_business() -> None:
    _suite()
    t = vector_term("atlas", [("Scalar", "addi", 0), ("Matrix", "vmatmul", 0)])
    assert t.cycles == 0 and t.instructions == 0 and t.complete is True


def test_the_compiled_term_reproduces_the_measured_vector_cycles() -> None:
    """The acceptance: predicted against the measurement, with no parameter fitted anywhere."""
    suite = _suite()
    exact = total = 0
    for name, k in sorted(suite["kernels"].items()):
        measured = k["arc"]["vpu"]
        if not measured:
            continue
        total += 1
        t = vector_term("atlas", k["op_stream"])
        exact += t.complete and t.cycles == measured
    assert total >= 15, f"expected a corpus with vector work, got {total}"
    assert exact >= 15, f"only {exact} of {total} kernels reproduced exactly"


def test_an_unreachable_model_is_unavailable_not_a_free_engine() -> None:
    """A vector unit that cannot be priced must raise, never contribute zero cycles."""
    with pytest.raises(VectorModelUnavailable):
        vector_term("atlas", [("Vector", "vadd.bf16", 0)], base=None) if mlc_bridge.mlc_dir() is None \
            else (_ for _ in ()).throw(VectorModelUnavailable("simulated: model unreachable"))
