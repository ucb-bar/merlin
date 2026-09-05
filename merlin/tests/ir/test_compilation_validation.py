"""Validating the artifact a BACKEND emitted, against the program it was handed.

This is the check that covers a compiler we did not write. Everything before it validated our own
passes on the `interface` plane — but `interface` is the *input* a backend receives, so a capsule-bench
agent's work sat entirely downstream of what was being checked.

The negative controls are the point. A validator that has never rejected anything has not been shown
to work, and one that rejects a CORRECT backend is worse than no validator at all — so both directions
are pinned here.
"""
from __future__ import annotations

import copy

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")

#: Small on purpose. Refutation cost climbs steeply with M*K*N (measured: sat in ~2 s at 4^3, no
#: verdict at all at 16^3 within 15 minutes), so the shape is pinned rather than left to a default.
_SHAPE = (2, 2, 2)
_TIMEOUT_MS = 180_000


def _pair():
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface

    m, k, n = _SHAPE
    iface, tc = _lower_to_interface(m, k, n, 2)
    return iface, _finish_lowering(iface, tc)


def test_a_faithful_backend_is_verified():
    """The control. If this ever fails, every refutation below is meaningless."""
    from merlin.verify.refine import validate_compilation

    iface, cb = _pair()
    v = validate_compilation(iface, cb, timeout_ms=_TIMEOUT_MS)
    assert v.status == "unsat", (
        f"the in-tree pipeline's own command buffer was not verified against the interface program "
        f"it came from (status={v.status}); the validator is rejecting correct output")


def _cb_fault_names() -> list[str]:
    from merlin.verify.faults import CB_CORPUS

    return [f.name for f in CB_CORPUS]


@pytest.mark.parametrize("fault_name", _cb_fault_names())
def test_a_miscompiling_backend_is_refuted_with_a_counterexample(fault_name):
    """Each seeded backend defect must be refuted, and must come with concrete inputs.

    A refutation without a counterexample is an assertion — the model is what makes it actionable and
    what lets the defect become a graded capsule.
    """
    from merlin.verify.faults import CB_CORPUS
    from merlin.verify.refine import validate_compilation

    fault = next(f for f in CB_CORPUS if f.name == fault_name)
    iface, cb = _pair()
    cb = copy.deepcopy(cb)
    fault.mutate(cb)

    v = validate_compilation(iface, cb, timeout_ms=_TIMEOUT_MS)
    if v.status == "unknown":
        pytest.fail(
            f"solver ABSTAINED on {fault_name} at {_SHAPE} within {_TIMEOUT_MS} ms — a budget "
            f"failure, NOT the validator accepting the miscompilation. Do not read it as a "
            f"correctness result.")
    assert v.refuted, f"validator ACCEPTED a miscompiling backend ({fault.summary}): {v.status}"
    assert v.model, f"refuted {fault_name} but produced no counterexample"


def test_a_backend_using_an_unencodable_opcode_abstains_rather_than_being_refuted():
    """Incompleteness in OUR encoder must never be reported as a defect in THEIR backend.

    This is the failure mode that would make the tool worse than useless: refuting correct work
    because we cannot model an opcode the target legitimately uses.
    """
    from merlin.verify.refine import validate_compilation
    from merlin.verify.smt_semantics import UnsupportedSemantics

    iface, cb = _pair()
    cb = copy.deepcopy(cb)
    cb["commands"][1]["opcode"] = "SOFTMAX"
    with pytest.raises(UnsupportedSemantics):
        validate_compilation(iface, cb, timeout_ms=_TIMEOUT_MS)


def test_a_buffer_for_a_different_program_abstains():
    """Comparing two artifacts that are not about the same program yields no meaningful verdict."""
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface
    from merlin.verify.refine import validate_compilation
    from merlin.verify.smt_semantics import UnsupportedSemantics

    iface, _ = _pair()
    other, tc = _lower_to_interface(4, 4, 4, 2)
    with pytest.raises(UnsupportedSemantics, match="not the same program|is \\(4, 4\\)"):
        validate_compilation(iface, _finish_lowering(other, tc), timeout_ms=_TIMEOUT_MS)
