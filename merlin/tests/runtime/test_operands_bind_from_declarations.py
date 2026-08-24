"""The grader must not decide which semantic families the benchmark can measure.

Every operand rule in ``muon_harness.args_from_cb`` is selected by opcode NAME, so the set of gradeable
capsules was the set of opcodes someone had already written a branch for. That is a closed vocabulary
presented as a general grader, and it mis-measured conformant backends twice this session:

  * ``RP10_gemv_batched_fp16_pt`` spelled a rank-3 contraction ``MATMUL``; it matched no branch and was
    reported as "no operand rule for the command shape ['MATMUL']" for two full A/B runs, while the exact
    same computation spelled ``MATMUL_BATCHED`` graded fine.
  * An earlier message blamed the command shape outright -- the one thing IDENTICAL between a submission
    that graded and one that did not. Fixing only the wording moved a baseline arm from 0/35 to 26/35.

The consequence is bigger than a missed capsule: a semantic family with no branch here cannot be graded
at all, so authoring a reduction or movement capsule would score a CORRECT submission as a failure. The
sibling target's oracle has zero opcode literals (``merlin/targets/gemmini`` binds through the cb's own
declarations), which is the proof that the general form works and this one had drifted.

These tests pin the two properties that keep the fallback honest: it binds from declarations for opcodes
nobody has heard of, and it REFUSES rather than guessing when a produced shape is not declared.
"""
from __future__ import annotations

import importlib

import pytest

try:
    from merlin.runtime.backends import base as _bk
    _MH = importlib.import_module(_bk.get_backend("muon").__name__ + ".muon_harness")
except Exception:  # noqa: BLE001 — SIMT backend absent in this env
    _MH = None

pytestmark = pytest.mark.skipif(_MH is None, reason="SIMT backend not present in this env")


def _cb(opcode, operands, tensors, *, canon=None):
    cb = {"abi_version": "1", "target": "radiance", "tensors": tensors,
          "commands": [{"opcode": opcode, "operands": operands}]}
    if canon:
        cb["canonical_inputs"] = canon
    return cb


def _t(shape, role, dtype="f32"):
    return {"shape": list(shape), "dtype": dtype, "role": role}


# ---------------------------------------------------------------------------------------------------
# 1. an opcode with no branch is still gradeable
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("opcode", ["REDUCE", "REDUCE_SUM", "MOVEMENT", "MVIN_MVOUT", "WHATEVER_V2"])
def test_an_unknown_opcode_binds_from_its_declarations(opcode):
    """The point of the change: the family a capsule exercises must not depend on this file's vocabulary."""
    cb = _cb(opcode, {"src": "X", "dst": "Y0"},
             {"X": _t([4, 8], "input"), "Y0": _t([4, 1], "output")})
    got = _MH.args_from_cb(cb)
    assert got is not None, f"opcode {opcode!r} produced no operand binding"
    in_args, out_args = got
    assert [a.name for a in in_args] == ["X"]
    assert out_args[0].name == "Y0"
    assert (out_args[0].rows, out_args[0].cols) == (4, 1)


def test_the_output_is_found_by_role_not_by_key_name():
    """`dst`/`out`/`result` are all spellings the schema allows; role is the declared fact."""
    for key in ("dst", "out", "result", "y"):
        cb = _cb("MOVEMENT", {"src": "X", key: "Y0"},
                 {"X": _t([2, 3], "input"), "Y0": _t([2, 3], "output")})
        got = _MH.args_from_cb(cb)
        assert got is not None and got[1][0].name == "Y0", f"key {key!r} failed"


def test_weight_role_is_ordered_first():
    """kernel_abi order is [weight] ++ [lhs] ++ [output]; the binder must honor it or the kernel and the
    harness disagree on operand order and the numbers come out wrong rather than absent."""
    cb = _cb("SOMETHING", {"a": "A0", "w": "W", "dst": "Y0"},
             {"A0": _t([4, 4], "input"), "W": _t([4, 4], "weight"), "Y0": _t([4, 4], "output")})
    got = _MH.args_from_cb(cb)
    assert got is not None
    assert [a.name for a in got[0]] == ["W", "A0"]


@pytest.mark.parametrize("shape,rc", [([6], (1, 6)), ([2, 3], (2, 3)), ([2, 3, 4], (6, 4))])
def test_any_rank_folds_row_major(shape, rc):
    """The ABI takes a flat row-major buffer, so folding the leading axes is exact -- and rank must not be
    a reason to refuse, which is the defect RP10 died on."""
    assert _MH._rowcol(shape) == rc


# ---------------------------------------------------------------------------------------------------
# 2. it refuses rather than guessing
# ---------------------------------------------------------------------------------------------------
def test_an_undeclared_output_shape_is_refused_not_inferred():
    """Inferring a PRODUCED shape from the inputs is exactly the per-op knowledge the fallback exists to
    avoid. Guessing here would manufacture a wrong answer, which is worse than an honest refusal."""
    cb = _cb("MOVEMENT", {"src": "X", "dst": "Y0"},
             {"X": _t([4, 8], "input"), "Y0": {"dtype": "f32", "role": "output"}})
    assert _MH.bind_from_declarations(cb, {"X": None}, lambda nm, t: [0.0] * 32) is None


def test_no_output_role_is_refused():
    cb = _cb("MOVEMENT", {"src": "X", "dst": "Y0"},
             {"X": _t([4, 8], "input"), "Y0": _t([4, 8], "input")})
    assert _MH.bind_from_declarations(cb, {"X": None}, lambda nm, t: [0.0] * 32) is None


def test_a_cb_with_no_tensors_is_refused():
    """`tensors` is optional in the schema but required to grade; with none there is nothing to bind."""
    cb = {"abi_version": "1", "target": "radiance", "tensors": {},
          "commands": [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y0"}}]}
    assert _MH.bind_from_declarations(cb, {}, lambda nm, t: [0.0]) is None


# ---------------------------------------------------------------------------------------------------
# 3. the existing rules still win — this change is additive
# ---------------------------------------------------------------------------------------------------
def test_a_plain_matmul_still_takes_its_own_rule():
    """The fallback must not shadow a branch that already derives; contraction operand ORDER and the
    produced (M, N) come from the matmul rule, not from the declarations."""
    cb = _cb("MATMUL", {"lhs": "A0", "rhs": "W", "dst": "Y0"},
             {"A0": _t([4, 8], "input"), "W": _t([8, 2], "weight"), "Y0": _t([4, 2], "output")})
    got = _MH.args_from_cb(cb)
    assert got is not None
    assert got[1][0].name == "Y0" and (got[1][0].rows, got[1][0].cols) == (4, 2)
