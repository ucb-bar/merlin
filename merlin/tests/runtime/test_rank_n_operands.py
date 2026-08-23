"""A batched operand must survive both the Tensor type and the harness's operand routing.

`RP10_gemv_batched_fp16_pt` failed for two full A/B runs, and neither failure was the submission's:

  1. Its command buffer spells a rank-3 contraction `MATMUL`. The batched operand rule triggered on the
     opcode NAME (`MATMUL_BATCHED` / `BATCHED_MATMUL`), so the identical computation was derivable under
     one spelling and reported "no operand rule for the command shape ['MATMUL']" under another. The file
     already refuses to guess operand KEY names for exactly this reason; the opcode deserved the same
     treatment. Rank is the structural fact.

  2. Underneath that, `Tensor.to_list()` did `rows, cols = self.shape` and raised on any rank-3 tensor.
     The muon harness reaches it through `_fl2`, a helper added *specifically* to be rank-agnostic after
     an earlier rank-3 crash -- so the caller was fixed while the type underneath was not, and the bug
     resurfaced three frames away as an opaque "cyclotron invocation failed: too many values to unpack".

Together they made a legal, schema-valid batched capsule ungradeable while looking like an agent defect.
"""
from __future__ import annotations

import importlib
import itertools

import pytest

from merlin.runtime.tensor import Tensor


def _flat(x):
    return list(itertools.chain.from_iterable(
        _flat(i) if isinstance(i, list) else [i] for i in x))


# --------------------------------------------------------------------------------------------
# 1. Tensor.to_list at any rank
# --------------------------------------------------------------------------------------------
def test_rank_1_and_2_are_byte_identical_to_before():
    """The fix must not perturb the shapes everything else already depends on."""
    assert Tensor((4,), list(range(4)), "f32").to_list() == [0, 1, 2, 3]
    assert Tensor((2, 3), list(range(6)), "f32").to_list() == [[0, 1, 2], [3, 4, 5]]


def test_a_rank_3_operand_serialises_instead_of_raising():
    t = Tensor((2, 3, 2), list(range(12)), "f32")
    assert t.to_list() == [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]


def test_the_batched_gemv_shape_that_broke_it():
    """RP10's actual operand: [2,16,16]. Previously ValueError, three frames from where it surfaced."""
    t = Tensor((2, 16, 16), list(range(512)), "f32")
    out = t.to_list()
    assert len(out) == 2 and len(out[0]) == 16 and len(out[0][0]) == 16


@pytest.mark.parametrize("shape,n", [((2, 3, 2), 12), ((2, 2, 2, 1), 8), ((1, 5, 3), 15)])
def test_row_major_order_is_preserved_at_every_rank(shape, n):
    """Only the nesting changes; the flat layout is what gets embedded, so it must not move."""
    assert _flat(Tensor(shape, list(range(n)), "f32").to_list()) == list(range(n))


# --------------------------------------------------------------------------------------------
# 2. batched routing keyed on rank, not on the opcode word
# --------------------------------------------------------------------------------------------
try:
    from merlin.runtime.backends import base as _bk
    _muon = _bk.get_backend("muon")
    _MH = importlib.import_module(_muon.__name__ + ".muon_harness")
except Exception:  # noqa: BLE001 — SIMT backend absent in this env
    _MH = None

pytestmark_backend = pytest.mark.skipif(_MH is None, reason="SIMT backend not present in this env")


def _batched_cb(opcode: str) -> dict:
    """RP10's shape: A0[2,16,16] @ A1[2,16,1] -> Y0[2,16,1], operands spelled lhs/rhs/dst."""
    return {
        "abi_version": "1", "target": "radiance",
        "tensors": {
            "A0": {"shape": [2, 16, 16], "dtype": "f32", "role": "input"},
            "A1": {"shape": [2, 16, 1], "dtype": "f32", "role": "weight"},
            "Y0": {"shape": [2, 16, 1], "dtype": "f32", "role": "output"},
        },
        "commands": [{"opcode": opcode, "operands": {"lhs": "A0", "rhs": "A1", "dst": "Y0"}}],
    }


@pytestmark_backend
@pytest.mark.parametrize("opcode", ["MATMUL", "MATMUL_BATCHED", "BATCHED_MATMUL"])
def test_a_rank_3_contraction_derives_under_every_spelling(opcode):
    """The whole point: the opcode word must not decide whether the capsule is gradeable."""
    derived = _MH.args_from_cb(_batched_cb(opcode))
    assert derived is not None, f"rank-3 contraction spelled {opcode!r} produced no operand rule"
    in_args, out_args = derived
    assert in_args and out_args
    assert out_args[0].name == "Y0"


@pytestmark_backend
def test_the_output_is_identified_by_its_declared_role():
    """Not by key name — the cb spells the destination `dst` here and `out` elsewhere."""
    cb = _batched_cb("MATMUL")
    cb["commands"][0]["operands"] = {"a": "A0", "w": "A1", "out": "Y0"}
    derived = _MH.args_from_cb(cb)
    assert derived is not None
    assert derived[1][0].name == "Y0"


@pytestmark_backend
def test_a_rank_2_matmul_is_untouched_by_the_rank_rule():
    """Plain 2-D matmul must keep whatever path it had; this fix is additive."""
    cb = _batched_cb("MATMUL")
    cb["tensors"] = {
        "A0": {"shape": [16, 16], "dtype": "f32", "role": "input"},
        "A1": {"shape": [16, 4], "dtype": "f32", "role": "weight"},
        "Y0": {"shape": [16, 4], "dtype": "f32", "role": "output"},
    }
    # must not raise; whether it derives is the pre-existing matmul path's business
    _MH.args_from_cb(cb)
