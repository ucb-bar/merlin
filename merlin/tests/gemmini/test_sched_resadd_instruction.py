"""The residual add as the device expresses it: LOOP_WS with one bit set.

ResNet-50 joins a residual at the end of every block, and the whole-model emission runs all of them as
host scalar code. The device needs no new instruction for it -- ``gemmini_loop_ws`` already carries an
``is_resadd`` flag, and the library's own ``sp_tiled_resadd`` emits exactly one such loop per tile.

What the flag changes is what the operands MEAN, which is why the checker could not simply be let
through. There is no contraction, so ``K`` carries no extent and must be 0 -- while ``K = 0`` must stay
refused everywhere else. And the addends never reach the mesh: ``sp_tiled_resadd`` mvins them to
``1 << (ADDR_LEN-1)`` and ``3 << (ADDR_LEN-2)``, the two halves of the ACCUMULATOR, and mvouts the sum
from the first. That is also why the library bounds its tiler at ``ACC_ROWS/2`` -- which is the same
number the header states as ``max_acc_rows``, so the existing accumulator check is already the library's
own bound and needed no resadd-specific spelling.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import TensorArg
from merlin.sched.isa import IsaError


@pytest.fixture(scope="module")
def iset():
    return base.get_backend("gemmini").sched_instruction_set()


@pytest.fixture(scope="module")
def instr(iset):
    return iset.instr("loop_ws")


@pytest.fixture(scope="module")
def sched():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


def _legal() -> dict:
    """One 16x16 addend pair stored through the accumulator readout."""
    return dict(
        I=1,
        J=1,
        K=0,
        pad_I=0,
        pad_J=0,
        pad_K=0,
        A=1,
        B=2,
        D=None,
        C=3,
        A_stride=16,
        B_stride=16,
        D_stride=0,
        C_stride=16,
        A_transpose=0,
        B_transpose=0,
        full_C=0,
        low_D=0,
        ex_accumulate=0,
        act=0,
        a_spad_id=0,
        b_spad_id=0,
        is_resadd=1,
    )


def _state() -> dict:
    return {
        "config_ld": {0: 16, 1: 16},
        "config_st": {"stride": 16, "acc_act": 0, "acc_scale": 1.0},
        "config_ex": {"A_transpose": 0, "B_transpose": 0},
    }


def test_a_residual_add_is_legal(instr):
    assert instr.check(_legal(), _state()) == []


def test_the_contraction_extent_must_be_absent_in_a_resadd(instr):
    assert any("no contraction extent" in e for e in instr.check(_legal() | {"K": 2}, _state()))


def test_a_zero_extent_stays_refused_everywhere_else(instr):
    """The guard this mode had to be threaded through, not removed: K=0 is legal ONLY for a resadd."""
    errs = instr.check(_legal() | {"is_resadd": 0, "K": 0}, _state())
    assert any("empty tile" in e for e in errs), errs


def test_every_mode_the_resadd_path_does_not_model_is_refused(instr):
    """The loop issues its own movement, so nothing downstream would catch a mode nobody reasoned about."""
    for field, value, wanted in (
        ("D", 9, "no bias"),
        ("C", None, "must store"),
        ("ex_accumulate", 1, "does not accumulate"),
        ("full_C", 1, "full_C"),
        ("low_D", 1, "low_D"),
        ("A_transpose", 1, "A_transpose"),
        ("B_transpose", 1, "B_transpose"),
    ):
        errs = instr.check(_legal() | {field: value}, _state())
        assert any(wanted in e for e in errs), (field, errs)


def test_scratchpad_id_reuse_is_still_refused_on_its_own(instr):
    """It used to share one refusal with resadd; splitting them must not have dropped it."""
    assert any("scratchpad-id" in e for e in instr.check(_legal() | {"a_spad_id": 1}, _state()))


def test_the_addends_are_not_reported_as_touching_no_bytes(instr):
    """The contraction spelling reads A and B through K, which is 0 here -- and `_extent` returns 0 for
    a zero column count. A footprint that understates a read admits a tile whose operands do not fit."""
    v = _legal() | {"I": 2, "J": 3, "A_stride": 48, "B_stride": 48, "C_stride": 48}
    fp = dict((n, b) for n, b, _ in instr.footprint(v))
    assert fp["A"] == fp["B"] == fp["C"] > 0
    assert fp["D"] == 0, "a resadd reads no bias"


# --- the recipe ----------------------------------------------------------------------------------


def test_the_recipe_is_one_loop_per_tile_behind_its_four_configurations(sched, iset):
    ops = {n: TensorArg(n, (56, 64), "i8", "write" if n == "c" else "read") for n in ("a", "b", "c")}
    k = sched.resadd_reference(name="r", rows=56, cols=64, operands=ops, relu=True, facts=iset.facts)
    assert [getattr(x, "instr", None) for x in k.body[:4]] == ["config_st", "config_ex", "config_ld", "config_ld"]
    assert check_kernel(k, iset) == []
    assert dict(k.attrs)["numerics"] == "matches the vendor library on gsim, ResNet-50 group model", (
        "the marker states what was measured"
    )


@pytest.mark.parametrize("h,channels", [(56, 256), (28, 512), (14, 1024), (7, 2048)])
def test_every_resnet50_residual_join_passes_the_static_check(sched, iset, h, channels):
    rows = h * h
    ops = {n: TensorArg(n, (rows, channels), "i8", "write" if n == "c" else "read") for n in ("a", "b", "c")}
    k = sched.resadd_reference(name="r", rows=rows, cols=channels, operands=ops, relu=True, facts=iset.facts)
    assert check_kernel(k, iset) == []


def test_the_tile_is_the_library_tilers_own(sched, iset):
    """Traced against `tiled_resadd_stride_auto`'s loop: from 224x224 it halves the row extent to 112,
    walks the column extent down by DIM until it is no longer the larger side, then halves again."""
    facts = iset.facts
    dim, budget = facts["dim"], facts["acc_rows_per_loop"]
    ti, tj = sched.choose_resadd_tiles(224, 224, facts)
    assert -(-ti // dim) * dim * -(-tj // dim) <= budget
    ref_ti, ref_tj = 224, 224

    def acc(a, b):
        return -(-a // dim) * dim * -(-b // dim)

    while acc(ref_ti, ref_tj) > budget:
        if ref_ti >= ref_tj or ref_tj <= dim:
            ref_ti //= 2
        else:
            ref_tj -= dim
    assert (ti, tj) == (ref_ti, ref_tj)


def test_a_tile_that_cannot_fit_fails_closed(sched, iset):
    """The library's own loop has no guard here and would divide to zero; this states the machine fact."""
    facts = dict(iset.facts)
    facts["acc_rows_per_loop"] = 1
    with pytest.raises(IsaError, match="accumulator rows"):
        sched.choose_resadd_tiles(64, 64, facts)


def test_an_empty_array_is_refused(sched, iset):
    with pytest.raises(IsaError, match="empty"):
        sched.choose_resadd_tiles(0, 64, iset.facts)
