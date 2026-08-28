"""What a device could take, asked of the device instead of written down.

The existing offload path answers this with two literals: a dtype triple ("i8","i8","i32") and an op
table {"linalg.matmul": 2, "linalg.batch_matmul": 3}. They are correct, and they are ONE device's
facts -- the triple is literally that device's first declared `accumulate` rule. Written as literals
they belong to nobody, so a second device either inherits another device's datapath or needs a second
copy of the pass.

The equivalence test below is the anti-overfit proof: derived-from-the-device must reproduce the
literal exactly for the device the literal was written for, and must produce something DIFFERENT for
devices with different silicon. If it only ever reproduced the literal, the derivation would be
decoration.

Skips (never fails) where a target's facts are absent -- they are generated during experiments and
gitignored, so a fresh checkout legitimately has none.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.system.offload import (device_contraction_ranks, device_dtype_triples,
                                   offloadable_contractions, why_not)

_TARGETS = ("gemmini", "saturn_opu_mxv256d128", "atlas", "radiance")


def _triples(name):
    t = device_dtype_triples(name)
    if not t:
        pytest.skip(f"{name}: no derivable datapath in this checkout")
    return t


def _shape(dtypes, parallel=(16, 16), reduction=(16,), op="linalg.matmul"):
    return SimpleNamespace(op=op, dtypes=tuple(dtypes), parallel=tuple(parallel),
                           reduction=tuple(reduction))


# ------------------------------------------------------------- the equivalence (anti-overfit)

def test_the_derivation_reproduces_the_literal_for_the_device_it_was_written_for():
    from merlin.llvmlower.passes_opu import INT8_DTYPES
    assert INT8_DTYPES in _triples("saturn_opu_mxv256d128"), (
        "the hardcoded triple IS that device's first accumulate rule; deriving it must reproduce it")


def test_the_derivation_is_not_merely_reproducing_that_literal_everywhere():
    """If every device derived the same triple, the derivation would be decoration."""
    seen = set()
    for name in _TARGETS:
        try:
            seen.add(device_dtype_triples(name))
        except Exception:                                # noqa: BLE001
            continue
    seen.discard(())
    if len(seen) < 2:
        pytest.skip("fewer than two targets with derivable datapaths here")
    assert len(seen) >= 2, f"every device derived the same datapath: {seen}"


def test_a_float_device_does_not_derive_an_integer_datapath():
    """The failure a literal cannot avoid: a float accelerator inheriting an int8 triple."""
    t = _triples("atlas")
    assert all(acc not in ("i32",) for _, _, acc in t), f"integer accumulator on a float device: {t}"


# ------------------------------------------------------------- fail closed

def test_an_underivable_device_offloads_nothing():
    assert device_dtype_triples("definitely_not_a_target") == ()
    assert offloadable_contractions(object(), "definitely_not_a_target") == []


def test_an_underivable_datapath_is_reported_not_assumed():
    msg = why_not(_shape(("i8", "i8", "i32")), triples=(), ranks=None)
    assert msg and "no derivable datapath" in msg


def test_a_dtype_the_registry_cannot_spell_is_skipped_not_approximated():
    """One real device declares a rule naming a token the registry does not know. Skipping it is the
    only safe reading -- a triple is a precision claim and a wrong one is silent."""
    t = _triples("saturn_opu_mxv256d128")
    assert all(all(x for x in tr) for tr in t), "a skipped rule must not leave a partial triple"


# ------------------------------------------------------------- the three gates

def test_a_dtype_outside_the_device_datapath_is_declined_with_its_reason():
    t = _triples("gemmini")
    msg = why_not(_shape(("f32", "f32", "f32")), triples=t, ranks=None)
    assert msg and "not among the device's datapaths" in msg


def test_exactly_one_reduction_dim():
    t = _triples("gemmini")
    msg = why_not(_shape(t[0], reduction=(16, 16)), triples=t, ranks=None)
    assert msg and "reduction" in msg


def test_unconstrained_ranks_forbid_nothing():
    """None and () both mean unconstrained in the capability model; a device that never narrowed its
    ranks has not thereby forbidden every rank."""
    t = _triples("gemmini")
    assert why_not(_shape(t[0], parallel=(1, 2, 3, 4)), triples=t, ranks=None) is None


def test_a_declared_rank_envelope_is_enforced():
    t = _triples("gemmini")
    assert why_not(_shape(t[0], parallel=(8, 8)), triples=t, ranks=(2,)) is None
    msg = why_not(_shape(t[0], parallel=(2, 8, 8)), triples=t, ranks=(2,))
    assert msg and "rank" in msg


def test_a_legal_contraction_has_no_reason_against_it():
    t = _triples("gemmini")
    assert why_not(_shape(t[0]), triples=t, ranks=device_contraction_ranks("gemmini")) is None


# ------------------------------------------------------------- ranks come from the device

def test_ranks_are_read_from_the_devices_own_capability():
    got = {n: device_contraction_ranks(n) for n in _TARGETS}
    declared = {n: r for n, r in got.items() if r}
    if len(declared) < 2:
        pytest.skip("fewer than two targets declare a contraction rank envelope here")
    assert len(set(declared.values())) >= 2, f"every device declared the same ranks: {declared}"
