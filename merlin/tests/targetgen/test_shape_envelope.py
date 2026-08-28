"""A unit's declared shape envelope, finally consultable by routing.

`SemanticCapability` has always carried ranks / batch / transpose / layouts -- the HARDWARE-truth
declaration the eligibility oracle reads as the ARR denominator. Routing could consult none of it,
because a demand had no shape beyond optional contraction extents. So a unit could declare "rank 2
only" and routing would still hand it a batched contraction, and the refusal would surface far away as
a backend that declined a shape nobody said it could not take.

`OpDemand.rank` is the missing half. These pin the property that makes adding it safe: **unknown
never narrows**. A producer that supplies no rank must not thereby make a unit refuse work it can do,
and a unit that declares no envelope must refuse nothing. Every demand producer in the tree omits the
rank today, so this is inert until one starts supplying it.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.compute_units import ComputeUnit, SemanticCapability
from merlin.targetgen.routing import OpDemand, _legal_on, _shape_envelope_ok

_I8 = {"in_fmt": "int8", "weight_fmt": "int8", "op": "matmul"}


def _unit(**caps):
    return ComputeUnit(name="u", kind="systolic", dtypes=("int8",), ops=("matmul",),
                       semantic_capabilities=(SemanticCapability(family="contraction",
                                                                 dtypes=("int8",), **caps),))


# ------------------------------------------------------------------ unknown never narrows

def test_a_demand_with_no_rank_is_never_refused_by_an_envelope():
    """The inert case, and the one every producer in the tree hits today."""
    unit = _unit(ranks=(2,), batch=False)
    assert _shape_envelope_ok(unit, OpDemand(site="x", **_I8)) is True
    assert _legal_on(unit, OpDemand(site="x", **_I8))[0] is True


def test_a_unit_declaring_no_envelope_refuses_nothing():
    bare = ComputeUnit(name="u", kind="systolic", dtypes=("int8",), ops=("matmul",))
    for rank in (2, 3, 5):
        assert _legal_on(bare, OpDemand(site="x", rank=rank, **_I8))[0] is True


# ------------------------------------------------------------------ a declared envelope is enforced

def test_a_rank_outside_the_declared_set_is_refused():
    unit = _unit(ranks=(2,))
    assert _legal_on(unit, OpDemand(site="x", rank=2, **_I8))[0] is True
    assert _legal_on(unit, OpDemand(site="x", rank=3, **_I8))[0] is False


def test_a_unit_that_declares_no_batch_refuses_a_batched_op():
    unit = _unit(batch=False)
    assert _legal_on(unit, OpDemand(site="x", rank=2, **_I8))[0] is True
    assert _legal_on(unit, OpDemand(site="x", rank=3, **_I8))[0] is False, "rank>2 carries a batch dim"


def test_batchedness_is_tri_state():
    """A unit declaring batch: False must refuse a batched op and must NOT refuse one whose
    batchedness nobody recorded."""
    assert OpDemand(site="x", **_I8).is_batched is None
    assert OpDemand(site="x", rank=2, **_I8).is_batched is False
    assert OpDemand(site="x", rank=4, **_I8).is_batched is True
    assert _legal_on(_unit(batch=False), OpDemand(site="x", **_I8))[0] is True


# ------------------------------------------------------------------ against the real targets

def test_a_real_targets_declared_ranks_are_honoured():
    from merlin.targetgen import compute_units as cu, target_registry as tr
    try:
        units = cu.compute_units(tr.load_contract("gemmini"))
    except Exception:                                     # noqa: BLE001
        pytest.skip("target not resolvable here")
    declared = {r for u in units for c in (u.semantic_capabilities or ())
                if c.family == "contraction" for r in (c.ranks or ())}
    if not declared:
        pytest.skip("this target declares no contraction rank envelope")
    outside = next((r for r in (2, 3, 4, 5) if r not in declared), None)
    if outside is None:
        pytest.skip("every probed rank is declared")
    assert any(_legal_on(u, OpDemand(site="x", rank=outside, **_I8))[0] is False for u in units)
