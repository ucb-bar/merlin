"""One placement decision over the host and the devices together.

Three surfaces decide host-vs-device today and disagree by construction: the contract router takes
the first legal unit in declaration order, the dispatch runtime matches the IR structurally and
ignores declared units, and the offload pass carries its own dtype triple. A model can be told a
layer belongs on the accelerator by one and off it by another.

What makes them reconcilable is representing the host. Today a routing gap silently BECOMES the host
lane, so "this landed on the host because nothing could take it" is indistinguishable from "this was
placed on the host". These tests pin both halves: that the host is now a unit with a recorded reason,
and that making it one changed nothing about where work actually goes.
"""
from __future__ import annotations

import pytest

from merlin.system.model import Device, Host, System
from merlin.system.place import HOST_DEVICE, host_units, place, units_for
from merlin.targetgen.routing import OpDemand

_MM_I8 = OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="mm")
_SOFTMAX = OpDemand(op="softmax", in_fmt="fp32", weight_fmt=None, site="sm")
_MM_MX = OpDemand(op="matmul", in_fmt="mxfp4", weight_fmt="mxfp4", site="mx")


def _sys(board="chipyard_kodiak", device="gemmini"):
    from merlin.system import system_for
    from merlin.system.derive import host_from_board
    s = system_for(device)
    if not s.devices or not units_for(s):
        pytest.skip("device not resolvable in this checkout")
    return System(host=host_from_board(board), devices=s.devices)


# ------------------------------------------------------------------ the host exists now

def test_the_host_is_a_unit_not_an_absence():
    units = units_for(_sys())
    assert any(d == HOST_DEVICE for d, _ in units), "the host must be a placement candidate"
    assert any(d != HOST_DEVICE for d, _ in units), "the device must still be one"


def test_devices_are_offered_before_the_host():
    """Declaration-order selection must keep preferring a device, or this would silently move work
    off the accelerator the moment the host became representable."""
    order = [d for d, _ in units_for(_sys())]
    assert order.index(HOST_DEVICE) == len(order) - order.count(HOST_DEVICE)


def test_a_vector_unit_is_synthesized_only_where_the_board_declares_one():
    """Inventing one places vector work on a core that traps on it -- a measured hang, not a slowdown."""
    assert [u.kind for u in host_units(Host("h", vector_harts=2))] == ["scalar", "vector"]
    assert [u.kind for u in host_units(Host("h", vector_harts=0))] == ["scalar"]
    assert [u.kind for u in host_units(Host("h"))] == ["scalar"], "unknown is not a licence to assume"
    assert [u.kind for u in host_units(None)] == ["scalar"]


# ------------------------------------------------------------------ inert on today's inputs

def test_an_op_the_device_accepts_still_goes_to_the_device():
    p = place([_MM_I8], _sys())
    assert p.placed[0].on_device and p.placed[0].lane == "on_mesh"


def test_an_op_the_device_refuses_goes_to_the_host_with_a_reason():
    p = place([_SOFTMAX], _sys())
    got = p.placed[0]
    assert not got.on_device and got.lane == "scalar_rvv_lane"
    assert got.unit is not None and "no device accepted it" in got.why


# ------------------------------------------------------------------ the case that was silent

def test_an_op_nothing_can_compute_is_reported_as_emulated():
    """Neither the device nor the host natively carries this format. Today that is indistinguishable
    from an ordinary host placement; it is the lowering having to emulate it."""
    p = place([_MM_MX], _sys())
    got = p.placed[0]
    assert got.emulated is True and got.unit is None
    assert "must emulate" in got.why
    assert p.emulated() == (got,)


def test_an_ordinary_host_placement_is_not_marked_emulated():
    assert place([_SOFTMAX], _sys()).emulated() == ()


# ------------------------------------------------------------------ cost is an argument, not a pass

def test_without_a_cost_model_placement_is_declaration_order():
    p = place([_MM_I8], _sys())
    assert "declaration order" in p.placed[0].why


def test_a_cost_model_can_move_work_off_the_device():
    """The decision becomes a decision: with the device priced as expensive, the host wins."""
    def cost(_demand, unit):
        return 100.0 if unit.kind == "systolic" else 1.0
    p = place([_MM_I8], _sys(), cost=cost)
    assert not p.placed[0].on_device
    assert "lowest cost" in p.placed[0].why


def test_a_cost_model_that_prices_nothing_keeps_the_legal_choice():
    """Declining to price is not declining to run."""
    p = place([_MM_I8], _sys(), cost=lambda _d, _u: None)
    assert p.placed[0].on_device and "could be priced" in p.placed[0].why


# ------------------------------------------------------------------ reporting

def test_every_op_gets_a_placement_in_order():
    demands = [_MM_I8, _SOFTMAX, _MM_MX]
    p = place(demands, _sys())
    assert [x.demand.site for x in p.placed] == ["mm", "sm", "mx"]
    assert sum(p.lanes().values()) == 3
    assert len(p.on_device()) + len(p.on_host()) == 3


def test_a_system_with_no_devices_still_places_everything_on_the_host():
    s = System(host=Host("h", vector_harts=1), devices=())
    p = place([_MM_I8, _SOFTMAX], s)
    assert all(not x.on_device for x in p.placed)
    assert p.emulated() == (), "the host natively carries these formats"
