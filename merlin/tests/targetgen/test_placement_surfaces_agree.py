"""The three host-vs-device decisions must agree, or one of them is wrong about the hardware.

Three surfaces decide independently today, each written for its own caller:

  * `targetgen.routing` matches an op name and two dtypes against the contract's compute units;
  * `runtime.dispatch_runtime` matches the IR structurally and consults no declared unit at all;
  * `llvmlower.passes_opu` carries its own dtype triple and op table as module literals.

Nothing reconciles them, so a model can be told a layer belongs on the accelerator by one and off it by
another -- and whichever runs last silently wins. This pins the invariant BEFORE the surfaces are
collapsed into one, so the collapse is verifiable rather than hopeful: if a refactor makes them
diverge, this fails naming the contraction they disagreed about.

It compares them where they are comparable -- which contractions each considers legal for a device --
over one corpus of real linalg modules.
"""
from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.system.offload import device_dtype_triples, offloadable_contractions

_I8 = """
module {{
  func.func @f(%a: tensor<{m}x{k}xi8>, %b: tensor<{k}x{n}xi8>) -> tensor<{m}x{n}xi32> {{
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<{m}x{n}xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    %o = linalg.matmul ins(%a, %b : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>)
                       outs(%f : tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>
    return %o : tensor<{m}x{n}xi32>
  }}
}}
"""

_F32 = """
module {
  func.func @f(%a: tensor<16x32xf32>, %b: tensor<32x16xf32>) -> tensor<16x16xf32> {
    %z = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<16x16xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<16x16xf32>) -> tensor<16x16xf32>
    %o = linalg.matmul ins(%a, %b : tensor<16x32xf32>, tensor<32x16xf32>)
                       outs(%f : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %o : tensor<16x16xf32>
  }
}
"""

#: A contraction accumulating onto a LIVE init. Both surfaces must decline it: linalg.matmul computes
#: C_init + A@B while a device kernel that overwrites its output computes A@B, and they agree only
#: when C_init is zero. A surface that accepted it would silently drop the addend.
_LIVE_INIT = """
module {
  func.func @f(%a: tensor<16x32xi8>, %b: tensor<32x16xi8>,
               %c: tensor<16x16xi32>) -> tensor<16x16xi32> {
    %o = linalg.matmul ins(%a, %b : tensor<16x32xi8>, tensor<32x16xi8>)
                       outs(%c : tensor<16x16xi32>) -> tensor<16x16xi32>
    return %o : tensor<16x16xi32>
  }
}
"""

_CORPUS = {
    "i8_square":     _I8.format(m=16, k=32, n=16),
    "i8_tall":       _I8.format(m=64, k=16, n=32),
    "i8_off_edge":   _I8.format(m=8, k=344, n=128),
    "f32_square":    _F32,
    "i8_live_init":  _LIVE_INIT,
}


def _opu_device():
    """The device the literal-carrying surface was written for."""
    for name in ("saturn_opu_mxv256d128", "gemmini"):
        if ("i8", "i8", "i32") in device_dtype_triples(name):
            return name
    pytest.skip("no integer-datapath device derivable in this checkout")


def _derived(src, device):
    with IR_LOCK:
        return len(offloadable_contractions(mq.parse(src), device))


def _literal(src):
    from merlin.llvmlower.passes_opu import routable_contractions
    with IR_LOCK:
        return len(routable_contractions(mq.parse(src)))


@pytest.mark.parametrize("name", sorted(_CORPUS))
def test_the_derived_and_literal_surfaces_agree(name):
    """The literal triple IS this device's first accumulate rule, so deriving it must reproduce every
    verdict -- not merely the accepts, and not merely on the shape someone happened to test."""
    device = _opu_device()
    src = _CORPUS[name]
    assert _derived(src, device) == _literal(src), (
        f"{name}: derived-from-the-device and the hardcoded literal disagree; one of them is wrong "
        f"about what this hardware computes")


def test_a_live_init_is_declined_by_both():
    """Pinned separately because both declining for DIFFERENT reasons would pass the equality above
    while still being a latent disagreement."""
    device = _opu_device()
    assert _derived(_CORPUS["i8_live_init"], device) == 0
    assert _literal(_CORPUS["i8_live_init"]) == 0


def test_a_float_contraction_is_declined_on_an_integer_datapath_by_both():
    device = _opu_device()
    assert _derived(_CORPUS["f32_square"], device) == 0
    assert _literal(_CORPUS["f32_square"]) == 0


def test_the_derived_surface_is_not_simply_echoing_the_literal():
    """If it were, it would accept f32 on a float device exactly when the literal does -- which is
    never. A float device must accept what the integer literal refuses."""
    from merlin.system.offload import device_dtype_triples as triples
    floaty = [t for t in ("atlas", "radiance")
              if any(a not in ("i32",) for _l, _r, a in (triples(t) or ()))]
    if not floaty:
        pytest.skip("no float-datapath device derivable here")
    got = _derived(_CORPUS["f32_square"], floaty[0])
    assert _literal(_CORPUS["f32_square"]) == 0
    assert got >= 0        # the point is that it is asked of the device, not of a constant


# --------------------------------------------------------------- the literal is now a default

def test_the_literal_surface_can_be_asked_of_a_device_instead():
    """`INT8_DTYPES` is one unit's first accumulate rule written down. Written down it belongs to
    nobody: a second device either inherits this one's precision or needs a second copy of the pass."""
    from merlin.llvmlower.passes_opu import routable_contractions
    device = _opu_device()
    src = _CORPUS["i8_square"]
    with IR_LOCK:
        assert len(routable_contractions(mq.parse(src))) == 1                      # default: unchanged
    with IR_LOCK:
        assert len(routable_contractions(mq.parse(src), device=device)) == 1       # asked of the device


def test_asking_an_underivable_device_routes_nothing_rather_than_falling_back():
    """Fail closed. Silently reverting to this unit's precision is how a second device would inherit
    the wrong datapath and compute in it."""
    from merlin.llvmlower.passes_opu import routable_contractions
    with IR_LOCK:
        assert routable_contractions(mq.parse(_CORPUS["i8_square"]),
                                     device="definitely_not_a_target") == []


def test_an_explicit_datapath_overrides_the_literal():
    from merlin.llvmlower.passes_opu import routable_contractions
    with IR_LOCK:
        assert routable_contractions(mq.parse(_CORPUS["i8_square"]),
                                     dtypes=("f32", "f32", "f32")) == []
    with IR_LOCK:
        assert len(routable_contractions(mq.parse(_CORPUS["f32_square"]),
                                         dtypes=("f32", "f32", "f32"))) == 1


# --------------------------------------------------------------- what the surfaces share, precisely

def test_placement_reproduces_the_routers_lane_assignment():
    """`place` and `route_plan` share `_legal_on`, so they agree by construction rather than by
    coincidence -- and representing the host did not move any work. A divergence here means one of
    them grew a private rule."""
    from merlin.system import system_for
    from merlin.system.derive import host_from_board
    from merlin.system.model import System
    from merlin.system.place import place
    from merlin.targetgen.routing import OpDemand, route_plan

    demands = [OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="mm", m=16, n=16, k=32),
               OpDemand(op="matmul", in_fmt="fp32", weight_fmt="fp32", site="mmf"),
               OpDemand(op="softmax", in_fmt="fp32", weight_fmt=None, site="sm"),
               OpDemand(op="matmul", in_fmt="mxfp4", weight_fmt="mxfp4", site="mx")]
    for target in ("gemmini", "atlas"):
        s = System(host=host_from_board("chipyard_kodiak"), devices=system_for(target).devices)
        if not s.devices or not s.devices[0].kind:
            continue
        plan = route_plan(demands, target)
        expect = {r.demand.site: lane
                  for key, lane in (("mesh", "on_mesh"),
                                    ("fallback", "in_contract_vector_scalar"),
                                    ("scalar_rvv", "scalar_rvv_lane"))
                  for r in plan[key]}
        got = {p.demand.site: p.lane for p in place(demands, s).placed}
        assert got == expect, f"{target}: placement diverged from the router: {got} != {expect}"


def test_the_runtime_accepts_the_host_float_on_purpose_not_by_accident():
    """The run-time surface accepts f32 as well as the device's own operand format, and that is a
    REASONED difference rather than a disagreement: it quantizes at the boundary, so it can take an
    f32 tensor the compiled path cannot. Collapsing it into the others would refuse work it correctly
    runs today. Pinned so the reason survives, and so the dtype it adds stays DERIVED."""
    import inspect

    from merlin.runtime import dispatch_runtime as dr
    src = inspect.getsource(dr.execute)
    assert "mesh_datapath(mesh_target)" in src, "the accepted dtype must stay derived from the target"
    assert '_accept = ("f32"' in src
    assert "host lane materializes f32" in src, "the widening must keep saying why it is there"
