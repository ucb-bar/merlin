"""Offloading contractions to a device from the whole-model build path.

Until now a device could only be reached by the host INTERPRETER: `kernel_backend="mesh"` is valid
only with `run="host"`, which walks the driver function in Python and ships each layer out of process
to a simulator. Nothing was emitted into a host binary, so no compiled artifact could run a model with
accelerator acceleration at all.

These tests pin the seam that changes that, and -- as importantly -- that it is INERT unless a
placement decision was actually made. A build that silently started moving contractions onto a device
would change what every existing image computes.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.device_offload import load_sidecar

_MODEL = """
module {
  func.func @forward(%a: tensor<16x32xi8>, %b: tensor<32x16xi8>) -> tensor<16x16xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x16xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<16x16xi32>) -> tensor<16x16xi32>
    %o = linalg.matmul ins(%a, %b : tensor<16x32xi8>, tensor<32x16xi8>)
                       outs(%f : tensor<16x16xi32>) -> tensor<16x16xi32>
    return %o : tensor<16x16xi32>
  }
}
"""


def _routing(**kw):
    return DeviceRouting(device=kw.pop("device", "gemmini"), package_dir="/nonexistent",
                         operand_dtype="int8", accum_dtype="i32", **kw)


def _prepare(tmp_path, device):
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering
    src = tmp_path / "model.mlir"
    src.write_text(_MODEL, encoding="utf-8")
    prepared, _features = prepare_for_lowering(src, tmp_path, blocking=False, device=device)
    return prepared.read_text(encoding="utf-8"), load_sidecar(tmp_path)


def test_no_routing_leaves_the_model_alone(tmp_path):
    """Every existing caller passes nothing here, and must be byte-identical."""
    text, side = _prepare(tmp_path, None)
    assert "linalg.matmul" in text and "func.call" not in text
    assert not side.get("signatures")


def test_a_routing_without_a_decision_is_still_inert(tmp_path):
    """The placement decision is made elsewhere and passed in. A routing that carries none must not
    cause the build to invent one."""
    text, side = _prepare(tmp_path, _routing())
    assert "linalg.matmul" in text and "func.call" not in text
    assert not side.get("signatures")


def test_a_routing_with_a_decision_moves_the_contraction(tmp_path):
    text, side = _prepare(tmp_path, _routing(select=lambda _s: True))
    assert "linalg.matmul" not in text, "the contraction should have become a call"
    assert text.count("func.call") == 1
    assert len(side.get("signatures") or {}) == 1


def test_the_offloaded_declaration_keeps_its_access_attributes(tmp_path):
    """Without these, one-shot-bufferize copies the weight operand of every routed contraction --
    silently, and at real cost in a shipped model."""
    text, _side = _prepare(tmp_path, _routing(select=lambda _s: True))
    assert text.count("bufferization.access") == 3


def test_a_device_that_declares_no_datapath_moves_nothing(tmp_path):
    """Fail closed: an underivable device offloads nothing rather than assuming a precision."""
    text, side = _prepare(tmp_path, _routing(device="definitely_not_a_target",
                                             select=lambda _s: True))
    assert "linalg.matmul" in text
    assert not side.get("signatures")


def test_the_link_refuses_offloaded_symbols_it_cannot_build(tmp_path):
    """A sidecar with signatures and no routing to build them against would link-error far away, so
    the build says which state it is in instead."""
    import inspect

    from merlin.runtime.backends import spike_model
    src = inspect.getsource(spike_model.build)
    assert "were offloaded but no `device=` routing" in src, (
        "the build must refuse offloaded signatures it has no way to build")


def test_device_objects_are_built_for_the_caller_s_isa(monkeypatch, tmp_path):
    """The device shim must be compiled for the ISA the rest of the image is built for.

    `_flags` defaulted to RISCV_FLAGS, which names `-march=rv64gcv` -- an assumption about the
    hardware. The kernels come out clean because they are translated from the target's own lowering,
    but this C shim was compiled against that default, so a whole-model image carried 224 vector
    instructions onto a Rocket whose own DTS reads `rv64imafdcbzicsr_..._xrocket`, with no `v`. It
    trapped mid-run on a `vsetvli` (`mcause=2`, mtval opcode 0x57) after 45M cycles of simulation --
    a failure that costs an hour to observe and names nothing about its cause.
    """
    from merlin.llvmlower import device_build as DB

    default = DB._flags("riscv")
    assert any(f.startswith("-march=") for f in default), default

    mine = ["--target=riscv64-unknown-elf", "-march=rv64gc_zba_zbb_zbs_zfh", "-mabi=lp64d"]
    assert DB._flags("riscv", mine) == mine
    assert not any("rv64gcv" in f for f in DB._flags("riscv", mine))


def test_build_device_objects_accepts_cflags():
    """The seam exists on the public entry point, not only on the private helper."""
    import inspect

    from merlin.llvmlower.device_build import build_device_objects

    assert "cflags" in inspect.signature(build_device_objects).parameters


# ------------------------------------------------- the decision now has a production source

def _placement_for(target, demands):
    from merlin.system.derive import system_for_experiment
    from merlin.system.place import place
    system, _why = system_for_experiment(target)
    return place(demands, system)


def _demands():
    """One contraction the mesh takes and one it cannot, at the extents `_MODEL` actually contains."""
    from merlin.targetgen.routing import OpDemand
    return [OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="mm", m=16, k=32, n=16),
            OpDemand(op="matmul", in_fmt="fp32", weight_fmt="fp32", site="host_mm", m=8, k=8, n=8)]


def test_the_selector_moves_what_the_placement_put_on_the_device_and_nothing_else():
    from merlin.kernels.shapes import contraction_shapes
    from merlin.system.place import device_selector

    placement = _placement_for("gemmini", _demands())
    on_dev = [p for p in placement.placed if p.on_device]
    if not on_dev:
        pytest.skip("no device placement derivable for gemmini in this checkout")
    select = device_selector(placement)
    shapes = contraction_shapes(_MODEL)
    assert shapes, "the fixture model must contain a contraction to decide about"
    assert [select(s) for s in shapes] == [True], (
        "the 16x32x16 i8 contraction is the one the placement put on the mesh")


def test_a_shape_the_placement_never_saw_is_not_moved():
    """Fail closed. A selector that defaulted to True would move a contraction whose placement nobody
    made -- which is exactly the silent-offload failure this seam exists to prevent."""
    from merlin.kernels.microkernel import ContractionShape
    from merlin.system.place import device_selector

    select = device_selector(_placement_for("gemmini", _demands()))
    assert select(ContractionShape(op="linalg.matmul", parallel=(999, 999), reduction=(999,))) is False


def test_one_extent_triple_placed_two_ways_is_declined_rather_than_guessed():
    """Ambiguity through the extent keying means decline, not pick-one. Once a cost model exists, the
    same M x K x N can be placed on the device at one site and on the host at another; applying one
    site's decision to the other is a silent miscompile."""
    from dataclasses import replace

    from merlin.kernels.microkernel import ContractionShape
    from merlin.system.place import Placed, Placement, device_selector

    d = _demands()[0]
    both = Placement(placed=(
        Placed(demand=d, device="gemmini", unit="mesh", lane="on_mesh", why="x"),
        Placed(demand=replace(d, site="elsewhere"), device="host", unit="host_scalar",
               lane="scalar_rvv_lane", why="y"),
    ))
    select = device_selector(both)
    assert select(ContractionShape(op="linalg.matmul", parallel=(16, 16), reduction=(32,))) is False


def test_a_routing_built_from_a_placement_carries_the_placements_own_datapath():
    """The formats are READ, never defaulted: the operand is what the router matched the contraction
    against, and the accumulate is either the rule the unit matched or -- when the contract declares no
    accumulate matrix, which the reference systolic mesh does not -- the device's own RTL datapath fact
    for that operand pair. A build that assumed either emits kernels in a precision nobody chose."""
    from merlin.llvmlower.device_build import routing_for_placement
    from merlin.system.offload import device_dtype_triples

    placement = _placement_for("gemmini", _demands())
    on_dev = [p for p in placement.placed if p.on_device]
    if not on_dev:
        pytest.skip("no device placement derivable for gemmini in this checkout")
    r = routing_for_placement(placement, on_dev[0].device, "/nonexistent")
    assert r.operand_dtype == on_dev[0].demand.in_fmt
    if on_dev[0].acc is not None:
        assert r.accum_dtype == on_dev[0].acc
    else:
        assert r.accum_dtype in {a for _i, _w, a in device_dtype_triples(on_dev[0].device)}
    assert r.select is not None


def test_an_accumulate_format_that_cannot_be_derived_refuses_rather_than_defaults():
    """Both derivations exhausted -- the unit matched no rule and the device's facts name no triple for
    this operand pair. Emitting a kernel anyway would pick the model's arithmetic by accident."""
    from merlin.llvmlower.device_build import routing_for_placement
    from merlin.system.place import Placed, Placement
    from merlin.targetgen.routing import OpDemand

    exotic = Placement(placed=(
        Placed(demand=OpDemand(op="matmul", in_fmt="mxfp4", weight_fmt="mxfp4", site="a",
                               m=16, k=32, n=16),
               device="gemmini", unit="systolic_mesh", lane="on_mesh", acc=None, why="x"),))
    with pytest.raises(ValueError, match="underivable"):
        routing_for_placement(exotic, "gemmini", "/nonexistent")


def test_a_placement_that_moved_nothing_is_a_caller_error_not_an_empty_build():
    from merlin.llvmlower.device_build import routing_for_placement
    from merlin.system.place import Placed, Placement

    empty = Placement(placed=(Placed(demand=_demands()[1], device="host", unit="host_scalar",
                                     lane="scalar_rvv_lane", why="host"),))
    with pytest.raises(ValueError, match="nothing to build"):
        routing_for_placement(empty, "gemmini", "/nonexistent")


def test_device_placements_that_disagree_about_the_datapath_refuse_to_become_one_image():
    """One ELF carries one device datapath. Picking the first and dropping the rest is how half a model
    gets computed in a precision nobody selected."""
    from merlin.llvmlower.device_build import routing_for_placement
    from merlin.system.place import Placed, Placement
    from merlin.targetgen.routing import OpDemand

    mixed = Placement(placed=(
        Placed(demand=OpDemand(op="matmul", in_fmt="int8", weight_fmt="int8", site="a",
                               m=16, k=32, n=16),
               device="gemmini", unit="mesh", lane="on_mesh", acc="i32", why="x"),
        Placed(demand=OpDemand(op="matmul", in_fmt="bf16", weight_fmt="bf16", site="b",
                               m=16, k=32, n=32),
               device="gemmini", unit="mesh", lane="on_mesh", acc="f32", why="x"),
    ))
    with pytest.raises(ValueError, match="disagree about the datapath"):
        routing_for_placement(mixed, "gemmini", "/nonexistent")


def test_the_offload_rewrite_accepts_the_placement_derived_selector(tmp_path):
    """End to end through the seam a whole-model build uses: a placement decides, the rewrite moves the
    contraction, and the sidecar the device build reads names a signature. Without this the fused path
    is reachable in principle and unreached in fact."""
    from merlin.llvmlower.device_build import routing_for_placement

    placement = _placement_for("gemmini", _demands())
    on_dev = [p for p in placement.placed if p.on_device]
    if not on_dev:
        pytest.skip("no device placement derivable for gemmini in this checkout")
    routing = routing_for_placement(placement, on_dev[0].device, "/nonexistent")
    text, side = _prepare(tmp_path, routing)
    assert side.get("signatures"), "the placement's decision must reach the device sidecar"
    assert "func.call" in text and "linalg.matmul" not in text
